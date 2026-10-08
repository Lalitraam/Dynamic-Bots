"""
Full-game integration tests through the HTTP API (Milestone 4, Phase 4).

Every response is checked against an independent python-chess "shadow" board,
so any drift between the server's board and the moves it reports is caught.
"""
import random
import threading
import time

import chess
import pytest
from fastapi.testclient import TestClient

from app import config
from app.main import app
from app.services import game_sessions, model_cache, player_paths
from app.services.game_sessions import SessionManager
from app.services.model_cache import LoadedBot, ModelCache

client = TestClient(app)


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def fake_cache(monkeypatch):
    cache = ModelCache(
        max_size=3,
        loader=lambda username, version: LoadedBot(username, object(), object(), version, time.time()),
    )
    monkeypatch.setattr(model_cache, "default_cache", cache)
    return cache


@pytest.fixture
def trained_player(data_root, fake_cache):
    d = data_root / "players" / "botplayer"
    d.mkdir(parents=True)
    (d / "model.pt").write_bytes(b"weights")
    return "botplayer"


def _check_against_shadow(state, shadow):
    """Bring the shadow board up to date with the reported moves and compare."""
    for record in state["moves"][shadow.ply():]:
        shadow.push_uci(record["uci"])
    assert len(state["moves"]) == shadow.ply()
    assert state["fen"] == shadow.fen()
    assert state["turn"] == ("white" if shadow.turn == chess.WHITE else "black")
    if state["human_to_move"]:
        assert sorted(state["legal_moves"]) == sorted(m.uci() for m in shadow.legal_moves)
    outcome = shadow.outcome(claim_draw=True)
    assert (state["status"] == "finished") == (outcome is not None)
    if outcome is not None:
        assert state["result"] == outcome.result()


@pytest.mark.parametrize("color", ["white", "black"])
def test_random_game_stays_consistent_with_python_chess(trained_player, monkeypatch, color):
    rng = random.Random(2024)

    def random_bot(bot, board, meta, temperature):
        return rng.choice(list(board.legal_moves))

    monkeypatch.setattr(game_sessions, "default_manager", SessionManager(mover=random_bot))

    state = client.post(
        "/api/play/sessions", json={"username": trained_player, "human_color": color}
    ).json()
    shadow = chess.Board()
    _check_against_shadow(state, shadow)

    for _ in range(120):
        if state["status"] == "finished":
            break
        assert state["human_to_move"] is True
        uci = rng.choice(state["legal_moves"])
        resp = client.post(f"/api/play/sessions/{state['session_id']}/moves", json={"move": uci})
        assert resp.status_code == 200, resp.text
        state = resp.json()
        assert state["moves"][shadow.ply()]["uci"] == uci      # our move was applied as sent
        _check_against_shadow(state, shadow)

    assert len(state["moves"]) >= 2


def test_simultaneous_identical_moves_over_http(trained_player, monkeypatch):
    def slow_bot(bot, board, meta, temperature):
        time.sleep(0.15)
        return next(iter(board.legal_moves))

    monkeypatch.setattr(game_sessions, "default_manager", SessionManager(mover=slow_bot))
    game = client.post("/api/play/sessions", json={"username": trained_player}).json()
    url = f"/api/play/sessions/{game['session_id']}/moves"
    barrier = threading.Barrier(2)
    statuses = []

    def attempt():
        local = TestClient(app)
        barrier.wait()
        statuses.append(local.post(url, json={"move": "e2e4"}).status_code)

    threads = [threading.Thread(target=attempt) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(statuses) == [200, 400]
    final = client.get(f"/api/play/sessions/{game['session_id']}").json()
    assert len(final["moves"]) == 2                    # one human move + one bot reply


def test_two_games_do_not_interfere(trained_player, monkeypatch):
    monkeypatch.setattr(
        game_sessions, "default_manager",
        SessionManager(mover=lambda bot, board, meta, t: next(iter(board.legal_moves))),
    )
    a = client.post("/api/play/sessions", json={"username": trained_player}).json()
    b = client.post("/api/play/sessions", json={"username": trained_player}).json()
    client.post(f"/api/play/sessions/{a['session_id']}/moves", json={"move": "e2e4"})
    assert client.get(f"/api/play/sessions/{b['session_id']}").json() == b


# ---------------------------------------------------------------------------
# Real model end to end (needs torch; skipped otherwise)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("color", ["white", "black"])
def test_real_untrained_model_plays_legal_game(data_root, monkeypatch, color):
    pytest.importorskip("torch")
    from app.ml import export
    from app.ml.model import ChessPolicyNet

    model = ChessPolicyNet(
        channels=config.MODEL_CHANNELS,
        num_res_blocks=config.MODEL_RES_BLOCKS,
        hidden_dim=config.MODEL_HIDDEN_DIM,
        dropout=config.MODEL_DROPOUT,
    )
    export.save_model(model, player_paths.model_path("realbot"))

    monkeypatch.setattr(model_cache, "default_cache", ModelCache())     # real loader
    monkeypatch.setattr(game_sessions, "default_manager", SessionManager())  # real mover

    ready = client.get("/api/play/realbot/ready").json()
    assert ready["state"] == "ready"

    state = client.post(
        "/api/play/sessions",
        json={"username": "realbot", "human_color": color, "temperature": 0.8},
    ).json()
    shadow = chess.Board()
    _check_against_shadow(state, shadow)

    for _ in range(8):
        if state["status"] == "finished":
            break
        uci = state["legal_moves"][0]
        resp = client.post(f"/api/play/sessions/{state['session_id']}/moves", json={"move": uci})
        assert resp.status_code == 200, resp.text
        state = resp.json()
        _check_against_shadow(state, shadow)

    assert len(state["moves"]) >= 2
    assert all(m["by"] in ("human", "bot") for m in state["moves"])
