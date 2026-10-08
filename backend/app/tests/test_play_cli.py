"""The terminal client logic, driven through the API with scripted input."""
import importlib.util
import time
from pathlib import Path

import chess
import pytest
from fastapi.testclient import TestClient

from app import config
from app.main import app
from app.services import game_sessions, model_cache
from app.services.game_sessions import SessionManager
from app.services.model_cache import LoadedBot, ModelCache

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "play_cli.py"
_spec = importlib.util.spec_from_file_location("play_cli", _SCRIPT)
play_cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(play_cli)

client = TestClient(app)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path)
    d = tmp_path / "players" / "botplayer"
    d.mkdir(parents=True)
    (d / "model.pt").write_bytes(b"weights")
    monkeypatch.setattr(
        model_cache, "default_cache",
        ModelCache(loader=lambda u, v: LoadedBot(u, object(), object(), v, time.time())),
    )

    class Scripted:
        def __init__(self, ucis):
            self.ucis = list(ucis)

        def __call__(self, bot, board, meta, t):
            return chess.Move.from_uci(self.ucis.pop(0))

    def install(ucis):
        monkeypatch.setattr(game_sessions, "default_manager", SessionManager(mover=Scripted(ucis)))

    return install


def run(inputs):
    feed = iter(inputs)
    out = []
    state = play_cli.play(
        client, "botplayer", input_fn=lambda prompt: next(feed), print_fn=out.append
    )
    return state, "\n".join(out)


def test_cli_plays_a_game_to_checkmate(setup):
    setup(["e7e5", "d8h4"])
    state, text = run(["f2f3", "g2g4"])
    assert state["status"] == "finished"
    assert "Bot plays Qh4#" in text
    assert "Game over: 0-1 (checkmate)" in text


def test_cli_reports_bad_moves_and_continues(setup):
    setup(["e7e5"])
    state, text = run(["nonsense", "e2e5", "e2e4", "q"])
    assert "! " in text                      # error lines were printed
    assert "Bot plays e5" in text
    assert len(state["moves"]) == 2


def test_cli_quits(setup):
    setup([])
    state, text = run(["q"])
    assert "Bye!" in text and state["moves"] == []


def test_cli_missing_model(setup, monkeypatch):
    out = []
    state = play_cli.play(client, "ghost_player", input_fn=lambda p: "q", print_fn=out.append)
    assert state is None
    assert "Could not start a game: 404" in "\n".join(out)


def test_render_board_orientation():
    start = chess.STARTING_FEN
    white_view = play_cli.render_board(start, "white").split("\n")
    black_view = play_cli.render_board(start, "black").split("\n")
    assert white_view[0].startswith("8  r n b q k b n r")
    assert black_view[0].startswith("1  R N B K Q B N R")
    assert white_view[-1].strip() == "a b c d e f g h"
    assert black_view[-1].strip() == "h g f e d c b a"
