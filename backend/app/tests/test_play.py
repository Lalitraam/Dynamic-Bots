"""Endpoint tests for the play router (Milestone 4)."""
import json
import time

import pytest
from fastapi.testclient import TestClient

from app import config
from app.main import app
from app.routers.ingest import _jobs
import chess

from app.services import game_sessions, model_cache
from app.services.game_sessions import SessionManager
from app.services.model_cache import LoadedBot, ModelCache, ModelLoadError

client = TestClient(app)


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def fake_cache(monkeypatch):
    """Replace the process-wide model cache with one that uses a fake loader."""
    state = {"fail": False}

    def loader(username, version):
        if state["fail"]:
            raise ModelLoadError("checkpoint is corrupt")
        return LoadedBot(username, object(), object(), version, time.time())

    cache = ModelCache(max_size=3, loader=loader)
    monkeypatch.setattr(model_cache, "default_cache", cache)
    cache.state = state
    return cache


# ---------------------------------------------------------------------------
# Phase 1: readiness endpoint
# ---------------------------------------------------------------------------

def test_ready_unknown_player(data_root, fake_cache):
    resp = client.get("/api/play/never_seen_player/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["state"] == "never_ingested"
    assert data["ready"] is False
    # Probing an unknown name must not create a folder.
    assert not (data_root / "players").exists()


@pytest.mark.parametrize("name", ["x", "bad.name", "a" * 31])
def test_ready_invalid_username_400(data_root, fake_cache, name):
    resp = client.get(f"/api/play/{name}/ready")
    assert resp.status_code == 400


def test_ready_is_case_insensitive(data_root, fake_cache):
    d = data_root / "players" / "mixed_case"
    d.mkdir(parents=True)
    (d / "info.json").write_text(json.dumps({"model_tier": "small"}))
    resp = client.get("/api/play/Mixed_Case/ready")
    assert resp.status_code == 200
    assert resp.json()["username"] == "mixed_case"
    assert resp.json()["state"] == "ingested_not_trained"


def test_ready_reflects_active_ingest_job(data_root, fake_cache, monkeypatch):
    monkeypatch.setitem(_jobs, "busy_player", {"status": "downloading"})
    resp = client.get("/api/play/busy_player/ready")
    assert resp.status_code == 200
    assert resp.json()["state"] == "ingesting"


# ---------------------------------------------------------------------------
# Phase 2: readiness verifies the model loads (via the cache)
# ---------------------------------------------------------------------------

def test_ready_trained_player(data_root, fake_cache):
    d = data_root / "players" / "trained_player"
    d.mkdir(parents=True)
    (d / "info.json").write_text(json.dumps({"model_tier": "full"}))
    (d / "model.pt").write_bytes(b"weights")

    resp = client.get("/api/play/trained_player/ready")

    assert resp.status_code == 200
    data = resp.json()
    assert data["state"] == "ready"
    assert data["ready"] is True
    assert data["model_tier"] == "full"
    assert fake_cache.keys() == ["trained_player"]    # cache warmed


def test_ready_model_unreadable(data_root, fake_cache):
    d = data_root / "players" / "broken_player"
    d.mkdir(parents=True)
    (d / "model.pt").write_bytes(b"weights")
    fake_cache.state["fail"] = True

    resp = client.get("/api/play/broken_player/ready")

    assert resp.status_code == 200
    data = resp.json()
    assert data["state"] == "model_unreadable"
    assert data["ready"] is False
    assert "corrupt" in data["detail"]


# ---------------------------------------------------------------------------
# Phase 3: game sessions (create + get)
# ---------------------------------------------------------------------------

def _first_legal_mover(bot, board, meta, temperature):
    return next(iter(board.legal_moves))


def _failing_mover(bot, board, meta, temperature):
    raise RuntimeError("model exploded")


@pytest.fixture
def fake_manager(monkeypatch):
    manager = SessionManager(mover=_first_legal_mover)
    monkeypatch.setattr(game_sessions, "default_manager", manager)
    return manager


@pytest.fixture
def trained_player(data_root, fake_cache):
    d = data_root / "players" / "botplayer"
    d.mkdir(parents=True)
    (d / "model.pt").write_bytes(b"weights")
    return "botplayer"


def test_create_session_as_white(trained_player, fake_manager):
    resp = client.post("/api/play/sessions", json={"username": "BotPlayer"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["username"] == "botplayer"
    assert data["human_color"] == "white" and data["human_to_move"] is True
    assert data["status"] == "active" and data["moves"] == []
    assert len(data["legal_moves"]) == 20
    assert data["temperature"] == config.INFERENCE_TEMPERATURE_DEFAULT
    assert len(fake_manager) == 1


def test_create_session_as_black_bot_opens(trained_player, fake_manager):
    resp = client.post(
        "/api/play/sessions",
        json={"username": trained_player, "human_color": "black", "temperature": 0.5},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["human_color"] == "black" and data["bot_color"] == "white"
    assert len(data["moves"]) == 1 and data["moves"][0]["by"] == "bot"
    assert data["last_bot_move"] == data["moves"][0]
    assert data["temperature"] == 0.5


def test_create_session_random_color(trained_player, fake_manager):
    resp = client.post(
        "/api/play/sessions", json={"username": trained_player, "human_color": "random"}
    )
    assert resp.status_code == 201
    assert resp.json()["human_color"] in ("white", "black")


def test_create_session_without_model_404(data_root, fake_cache, fake_manager):
    resp = client.post("/api/play/sessions", json={"username": "ghost_player"})
    assert resp.status_code == 404
    assert "ready" in resp.json()["detail"]
    assert len(fake_manager) == 0


def test_create_session_invalid_username_400(data_root, fake_cache, fake_manager):
    resp = client.post("/api/play/sessions", json={"username": "../hack"})
    assert resp.status_code == 400


def test_create_session_corrupt_model_500(trained_player, fake_cache, fake_manager):
    fake_cache.state["fail"] = True
    resp = client.post("/api/play/sessions", json={"username": trained_player})
    assert resp.status_code == 500
    assert len(fake_manager) == 0


@pytest.mark.parametrize(
    "body",
    [
        {"username": "botplayer", "temperature": 0.0},
        {"username": "botplayer", "temperature": 9.0},
        {"username": "botplayer", "human_color": "purple"},
        {"username": "botplayer", "human_rating": 5},
        {"username": "botplayer", "bot_rating": 99999},
        {"human_color": "white"},
    ],
)
def test_create_session_validates_body_422(trained_player, fake_manager, body):
    resp = client.post("/api/play/sessions", json=body)
    assert resp.status_code == 422


def test_create_session_bot_failure_when_opening_500(trained_player, monkeypatch):
    manager = SessionManager(mover=_failing_mover)
    monkeypatch.setattr(game_sessions, "default_manager", manager)
    resp = client.post(
        "/api/play/sessions", json={"username": trained_player, "human_color": "black"}
    )
    assert resp.status_code == 500
    assert len(manager) == 0                      # nothing half-created


def test_create_session_cap_503(trained_player, monkeypatch):
    manager = SessionManager(mover=_first_legal_mover, max_sessions=1)
    monkeypatch.setattr(game_sessions, "default_manager", manager)
    assert client.post("/api/play/sessions", json={"username": trained_player}).status_code == 201
    assert client.post("/api/play/sessions", json={"username": trained_player}).status_code == 503


def test_get_session_roundtrip(trained_player, fake_manager):
    created = client.post("/api/play/sessions", json={"username": trained_player}).json()
    resp = client.get(f"/api/play/sessions/{created['session_id']}")
    assert resp.status_code == 200
    assert resp.json() == created


def test_get_unknown_session_404(fake_manager):
    resp = client.get("/api/play/sessions/doesnotexist")
    assert resp.status_code == 404
