"""Endpoint tests for the play router (Milestone 4)."""
import json
import time

import pytest
from fastapi.testclient import TestClient

from app import config
from app.main import app
from app.routers.ingest import _jobs
from app.services import model_cache
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
