"""
Tests for app/routers/train.py.
"""
import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app
from app import config
from app.routers.train import _train_jobs
from app.routers.ingest import _jobs as _ingest_jobs
from app.services.storage import player_dir

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_job_states():
    _train_jobs.clear()
    _ingest_jobs.clear()
    yield
    _train_jobs.clear()
    _ingest_jobs.clear()


def _setup_ingested_user(tmp_path, username="test_trainer", model_tier="small"):
    pdir = player_dir(username, create=True)
    info_path = pdir / "info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump({"total_samples": 500, "model_tier": model_tier}, f)
    samples_path = pdir / "samples.parquet"
    samples_path.write_bytes(b"dummy parquet bytes")
    return pdir


def test_train_invalid_username(tmp_path, monkeypatch):
    """FIX 4: 422 on invalid username before touching state or disk."""
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path)
    for bad in ["a", "bad name", "a.b", "x" * 31, "-abc"]:
        resp_post = client.post(f"/api/train/{bad}")
        assert resp_post.status_code == 422
        resp_get = client.get(f"/api/train_status/{bad}")
        assert resp_get.status_code == 422


def test_train_404_never_ingested(tmp_path, monkeypatch):
    """FIX 4: 404 if info.json or samples.parquet missing ('never ingested')."""
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path)
    resp = client.post("/api/train/nonexistent_user")
    assert resp.status_code == 404
    assert "never been ingested" in resp.json()["detail"]


def test_train_400_rejected_tier(tmp_path, monkeypatch):
    """FIX 4: 400 if model_tier == 'rejected'."""
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path)
    _setup_ingested_user(tmp_path, username="rejected_user", model_tier="rejected")
    resp = client.post("/api/train/rejected_user")
    assert resp.status_code == 400
    assert "rejected" in resp.json()["detail"]


def test_train_409_while_ingest_active(tmp_path, monkeypatch):
    """FIX 4: 409 if an ingest job for that user is in ingest._ACTIVE_STATUSES."""
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path)
    _setup_ingested_user(tmp_path, username="busy_user")
    _ingest_jobs["busy_user"] = {"status": "downloading"}

    resp = client.post("/api/train/busy_user")
    assert resp.status_code == 409
    assert "Ingest job" in resp.json()["detail"]


def test_train_409_while_train_active(tmp_path, monkeypatch):
    """FIX 4: 409 if a train job is already active."""
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path)
    _setup_ingested_user(tmp_path, username="training_user")
    _train_jobs["training_user"] = {"status": "running"}

    resp = client.post("/api/train/training_user")
    assert resp.status_code == 409
    assert "Train job" in resp.json()["detail"]


def test_train_success_and_completed_payload(tmp_path, monkeypatch):
    """FIX 4: 202 status code and completed payload with training metadata."""
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path)
    _setup_ingested_user(tmp_path, username="happy_user")

    fake_result = {
        "epochs_run": 10,
        "best_val_loss": 3.1415,
        "device": "cpu",
        "model_tier": "small",
        "created_at": "2026-10-09T00:00:00Z",
    }

    with patch("app.routers.train.train_player_model", return_value=fake_result):
        resp = client.post("/api/train/happy_user")
        assert resp.status_code == 202
        assert resp.json()["status"] == "queued"

    # Background task finishes under TestClient
    status_resp = client.get("/api/train_status/happy_user")
    assert status_resp.status_code == 200
    data = status_resp.json()
    assert data["status"] == "completed"
    assert data["epochs_run"] == 10
    assert data["best_val_loss"] == 3.1415
    assert data["device"] == "cpu"
    assert data["model_tier"] == "small"
    assert data["created_at"] == "2026-10-09T00:00:00Z"


def test_train_failed_path_preserves_error(tmp_path, monkeypatch):
    """FIX 4: Failed path when train_player_model raises; error text preserved."""
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path)
    _setup_ingested_user(tmp_path, username="fail_train_user")

    with patch("app.routers.train.train_player_model", side_effect=RuntimeError("GPU explosive failure")):
        resp = client.post("/api/train/fail_train_user")
        assert resp.status_code == 202

    status_resp = client.get("/api/train_status/fail_train_user")
    assert status_resp.status_code == 200
    data = status_resp.json()
    assert data["status"] == "failed"
    assert "GPU explosive failure" in data["error"]


def test_train_status_disk_fallback_and_not_trained(tmp_path, monkeypatch):
    """FIX 4: Disk fallback reads training_info.json, returns not_trained if only info.json,
    and 404 if no user data exists."""
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path)

    # 1. Disk fallback when training_info.json exists
    pdir_disk = _setup_ingested_user(tmp_path, username="disk_user")
    with open(pdir_disk / "training_info.json", "w", encoding="utf-8") as f:
        json.dump({
            "epochs_run": 8,
            "best_val_loss": 2.718,
            "device": "cuda",
            "model_tier": "small",
            "created_at": "2026-10-09T00:00:00Z",
        }, f)

    _train_jobs.pop("disk_user", None)
    resp_disk = client.get("/api/train_status/disk_user")
    assert resp_disk.status_code == 200
    data_disk = resp_disk.json()
    assert data_disk["status"] == "completed"
    assert data_disk["source"] == "disk"
    assert data_disk["epochs_run"] == 8
    assert data_disk["best_val_loss"] == 2.718
    assert data_disk["device"] == "cuda"

    # 2. Ingested only (info.json exists, but not training_info.json) -> not_trained
    _setup_ingested_user(tmp_path, username="untrained_user")
    _train_jobs.pop("untrained_user", None)
    resp_untrained = client.get("/api/train_status/untrained_user")
    assert resp_untrained.status_code == 200
    assert resp_untrained.json()["status"] == "not_trained"

    # 3. Neither exists -> 404
    resp_ghost = client.get("/api/train_status/ghost_user")
    assert resp_ghost.status_code == 404
