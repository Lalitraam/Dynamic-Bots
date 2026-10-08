import json

import pytest

from app import config
from app.services import readiness
from app.services.player_paths import InvalidUsernameError


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path)
    return tmp_path


def _player_dir(root, name="alice"):
    d = root / "players" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_never_ingested_and_creates_nothing(data_root):
    res = readiness.get_readiness("Alice")
    assert res["state"] == "never_ingested"
    assert res["ready"] is False
    assert res["username"] == "alice"
    assert not (data_root / "players").exists()


def test_invalid_username_raises(data_root):
    with pytest.raises(InvalidUsernameError):
        readiness.get_readiness("../hack")


def test_ingesting_from_active_job(data_root):
    res = readiness.get_readiness("alice", ingest_job={"status": "downloading"})
    assert res["state"] == "ingesting"
    assert res["ingest_status"] == "downloading"
    assert res["ready"] is False


def test_ingest_failed_reports_error(data_root):
    job = {"status": "failed", "error": "Not enough valid games: 12 (minimum 150)"}
    res = readiness.get_readiness("alice", ingest_job=job)
    assert res["state"] == "ingest_failed"
    assert "Not enough valid games" in res["detail"]


def test_completed_job_without_disk_data_is_never_ingested(data_root):
    res = readiness.get_readiness("alice", ingest_job={"status": "completed"})
    assert res["state"] == "never_ingested"


def test_rejected_tier_on_disk(data_root):
    d = _player_dir(data_root)
    (d / "info.json").write_text(json.dumps({"model_tier": "rejected"}))
    res = readiness.get_readiness("alice")
    assert res["state"] == "rejected"
    assert res["model_tier"] == "rejected"
    assert res["ready"] is False


def test_ingested_not_trained(data_root):
    d = _player_dir(data_root)
    (d / "info.json").write_text(json.dumps({"model_tier": "standard"}))
    res = readiness.get_readiness("alice")
    assert res["state"] == "ingested_not_trained"
    assert res["model_tier"] == "standard"


def test_corrupt_info_json_counts_as_ingested_not_trained(data_root):
    d = _player_dir(data_root)
    (d / "info.json").write_text("{not json")
    res = readiness.get_readiness("alice")
    assert res["state"] == "ingested_not_trained"
    assert res["model_tier"] is None


def test_ready_when_model_exists(data_root):
    d = _player_dir(data_root)
    (d / "info.json").write_text(json.dumps({"model_tier": "full"}))
    (d / "model.pt").write_bytes(b"weights")
    res = readiness.get_readiness("alice")
    assert res["state"] == "ready"
    assert res["ready"] is True
    assert res["model_tier"] == "full"


def test_empty_model_file_is_not_ready(data_root):
    d = _player_dir(data_root)
    (d / "info.json").write_text(json.dumps({"model_tier": "full"}))
    (d / "model.pt").write_bytes(b"")
    res = readiness.get_readiness("alice")
    assert res["state"] == "ingested_not_trained"


def test_model_wins_over_active_ingest_job(data_root):
    d = _player_dir(data_root)
    (d / "model.pt").write_bytes(b"weights")
    res = readiness.get_readiness("alice", ingest_job={"status": "downloading"})
    assert res["state"] == "ready"
    assert res["ingest_status"] == "downloading"


def test_disk_data_wins_over_failed_job(data_root):
    d = _player_dir(data_root)
    (d / "info.json").write_text(json.dumps({"model_tier": "small"}))
    res = readiness.get_readiness(
        "alice", ingest_job={"status": "failed", "error": "network down"}
    )
    assert res["state"] == "ingested_not_trained"


def test_training_info_tier_preferred(data_root):
    d = _player_dir(data_root)
    (d / "info.json").write_text(json.dumps({"model_tier": "standard"}))
    (d / "training_info.json").write_text(json.dumps({"model_tier": "full"}))
    (d / "model.pt").write_bytes(b"weights")
    assert readiness.get_readiness("alice")["model_tier"] == "full"
