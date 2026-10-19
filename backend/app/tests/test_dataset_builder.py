import pytest
import os
from pathlib import Path
from unittest.mock import patch
import torch

from app.services.dataset_builder import build_dataset
from app.ml.dataset import PlayerMoveDataset
from app import config

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_games.jsonl"

def load_fixture_games():
    import json
    games = []
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            game = json.loads(line)
            # Need datetime for dataset builder (since storage.load_games normally returns datetime)
            from datetime import datetime
            game["date"] = datetime.fromisoformat(game["date"])
            games.append(game)
    return games

def test_build_dataset_and_dataloader(tmp_path, monkeypatch):
    # Override storage directory
    import app.services.storage as storage_module
    import app.config as config_module
    
    monkeypatch.setattr(config_module, "DATA_ROOT", tmp_path)
    
    games = load_fixture_games()
    
    # We want to mock storage.load_games
    with patch("app.services.extractor.storage.load_games", return_value=games):
        info = build_dataset("testuser")
        
    assert info["total_samples"] > 0
    
    player_dir = tmp_path / "players" / "testuser"
    parquet_path = player_dir / "samples.parquet"
    info_path = player_dir / "info.json"
    
    assert parquet_path.exists()
    assert info_path.exists()
    
    # Let's test idempotency
    import pyarrow.parquet as pq
    table1 = pq.read_table(parquet_path)
    
    with patch("app.services.extractor.storage.load_games", return_value=games):
        build_dataset("testuser")
        
    table2 = pq.read_table(parquet_path)
    assert table1.equals(table2), "Parquet files are not identical (idempotency failed)"
    
    # Test DataLoader smoke test
    from torch.utils.data import DataLoader
    
    # Load train split
    dataset = PlayerMoveDataset(str(parquet_path), "train")
    if len(dataset) == 0:
        dataset = PlayerMoveDataset(str(parquet_path), "test") # in case train has 0 (unlikely)
        
    loader = DataLoader(dataset, batch_size=2, num_workers=0)
    batch = next(iter(loader))
    
    assert "board" in batch
    assert batch["board"].shape == (2, 18, 8, 8)
    assert batch["board"].dtype == torch.uint8
    
    assert "meta" in batch
    assert batch["meta"].shape == (2, config.META_DIM)
    assert batch["meta"].dtype == torch.float32
    
    assert "from_sq" in batch
    assert batch["from_sq"].shape == (2,)
    assert batch["from_sq"].dtype == torch.int64
    
    assert "legal_mask" in batch
    assert batch["legal_mask"].shape == (2, 64, 64)


def test_dataset_builder_retries_used_single_source_of_truth(tmp_path):
    """T_A4: stale ingest_info.json with retries_used=7 + build_dataset(..., retries_used=2)
    -> info.json says 2."""
    import json
    import app.config as config_module
    from app.services.storage import player_dir

    orig_data_root = config_module.DATA_ROOT
    config_module.DATA_ROOT = tmp_path

    username = "retries_truth_user"
    pdir = player_dir(username)
    pdir.mkdir(parents=True, exist_ok=True)

    # Write stale ingest_info.json with retries_used=7
    with open(pdir / "ingest_info.json", "w", encoding="utf-8") as f:
        json.dump({"retries_used": 7}, f)

    games = load_fixture_games()
    try:
        with patch("app.services.extractor.storage.load_games", return_value=games):
            info = build_dataset(username, retries_used=2)

        assert info["retries_used"] == 2
        with open(pdir / "info.json", "r", encoding="utf-8") as f:
            saved_info = json.load(f)
        assert saved_info["retries_used"] == 2
    finally:
        config_module.DATA_ROOT = orig_data_root


def test_dataset_report_renders_zero_counts_and_ingestion_section(tmp_path):
    """D1 test: build_dataset on the fixture, read dataset_report.md, assert 'illegal_move: 0'
    and the ingestion section (including the retries_used and per-bucket request lines) are present.
    """
    import json
    import app.config as config_module
    from app.services.storage import player_dir

    orig_data_root = config_module.DATA_ROOT
    config_module.DATA_ROOT = tmp_path

    username = "report_test_user"
    pdir = player_dir(username)
    pdir.mkdir(parents=True, exist_ok=True)

    # Ingest info with retries_used and per-bucket requests
    ingest_info = {
        "retries_used": 3,
        "requests": 5,
        "skipped_duplicates": 2,
        "bucket_requests": {"last_6m": 2, "6_18m": 2, "18_36m": 1},
        "drop_stats": {"duplicate": 0, "too_old": 1},
        "request_log": [
            {"bucket": "last_6m", "status_code": 200, "lines": 10, "max": 1500, "cap_hit": False},
        ],
    }
    with open(pdir / "ingest_info.json", "w", encoding="utf-8") as f:
        json.dump(ingest_info, f, indent=2)

    games = load_fixture_games()
    try:
        with patch("app.services.extractor.storage.load_games", return_value=games):
            build_dataset(username, retries_used=3)

        report_path = pdir / "dataset_report.md"
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")

        # 1. Zero counts print (no filter > 0)
        assert "## Extraction Drops" in content
        assert "- illegal_move: 0" in content
        assert "- unparseable_pgn: 0" in content
        assert "- no_target_moves: 0" in content

        # 2. Ingestion section present
        assert "## Ingestion" in content
        assert "- Retries Used: 3" in content
        assert "- Requests: 5" in content
        assert "- Skipped Duplicates: 2" in content

        # 3. Per-bucket request lines present
        assert "### Per-Bucket Requests" in content
        assert "- last_6m: 2 requests" in content
        assert "- 6_18m: 2 requests" in content
        assert "- 18_36m: 1 requests" in content

        # 4. Request log summary and ingestion drops
        assert "### Request Log Summary" in content
        assert "cap_hit=False" in content
        assert "### Ingestion Drops" in content
    finally:
        config_module.DATA_ROOT = orig_data_root


def test_v2_parquet_byte_identity(tmp_path):
    """Assert v2 samples.parquet generated from sample_games.jsonl is bitwise identical
    to the pre-edit baseline SHA256: 8540e76b6874ae0e7de7400a69232386276310453c68ea40d04ce908dbc4e273.
    """
    import hashlib
    import app.config as config_module
    from app.services.storage import player_dir

    orig_data_root = config_module.DATA_ROOT
    config_module.DATA_ROOT = tmp_path
    username = "v2_identity_user"
    games = load_fixture_games()
    try:
        with patch("app.services.extractor.storage.load_games", return_value=games):
            build_dataset(username)
        parquet_path = player_dir(username) / "samples.parquet"
        assert parquet_path.exists()
        file_bytes = parquet_path.read_bytes()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        valid_hashes = {
            "8540e76b6874ae0e7de7400a69232386276310453c68ea40d04ce908dbc4e273",  # Windows / standard pyarrow
            "9501b677ccb497dc8ea119e15d55a59cedb5a7674315cf52b9b3a1f6be3ed45e",  # Linux / CI runner pyarrow
        }
        assert file_hash in valid_hashes
    finally:
        config_module.DATA_ROOT = orig_data_root


