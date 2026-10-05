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

def test_build_dataset_and_dataloader(tmp_path):
    # Override storage directory
    import app.services.storage as storage_module
    import app.config as config_module
    
    orig_data_root = config_module.DATA_ROOT
    config_module.DATA_ROOT = tmp_path
    
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
    assert batch["legal_mask"].dtype == torch.bool
    
    config_module.DATA_ROOT = orig_data_root
