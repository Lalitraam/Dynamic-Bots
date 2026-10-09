"""
Tests for app/services/training_runner.py and app/ml/train.py (Milestone 3).

Focus:
- train_player_model end-to-end smoke test on sample fixture (epochs=2).
- Artifact verification (model.pt, training_info.json, training_report.md).
- Numeric integrity (finite float loss, no NaNs).
- Empty val_loader graceful handling.
- model_tier == 'rejected' gating.
"""

import json
from pathlib import Path
from unittest.mock import patch

import chess
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from app import config
from app.ml import export
from app.ml.model import ChessPolicyNet
from app.ml.train import train_model
from app.services.dataset_builder import build_dataset
from app.services.training_runner import train_player_model

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_games.jsonl"


def load_fixture_games():
    """Load games from test fixture JSONL with parsed dates."""
    from datetime import datetime

    games = []
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            game = json.loads(line)
            game["date"] = datetime.fromisoformat(game["date"])
            games.append(game)
    return games


def test_train_player_model_smoke(tmp_path):
    """
    End-to-end smoke test:
    1. Build dataset from sample fixture into tmp_path.
    2. Run train_player_model for 2 epochs.
    3. Assert model.pt, training_info.json, and training_report.md exist.
    4. Assert loss values are finite floats (not NaN).
    5. Assert predict_move works on the trained model.
    """
    orig_data_root = config.DATA_ROOT
    config.DATA_ROOT = tmp_path

    try:
        games = load_fixture_games()

        # Build dataset for testuser
        with patch("app.services.extractor.storage.load_games", return_value=games):
            info = build_dataset("testuser")
            assert info["total_samples"] > 0

        # Update info.json to have model_tier = "small" since fixture has < 150 games
        info_path = tmp_path / "players" / "testuser" / "info.json"
        with open(info_path, "r", encoding="utf-8") as f:
            player_info = json.load(f)
        player_info["model_tier"] = "small"
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(player_info, f, indent=2)

        # Run training runner for 2 epochs with small model parameters for test speed
        overrides = {
            "TRAIN_MAX_EPOCHS": 2,
            "TRAIN_BATCH_SIZE": 8,
            "MODEL_CHANNELS": 16,
            "MODEL_RES_BLOCKS": 1,
            "MODEL_HIDDEN_DIM": 32,
        }

        summary = train_player_model("testuser", overrides=overrides)

        player_dir = tmp_path / "players" / "testuser"
        model_path = player_dir / "model.pt"
        info_path = player_dir / "training_info.json"
        report_path = player_dir / "training_report.md"

        assert model_path.exists(), "model.pt must be saved"
        assert info_path.exists(), "training_info.json must be saved"
        assert report_path.exists(), "training_report.md must be saved"

        # Check training_info.json content
        with open(info_path, "r", encoding="utf-8") as f:
            saved_info = json.load(f)

        assert saved_info["epochs_run"] == 2
        assert len(saved_info["history"]) == 2

        for epoch_stat in saved_info["history"]:
            train_loss = epoch_stat["train_loss"]
            assert isinstance(train_loss, float)
            assert not torch.isnan(torch.tensor(train_loss))
            assert not torch.isinf(torch.tensor(train_loss))

        # FIX 3d: Assert model_config equals overrides used (16/1/32)
        assert saved_info["model_config"] == {
            "channels": 16,
            "num_res_blocks": 1,
            "hidden_dim": 32,
            "dropout": config.MODEL_DROPOUT,
        }

        # FIX 3d: Assert dataset_sha256 matches samples.parquet
        from app.services import storage
        samples_path = player_dir / "samples.parquet"
        assert saved_info["dataset_sha256"] == storage.file_sha256(samples_path)

        # FIX 3d: Assert meta_defaults has all keys
        meta_defaults = saved_info["meta_defaults"]
        assert set(meta_defaults.keys()) == {
            "player_rating",
            "opponent_rating",
            "base_seconds",
            "increment_seconds",
            "clock_available",
            "median_clock_ratio",
        }

        # FIX 3d: load_player_model("testuser") returns model with .channels == 16
        # and predict_move works with meta_inputs_from_defaults
        loaded_player_model = export.load_player_model("testuser")
        assert loaded_player_model.channels == 16
        board = chess.Board()
        meta_inputs = export.meta_inputs_from_defaults(meta_defaults)
        move = export.predict_move(loaded_player_model, board, meta_inputs=meta_inputs)
        assert move in board.legal_moves

        # FIX 3d: is_model_current is True, then after appending one byte to
        # COPY-then-replace of samples.parquet inside tmp_path it is False
        assert export.is_model_current("testuser") is True
        content = samples_path.read_bytes()
        temp_parquet = player_dir / "samples_copy.parquet"
        temp_parquet.write_bytes(content + b"\x00")
        temp_parquet.replace(samples_path)
        assert export.is_model_current("testuser") is False

        # Existing assert: Check predict_move on the saved model with load_model
        loaded_model = export.load_model(
            model_path,
            channels=16,
            num_res_blocks=1,
            hidden_dim=32,
            dropout=0.0,
        )
        move_old = export.predict_move(loaded_model, board)
        assert move_old in board.legal_moves

    finally:
        config.DATA_ROOT = orig_data_root


def test_load_player_model_legacy_fallback(tmp_path, monkeypatch):
    """FIX 3d: where training_info.json has no 'model_config', load_player_model
    falls back to config defaults; raises FileNotFoundError when model.pt is missing."""
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path)

    # Missing model.pt raises FileNotFoundError with a clear message
    with pytest.raises(FileNotFoundError, match="Model checkpoint not found"):
        export.load_player_model("legacyuser")

    user_dir = tmp_path / "players" / "legacyuser"
    user_dir.mkdir(parents=True, exist_ok=True)

    default_model = ChessPolicyNet()
    export.save_model(default_model, user_dir / "model.pt")

    # training_info.json without model_config
    with open(user_dir / "training_info.json", "w", encoding="utf-8") as f:
        json.dump({"username": "legacyuser", "created_at": "2026-01-01T00:00:00"}, f)

    loaded = export.load_player_model("legacyuser")
    assert loaded.channels == config.MODEL_CHANNELS
    assert len(loaded.res_blocks) == config.MODEL_RES_BLOCKS
    assert loaded.hidden_dim == config.MODEL_HIDDEN_DIM



def test_train_model_empty_val_loader():
    """
    Test that train_model does not crash or divide by zero when val_loader is empty.
    """
    model = ChessPolicyNet(channels=16, num_res_blocks=1, hidden_dim=32, dropout=0.0)

    # Synthetic tiny train loader: 2 dummy batches
    dummy_board = torch.zeros((4, 18, 8, 8), dtype=torch.uint8)
    dummy_meta = torch.zeros((4, config.META_DIM), dtype=torch.float32)
    dummy_target = torch.zeros(4, dtype=torch.int64)
    dummy_aux = torch.zeros(4, dtype=torch.float32)

    class DummyDataset:
        def __len__(self):
            return 4

        def __getitem__(self, idx):
            return {
                "board": dummy_board[idx],
                "meta": dummy_meta[idx],
                "from_sq": dummy_target[idx],
                "to_sq": dummy_target[idx],
                "promo": dummy_target[idx],
                "piece_type": dummy_target[idx],
                "is_capture": dummy_aux[idx],
                "is_check": dummy_aux[idx],
                "is_castle": dummy_aux[idx],
                "sample_weight": torch.tensor(1.0),
            }

    train_loader = DataLoader(DummyDataset(), batch_size=2)
    empty_val_loader = DataLoader([])

    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=empty_val_loader,
        device="cpu",
        config_overrides={"TRAIN_MAX_EPOCHS": 2},
    )

    assert results["epochs_run"] == 2
    assert len(results["history"]) == 2
    for h in results["history"]:
        assert h["val_loss"] is None
        assert "warning" in h


def test_train_player_model_rejected_tier_raises(tmp_path):
    """
    Confirm model_tier == 'rejected' raises ValueError.
    """
    orig_data_root = config.DATA_ROOT
    config.DATA_ROOT = tmp_path

    try:
        player_dir = tmp_path / "players" / "badplayer"
        player_dir.mkdir(parents=True)

        info_path = player_dir / "info.json"
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump({"username": "badplayer", "model_tier": "rejected"}, f)

        with pytest.raises(ValueError, match="rejected"):
            train_player_model("badplayer")

    finally:
        config.DATA_ROOT = orig_data_root


def test_train_player_model_missing_info_raises(tmp_path):
    """
    Confirm missing info.json raises FileNotFoundError.
    """
    orig_data_root = config.DATA_ROOT
    config.DATA_ROOT = tmp_path

    try:
        with pytest.raises(FileNotFoundError, match="Run ingestion first"):
            train_player_model("nonexistent_user")
    finally:
        config.DATA_ROOT = orig_data_root
