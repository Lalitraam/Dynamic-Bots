"""
Tests for train_v2: smoke run, checkpointing, and resume determinism on CPU.
Uses synthetic parquet shard fixture (no real data access).
"""
from __future__ import annotations

import math
from pathlib import Path
import chess
import pandas as pd
import pytest
import torch

from scripts.train_v2 import train_model


@pytest.fixture
def synthetic_shards_dir(tmp_path: Path) -> Path:
    """Build a tiny synthetic shard fixture (10 games x 20 positions, split across train/val)."""
    rows = []
    moves = [
        "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7",
        "f1e1", "b7b5", "a4b3", "d7d6", "c2c3", "e8g8", "h2h3", "c6b8", "d2d4", "b8d7",
    ]

    for g_idx in range(10):
        board = chess.Board()
        split = "train" if g_idx < 8 else "val"
        for ply, uci in enumerate(moves):
            rows.append({
                "game_id": f"synthetic_g_{g_idx}",
                "ply": ply,
                "fen": board.fen(),
                "uci": uci,
                "date": "2026-01-10",
                "category": "blitz",
                "is_fast": True,
                "time_control": "180+0",
                "player_rating": 1500,
                "opponent_rating": 1500,
                "color": "white" if board.turn == chess.WHITE else "black",
                "termination": "Normal",
                "clock_before_move": 180.0,
                "halfmove_clock": board.halfmove_clock,
                "sample_weight": 1.0,
                "split": split,
                "schema_version": "v3",
                "target_berserk": False,
                "prev_move_1": None,
                "prev_move_2": None,
                "clock_after_move": 178.0,
                "time_spent_s": 2.0,
                "increment_s": 0,
                "base_s": 180,
                "created_at_ms": None,
                "game_result_for_player": 0.5,
                "num_legal_moves": board.legal_moves.count(),
            })
            board.push(chess.Move.from_uci(uci))

    shards_dir = tmp_path / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(shards_dir / "shard_0000.parquet", index=False)
    return shards_dir


def test_smoke_train_v2(synthetic_shards_dir: Path, tmp_path: Path):
    """
    Smoke test:
    1 epoch, batch_size 4, max_batches 5, device cpu.
    Assert outputs written:
      checkpoint_latest.pt, base_v1.pt, training_info_v2.json, model_card_v2.md.
    Assert val loss is finite.
    """
    out_dir = tmp_path / "smoke_out"
    info = train_model(
        shards_dir=synthetic_shards_dir,
        out_dir=out_dir,
        epochs=1,
        batch_size=4,
        lr=2e-3,
        max_batches=5,
        device_str="cpu",
        base_version=1,
    )

    # Check written artifacts
    assert (out_dir / "checkpoint_latest.pt").exists(), "checkpoint_latest.pt missing"
    assert (out_dir / "base_v1.pt").exists(), "base_v1.pt missing"
    assert (out_dir / "training_info_v2.json").exists(), "training_info_v2.json missing"
    assert (out_dir / "model_card_v2.md").exists(), "model_card_v2.md missing"

    # Check validation metrics
    assert info.get("val_metric") == "policy_nll_eps0_no_promo"
    assert info.get("train_metric") == "smoothed_policy_ce_0.05_plus_0.2_promo"
    assert len(info["history"]) >= 1
    val_loss = info["history"][0]["val_loss"]
    assert math.isfinite(val_loss), f"val_loss is not finite: {val_loss}"
    assert val_loss > 0.0


def test_resume_checkpoint_determinism(synthetic_shards_dir: Path, tmp_path: Path):
    """
    Resume test:
    Run 1 epoch, record loss.
    Rerun with --resume for 1 more epoch.
    Assert the second run's FIRST step loss equals the first run's LAST step loss within 1e-4.
    """
    out_dir = tmp_path / "resume_out"

    # Run 1 epoch
    res1 = train_model(
        shards_dir=synthetic_shards_dir,
        out_dir=out_dir,
        epochs=1,
        batch_size=4,
        lr=2e-3,
        max_batches=5,
        device_str="cpu",
        base_version=1,
        resume=False,
    )
    last_step_loss = res1["last_step_loss"]
    assert (out_dir / "checkpoint_latest.pt").exists(), "checkpoint_latest.pt missing from run 1"

    # Run second epoch with resume=True
    res2 = train_model(
        shards_dir=synthetic_shards_dir,
        out_dir=out_dir,
        epochs=1,
        batch_size=4,
        lr=2e-3,
        max_batches=5,
        device_str="cpu",
        base_version=1,
        resume=True,
    )

    resumed_first_loss = res2.get("resumed_first_step_loss")
    assert resumed_first_loss is not None, "resumed_first_step_loss was not recorded on resume"
    assert abs(resumed_first_loss - last_step_loss) < 1e-4, (
        f"Mismatch: resumed {resumed_first_loss} vs previous last {last_step_loss}"
    )


def test_parquet_v3_dataset_target_in_legal_slice(tmp_path: Path):
    """
    Test Patch A: assert target move is in legal slice for ParquetV3Dataset.
    Tests both White and Black moves (verifying mirror logic for Black).
    """
    from scripts.train_v2 import ParquetV3Dataset

    # 3-row synthetic dataset: Row 0 White, Row 1 Black, Row 2 White
    rows = [
        {
            "game_id": "test_g_0",
            "ply": 0,
            "fen": chess.STARTING_FEN,
            "uci": "e2e4",
            "split": "train",
            "player_rating": 1500,
            "opponent_rating": 1500,
            "base_s": 180,
            "increment_s": 0,
            "clock_before_move": 180.0,
            "halfmove_clock": 0,
            "num_legal_moves": 20,
            "target_berserk": False,
            "sample_weight": 1.0,
        },
        {
            "game_id": "test_g_0",
            "ply": 1,
            "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "uci": "e7e5",
            "split": "train",
            "player_rating": 1500,
            "opponent_rating": 1500,
            "base_s": 180,
            "increment_s": 0,
            "clock_before_move": 180.0,
            "halfmove_clock": 0,
            "num_legal_moves": 20,
            "target_berserk": False,
            "sample_weight": 1.0,
        },
        {
            "game_id": "test_g_0",
            "ply": 2,
            "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "uci": "g1f3",
            "split": "train",
            "player_rating": 1500,
            "opponent_rating": 1500,
            "base_s": 180,
            "increment_s": 0,
            "clock_before_move": 178.0,
            "halfmove_clock": 0,
            "num_legal_moves": 29,
            "target_berserk": False,
            "sample_weight": 1.0,
        },
    ]

    dataset = ParquetV3Dataset(rows, history=True)
    assert len(dataset) == 3

    for i in range(3):
        item = dataset[i]
        tgt_idx = item["target_move_idx"].item()
        assert item["legal_mask"][tgt_idx].item() is True, (
            f"Row {i} target move {rows[i]['uci']} idx {tgt_idx} is not in legal_mask!"
        )


def test_train_v2_dump_config_and_time_50_batches(synthetic_shards_dir: Path, tmp_path: Path, capsys):
    """Test --dump-config and --time-50-batches flags."""
    import json
    cfg_path = tmp_path / "model_config.json"
    bench_dir = tmp_path / "bench_out"

    res = train_model(
        shards_dir=synthetic_shards_dir,
        out_dir=bench_dir,
        epochs=1,
        batch_size=4,
        device_str="cpu",
        dump_config=cfg_path,
        time_50_batches=True,
    )

    # Check dump_config output
    assert cfg_path.exists(), "dump-config JSON was not written"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for key in [
        "arch", "num_planes", "meta_dim", "batch_size", "lr", "wd",
        "epochs", "shards_dir", "out_dir", "seed", "device",
        "amp_dtype", "torch_version", "cuda_version", "gpu_name",
    ]:
        assert key in cfg, f"Missing key {key} in dumped config"
    assert cfg["arch"] == "ChessPolicyNetV2"

    # Check time_50_batches behavior: executed 50 steps without saving model
    assert res.get("time_50_batches_completed") is True
    assert not (bench_dir / "base_v1.pt").exists(), "base_v1.pt should NOT be saved in time-50-batches mode"
    assert not (bench_dir / "checkpoint_latest.pt").exists(), "checkpoint_latest.pt should NOT be saved in time-50-batches mode"

    captured = capsys.readouterr()
    assert "STEP 1 | loss" in captured.out
    assert "STEP 50 | loss" in captured.out

