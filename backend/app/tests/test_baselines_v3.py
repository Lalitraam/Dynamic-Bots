"""
Tests for B.6 baselines (app/tests/test_baselines_v3.py).

Verifies:
1. Uniform batch fn on fixture cache equals evaluate(UniformLegal) on the same rows,
   matched by (game_id, ply).
2. PositionFrequency is deterministic; unseen FEN key yields uniform distribution.
3. BookThenModel with an empty BookV2 equals CurrentModel output bit-for-bit.
4. v2 model slicing with planes[:, :18] and meta[:, :8] does not raise and produces
   valid normalized log probabilities.
"""

from datetime import datetime
import json
import math
from pathlib import Path
from unittest.mock import patch

import chess
import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from app import config
from app.ml.baselines import CurrentModel, PositionFrequency, UniformLegal
from app.ml.evaluate import evaluate, evaluate_cached, row_to_ctx
from app.ml.model import ChessPolicyNet
from app.ml.tensor_cache import build_cache
from app.services.dataset_builder_v3 import build_dataset_v3
from app.services.opening_book_v2 import BookV2
from scripts.baselines_v3 import BookV2ThenModel, make_model_batch_logprob_fn

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_games.jsonl"


def _load_fixture_games():
    games = []
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            game = json.loads(line)
            game["date"] = datetime.fromisoformat(game["date"])
            games.append(game)
    return games


def _setup_fixture_cache(tmp_path: Path, username: str = "baselines_v3_fixture_user") -> tuple[Path, Path]:
    orig_data_root = config.DATA_ROOT
    config.DATA_ROOT = tmp_path
    games = _load_fixture_games()
    try:
        with patch("app.services.extractor.storage.load_games", return_value=games):
            build_dataset_v3(username)
        cache_dir = build_cache(username, data_root=tmp_path, history=True)
        parquet_path = tmp_path / "players" / username / "samples_v3.parquet"
        return cache_dir, parquet_path
    finally:
        config.DATA_ROOT = orig_data_root


def uniform_batch_fn(batch: dict[str, np.ndarray]) -> np.ndarray:
    mask = batch["legal_mask"]
    B = mask.shape[0]
    out = np.full((B, 4096), -np.inf, dtype=np.float64)
    for b in range(B):
        legals = np.where(mask[b])[0]
        out[b, legals] = -np.log(len(legals))
    return out


def test_1_uniform_batch_fn_equals_uniform_legal_on_fixture(tmp_path):
    """
    Test 1: Uniform batch fn on fixture cache equals evaluate(UniformLegal)
    matched by (game_id, ply) on non-promotion positions.
    """
    cache_dir, parquet_path = _setup_fixture_cache(tmp_path)
    df = pq.read_table(parquet_path).to_pandas()

    test_rows = df[df["split"] == "test"].to_dict(orient="records")
    uncached_res = evaluate(UniformLegal(), test_rows)
    cached_res = evaluate_cached(
        uniform_batch_fn,
        cache_dir,
        parquet_path,
        split="test",
        berserk_filter="all",
        batch_size=64,
    )

    cached_by_key = {(p["game_id"], p["ply"]): p for p in cached_res["per_position"]}
    uncached_by_key = {(p["game_id"], p["ply"]): p for p in uncached_res["per_position"]}

    for row in test_rows:
        key = (row["game_id"], row["ply"])
        board = chess.Board(row["fen"])
        has_promo = any(m.promotion is not None for m in board.legal_moves)
        if not has_promo:
            c_pos = cached_by_key[key]
            u_pos = uncached_by_key[key]
            assert abs(c_pos["top1"] - u_pos["top1"]) < 1e-9
            assert abs(c_pos["top3"] - u_pos["top3"]) < 1e-9
            assert abs(c_pos["nll"] - u_pos["nll"]) < 1e-9
            assert abs(c_pos["rank"] - u_pos["rank"]) < 1e-9


def test_2_position_frequency_deterministic_and_unseen_uniform(tmp_path):
    """
    Test 2: PositionFrequency is deterministic and unseen FEN keys yield uniform distribution.
    """
    train_rows = [
        {"fen": chess.STARTING_FEN, "uci": "e2e4"},
        {"fen": chess.STARTING_FEN, "uci": "e2e4"},
        {"fen": chess.STARTING_FEN, "uci": "d2d4"},
    ]
    pf = PositionFrequency(train_rows)
    board = chess.Board()
    ctx = {}

    logps_1 = pf.legal_log_probs(board, ctx)
    logps_2 = pf.legal_log_probs(board, ctx)
    assert logps_1 == logps_2

    # Unseen position
    unseen_board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    unseen_logps = pf.legal_log_probs(unseen_board, ctx)
    n_legal = len(list(unseen_board.legal_moves))
    expected_uniform = -math.log(n_legal)
    for lp in unseen_logps.values():
        assert abs(lp - expected_uniform) < 1e-9


def test_3_book_then_model_empty_book_equals_current_model():
    """
    Test 3: BookThenModel with an empty BookV2 equals CurrentModel output.
    """
    torch.manual_seed(0)
    model = ChessPolicyNet(channels=16, num_res_blocks=1, hidden_dim=32, dropout=0.0)
    model.eval()

    current_model = CurrentModel(model, device="cpu")
    empty_book = BookV2({"meta": {}, "positions": {}})
    book_then_model = BookV2ThenModel(current_model, empty_book)

    board = chess.Board()
    ctx = row_to_ctx({
        "player_rating": 1500,
        "opponent_rating": 1500,
        "time_control": "180+0",
        "ply": 0,
        "clock_before_move": 180.0,
        "halfmove_clock": 0,
    })

    logps_cm = current_model.legal_log_probs(board, ctx)
    logps_btm = book_then_model.legal_log_probs(board, ctx)

    assert logps_cm == logps_btm


def test_4_v2_model_slicing_does_not_raise():
    """
    Test 4: v2 model slicing with planes[:, :18] and meta[:, :8] does not raise
    and produces valid normalized distributions.
    """
    torch.manual_seed(0)
    model = ChessPolicyNet(channels=16, num_res_blocks=1, hidden_dim=32, dropout=0.0)
    model.eval()

    fn = make_model_batch_logprob_fn(model, device="cpu")

    B = 3
    planes = np.zeros((B, 22, 8, 8), dtype=np.uint8)
    meta = np.zeros((B, 10), dtype=np.float32)
    legal_mask = np.zeros((B, 4096), dtype=bool)

    legal_mask[0, [64 * 12 + 28, 64 * 12 + 20]] = True
    legal_mask[1, [64 * 1 + 18, 64 * 6 + 21]] = True
    legal_mask[2, [64 * 8 + 16, 64 * 9 + 17]] = True

    batch = {
        "planes": planes,
        "meta": meta,
        "legal_mask": legal_mask,
    }

    log_probs = fn(batch)
    assert log_probs.shape == (B, 4096)

    for b in range(B):
        legals = np.where(legal_mask[b])[0]
        lp_legals = log_probs[b, legals]
        assert np.all(np.isfinite(lp_legals))
        total_p = np.sum(np.exp(lp_legals))
        assert abs(total_p - 1.0) < 1e-4
