"""
Tests for evaluate_cached (Phase B cached evaluation).

Verifies:
1. Exact equality against evaluate(UniformLegal) on non-promotion positions;
   promotion positions have nll == ln(len(legal_slice)).
2. Exact aggregate equality for non-uniform distribution softmax(-legal_idx / 50).
3. Partition invariant: n(non_berserk) + n(berserk) == n(all) on train.
4. Batch size invariance: batch_size=3 vs 512 yields identical metrics.
5. Invalidation & checks: sum(exp) != 1 raises ValueError, target not in legal raises ValueError.
6. Stale cache (parquet bytes modified) raises ValueError.
"""

from datetime import datetime
import json
import math
from pathlib import Path
import shutil
from unittest.mock import patch

import chess
import numpy as np
import pyarrow.parquet as pq
import pytest

from app import config
from app.ml import encoding
from app.ml.evaluate import evaluate, evaluate_cached
from app.ml.tensor_cache import build_cache
from app.services.dataset_builder_v3 import build_dataset_v3

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


def _setup_fixture_cache(tmp_path: Path, username: str = "eval_cached_user") -> tuple[Path, Path]:
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


class UniformLegalPredictor:
    """Predictor assigning uniform log probability across all legal UCI moves."""

    def legal_log_probs(self, board: chess.Board, ctx: dict) -> dict[str, float]:
        legals = list(board.legal_moves)
        n = len(legals)
        lp = -math.log(n)
        return {m.uci(): lp for m in legals}


def uniform_batch_fn(batch: dict[str, np.ndarray]) -> np.ndarray:
    """Uniform probability distribution over legal moves in the CSR mask."""
    mask = batch["legal_mask"]
    B = mask.shape[0]
    out = np.full((B, 4096), -np.inf, dtype=np.float64)
    for b in range(B):
        legals = np.where(mask[b])[0]
        out[b, legals] = -np.log(len(legals))
    return out


def test_1_uniform_cached_vs_uncached_non_promotion_equality(tmp_path):
    """
    Test 1: Uniform batch fn on test + train splits.

    For non-promotion-legal positions, per-position top1, top3, nll, rank from
    evaluate_cached match evaluate(UniformLegalPredictor) within 1e-9 tolerance.
    For promotion-legal positions, cached nll equals ln(len(legal_slice)).
    """
    cache_dir, parquet_path = _setup_fixture_cache(tmp_path)
    df = pq.read_table(parquet_path).to_pandas()

    for split_name in ["train", "test"]:
        cached_res = evaluate_cached(
            uniform_batch_fn,
            cache_dir,
            parquet_path,
            split=split_name,
            berserk_filter="all",
            batch_size=64,
        )

        split_rows = df[df["split"] == split_name].to_dict(orient="records")
        uncached_res = evaluate(
            UniformLegalPredictor(),
            split_rows,
        )

        cached_by_key = {(p["game_id"], p["ply"]): p for p in cached_res["per_position"]}
        uncached_by_key = {(p["game_id"], p["ply"]): p for p in uncached_res["per_position"]}

        promo_pos_count = 0
        non_promo_count = 0

        for row in split_rows:
            key = (row["game_id"], row["ply"])
            board = chess.Board(row["fen"])
            has_promo_legal = any(m.promotion is not None for m in board.legal_moves)

            c_pos = cached_by_key[key]
            u_pos = uncached_by_key[key]

            if not has_promo_legal:
                non_promo_count += 1
                assert abs(c_pos["top1"] - u_pos["top1"]) < 1e-9
                assert abs(c_pos["top3"] - u_pos["top3"]) < 1e-9
                assert abs(c_pos["nll"] - u_pos["nll"]) < 1e-9
                assert abs(c_pos["rank"] - u_pos["rank"]) < 1e-9
            else:
                promo_pos_count += 1
                # In cached representation, promotions collapse into single (from, to) squares.
                # Expected nll is ln(n_unique_from_to_pairs).
                # Move pairs:
                unique_pairs = {(m.from_square, m.to_square) for m in board.legal_moves}
                expected_nll = math.log(len(unique_pairs))
                assert abs(c_pos["nll"] - expected_nll) < 1e-9

        print(f"Split {split_name}: {non_promo_count} non-promo positions, {promo_pos_count} promo positions.")


def test_2_deterministic_non_uniform_distribution_equality(tmp_path):
    """
    Test 2: Deterministic non-uniform log-prob fn:
    logp proportional to softmax(-legal_index / 50.0).

    Verifies identical metrics against evaluate() on non-promotion positions.
    """
    cache_dir, parquet_path = _setup_fixture_cache(tmp_path)
    df = pq.read_table(parquet_path).to_pandas()

    def non_uniform_batch_fn(batch: dict[str, np.ndarray]) -> np.ndarray:
        mask = batch["legal_mask"]
        B = mask.shape[0]
        out = np.full((B, 4096), -np.inf, dtype=np.float64)
        for b in range(B):
            legals = np.where(mask[b])[0]
            logits = -legals / 50.0
            lse = np.log(np.sum(np.exp(logits - np.max(logits)))) + np.max(logits)
            out[b, legals] = logits - lse
        return out

    class NonUniformPredictor:
        def legal_log_probs(self, board: chess.Board, ctx: dict) -> dict[str, float]:
            legals = list(board.legal_moves)
            indices = []
            for m in legals:
                enc = encoding.encode_move(board, m)
                indices.append(enc["from_sq"] * 64 + enc["to_sq"])
            logits = np.array([-idx / 50.0 for idx in indices], dtype=np.float64)
            lse = np.log(np.sum(np.exp(logits - np.max(logits)))) + np.max(logits)
            logps = logits - lse
            return {m.uci(): float(lp) for m, lp in zip(legals, logps)}

    cached_res = evaluate_cached(
        non_uniform_batch_fn,
        cache_dir,
        parquet_path,
        split="train",
        berserk_filter="all",
    )

    train_rows = df[df["split"] == "train"].to_dict(orient="records")
    uncached_res = evaluate(
        NonUniformPredictor(),
        train_rows,
    )

    cached_by_key = {(p["game_id"], p["ply"]): p for p in cached_res["per_position"]}
    uncached_by_key = {(p["game_id"], p["ply"]): p for p in uncached_res["per_position"]}

    # Filter to non-promotion positions
    non_promo_cached = []
    non_promo_uncached = []

    for row in train_rows:
        key = (row["game_id"], row["ply"])
        board = chess.Board(row["fen"])
        if not any(m.promotion is not None for m in board.legal_moves):
            non_promo_cached.append(cached_by_key[key])
            non_promo_uncached.append(uncached_by_key[key])

    assert len(non_promo_cached) > 0

    for c, u in zip(non_promo_cached, non_promo_uncached):
        assert abs(c["top1"] - u["top1"]) < 1e-9
        assert abs(c["top3"] - u["top3"]) < 1e-9
        assert abs(c["nll"] - u["nll"]) < 1e-9
        assert abs(c["rank"] - u["rank"]) < 1e-9


def test_3_berserk_filter_partitions_train_split(tmp_path):
    """
    Test 3: Partition invariant:
    n(non_berserk) + n(berserk) == n(all) for the train split.
    """
    cache_dir, parquet_path = _setup_fixture_cache(tmp_path)

    res_all = evaluate_cached(uniform_batch_fn, cache_dir, parquet_path, split="train", berserk_filter="all")
    res_nb = evaluate_cached(uniform_batch_fn, cache_dir, parquet_path, split="train", berserk_filter="non_berserk")
    res_b = evaluate_cached(uniform_batch_fn, cache_dir, parquet_path, split="train", berserk_filter="berserk")

    assert res_nb["n_positions"] + res_b["n_positions"] == res_all["n_positions"]


def test_4_batch_size_invariance(tmp_path):
    """
    Test 4: Batching invariance: batch_size=3 vs 512 give identical results.
    """
    cache_dir, parquet_path = _setup_fixture_cache(tmp_path)

    res_3 = evaluate_cached(uniform_batch_fn, cache_dir, parquet_path, split="train", batch_size=3)
    res_512 = evaluate_cached(uniform_batch_fn, cache_dir, parquet_path, split="train", batch_size=512)

    assert res_3["n_positions"] == res_512["n_positions"]
    assert abs(res_3["top1"] - res_512["top1"]) < 1e-9
    assert abs(res_3["top3"] - res_512["top3"]) < 1e-9
    assert abs(res_3["nll"] - res_512["nll"]) < 1e-9
    assert abs(res_3["rank"] - res_512["rank"]) < 1e-9


def test_5_validation_sum_probs_and_illegal_target(tmp_path):
    """
    Test 5:
    - Batch fn returning probabilities not summing to 1 -> ValueError.
    - True target not in legal slice (tampered cache) -> ValueError.
    """
    cache_dir, parquet_path = _setup_fixture_cache(tmp_path)

    # 1. Probabilities do not sum to 1
    def bad_sum_fn(batch: dict[str, np.ndarray]) -> np.ndarray:
        mask = batch["legal_mask"]
        B = mask.shape[0]
        out = np.full((B, 4096), -np.inf, dtype=np.float64)
        for b in range(B):
            legals = np.where(mask[b])[0]
            out[b, legals] = 0.0  # exp(0) = 1 for each legal move, sum = len(legals) != 1
        return out

    with pytest.raises(ValueError, match="Sum of exp"):
        evaluate_cached(bad_sum_fn, cache_dir, parquet_path, split="train")

    # 2. Tampered cache: target not in legal slice
    tamper_dir = tmp_path / "tampered_cache"
    shutil.copytree(cache_dir, tamper_dir)

    target_arr = np.load(tamper_dir / "target.npy")
    target_arr[0] = 4000  # not among legal moves of row 0
    np.save(tamper_dir / "target.npy", target_arr)

    with pytest.raises(ValueError, match="True target"):
        evaluate_cached(uniform_batch_fn, tamper_dir, parquet_path, split="train")


def test_6_stale_cache_raises(tmp_path):
    """
    Test 6: Modifying parquet bytes invalidates cache -> evaluate_cached raises ValueError.
    """
    cache_dir, parquet_path = _setup_fixture_cache(tmp_path)

    # Modify parquet file by appending a byte
    with open(parquet_path, "ab") as f:
        f.write(b"\x00")

    with pytest.raises(ValueError, match="Cache is invalid"):
        evaluate_cached(uniform_batch_fn, cache_dir, parquet_path, split="train")
