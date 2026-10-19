"""Tests for ThinkTimeModel (backend/app/engine/think_time.py)."""

from pathlib import Path
import random
import chess
import pytest

from app import config
from app.engine.provider import MoveContext
from app.engine.think_time import (
    ThinkTimeModel,
    _bucket_key,
    _clock_ratio_bucket,
    _phase_from_ply,
    _speed_from_time_control,
)


def _make_ctx(
    ply: int = 10,
    base_s: int = 180,
    inc_s: int = 2,
    bot_clock_ms: int = 120000,
    opp_clock_ms: int = 120000,
) -> MoveContext:
    return MoveContext(
        board=chess.Board(),
        prev_moves=[],
        bot_rating=1800,
        opp_rating=1800,
        base_seconds=base_s,
        increment_seconds=inc_s,
        bot_clock_ms=bot_clock_ms,
        opp_clock_ms=opp_clock_ms,
        ply=ply,
    )


def test_1_phase_from_ply():
    assert _phase_from_ply(10) == "opening"
    assert _phase_from_ply(30) == "middlegame"
    assert _phase_from_ply(80) == "endgame"


def test_2_clock_ratio_bucket():
    assert _clock_ratio_bucket(0.05) == "<0.1"
    assert _clock_ratio_bucket(0.2) == "0.1-0.3"
    assert _clock_ratio_bucket(0.5) == ">0.3"
    assert _clock_ratio_bucket(None) == "unknown"


def test_3_speed_from_time_control():
    assert _speed_from_time_control(60, 0) == "bullet"
    assert _speed_from_time_control(180, 0) == "blitz"
    assert _speed_from_time_control(600, 0) == "rapid"
    assert _speed_from_time_control(120, 1) == "blitz"
    assert _speed_from_time_control(60, 0) == "bullet"


def test_4_bucket_key_format():
    key = _bucket_key("blitz", "opening", 0.5, False)
    assert key == ("blitz", "opening", ">0.3", False)
    assert isinstance(key, tuple)
    assert len(key) == 4
    assert isinstance(key[0], str)
    assert isinstance(key[1], str)
    assert isinstance(key[2], str)
    assert isinstance(key[3], bool)


def test_5_sample_clamps_to_floor():
    # Bucket with all 0.0 samples
    key = ("blitz", "opening", ">0.3", False)
    model = ThinkTimeModel({key: [0.0] * 30})
    ctx = _make_ctx(ply=10, base_s=180, inc_s=0, bot_clock_ms=100000)
    rng = random.Random(42)
    think_ms = model.sample(ctx, rng)
    assert think_ms >= config.THINK_FLOOR_MS


def test_6_sample_clamps_to_clock():
    # Set bot_clock_ms=500, bucket sample=10000ms -> returns <= 200ms (500-300)
    key = ("blitz", "opening", ">0.3", False)
    model = ThinkTimeModel({key: [10.0] * 30})
    ctx = _make_ctx(ply=10, base_s=180, inc_s=0, bot_clock_ms=500)
    rng = random.Random(42)
    think_ms = model.sample(ctx, rng)
    assert think_ms <= 200
    assert think_ms >= config.THINK_FLOOR_MS


def test_7_reservoir_sampling_deterministic(tmp_path: Path):
    p = Path("../data/players/the_king_crusher/samples_v3.parquet")
    if not p.exists():
        import pandas as pd
        p = tmp_path / "samples_v3.parquet"
        df = pd.DataFrame([
            {
                "base_s": 180,
                "increment_s": 2,
                "ply": i,
                "clock_before_move": 150.0 - i,
                "num_legal_moves": 20,
                "time_spent_s": 1.5 + (i % 3),
            }
            for i in range(120)
        ])
        df.to_parquet(p)

    m1 = ThinkTimeModel.from_parquet(p, max_samples_per_bucket=50, seed=0)
    m2 = ThinkTimeModel.from_parquet(p, max_samples_per_bucket=50, seed=0)
    assert set(m1.buckets.keys()) == set(m2.buckets.keys())
    for k in m1.buckets:
        assert m1.buckets[k] == m2.buckets[k]


def test_8_fallback_to_pooled_when_bucket_empty():
    # Case 1: When exact clock-ratio bucket has >= 20 samples, it uses that bucket
    specific_key = ("blitz", "opening", ">0.3", False)
    other_key = ("rapid", "endgame", "<0.1", True)

    specific_samples = [0.5] * 25
    pooled_other_samples = [10.0, 12.0, 14.0]

    model_with_enough = ThinkTimeModel({
        specific_key: specific_samples,
        other_key: pooled_other_samples,
    })
    ctx_match = _make_ctx(ply=10, base_s=180, inc_s=0, bot_clock_ms=100000)
    # With enough samples in specific bucket, should return 500 ms (0.5s)
    res_specific = model_with_enough.sample(ctx_match, random.Random(0))
    assert res_specific == 500

    # Case 2: When exact bucket has only 3 samples (< 20), it falls back to pooled
    sparse_key = ("blitz", "opening", ">0.3", False)
    sparse_samples = [1.0, 2.0, 3.0]
    model_sparse = ThinkTimeModel({
        sparse_key: sparse_samples,
        other_key: pooled_other_samples,
    })
    # Pooled list is [1.0, 2.0, 3.0, 10.0, 12.0, 14.0]
    # Hand-computed: random.Random(0).choice([1.0, 2.0, 3.0, 10.0, 12.0, 14.0]) is 10.0
    res_pooled = model_sparse.sample(ctx_match, random.Random(0))
    assert res_pooled == 2000
    assert 1000 <= res_pooled <= 14000
