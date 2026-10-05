"""
Tests for the sampler service.

Covers:
- Total sampled <= max_total
- No duplicate games (by game_id)
- Under-filled scenario gives expected counts
- 2000 games all in last_6m → exactly 1200 sampled
- Fewer total games than max_total → all returned
- Same seed gives same result
"""

import pytest
from app.services.sampler import sample_games
from app import config


def _make_games(bucket: str, n: int, category: str = "blitz", start_id: int = 0) -> list[dict]:
    """Helper: create n minimal game dicts with unique game_ids."""
    return [
        {
            "game_id": f"{bucket}_{start_id + i}",
            "category": category,
            "is_fast": category == "bullet",
            "player_color": "white",
            "rating": 1500,
            "opponent_rating": 1500,
            "time_control": "180+2",
            "termination": "Normal",
            "pgn": f"1. e4 e5 *",
        }
        for i in range(n)
    ]


def _make_buckets(last_6m=0, six_18m=0, eighteen_36m=0, category="blitz"):
    return {
        "last_6m": _make_games("last_6m", last_6m, category),
        "6_18m": _make_games("6_18m", six_18m, category, start_id=last_6m),
        "18_36m": _make_games("18_36m", eighteen_36m, category, start_id=last_6m + six_18m),
    }


# ---------------------------------------------------------------------------
# Total sampled <= max_total
# ---------------------------------------------------------------------------

def test_total_never_exceeds_max():
    buckets = _make_buckets(last_6m=600, six_18m=600, eighteen_36m=600)
    result = sample_games(buckets, seed=42)
    assert len(result["games"]) <= config.MAX_TOTAL_SAMPLED


def test_total_exactly_max_when_enough_games():
    """When there are more games than max_total, we should get exactly max_total."""
    buckets = _make_buckets(last_6m=600, six_18m=600, eighteen_36m=600)
    result = sample_games(buckets, seed=42)
    # 1800 games available, max_total=1200
    assert len(result["games"]) == config.MAX_TOTAL_SAMPLED


# ---------------------------------------------------------------------------
# No duplicate game_ids
# ---------------------------------------------------------------------------

def test_no_duplicates():
    buckets = _make_buckets(last_6m=600, six_18m=500, eighteen_36m=400)
    result = sample_games(buckets, seed=99)
    ids = [g["game_id"] for g in result["games"]]
    assert len(ids) == len(set(ids)), "Duplicate game_ids found in sampled output"


# ---------------------------------------------------------------------------
# Under-filled scenario
# ---------------------------------------------------------------------------

def test_underfilled_scenario():
    """
    last_6m: 100 games  (quota 500 → takes all 100, 400 carry over)
    6_18m:   50 games   (quota 400 + 400 carry = 800 → takes all 50)
    18_36m:  30 games   (quota 300 + leftover → takes all 30)
    Total: 180 games, all returned.
    """
    buckets = _make_buckets(last_6m=100, six_18m=50, eighteen_36m=30)
    result = sample_games(buckets, seed=0)
    assert len(result["games"]) == 180
    assert result["quota_used"]["last_6m"] == 100
    assert result["quota_used"]["6_18m"] == 50
    assert result["quota_used"]["18_36m"] == 30


# ---------------------------------------------------------------------------
# 2000 games all in last_6m → exactly 1200
# ---------------------------------------------------------------------------

def test_all_in_last_6m_2000_games():
    """
    2000 games in last_6m, none elsewhere.
    Pass 1: takes 500 (quota).
    Pass 2: no redistribution needed (other buckets empty).
    Pass 3: fills from last_6m leftovers up to 1200-500=700 more.
    Total: 1200.
    """
    buckets = _make_buckets(last_6m=2000, six_18m=0, eighteen_36m=0)
    result = sample_games(buckets, seed=7)
    assert len(result["games"]) == config.MAX_TOTAL_SAMPLED, (
        f"Expected {config.MAX_TOTAL_SAMPLED}, got {len(result['games'])}"
    )


# ---------------------------------------------------------------------------
# Fewer total games than max_total → all returned
# ---------------------------------------------------------------------------

def test_fewer_than_max_returns_all():
    total = 300
    buckets = _make_buckets(last_6m=100, six_18m=100, eighteen_36m=100)
    result = sample_games(buckets, seed=1)
    assert len(result["games"]) == total


def test_zero_games():
    buckets = _make_buckets(last_6m=0, six_18m=0, eighteen_36m=0)
    result = sample_games(buckets, seed=0)
    assert len(result["games"]) == 0


# ---------------------------------------------------------------------------
# Same seed gives same result
# ---------------------------------------------------------------------------

def test_reproducible_with_same_seed():
    buckets = _make_buckets(last_6m=800, six_18m=600, eighteen_36m=400)
    result_a = sample_games(buckets, seed=42)
    result_b = sample_games(buckets, seed=42)
    ids_a = [g["game_id"] for g in result_a["games"]]
    ids_b = [g["game_id"] for g in result_b["games"]]
    assert ids_a == ids_b, "Same seed should produce identical results"


def test_different_seeds_give_different_results():
    buckets = _make_buckets(last_6m=800, six_18m=600, eighteen_36m=400)
    result_a = sample_games(buckets, seed=1)
    result_b = sample_games(buckets, seed=2)
    ids_a = set(g["game_id"] for g in result_a["games"])
    ids_b = set(g["game_id"] for g in result_b["games"])
    # With large buckets and random shuffle, different seeds should (almost always) differ
    assert ids_a != ids_b, "Different seeds should give different results"


# ---------------------------------------------------------------------------
# Category counts
# ---------------------------------------------------------------------------

def test_category_counts():
    buckets = {
        "last_6m": _make_games("last_6m", 100, "bullet") + _make_games("last_6m", 200, "blitz", start_id=100),
        "6_18m": _make_games("6_18m", 150, "rapid"),
        "18_36m": _make_games("18_36m", 50, "blitz", start_id=150),
    }
    result = sample_games(buckets, seed=0)
    total = len(result["games"])
    assert result["bullet_count"] + result["blitz_count"] + result["rapid_count"] == total
    assert result["bullet_count"] <= 100
    assert result["rapid_count"] <= 150


# ---------------------------------------------------------------------------
# Redistribution: active player with everything in last_6m gets 1200
# ---------------------------------------------------------------------------

def test_active_player_gets_1200():
    """Quota redistribution + fill pass must allow 1200 games even if all in last_6m."""
    buckets = _make_buckets(last_6m=2000, six_18m=0, eighteen_36m=0)
    result = sample_games(buckets, seed=5)
    assert len(result["games"]) == config.MAX_TOTAL_SAMPLED
