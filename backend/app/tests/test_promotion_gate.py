"""
Unit tests for promotion gate (app/ml/promotion_gate.py).
Tests dual-metric significance testing, game-level paired bootstrap,
and sample size decision branches on synthetic fixtures.
"""
from __future__ import annotations

import pytest
from app.ml.promotion_gate import paired_game_bootstrap, should_promote


def _make_dummy_positions(n_games: int, top1_val: float, nll_val: float) -> list[dict]:
    positions = []
    for g in range(n_games):
        for ply in range(1, 3):
            positions.append({
                "game_id": f"game_{g}",
                "ply": ply,
                "top1": float(top1_val),
                "top3": float(min(1.0, top1_val + 0.1)),
                "nll": float(nll_val),
                "rank": 1.0,
            })
    return positions


def test_1_gate_promotes_when_ft_better():
    """Verify that FT is promoted when NLL is significantly lower and Top-1 significantly higher."""
    base_pos = _make_dummy_positions(25, top1_val=0.3, nll_val=2.5)
    ft_pos = _make_dummy_positions(25, top1_val=0.7, nll_val=1.2)

    res = should_promote(base_pos, ft_pos, n_test_games=25)

    assert res["decision"] == "promote_ft"
    assert res["ci_nll"][1] < 0.0, f"Expected NLL CI upper < 0, got {res['ci_nll'][1]}"
    assert res["ci_top1"][0] > 0.0, f"Expected Top1 CI lower > 0, got {res['ci_top1'][0]}"


def test_2_gate_keeps_base_when_ft_worse():
    """Verify that base is kept when FT is worse or not significantly better."""
    base_pos = _make_dummy_positions(25, top1_val=0.7, nll_val=1.2)
    ft_pos = _make_dummy_positions(25, top1_val=0.3, nll_val=2.5)

    res = should_promote(base_pos, ft_pos, n_test_games=25)

    assert res["decision"] == "promote_base"


def test_3_gate_insufficient_for_small_sample():
    """Verify that gate returns 'insufficient' when test_games < 20 and no val halves provided."""
    base_pos = _make_dummy_positions(10, top1_val=0.3, nll_val=2.5)
    ft_pos = _make_dummy_positions(10, top1_val=0.7, nll_val=1.2)

    res = should_promote(base_pos, ft_pos, n_test_games=10)

    assert res["decision"] == "insufficient"


def test_4_gate_uses_val_halves_when_available():
    """Verify small-sample fallback uses validation halves when test_games < 20."""
    base_pos = _make_dummy_positions(10, top1_val=0.4, nll_val=2.0)
    ft_pos = _make_dummy_positions(10, top1_val=0.6, nll_val=1.5)

    # Subcase A: FT NLL lower on BOTH halves -> promote_ft
    val_h1_base = {"nll": 2.2}
    val_h1_ft = {"nll": 1.8}
    val_h2_base = {"nll": 2.1}
    val_h2_ft = {"nll": 1.7}

    res_pass = should_promote(
        base_pos,
        ft_pos,
        n_test_games=10,
        val_split_half1_base=val_h1_base,
        val_split_half1_ft=val_h1_ft,
        val_split_half2_base=val_h2_base,
        val_split_half2_ft=val_h2_ft,
    )
    assert res_pass["decision"] == "promote_ft"

    # Subcase B: FT NLL higher on one half -> promote_base
    val_h2_ft_worse = {"nll": 2.4}
    res_fail = should_promote(
        base_pos,
        ft_pos,
        n_test_games=10,
        val_split_half1_base=val_h1_base,
        val_split_half1_ft=val_h1_ft,
        val_split_half2_base=val_h2_base,
        val_split_half2_ft=val_h2_ft_worse,
    )
    assert res_fail["decision"] == "promote_base"


def test_5_paired_bootstrap_raises_on_mismatch():
    """Verify that paired_game_bootstrap raises ValueError on mismatched game/ply keys."""
    base_pos = [{"game_id": "game_1", "ply": 1, "nll": 2.0}]
    ft_pos = [{"game_id": "game_2", "ply": 1, "nll": 1.5}]

    with pytest.raises(ValueError):
        paired_game_bootstrap(base_pos, ft_pos, "nll")


def test_6_reason_string_reflects_nll_only():
    """Regression: when FT has better NLL but worse top-1, the reason
    string must NOT falsely claim a top-1 improvement.

    Hand-computed scenario:
      base_pos: 30 games x 3 positions, top1=0.5, nll=2.0
      ft_pos:   30 games x 3 positions, top1=0.4, nll=1.0
    Expected: decision="promote_ft" (NLL clearly better),
              and "Top1" must not appear in the reason string.
    """
    base_pos = _make_dummy_positions(30, top1_val=0.5, nll_val=2.0)
    ft_pos = _make_dummy_positions(30, top1_val=0.4, nll_val=1.0)

    res = should_promote(base_pos, ft_pos, n_test_games=30)

    assert res["decision"] == "promote_ft", res
    assert "Top1" not in res["reason"], f"reason contains 'Top1': {res['reason']}"
    assert "CI =" in res["reason"], res["reason"]

