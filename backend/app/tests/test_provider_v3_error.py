"""Unit tests for V3Provider pattern-conditioned error model integration (Phase P)."""

import math
from unittest.mock import MagicMock, patch
import chess
import numpy as np
import pytest
import random

from app import config
from app.analysis.error_model import ErrorModel
from app.engine.provider import MoveContext
from app.engine.provider_v3 import V3Provider


def _make_dummy_provider(error_model=None, temperature=1.0, stockfish_analyser=None):
    mock_cache = MagicMock()
    mock_model = MagicMock()

    # Logits: e2e4 (idx 796) has logit 3.0, d2d4 (idx 731) has logit 2.0, others -2.0
    logits = np.full(4096, -2.0, dtype=np.float32)
    logits[12 * 64 + 28] = 3.0  # e2e4
    logits[11 * 64 + 27] = 2.0  # d2d4

    import torch
    mock_model.return_value = {"policy_logits": torch.from_numpy(logits).unsqueeze(0)}
    mock_cache.get.return_value = mock_model

    mock_tt = MagicMock()
    mock_tt.sample.return_value = 500

    provider = V3Provider(
        username="the_king_crusher",
        model_version=1,
        model_cache=mock_cache,
        think_time_model=mock_tt,
        temperature=temperature,
        error_model=error_model,
        stockfish_analyser=stockfish_analyser,
    )
    provider.temperature = float(temperature) if isinstance(temperature, (int, float)) else temperature
    return provider


def test_provider_when_error_model_disabled():
    """When ERROR_MODEL_ENABLED=False, skip error model and assert top5 matches temperature-only sample."""
    mock_em = MagicMock(spec=ErrorModel)
    provider = _make_dummy_provider(error_model=mock_em, temperature=1.0)

    board = chess.Board()
    ctx = MoveContext(
        board=board,
        prev_moves=[],
        bot_rating=1800,
        opp_rating=1800,
        base_seconds=180,
        increment_seconds=2,
        bot_clock_ms=100000,
        opp_clock_ms=100000,
        ply=0,
    )

    with patch.object(config, "ERROR_MODEL_ENABLED", False):
        rng = MagicMock()
        rng.random.return_value = 0.05
        rng.choices = random.Random(42).choices

        dec = provider.decide(ctx, rng)

        # Assert error model was NOT called
        assert not mock_em.predict.called
        # Assert p_err is None in telemetry
        assert dec.p_err is None
        # Assert top5 matches clean temperature-only sample (e2e4 > d2d4 with e2e4 having highest probability)
        assert dec.top5[0][0] == "e2e4"
        top5_dict = dict(dec.top5)
        assert top5_dict["e2e4"] > top5_dict["d2d4"]


def test_provider_error_model_prob_and_temperature_branches():
    if not getattr(config, "ERROR_MODEL_ENABLED", False):
        pytest.skip("ERROR_MODEL_ENABLED is False in config")

    mock_em = MagicMock(spec=ErrorModel)
    mock_em.predict.return_value = np.array([0.5])

    provider = _make_dummy_provider(error_model=mock_em, temperature=1.0)
    t_base = provider.temperature

    board = chess.Board()
    ctx = MoveContext(
        board=board,
        prev_moves=[],
        bot_rating=1800,
        opp_rating=1800,
        base_seconds=180,
        increment_seconds=2,
        bot_clock_ms=100000,
        opp_clock_ms=100000,
        ply=0,
    )

    # 1. Test error branch: u = 0.2 < p_err = 0.5 -> eff_temp = 1.1 * t_base
    rng_error = MagicMock()
    rng_error.random.return_value = 0.2
    rng_error.choices = random.Random(42).choices

    dec_err = provider.decide(ctx, rng_error)
    assert dec_err.p_err == 0.5
    top5_err = dict(dec_err.top5)

    # 2. Test clean branch: u = 0.8 >= p_err = 0.5 -> eff_temp = t_base
    rng_clean = MagicMock()
    rng_clean.random.return_value = 0.8
    rng_clean.choices = random.Random(42).choices

    dec_clean = provider.decide(ctx, rng_clean)
    assert dec_clean.p_err == 0.5
    top5_clean = dict(dec_clean.top5)

    # Under error branch, eff_temp is strictly 1.1 * t_base
    clean_log_ratio = math.log(top5_clean["e2e4"]) - math.log(top5_clean["d2d4"])
    err_log_ratio = math.log(top5_err["e2e4"]) - math.log(top5_err["d2d4"])
    eff_temp_measured = clean_log_ratio * t_base / err_log_ratio
    assert eff_temp_measured == pytest.approx(1.1 * t_base, rel=1e-3)

    # Under error branch, probability of top-1 move should be lower than under clean branch
    top_move = dec_clean.top5[0][0]
    p_clean_top = top5_clean[top_move]
    p_err_top = top5_err[top_move]
    assert p_err_top < p_clean_top, "Error branch temperature (1.1 * t_base) must flatten the distribution"


def test_provider_blunder_capping_resamples():
    if not getattr(config, "ERROR_MODEL_ENABLED", False):
        pytest.skip("ERROR_MODEL_ENABLED is False in config")

    mock_em = MagicMock(spec=ErrorModel)
    mock_em.predict.return_value = np.array([0.9])  # Always error branch

    # Mock Stockfish analyzer that reports cp_loss = 800 (exceeds cap of 600) for first move
    mock_sf = MagicMock()

    provider = _make_dummy_provider(
        error_model=mock_em,
        temperature=1.0,
        stockfish_analyser=mock_sf,
    )
    provider.player_p99 = 500.0  # Cap is min(500, 600) = 500

    board = chess.Board()
    ctx = MoveContext(
        board=board,
        prev_moves=[],
        bot_rating=1800,
        opp_rating=1800,
        base_seconds=180,
        increment_seconds=2,
        bot_clock_ms=100000,
        opp_clock_ms=100000,
        ply=0,
    )

    # Set up _compute_cp_loss so the first move triggers blunder cap (> 500), second doesn't
    call_count = 0
    def mock_loss(b, m):
        nonlocal call_count
        call_count += 1
        return 800.0 if call_count == 1 else 50.0

    provider._compute_cp_loss = mock_loss

    rng = MagicMock()
    rng.random.return_value = 0.1  # u < 0.9 -> error branch
    rng.choices = random.Random(123).choices

    dec = provider.decide(ctx, rng)
    assert call_count >= 1, "Expected blunder check to be called"
    assert dec.p_err == 0.9
