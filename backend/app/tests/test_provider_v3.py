"""Tests for V3Provider (backend/app/engine/provider_v3.py)."""

from pathlib import Path
import random
import chess
import numpy as np
import pytest
import torch

from app import config
from app.engine.model_cache import ModelCache
from app.engine.provider import MoveContext, MoveDecision
from app.engine.provider_v3 import (
    V3Provider,
    _apply_top_p,
    _build_board_planes,
    _build_meta,
    legal_indices,
)
from app.engine.think_time import ThinkTimeModel
from app.ml.model_v2 import ChessPolicyNetV2


def _make_dummy_provider(data_root: Path | None = None) -> V3Provider:
    cache = ModelCache(capacity=8, data_root=data_root or config.DATA_ROOT)
    # Dummy think-time model with sufficient samples for default bucket
    key = ("blitz", "opening", ">0.3", False)
    tt_model = ThinkTimeModel({key: [0.5] * 25})
    return V3Provider(
        username="dummy_player",
        model_version=1,
        model_cache=cache,
        think_time_model=tt_model,
        book=None,
        temperature=1.0,
        data_root=data_root or config.DATA_ROOT,
    )


def test_1_provider_returns_legal_move():
    provider = _make_dummy_provider()
    ctx = MoveContext(
        board=chess.Board(),
        prev_moves=[],
        bot_rating=1800,
        opp_rating=1800,
        base_seconds=180,
        increment_seconds=2,
        bot_clock_ms=120000,
        opp_clock_ms=120000,
        ply=0,
    )
    rng = random.Random(42)
    decision = provider.decide(ctx, rng)
    assert isinstance(decision, MoveDecision)
    assert decision.move in ctx.board.legal_moves
    assert decision.source in ("book", "policy")
    assert decision.think_ms >= config.THINK_FLOOR_MS


def test_2_provider_deterministic_with_seed():
    provider = _make_dummy_provider()
    ctx = MoveContext(
        board=chess.Board(),
        prev_moves=[],
        bot_rating=1900,
        opp_rating=1850,
        base_seconds=180,
        increment_seconds=2,
        bot_clock_ms=100000,
        opp_clock_ms=100000,
        ply=0,
    )
    d1 = provider.decide(ctx, random.Random(12345))
    d2 = provider.decide(ctx, random.Random(12345))
    assert d1.move == d2.move
    assert d1.think_ms == d2.think_ms
    assert d1.top5 == d2.top5
    assert d1.source == d2.source


def test_3_provider_think_time_within_bounds():
    provider = _make_dummy_provider()
    bot_clock_ms = 50000
    ctx = MoveContext(
        board=chess.Board(),
        prev_moves=[],
        bot_rating=1800,
        opp_rating=1800,
        base_seconds=180,
        increment_seconds=2,
        bot_clock_ms=bot_clock_ms,
        opp_clock_ms=50000,
        ply=10,
    )
    rng = random.Random(99)
    decision = provider.decide(ctx, rng)
    min_allowed = config.THINK_FLOOR_MS
    max_allowed = bot_clock_ms - config.THINK_MARGIN_MS
    assert min_allowed <= decision.think_ms <= max_allowed


def test_4_meta_vector_reaches_model(monkeypatch):
    provider = _make_dummy_provider()
    captured_meta = []
    real_forward = ChessPolicyNetV2.forward

    def mock_forward(self, board, meta, **kwargs):
        captured_meta.append(meta.clone())
        return real_forward(self, board, meta, **kwargs)

    monkeypatch.setattr(ChessPolicyNetV2, "forward", mock_forward)

    bot_rating = 2300
    base_seconds = 180
    bot_clock_ms = 90000
    ctx = MoveContext(
        board=chess.Board(),
        prev_moves=[],
        bot_rating=bot_rating,
        opp_rating=1800,
        base_seconds=base_seconds,
        increment_seconds=2,
        bot_clock_ms=bot_clock_ms,
        opp_clock_ms=90000,
        ply=15,
    )
    provider.decide(ctx, random.Random(42))

    assert len(captured_meta) == 1
    meta_vec = captured_meta[0][0].cpu().numpy()
    expected_meta_0 = (bot_rating - 1500) / 400.0
    expected_meta_5 = bot_clock_ms / (base_seconds * 1000.0)

    assert np.isclose(meta_vec[0], expected_meta_0, atol=1e-5)
    assert np.isclose(meta_vec[5], expected_meta_5, atol=1e-5)


def test_5_promotion_legality_1000_random_positions():
    provider = _make_dummy_provider()
    rng = random.Random(777)

    # Diverse test positions ensuring standard games, promotion opportunities, and en passant
    start_fens = [
        chess.STARTING_FEN,
        "8/4P3/8/8/8/8/8/k6K w - - 0 1",  # White imminent promotion
        "k6K/8/8/8/8/8/4p3/8 b - - 0 1",  # Black imminent promotion
        "rnbqkbnr/pppp1ppp/8/4pP2/8/8/PPPPP1PP/RNBQKBNR w KQkq e6 0 2",  # White en passant available
        "rnbqkbnr/ppppp1pp/8/8/4Pp2/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 2",  # Black en passant available
    ]

    board = chess.Board(start_fens[0])
    fen_idx = 0

    for _ in range(1000):
        if board.is_game_over() or board.ply() > 80:
            fen_idx = (fen_idx + 1) % len(start_fens)
            board = chess.Board(start_fens[fen_idx])

        ctx = MoveContext(
            board=board,
            prev_moves=[m.uci() for m in board.move_stack[-2:]],
            bot_rating=1800,
            opp_rating=1800,
            base_seconds=180,
            increment_seconds=2,
            bot_clock_ms=60000,
            opp_clock_ms=60000,
            ply=board.ply(),
        )

        decision = provider.decide(ctx, rng)
        assert decision.move in board.legal_moves
        board.push(decision.move)


def test_6_top_p_filter_keeps_cumulative_leq_p():
    probs = np.array([0.55, 0.22, 0.12, 0.06, 0.03, 0.01, 0.01])
    filtered = _apply_top_p(probs, p=0.95)

    assert np.isclose(filtered[0], 0.55)
    assert np.isclose(filtered[1], 0.22)
    assert np.isclose(filtered[2], 0.12)
    assert np.isclose(filtered[3], 0.06)
    assert filtered[4] == 0.0
    assert filtered[5] == 0.0
    assert filtered[6] == 0.0


def test_7_provider_decision_includes_p_err():
    provider = _make_dummy_provider()
    ctx = MoveContext(
        board=chess.Board(),
        prev_moves=[],
        bot_rating=1800,
        opp_rating=1800,
        base_seconds=180,
        increment_seconds=2,
        bot_clock_ms=120000,
        opp_clock_ms=120000,
        ply=0,
    )
    if not getattr(config, "ERROR_MODEL_ENABLED", False):
        decision = provider.decide(ctx, random.Random(42))
        assert decision.p_err is None
    else:
        decision = provider.decide(ctx, random.Random(42))
        assert decision.p_err is not None
        assert 0.0 <= decision.p_err <= 1.0

