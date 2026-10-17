"""Tests for F0 frozen interfaces (app/engine/provider.py)."""

import inspect
import random
import chess
from app.engine.provider import MoveContext, MoveDecision, MoveProvider


def test_1_move_context_dataclass_constructs():
    board = chess.Board()
    ctx = MoveContext(
        board=board,
        prev_moves=["e2e4", "e7e5"],
        bot_rating=1800,
        opp_rating=1750,
        base_seconds=180,
        increment_seconds=2,
        bot_clock_ms=120000,
        opp_clock_ms=130000,
        ply=2,
    )
    assert ctx.board == board
    assert ctx.prev_moves == ["e2e4", "e7e5"]
    assert ctx.bot_rating == 1800
    assert ctx.opp_rating == 1750
    assert ctx.base_seconds == 180
    assert ctx.increment_seconds == 2
    assert ctx.bot_clock_ms == 120000
    assert ctx.opp_clock_ms == 130000
    assert ctx.ply == 2


def test_2_move_decision_dataclass_constructs():
    move = chess.Move.from_uci("g1f3")
    dec = MoveDecision(
        move=move,
        source="policy",
        think_ms=450,
        top5=[("g1f3", 0.45), ("d2d4", 0.30)],
        p_err=None,
        model_version="test:v1",
    )
    assert dec.move == move
    assert dec.source == "policy"
    assert dec.think_ms == 450
    assert len(dec.top5) == 2
    assert dec.p_err is None
    assert dec.model_version == "test:v1"


def test_3_move_provider_protocol_has_decide_method():
    assert hasattr(MoveProvider, "decide")
    sig = inspect.signature(MoveProvider.decide)
    params = list(sig.parameters.keys())
    assert params == ["self", "ctx", "rng"]

