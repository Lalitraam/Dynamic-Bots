"""Tests exercising real fine-tuned models through V3Provider (Phase E follow-up)."""

from pathlib import Path
import random
import chess

from app import config
from app.engine.model_cache import ModelCache
from app.engine.provider import MoveContext, MoveDecision
from app.engine.provider_v3 import V3Provider
from app.engine.think_time import ThinkTimeModel


def test_1_ft_model_loads_through_cache():
    cache = ModelCache()
    model = cache.get("the_king_crusher", 1)
    assert model is not None

    weights_file = config.DATA_ROOT / "players" / "the_king_crusher" / "models" / "v1" / "weights.pt"
    assert weights_file.exists(), "the_king_crusher weights.pt must exist for FT test"

    tt_model = ThinkTimeModel({("blitz", "opening", ">0.3", False): [0.5] * 25})

    provider = V3Provider(
        username="the_king_crusher",
        model_version=1,
        model_cache=cache,
        think_time_model=tt_model,
        book=None,
    )
    assert not provider.is_base_fallback

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
    assert decision.model_version == "the_king_crusher:v1"
    assert decision.move in ctx.board.legal_moves
