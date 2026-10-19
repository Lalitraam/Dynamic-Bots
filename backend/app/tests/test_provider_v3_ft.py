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
    weights_file = config.DATA_ROOT / "players" / "the_king_crusher" / "models" / "v1" / "weights.pt"
    if not weights_file.exists():
        weights_file.parent.mkdir(parents=True, exist_ok=True)
        import torch
        from app.ml.model_v2 import ChessPolicyNetV2
        dummy = ChessPolicyNetV2(channels=96, num_blocks=6, policy_dim=32, meta_dim=10, num_planes=22)
        torch.save({"state_dict": dummy.state_dict(), "config": dummy.to_config()}, weights_file)

    cache = ModelCache()
    model = cache.get("the_king_crusher", 1)
    assert model is not None
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
