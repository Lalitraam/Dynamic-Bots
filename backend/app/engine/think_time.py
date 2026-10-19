"""Empirical think-time model (Phase E)."""

from __future__ import annotations
from pathlib import Path
import random
from typing import TYPE_CHECKING, Any
import pandas as pd

from app import config

if TYPE_CHECKING:
    from app.engine.provider import MoveContext


def _phase_from_ply(ply: int) -> str:
    """Return game phase string from ply: opening (<=20), middlegame (21-60), endgame (>60)."""
    if ply <= 20:
        return "opening"
    elif ply <= 60:
        return "middlegame"
    return "endgame"


def _clock_ratio_bucket(ratio: float | None) -> str:
    """Return clock ratio category: <0.1, 0.1-0.3, >0.3, or unknown."""
    if ratio is None:
        return "unknown"
    if ratio < 0.1:
        return "<0.1"
    elif ratio <= 0.3:
        return "0.1-0.3"
    return ">0.3"


def _speed_from_time_control(base_s: int, inc_s: int) -> str:
    """Return speed category (bullet, blitz, rapid) from base and increment seconds."""
    estimated = base_s + 60 * inc_s
    if estimated < 180:
        return "bullet"
    elif estimated < 600:
        return "blitz"
    return "rapid"


def _bucket_key(
    speed: str,
    phase: str,
    clock_ratio: float | str | None,
    is_forced: bool,
) -> tuple[str, str, str, bool]:
    """Return the bucket key tuple: (speed, phase, clock_ratio_bucket, is_forced)."""
    ratio_bucket = (
        clock_ratio
        if isinstance(clock_ratio, str)
        else _clock_ratio_bucket(clock_ratio)
    )
    return (speed, phase, ratio_bucket, bool(is_forced))


class ThinkTimeModel:
    """Empirical think-time sampler, bucketed by (speed, phase, clock_ratio, forced)."""

    def __init__(self, buckets: dict | None = None):
        """buckets: mapping from bucket key to list of time_spent_s samples."""
        self.buckets = {k: list(v) for k, v in buckets.items()} if buckets else {}

    @classmethod
    def from_parquet(
        cls,
        parquet_path: Path,
        max_samples_per_bucket: int = config.THINK_MAX_SAMPLES_PER_BUCKET,
        seed: int = 0,
    ) -> "ThinkTimeModel":
        """Build buckets from a player's samples_v3.parquet.
        Only rows with non-null time_spent_s are used.
        Each bucket key = (speed_category, phase, clock_ratio_bucket, is_forced).
        Reservoir-sample each bucket to at most max_samples_per_bucket values.
        Uses fixed seed for determinism.
        """
        df = pd.read_parquet(parquet_path)
        valid = df[df["time_spent_s"].notnull()]
        cols = [c for c in ["base_s", "increment_s", "ply", "clock_before_move", "num_legal_moves", "time_spent_s"] if c in valid.columns]
        records = valid[cols].to_dict("records")

        bucket_raw: dict[tuple, list[float]] = {}
        for row in records:
            base_s = int(row.get("base_s", 180))
            inc_s = int(row.get("increment_s", 0))
            speed = _speed_from_time_control(base_s, inc_s)
            ply = int(row.get("ply", 0))
            phase = _phase_from_ply(ply)

            cb = row.get("clock_before_move")
            ratio = (float(cb) / base_s) if (cb is not None and pd.notnull(cb) and base_s > 0) else None
            ratio_b = _clock_ratio_bucket(ratio)

            num_legal = row.get("num_legal_moves")
            is_forced = bool(num_legal == 1) if (num_legal is not None and pd.notnull(num_legal)) else False

            key = (speed, phase, ratio_b, is_forced)
            t_s = float(row["time_spent_s"])
            bucket_raw.setdefault(key, []).append(t_s)

        # Deterministic reservoir sampling per bucket
        rng = random.Random(seed)
        sampled_buckets: dict[tuple, list[float]] = {}
        for key in sorted(bucket_raw.keys()):
            vals = bucket_raw[key]
            if len(vals) <= max_samples_per_bucket:
                sampled_buckets[key] = list(vals)
            else:
                res = vals[:max_samples_per_bucket].copy()
                for i in range(max_samples_per_bucket, len(vals)):
                    j = rng.randint(0, i)
                    if j < max_samples_per_bucket:
                        res[j] = vals[i]
                sampled_buckets[key] = res

        return cls(sampled_buckets)

    def sample(self, ctx: MoveContext, rng: random.Random) -> int:
        """Return think_ms for the given context.
        1. Compute bucket from ctx.
        2. If bucket has >= THINK_MIN_SAMPLES_PER_BUCKET samples, sample from it.
        3. Else fall back: same speed+phase, then same speed, then pooled.
        4. If NO samples exist at all, return uniform in [500, 3000] ms.
        5. Clamp: max(THINK_FLOOR_MS, min(sample, bot_clock_ms - THINK_MARGIN_MS)).
        """
        speed = _speed_from_time_control(ctx.base_seconds, ctx.increment_seconds)
        phase = _phase_from_ply(ctx.ply)
        ratio = (
            (ctx.bot_clock_ms / 1000.0) / ctx.base_seconds
            if ctx.base_seconds > 0 and ctx.bot_clock_ms is not None
            else None
        )
        is_forced = (ctx.board.legal_moves.count() == 1)
        target_key = _bucket_key(speed, phase, ratio, is_forced)

        min_samples = config.THINK_MIN_SAMPLES_PER_BUCKET

        candidates: list[float] = []
        # 1. Exact bucket
        if target_key in self.buckets and len(self.buckets[target_key]) >= min_samples:
            candidates = self.buckets[target_key]
        else:
            # 2. Same speed + phase
            same_sp_ph: list[float] = []
            for k, vals in self.buckets.items():
                if k[0] == speed and k[1] == phase:
                    same_sp_ph.extend(vals)
            if len(same_sp_ph) >= min_samples:
                candidates = same_sp_ph
            else:
                # 3. Same speed
                same_sp: list[float] = []
                for k, vals in self.buckets.items():
                    if k[0] == speed:
                        same_sp.extend(vals)
                if len(same_sp) >= min_samples:
                    candidates = same_sp
                else:
                    # 4. Pooled across all buckets
                    pooled: list[float] = []
                    for vals in self.buckets.values():
                        pooled.extend(vals)
                    if len(pooled) > 0:
                        candidates = pooled



        # 4. If NO samples exist at all
        if not candidates:
            sample_ms = rng.randint(500, 3000)
        else:
            sample_s = rng.choice(candidates)
            sample_ms = int(round(sample_s * 1000))

        # 5. Clamp
        think_cap = getattr(config, "THINK_CAP_MS", 6000)
        is_fast = ctx.base_seconds < 480  # Blitz or Bullet
        if is_fast:
            max_allowed = min(think_cap, int(ctx.bot_clock_ms * 0.02))
        else:
            max_allowed = min(10000, int(ctx.bot_clock_ms * 0.05))

        # Ensure we never go below the floor (150ms)
        clamped = max(config.THINK_FLOOR_MS, min(sample_ms, max_allowed))
        return int(clamped)

