"""
Sampler service: draw a stratified random sample from the bucketed games.

Sampling strategy
-----------------
1. Draw up to BUCKET_QUOTAS["last_6m"] games from the last_6m bucket.
2. Redistribute any unused quota from last_6m to 6_18m (up to its quota).
3. Redistribute any unused quota from 6_18m  to 18_36m (up to its quota).
4. If the total is still below MAX_TOTAL_SAMPLED and any bucket has unused
   games, fill greedily from unused games (newest-first: last_6m, 6_18m,
   18_36m) until MAX_TOTAL_SAMPLED is reached or games run out.

Returns
-------
dict with keys:
  games         : list[dict]   – sampled games, newest first
  bullet_count  : int
  blitz_count   : int
  rapid_count   : int
  quota_used    : dict[str, int]  – how many taken from each bucket
"""

import random
from .. import config

_BUCKET_ORDER = ["last_6m", "6_18m", "18_36m"]


def sample_games(
    buckets: dict[str, list[dict]],
    seed: int | None = None,
) -> dict:
    """
    Draw a stratified sample from *buckets* according to config quotas.

    Parameters
    ----------
    buckets : dict[str, list[dict]]
        Game lists keyed by bucket name.  Each game dict must have at least
        a "category" field.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    dict with keys: games, bullet_count, blitz_count, rapid_count, quota_used
    """
    rng = random.Random(seed)
    quotas = config.BUCKET_QUOTAS  # {"last_6m": 500, "6_18m": 400, "18_36m": 300}
    max_total = config.MAX_TOTAL_SAMPLED  # 1200

    # Make shuffled copies so we don't mutate the caller's lists.
    shuffled: dict[str, list[dict]] = {}
    for bucket in _BUCKET_ORDER:
        games = list(buckets.get(bucket, []))
        rng.shuffle(games)
        shuffled[bucket] = games

    selected: dict[str, list[dict]] = {b: [] for b in _BUCKET_ORDER}
    remaining_quota = {b: quotas.get(b, 0) for b in _BUCKET_ORDER}

    # --- Pass 1: each bucket takes up to its own quota ---
    for bucket in _BUCKET_ORDER:
        take = min(remaining_quota[bucket], len(shuffled[bucket]))
        selected[bucket] = shuffled[bucket][:take]
        shuffled[bucket] = shuffled[bucket][take:]  # leftovers

    # --- Pass 2: redistribute unused quota toward newer→older ---
    carried_over = 0
    for i, bucket in enumerate(_BUCKET_ORDER):
        remaining_quota[bucket] = remaining_quota[bucket] - len(selected[bucket]) + carried_over
        if remaining_quota[bucket] > 0 and i + 1 < len(_BUCKET_ORDER):
            next_bucket = _BUCKET_ORDER[i + 1]
            extra = min(remaining_quota[bucket], len(shuffled[next_bucket]))
            selected[next_bucket].extend(shuffled[next_bucket][:extra])
            shuffled[next_bucket] = shuffled[next_bucket][extra:]
            carried_over = remaining_quota[bucket] - extra
        else:
            carried_over = 0

    # --- Pass 3: fill remaining quota from any bucket, newest first ---
    total_so_far = sum(len(v) for v in selected.values())
    if total_so_far < max_total:
        for bucket in _BUCKET_ORDER:
            if total_so_far >= max_total:
                break
            can_take = min(max_total - total_so_far, len(shuffled[bucket]))
            if can_take > 0:
                selected[bucket].extend(shuffled[bucket][:can_take])
                shuffled[bucket] = shuffled[bucket][can_take:]
                total_so_far += can_take

    # --- Assemble result: merge buckets newest-first, preserving order ---
    all_games: list[dict] = []
    for bucket in _BUCKET_ORDER:
        all_games.extend(selected[bucket])

    # Count by category
    bullet_count = sum(1 for g in all_games if g.get("category") == "bullet")
    blitz_count = sum(1 for g in all_games if g.get("category") == "blitz")
    rapid_count = sum(1 for g in all_games if g.get("category") == "rapid")

    quota_used = {b: len(selected[b]) for b in _BUCKET_ORDER}

    return {
        "games": all_games,
        "bullet_count": bullet_count,
        "blitz_count": blitz_count,
        "rapid_count": rapid_count,
        "quota_used": quota_used,
    }
