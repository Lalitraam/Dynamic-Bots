from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import pyarrow.parquet as pq

from app import config


@lru_cache(maxsize=32)
def get_player_median_rating(username: str) -> int:
    """Compute and cache the player's median rating from samples_v3.parquet (all splits)."""
    p = Path(config.DATA_ROOT) / "players" / username / "samples_v3.parquet"
    if not p.exists():
        return config.DEFAULT_RATING
    df = pq.read_table(p, columns=['player_rating']).to_pandas()
    return int(df['player_rating'].median())
