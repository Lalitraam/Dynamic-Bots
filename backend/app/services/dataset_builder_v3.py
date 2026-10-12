"""
Dataset builder v3 service: orchestrates extraction, splitting, recency weighting,
and saving to Parquet with Phase B schema.
"""
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
from datetime import datetime, timezone
import logging

from . import extractor
from . import splitter
from . import storage
from .. import config

logger = logging.getLogger(__name__)

# Map splits to an order for stable sorting
_SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}


def build_dataset_v3(username: str, retries_used: int = 0) -> dict:
    """
    Builds the v3 dataset for a player: extracts, splits, applies recency weighting,
    and writes samples_v3.parquet and info_v3.json. Does not write opening book or report.

    Parameters
    ----------
    username : str
        The player's username.
    retries_used : int
        Number of download retries used for this player's download stream.

    Returns
    -------
    dict
        Summary information written to info_v3.json.
    """
    # 1. Extract
    samples, drop_stats = extractor.extract_samples(username)

    # 2. Split
    samples = splitter.assign_splits(samples)

    # Add schema_version to all samples
    for row in samples:
        row["schema_version"] = config.SCHEMA_VERSION_V3

    out_dir = storage.player_dir(username)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not samples:
        info = {
            "total_samples": 0,
            "drop_stats": drop_stats,
            "model_tier": "rejected",
            "schema_version": config.SCHEMA_VERSION_V3,
            "build_date": datetime.now(timezone.utc).isoformat(),
            "retries_used": retries_used,
            "berserk_positions_fraction": 0.0,
        }
        with open(out_dir / "info_v3.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        return info

    # 3. Create DataFrame
    df = pd.DataFrame(samples)
    df["date"] = pd.to_datetime(df["date"], utc=True)

    # 4. Stable row order: sort by (split_order, date descending, game_id, ply)
    df["_split_order"] = df["split"].map(_SPLIT_ORDER)
    df = df.sort_values(
        by=["_split_order", "date", "game_id", "ply"],
        ascending=[True, False, True, True]
    ).drop(columns=["_split_order"])
    df.reset_index(drop=True, inplace=True)

    # 5. Recency weighting (Phase B.2)
    # sample_weight = 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)
    # age measured from the NEWEST game in the player's dataset (never wall-clock)
    # normalise to mean 1.0 over the TRAIN split; val/test weights = 1.0.
    newest_date = df["date"].max()
    age_days = (newest_date - df["date"]).dt.total_seconds() / 86400.0
    raw_weights = 0.5 ** (age_days / config.RECENCY_HALF_LIFE_DAYS)

    train_mask = (df["split"] == "train")
    if train_mask.sum() > 0:
        train_mean = raw_weights[train_mask].mean()
        if train_mean > 0:
            df.loc[train_mask, "sample_weight"] = (raw_weights[train_mask] / train_mean).astype("float32")
        else:
            df.loc[train_mask, "sample_weight"] = 1.0
    else:
        df["sample_weight"] = 1.0

    df.loc[~train_mask, "sample_weight"] = 1.0
    df["sample_weight"] = df["sample_weight"].astype("float32")

    # 6. Enforce fixed dtypes and explicit v3 schema for pyarrow
    schema_v3 = pa.schema([
        pa.field('game_id', pa.string()),
        pa.field('ply', pa.int32()),
        pa.field('fen', pa.string()),
        pa.field('uci', pa.string()),
        pa.field('date', pa.timestamp('ms', tz='UTC')),
        pa.field('category', pa.string()),
        pa.field('is_fast', pa.bool_()),
        pa.field('time_control', pa.string()),
        pa.field('player_rating', pa.int32()),
        pa.field('opponent_rating', pa.int32()),
        pa.field('color', pa.string()),
        pa.field('termination', pa.string()),
        pa.field('clock_before_move', pa.float64()),
        pa.field('halfmove_clock', pa.int32()),
        pa.field('sample_weight', pa.float32()),
        pa.field('split', pa.string()),
        pa.field('schema_version', pa.string()),
        pa.field('target_berserk', pa.bool_()),
        # Phase B.1 additive columns
        pa.field('prev_move_1', pa.string()),
        pa.field('prev_move_2', pa.string()),
        pa.field('clock_after_move', pa.float64()),
        pa.field('time_spent_s', pa.float64()),
        pa.field('increment_s', pa.int32()),
        pa.field('base_s', pa.int32()),
        pa.field('created_at_ms', pa.int64()),
        pa.field('game_result_for_player', pa.float32()),
        pa.field('num_legal_moves', pa.int32()),
    ])

    v3_columns = [f.name for f in schema_v3]
    df = df[v3_columns]

    table = pa.Table.from_pandas(df, schema=schema_v3, preserve_index=False)

    # 7. Save to samples_v3.parquet
    parquet_path = out_dir / "samples_v3.parquet"
    pq.write_table(table, parquet_path, compression="snappy")

    # 8. Compute summary for info_v3.json
    n_games = df["game_id"].nunique()
    model_tier = config.model_tier(n_games)
    split_counts = df["split"].value_counts().to_dict()
    berserk_fraction = float(df["target_berserk"].mean()) if (len(df) > 0 and "target_berserk" in df) else 0.0

    info = {
        "total_samples": len(df),
        "split_counts": split_counts,
        "model_tier": model_tier,
        "schema_version": config.SCHEMA_VERSION_V3,
        "build_date": datetime.now(timezone.utc).isoformat(),
        "drop_stats": drop_stats,
        "retries_used": retries_used,
        "berserk_positions_fraction": berserk_fraction,
    }

    with open(out_dir / "info_v3.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    return info
