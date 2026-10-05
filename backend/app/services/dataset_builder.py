"""
Dataset builder service: orchestrates extraction, splitting, and saving to Parquet.
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

def build_dataset(username: str) -> dict:
    """
    Builds the dataset for a player: extracts, splits, and writes samples.parquet.
    Also produces opening_book.json, dataset_report.md, info.json (delegated later).

    Parameters
    ----------
    username : str
        The player's username.
        
    Returns
    -------
    dict
        Summary information to be written to info.json and returned to the router.
    """
    # 1. Extract
    samples, drop_stats = extractor.extract_samples(username)
    
    # 2. Split
    samples = splitter.assign_splits(samples)
    
    # Add schema_version to all samples
    for row in samples:
        row["schema_version"] = config.SCHEMA_VERSION
        
    if not samples:
        # No samples to save
        return {
            "total_samples": 0,
            "drop_stats": drop_stats,
            "model_tier": "rejected",
            "schema_version": config.SCHEMA_VERSION,
            "build_date": datetime.now(timezone.utc).isoformat()
        }

    # 3. Create DataFrame
    df = pd.DataFrame(samples)
    
    # Convert 'date' to sortable if it's datetime
    # We can just use the datetime objects directly for sorting
    
    # 4. Stable row order: sort by (split_order, date descending, game_id, ply)
    # The user asked for "date descending" in the review? Let me re-read:
    # "(split_order, date descending, game_id, ply)"
    df["_split_order"] = df["split"].map(_SPLIT_ORDER)
    df = df.sort_values(
        by=["_split_order", "date", "game_id", "ply"],
        ascending=[True, False, True, True]
    ).drop(columns=["_split_order"])
    
    df.reset_index(drop=True, inplace=True)
    
    # 5. Enforce fixed dtypes for idempotency
    # We define the schema for pyarrow explicitly to ensure exact match
    schema = pa.schema([
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
    ])

    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    
    # 6. Save to Parquet
    out_dir = storage.player_dir(username)
    parquet_path = out_dir / "samples.parquet"
    
    pq.write_table(table, parquet_path, compression="snappy")
    
    # Compute summary for info.json
    n_games = df["game_id"].nunique()
    model_tier = config.model_tier(n_games)
    
    split_counts = df["split"].value_counts().to_dict()
    
    info = {
        "total_samples": len(df),
        "split_counts": split_counts,
        "model_tier": model_tier,
        "schema_version": config.SCHEMA_VERSION,
        "build_date": datetime.now(timezone.utc).isoformat(),
        "drop_stats": drop_stats,
    }
    
    # Write info.json
    with open(out_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
        
    # 7. Opening book
    from . import opening_book
    opening_book.build_opening_book(samples, username)
    
    # 8. Report
    from . import report
    report.generate_report(username, samples, info)
        
    return info
