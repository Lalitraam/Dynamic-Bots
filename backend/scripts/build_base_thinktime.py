"""Build pooled base think-time distribution from base parquet shards (Phase SP)."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import sys

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from app import config


def build_base_thinktime(
    shards_dir: Path | str,
    out_dir: Path | str,
    max_samples_per_bucket: int = config.BASE_THINKTIME_MAX_PER_BUCKET,
    seed: int = config.BASE_THINKTIME_SEED,
) -> Path:
    shards_p = Path(shards_dir)
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(shards_p.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {shards_p}")

    cols = [
        "player_rating",
        "base_s",
        "increment_s",
        "ply",
        "clock_before_move",
        "num_legal_moves",
        "time_spent_s",
    ]

    bucket_raw: dict[str, list[float]] = {}

    for pf in parquet_files:
        print(f"Reading {pf.name}...")
        df = pq.read_table(pf, columns=cols).to_pandas()
        df = df[df["time_spent_s"].notnull()]
        if len(df) == 0:
            continue

        ratings = df["player_rating"].fillna(1500).values
        rating_bands = (np.round(ratings / 200.0) * 200).astype(int)

        base_s = df["base_s"].fillna(180).values.astype(int)
        inc_s = df["increment_s"].fillna(0).values.astype(int)
        est = base_s + 60 * inc_s
        speeds = np.where(est < 180, "bullet", np.where(est < 600, "blitz", "rapid"))

        plies = df["ply"].fillna(0).values.astype(int)
        phases = np.where(plies <= 20, "opening", np.where(plies <= 60, "middlegame", "endgame"))

        clocks = df["clock_before_move"].values
        has_clock = ~pd.isna(clocks) & (base_s > 0)
        ratios = np.where(has_clock, clocks / np.maximum(base_s, 1), np.nan)
        clock_ratios = np.where(
            np.isnan(ratios),
            "unknown",
            np.where(ratios < 0.1, "<0.1", np.where(ratios <= 0.3, "0.1-0.3", ">0.3")),
        )

        is_forced = (df["num_legal_moves"].fillna(0).values == 1)

        keys = (
            rating_bands.astype(str)
            + "|"
            + speeds
            + "|"
            + phases
            + "|"
            + clock_ratios
            + "|"
            + is_forced.astype(str)
        )
        df["key"] = keys

        grouped = df.groupby("key")["time_spent_s"].apply(list).to_dict()
        for k, v in grouped.items():
            bucket_raw.setdefault(k, []).extend(v)

    print(f"Total unique raw buckets: {len(bucket_raw)}")

    # Reservoir sample each bucket deterministically
    rng = random.Random(seed)
    sampled_buckets: dict[str, list[float]] = {}
    for key in sorted(bucket_raw.keys()):
        vals = bucket_raw[key]
        if len(vals) <= max_samples_per_bucket:
            sampled_buckets[key] = [round(float(x), 4) for x in vals]
        else:
            reservoir = vals[:max_samples_per_bucket].copy()
            for i in range(max_samples_per_bucket, len(vals)):
                j = rng.randint(0, i)
                if j < max_samples_per_bucket:
                    reservoir[j] = vals[i]
            sampled_buckets[key] = [round(float(x), 4) for x in reservoir]

    out_file = out_p / "pooled_thinktime.json"
    result = {
        "metadata": {
            "seed": seed,
            "max_samples_per_bucket": max_samples_per_bucket,
            "shards_dir": str(shards_p.resolve()),
        },
        "buckets": sampled_buckets,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    sizes = [len(v) for v in sampled_buckets.values()]
    print("==================================================")
    print("BASE THINK-TIME SUMMARY:")
    print(f"  Num buckets:        {len(sampled_buckets)}")
    print(f"  Total samples:      {sum(sizes)}")
    print(f"  Min bucket size:    {min(sizes) if sizes else 0}")
    print(f"  Max bucket size:    {max(sizes) if sizes else 0}")
    print(f"  Output saved to:    {out_file}")
    print("==================================================")

    return out_file


def main():
    parser = argparse.ArgumentParser(description="Build pooled base think-time distribution.")
    parser.add_argument(
        "--shards-dir",
        type=str,
        default=str(config.DATA_ROOT / "base" / "shards" / "base_shards_clean2"),
        help="Directory containing base parquet shards.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(config.BASE_THINKTIME_DIR),
        help="Output directory for pooled_thinktime.json.",
    )
    parser.add_argument(
        "--max-samples-per-bucket",
        type=int,
        default=config.BASE_THINKTIME_MAX_PER_BUCKET,
        help="Maximum samples per bucket.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.BASE_THINKTIME_SEED,
        help="Random seed for deterministic reservoir sampling.",
    )
    args = parser.parse_args()

    build_base_thinktime(
        shards_dir=args.shards_dir,
        out_dir=args.out_dir,
        max_samples_per_bucket=args.max_samples_per_bucket,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
