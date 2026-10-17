"""
Tensor cache builder and validator for Phase B chess imitation learning.

Precomputes and serializes packed board planes, metadata, targets, and legal move
indices in CSR format to disk for high-throughput training data loading.

Dependencies: numpy, pyarrow, config, encoding_v3 ONLY (no torch).
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from .. import config
from ..services import storage
from . import encoding_v3

SPLIT_MAP: dict[str, int] = {
    "train": 0,
    "val": 1,
    "test": 2,
}

ARRAY_FILES: list[str] = [
    "planes.npy",
    "meta.npy",
    "target.npy",
    "promo.npy",
    "legal_ptr.npy",
    "legal_idx.npy",
    "weight.npy",
    "split.npy",
    "game_idx.npy",
    "berserk.npy",
]


def build_cache(
    username: str,
    data_root: Path | None = None,
    history: bool = config.HISTORY_PLANES,
) -> Path:
    """Build the tensor cache for a player from samples_v3.parquet.

    Reads samples_v3.parquet in stored row order and writes cache_v3/:
    - planes.npy: uint8 (N, ceil(P*64/8)) = np.packbits of flattened (P,8,8)
    - meta.npy: float32 (N, 10)
    - target.npy: int16 (N,)
    - promo.npy: int8 (N,)
    - legal_ptr.npy: int64 (N+1,) CSR row pointers
    - legal_idx.npy: int16 (total_legal,)
    - weight.npy: float32 (N,)
    - split.npy: int8 (N,) (0=train, 1=val, 2=test)
    - game_idx.npy: int32 (N,) index into sorted unique game_ids
    - berserk.npy: int8 (N,) (1 if target_berserk else 0)
    - manifest.json: metadata written LAST

    Arrays are byte-deterministic across builds.

    Parameters
    ----------
    username : str
        Player username.
    data_root : Path | None
        Data root directory (defaults to config.DATA_ROOT).
    history : bool
        Whether to generate history planes (defaults to config.HISTORY_PLANES).

    Returns
    -------
    Path
        Path to the generated cache_v3 directory.
    """
    if data_root is not None:
        p_dir = data_root / "players" / username
    else:
        p_dir = storage.player_dir(username, create=False)

    parquet_path = p_dir / "samples_v3.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"samples_v3.parquet not found at {parquet_path}")

    parquet_bytes = parquet_path.read_bytes()
    parquet_sha256 = hashlib.sha256(parquet_bytes).hexdigest()

    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    N = len(df)

    P = 22 if history else 18
    packed_dim = math.ceil(P * 64 / 8)

    all_planes = np.zeros((N, packed_dim), dtype=np.uint8)
    all_meta = np.zeros((N, config.META_DIM_V3), dtype=np.float32)
    all_target = np.zeros(N, dtype=np.int16)
    all_promo = np.zeros(N, dtype=np.int8)
    all_weight = np.zeros(N, dtype=np.float32)
    all_split = np.zeros(N, dtype=np.int8)
    all_berserk = np.zeros(N, dtype=np.int8)
    legal_ptr = np.zeros(N + 1, dtype=np.int64)

    # Sorted unique game_ids and lookup
    sorted_game_ids = sorted(list(set(df["game_id"].tolist())))
    game_id_to_idx = {gid: idx for idx, gid in enumerate(sorted_game_ids)}
    game_idx_arr = np.array([game_id_to_idx[gid] for gid in df["game_id"]], dtype=np.int32)

    legal_idx_list: list[np.ndarray] = []

    records = df.to_dict(orient="records")
    for i, row in enumerate(records):
        enc = encoding_v3.encode_row_v3(row, history=history)

        # Pack board planes with default bitorder ("big")
        planes_flat = enc["planes"].reshape(-1)
        all_planes[i] = np.packbits(planes_flat)

        all_meta[i] = enc["meta"]
        all_target[i] = enc["move_idx"]
        all_promo[i] = enc["promo"]
        all_weight[i] = enc["weight"]

        split_str = str(row.get("split", "train"))
        all_split[i] = SPLIT_MAP.get(split_str, 0)

        all_berserk[i] = 1 if enc["berserk"] else 0

        l_idx = enc["legal_idx"]
        legal_idx_list.append(l_idx)
        legal_ptr[i + 1] = legal_ptr[i] + len(l_idx)

    if legal_idx_list:
        legal_idx_arr = np.concatenate(legal_idx_list).astype(np.int16)
    else:
        legal_idx_arr = np.array([], dtype=np.int16)

    # CSR invariants check
    if legal_ptr[-1] != len(legal_idx_arr):
        raise ValueError(
            f"CSR invariant broken: ptr[-1] ({legal_ptr[-1]}) != len(legal_idx) ({len(legal_idx_arr)})"
        )
    if not np.all(np.diff(legal_ptr) >= 0):
        raise ValueError("CSR invariant broken: legal_ptr is not monotonically non-decreasing")

    cache_dir = p_dir / "cache_v3"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Write all numpy arrays first (byte-deterministic across builds)
    np.save(cache_dir / "planes.npy", all_planes)
    np.save(cache_dir / "meta.npy", all_meta)
    np.save(cache_dir / "target.npy", all_target)
    np.save(cache_dir / "promo.npy", all_promo)
    np.save(cache_dir / "legal_ptr.npy", legal_ptr)
    np.save(cache_dir / "legal_idx.npy", legal_idx_arr)
    np.save(cache_dir / "weight.npy", all_weight)
    np.save(cache_dir / "split.npy", all_split)
    np.save(cache_dir / "game_idx.npy", game_idx_arr)
    np.save(cache_dir / "berserk.npy", all_berserk)

    # Write manifest LAST
    manifest = {
        "schema": "v3",
        "planes": P,
        "history": bool(history),
        "meta_dim": config.META_DIM_V3,
        "n": N,
        "parquet_sha256": parquet_sha256,
        "bit_order": "big",
        "game_ids": sorted_game_ids,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(cache_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return cache_dir


def cache_is_valid(
    cache_dir: Path | str,
    parquet_path: Path | str,
    history: bool = config.HISTORY_PLANES,
    meta_dim: int = config.META_DIM_V3,
) -> tuple[bool, str]:
    """Verify validity and consistency of a tensor cache directory.

    Checks:
    - manifest.json exists and is valid JSON
    - schema is "v3"
    - planes matches 18 + 4*int(history)
    - history flag matches
    - meta_dim matches
    - parquet SHA256 matches current file bytes
    - all 10 array files exist

    Returns
    -------
    tuple[bool, str]
        (True, "valid") if cache is completely valid, else (False, reason).
    """
    cache_dir = Path(cache_dir)
    parquet_path = Path(parquet_path)

    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return False, "manifest missing"

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        return False, f"invalid manifest json: {e}"

    if manifest.get("schema") != "v3":
        return False, f"schema mismatch: {manifest.get('schema')} != v3"

    expected_planes = 18 + 4 * int(history)
    if manifest.get("planes") != expected_planes:
        return False, f"planes mismatch: {manifest.get('planes')} != {expected_planes}"

    if manifest.get("history") != history:
        return False, f"history mismatch: {manifest.get('history')} != {history}"

    if manifest.get("meta_dim") != meta_dim:
        return False, f"meta_dim mismatch: {manifest.get('meta_dim')} != {meta_dim}"

    if not parquet_path.exists():
        return False, f"parquet file missing: {parquet_path}"

    current_parquet_sha = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
    if manifest.get("parquet_sha256") != current_parquet_sha:
        return False, f"parquet sha256 mismatch: {manifest.get('parquet_sha256')} != {current_parquet_sha}"

    for array_file in ARRAY_FILES:
        if not (cache_dir / array_file).exists():
            return False, f"missing array file: {array_file}"

    return True, "valid"
