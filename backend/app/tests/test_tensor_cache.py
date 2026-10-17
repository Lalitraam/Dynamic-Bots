"""
Tests for Phase B tensor cache and CachedDataset (app/ml/tensor_cache.py, app/ml/dataset_v3.py).

Verifies:
5. Cache vs on-the-fly: on the fixture games build the cache and assert bit-exact equality
   of planes (after unpackbits), meta, target, promo, weight, berserk and the legal mask
   against encode_row_v3 for EVERY row; CSR invariants (ptr monotone, ptr[-1] == len(legal_idx),
   target in its own legal slice for every row).
6. Two builds -> every .npy file has identical sha256. Cache invalidation: modify a copy of
   the parquet (append one byte) -> cache_is_valid False with a reason; wrong history flag -> False;
   fresh cache -> True.
7. CachedDataset: split sizes equal the parquet's, berserk_filter partitions the rows
   (non_berserk + berserk == all), DataLoader batch of 4 has the right shapes/dtypes.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import DataLoader

from app import config
from app.ml import encoding_v3
from app.ml.dataset_v3 import CachedDataset
from app.ml.tensor_cache import ARRAY_FILES, build_cache, cache_is_valid
from app.services.dataset_builder_v3 import build_dataset_v3

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_games.jsonl"


def _load_fixture_games():
    games = []
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            game = json.loads(line)
            game["date"] = datetime.fromisoformat(game["date"])
            games.append(game)
    return games


def _setup_fixture_cache(tmp_path: Path, username: str = "cache_fixture_user") -> tuple[Path, Path]:
    orig_data_root = config.DATA_ROOT
    config.DATA_ROOT = tmp_path
    games = _load_fixture_games()
    try:
        with patch("app.services.extractor.storage.load_games", return_value=games):
            build_dataset_v3(username)
        cache_dir = build_cache(username, data_root=tmp_path, history=True)
        parquet_path = tmp_path / "players" / username / "samples_v3.parquet"
        return cache_dir, parquet_path
    finally:
        config.DATA_ROOT = orig_data_root


def test_5_cache_vs_on_the_fly_equality_and_csr_invariants(tmp_path):
    """
    Test 5: Bit-exact equality of cache against encode_row_v3 for EVERY row
    and CSR legal index invariants.

    Hand-computed invariants:
    - planes: unpacking np.unpackbits of planes.npy for each row i yields bit-exact
      equality with encode_row_v3(row)['planes'] of shape (22, 8, 8).
    - meta: float32 values match within 1e-6 tolerance.
    - target: int16 move_idx matches encode_row_v3(row)['move_idx'] exactly.
    - promo: int8 promotion class matches encode_row_v3(row)['promo'] exactly.
    - weight: sample_weight matches encode_row_v3(row)['weight'] within 1e-6.
    - berserk: berserk boolean matches encode_row_v3(row)['berserk'] exactly.
    - legal_mask: expanding the CSR slice legal_idx[ptr[i]:ptr[i+1]] produces
      an identical (4096,) boolean mask to encode_row_v3(row)['legal_idx'].
    - CSR invariants:
      1. ptr[0] == 0, ptr is monotonically non-decreasing (ptr[i+1] >= ptr[i]).
      2. ptr[-1] == total legal move count == len(legal_idx).
      3. For every row i, target[i] is present in legal_idx[ptr[i]:ptr[i+1]].
    """
    cache_dir, parquet_path = _setup_fixture_cache(tmp_path)
    assert parquet_path.exists()
    assert cache_dir.exists()

    df = pq.read_table(parquet_path).to_pandas()
    N = len(df)
    assert N > 0

    planes_arr = np.load(cache_dir / "planes.npy")
    meta_arr = np.load(cache_dir / "meta.npy")
    target_arr = np.load(cache_dir / "target.npy")
    promo_arr = np.load(cache_dir / "promo.npy")
    weight_arr = np.load(cache_dir / "weight.npy")
    berserk_arr = np.load(cache_dir / "berserk.npy")
    legal_ptr = np.load(cache_dir / "legal_ptr.npy")
    legal_idx = np.load(cache_dir / "legal_idx.npy")

    # CSR structural invariants
    assert legal_ptr[0] == 0
    assert len(legal_ptr) == N + 1
    assert np.all(np.diff(legal_ptr) >= 0)
    assert legal_ptr[-1] == len(legal_idx)

    records = df.to_dict(orient="records")
    for i, row in enumerate(records):
        enc = encoding_v3.encode_row_v3(row, history=True)

        # 1. Unpacked planes bit-exact match
        packed = planes_arr[i]
        unpacked = np.unpackbits(packed)[: 22 * 64].reshape(22, 8, 8)
        np.testing.assert_array_equal(unpacked, enc["planes"])

        # 2. Meta vector match
        np.testing.assert_allclose(meta_arr[i], enc["meta"], atol=1e-6)

        # 3. Target and Promo match
        assert target_arr[i] == enc["move_idx"]
        assert promo_arr[i] == enc["promo"]

        # 4. Weight match
        assert pytest.approx(float(weight_arr[i]), abs=1e-5) == float(enc["weight"])

        # 5. Berserk flag match
        assert bool(berserk_arr[i] == 1) == enc["berserk"]

        # 6. Legal mask and slice match
        slice_start = legal_ptr[i]
        slice_end = legal_ptr[i + 1]
        row_legals = legal_idx[slice_start:slice_end]

        np.testing.assert_array_equal(row_legals, enc["legal_idx"])

        mask_cached = np.zeros(4096, dtype=bool)
        mask_cached[row_legals] = True

        mask_expected = np.zeros(4096, dtype=bool)
        mask_expected[enc["legal_idx"]] = True
        np.testing.assert_array_equal(mask_cached, mask_expected)

        # 7. Target move is in its own legal move slice
        assert target_arr[i] in row_legals


def test_6_determinism_and_cache_invalidation(tmp_path):
    """
    Test 6: Two builds write byte-identical arrays, and cache_is_valid correctly
    invalidates when parquet or configuration parameters change.

    Hand-computed invariants:
    - Byte determinism: building the cache twice on the exact same parquet produces
      identical SHA256 hashes for all 10 .npy array files.
    - Fresh cache: cache_is_valid returns (True, "valid").
    - Parquet alteration: appending 1 byte changes SHA256, causing cache_is_valid to
      return (False, "parquet sha256 mismatch: ...").
    - History mismatch: passing history=False on a history=True cache returns False.
    - Meta dim mismatch: passing meta_dim=8 on a meta_dim=10 cache returns False.
    """
    orig_data_root = config.DATA_ROOT
    config.DATA_ROOT = tmp_path
    username = "det_user"
    games = _load_fixture_games()

    try:
        with patch("app.services.extractor.storage.load_games", return_value=games):
            build_dataset_v3(username)
        pdir = tmp_path / "players" / username
        parquet_path = pdir / "samples_v3.parquet"

        # Build 1
        cache_dir = build_cache(username, data_root=tmp_path, history=True)
        hashes_1 = {f: hashlib.sha256((cache_dir / f).read_bytes()).hexdigest() for f in ARRAY_FILES}

        # Build 2
        build_cache(username, data_root=tmp_path, history=True)
        hashes_2 = {f: hashlib.sha256((cache_dir / f).read_bytes()).hexdigest() for f in ARRAY_FILES}

        for f in ARRAY_FILES:
            assert hashes_1[f] == hashes_2[f], f"File {f} is not byte-deterministic across builds!"

        # Fresh cache is valid
        is_valid, reason = cache_is_valid(cache_dir, parquet_path, history=True, meta_dim=10)
        assert is_valid is True
        assert reason == "valid"

        # Invalidate via modified parquet copy (append one byte)
        mod_parquet = pdir / "modified_samples_v3.parquet"
        mod_parquet.write_bytes(parquet_path.read_bytes() + b"\x00")
        is_valid_mod, reason_mod = cache_is_valid(cache_dir, mod_parquet, history=True, meta_dim=10)
        assert is_valid_mod is False
        assert "parquet sha256 mismatch" in reason_mod

        # Invalidate via history flag mismatch
        is_valid_hist, reason_hist = cache_is_valid(cache_dir, parquet_path, history=False, meta_dim=10)
        assert is_valid_hist is False
        assert "history mismatch" in reason_hist or "planes mismatch" in reason_hist

        # Invalidate via meta_dim mismatch
        is_valid_dim, reason_dim = cache_is_valid(cache_dir, parquet_path, history=True, meta_dim=8)
        assert is_valid_dim is False
        assert "meta_dim mismatch" in reason_dim

        # Invalidate via missing array file
        target_npy = cache_dir / "target.npy"
        target_backup = target_npy.read_bytes()
        try:
            target_npy.unlink()
            is_valid_miss, reason_miss = cache_is_valid(cache_dir, parquet_path, history=True, meta_dim=10)
            assert is_valid_miss is False
            assert "missing array file: target.npy" in reason_miss
        finally:
            target_npy.write_bytes(target_backup)

    finally:
        config.DATA_ROOT = orig_data_root


def test_7_cached_dataset_splits_berserk_partition_and_dataloader_batch(tmp_path):
    """
    Test 7: CachedDataset split sizes, berserk partitioning, and DataLoader batch shapes/dtypes.

    Hand-computed invariants:
    - Split sizes: len(CachedDataset(split=s)) == len(parquet[split=s]) for s in ('train', 'val', 'test').
    - Berserk partitioning:
      len(non_berserk) + len(berserk) == len(all) for the train split.
    - DataLoader batch of 4 (batch_size=4, num_workers=0):
      board: torch.uint8 (4, 22, 8, 8)
      meta: torch.float32 (4, 10)
      target: torch.int64 (4,)
      from_sq: torch.int64 (4,)
      to_sq: torch.int64 (4,)
      promo: torch.int64 (4,)
      legal_mask: torch.bool (4, 4096)
      sample_weight: torch.float32 (4,)
      berserk: torch.bool (4,)
      game_idx: torch.int64 (4,)
    """
    cache_dir, parquet_path = _setup_fixture_cache(tmp_path)
    df = pq.read_table(parquet_path).to_pandas()

    # 1. Split sizes equal parquet splits
    ds_train = CachedDataset(cache_dir, split="train", berserk_filter="all")
    ds_val = CachedDataset(cache_dir, split="val", berserk_filter="all")
    ds_test = CachedDataset(cache_dir, split="test", berserk_filter="all")

    n_train_df = len(df[df["split"] == "train"])
    n_val_df = len(df[df["split"] == "val"])
    n_test_df = len(df[df["split"] == "test"])

    assert len(ds_train) == n_train_df
    assert len(ds_val) == n_val_df
    assert len(ds_test) == n_test_df

    # 2. Berserk filter partitions the rows
    ds_non_berserk = CachedDataset(cache_dir, split="train", berserk_filter="non_berserk")
    ds_berserk = CachedDataset(cache_dir, split="train", berserk_filter="berserk")

    assert len(ds_non_berserk) + len(ds_berserk) == len(ds_train)

    for i in range(len(ds_non_berserk)):
        assert ds_non_berserk[i]["berserk"] is False
    for i in range(len(ds_berserk)):
        assert ds_berserk[i]["berserk"] is True

    # 3. DataLoader batch of 4
    train_loader = DataLoader(ds_train, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(train_loader))

    assert batch["board"].shape == (4, 22, 8, 8)
    assert batch["board"].dtype == torch.uint8

    assert batch["meta"].shape == (4, 10)
    assert batch["meta"].dtype == torch.float32

    assert batch["target"].shape == (4,)
    assert batch["target"].dtype == torch.int64

    assert batch["from_sq"].shape == (4,)
    assert batch["from_sq"].dtype == torch.int64

    assert batch["to_sq"].shape == (4,)
    assert batch["to_sq"].dtype == torch.int64

    assert batch["promo"].shape == (4,)
    assert batch["promo"].dtype == torch.int64

    assert batch["legal_mask"].shape == (4, 4096)
    assert batch["legal_mask"].dtype == torch.bool

    assert batch["sample_weight"].shape == (4,)
    assert batch["sample_weight"].dtype == torch.float32

    assert batch["berserk"].shape == (4,)
    assert batch["berserk"].dtype == torch.bool

    assert batch["game_idx"].shape == (4,)
    assert batch["game_idx"].dtype == torch.int64
