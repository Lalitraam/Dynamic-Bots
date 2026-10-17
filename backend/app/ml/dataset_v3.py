"""
PyTorch Dataset using memory-mapped tensor cache for Phase B chess imitation learning.

Loads pre-packed board planes, metadata, targets, and legal masks with minimal overhead.
Safe for num_workers=0 on Windows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .. import config
from .tensor_cache import SPLIT_MAP


class CachedDataset(Dataset):
    """PyTorch Dataset reading memory-mapped tensor cache.

    Parameters
    ----------
    cache_dir : str | Path
        Directory containing cache_v3 numpy files and manifest.json.
    split : str
        One of 'train', 'val', 'test'.
    berserk_filter : str
        Filtering mode: 'all', 'non_berserk', or 'berserk'.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        split: str,
        berserk_filter: str = "all",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.berserk_filter = berserk_filter

        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found in {self.cache_dir}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        self.planes: int = int(self.manifest.get("planes", 22 if config.HISTORY_PLANES else 18))
        self.meta_dim: int = int(self.manifest.get("meta_dim", config.META_DIM_V3))

        # Memory-map the arrays in read-only mode
        self.planes_arr = np.load(self.cache_dir / "planes.npy", mmap_mode="r")
        self.meta_arr = np.load(self.cache_dir / "meta.npy", mmap_mode="r")
        self.target_arr = np.load(self.cache_dir / "target.npy", mmap_mode="r")
        self.promo_arr = np.load(self.cache_dir / "promo.npy", mmap_mode="r")
        self.legal_ptr_arr = np.load(self.cache_dir / "legal_ptr.npy", mmap_mode="r")
        self.legal_idx_arr = np.load(self.cache_dir / "legal_idx.npy", mmap_mode="r")
        self.weight_arr = np.load(self.cache_dir / "weight.npy", mmap_mode="r")
        self.split_arr = np.load(self.cache_dir / "split.npy", mmap_mode="r")
        self.game_idx_arr = np.load(self.cache_dir / "game_idx.npy", mmap_mode="r")
        self.berserk_arr = np.load(self.cache_dir / "berserk.npy", mmap_mode="r")

        if split not in SPLIT_MAP:
            raise ValueError(f"Unknown split '{split}', expected one of {list(SPLIT_MAP.keys())}")
        split_code = SPLIT_MAP[split]

        mask = (self.split_arr == split_code)
        if berserk_filter == "all":
            pass
        elif berserk_filter == "non_berserk":
            mask = mask & (self.berserk_arr == 0)
        elif berserk_filter == "berserk":
            mask = mask & (self.berserk_arr == 1)
        else:
            raise ValueError(
                f"Unknown berserk_filter '{berserk_filter}', expected 'all', 'non_berserk', or 'berserk'"
            )

        self.indices: np.ndarray = np.where(mask)[0]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row_idx = int(self.indices[idx])

        # Board planes: unpack bits into (P, 8, 8)
        packed = self.planes_arr[row_idx]
        unpacked = np.unpackbits(packed)[: self.planes * 64]
        board_np = unpacked.reshape(self.planes, 8, 8)
        board = torch.from_numpy(board_np).to(torch.uint8)

        # Meta vector
        meta_np = np.array(self.meta_arr[row_idx], dtype=np.float32)
        meta = torch.from_numpy(meta_np).to(torch.float32)

        # Target and from/to
        target_val = int(self.target_arr[row_idx])
        target = torch.tensor(target_val, dtype=torch.int64)
        from_sq = torch.tensor(target_val // 64, dtype=torch.int64)
        to_sq = torch.tensor(target_val % 64, dtype=torch.int64)
        promo = torch.tensor(int(self.promo_arr[row_idx]), dtype=torch.int64)

        # Legal move mask from CSR slice
        ptr_start = self.legal_ptr_arr[row_idx]
        ptr_end = self.legal_ptr_arr[row_idx + 1]
        indices = self.legal_idx_arr[ptr_start:ptr_end]
        mask_np = np.zeros(4096, dtype=bool)
        mask_np[indices] = True
        legal_mask = torch.from_numpy(mask_np).to(torch.bool)

        sample_weight = torch.tensor(float(self.weight_arr[row_idx]), dtype=torch.float32)
        berserk = bool(self.berserk_arr[row_idx] == 1)
        game_idx = torch.tensor(int(self.game_idx_arr[row_idx]), dtype=torch.int64)

        return {
            "board": board,
            "meta": meta,
            "target": target,
            "from_sq": from_sq,
            "to_sq": to_sq,
            "promo": promo,
            "legal_mask": legal_mask,
            "sample_weight": sample_weight,
            "berserk": berserk,
            "game_idx": game_idx,
        }
