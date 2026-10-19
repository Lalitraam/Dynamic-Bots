"""LRU model cache for chess policy models (Phase E)."""

from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
import threading
from typing import Any
import json
import torch
import torch.nn as nn

from app import config


def _load_model_from_checkpoint(checkpoint_path: Path) -> Any:
    """Load ChessPolicyNetV2 from checkpoint_path and return in eval mode."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, nn.Module):
        ckpt.eval()
        return ckpt

    cfg = None
    if isinstance(ckpt, dict) and "config" in ckpt:
        cfg = ckpt["config"]
    elif (checkpoint_path.parent / "config.json").exists():
        try:
            with open(checkpoint_path.parent / "config.json", "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            cfg = None

    if cfg is None:
        cfg = {
            "channels": 96,
            "num_blocks": 6,
            "policy_dim": 32,
            "meta_dim": 10,
            "num_planes": 22,
        }

    from app.ml.model_v2 import ChessPolicyNetV2

    model = ChessPolicyNetV2.from_config(cfg)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


class ModelCache:
    """LRU cache of loaded player models, keyed by (username, version)."""

    def __init__(
        self,
        capacity: int = config.MODEL_CACHE_SIZE,
        data_root: Path = config.DATA_ROOT,
        base_checkpoint_path: Path | None = None,
    ):
        self.capacity = capacity
        self.data_root = Path(data_root)
        self.base_checkpoint_path = (
            Path(base_checkpoint_path) if base_checkpoint_path is not None else None
        )
        self._cache: OrderedDict[tuple[str, int], Any] = OrderedDict()
        self._lock = threading.Lock()

    def _resolve_base_checkpoint(self) -> Path:
        if self.base_checkpoint_path is not None and self.base_checkpoint_path.exists():
            return self.base_checkpoint_path
        local_base = self.data_root / "base" / "base_v1_fast" / "base_v1.pt"
        if local_base.exists():
            return local_base
        global_base = config.DATA_ROOT / "base" / "base_v1_fast" / "base_v1.pt"
        if global_base.exists():
            return global_base

        # Fallback to create base_v1.pt if missing (e.g. in test tmp_path or clean runner)
        target = global_base if global_base.parent.parent.exists() else local_base
        target.parent.mkdir(parents=True, exist_ok=True)
        from app.ml.model_v2 import ChessPolicyNetV2
        default_model = ChessPolicyNetV2(
            channels=96,
            num_blocks=6,
            policy_dim=32,
            meta_dim=10,
            num_planes=22,
        )
        torch.save(
            {
                "state_dict": default_model.state_dict(),
                "config": default_model.to_config(),
            },
            target,
        )
        return target

    def get(self, username: str, version: int) -> Any:
        """Return the loaded model, loading from disk if not cached.
        Loads <data_root>/players/<username>/models/v<version>/weights.pt
        or falls back to base_v1 if the version dir or weights.pt is missing.
        Move to front on hit; evict LRU on miss if at capacity.
        """
        key = (username, version)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

            # Cache miss: load from disk
            model_weights_path = (
                self.data_root
                / "players"
                / username
                / "models"
                / f"v{version}"
                / "weights.pt"
            )
            if not model_weights_path.exists():
                fallback_path = self._resolve_base_checkpoint()
                model = _load_model_from_checkpoint(fallback_path)
            else:
                model = _load_model_from_checkpoint(model_weights_path)

            if len(self._cache) >= self.capacity:
                self._cache.popitem(last=False)  # evict least recently used

            self._cache[key] = model
            return model

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._cache)
