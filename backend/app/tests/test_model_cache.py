"""Tests for ModelCache (backend/app/engine/model_cache.py)."""

from pathlib import Path
import torch
from app import config
from app.engine.model_cache import ModelCache
from app.ml.model_v2 import ChessPolicyNetV2


def _create_dummy_checkpoint(path: Path) -> ChessPolicyNetV2:
    model = ChessPolicyNetV2(channels=16, num_blocks=1, policy_dim=8, meta_dim=10, num_planes=22)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": model.to_config(),
        },
        path,
    )
    return model


def test_1_cache_get_miss_then_hit(tmp_path: Path):
    weights_path = tmp_path / "players" / "user1" / "models" / "v1" / "weights.pt"
    _create_dummy_checkpoint(weights_path)

    cache = ModelCache(capacity=8, data_root=tmp_path)
    assert cache.size() == 0

    # Miss: loads from disk
    m1 = cache.get("user1", 1)
    assert m1 is not None
    assert cache.size() == 1

    # Hit: returns cached instance
    m2 = cache.get("user1", 1)
    assert m2 is m1
    assert cache.size() == 1


def test_2_cache_evicts_lru(tmp_path: Path):
    _create_dummy_checkpoint(tmp_path / "players" / "user1" / "models" / "v1" / "weights.pt")
    _create_dummy_checkpoint(tmp_path / "players" / "user2" / "models" / "v1" / "weights.pt")
    _create_dummy_checkpoint(tmp_path / "players" / "user3" / "models" / "v1" / "weights.pt")

    cache = ModelCache(capacity=2, data_root=tmp_path)
    cache.get("user1", 1)
    cache.get("user2", 1)
    assert cache.size() == 2
    assert ("user1", 1) in cache._cache
    assert ("user2", 1) in cache._cache

    # Loading user3 should evict the least-recently used (user1)
    cache.get("user3", 1)
    assert cache.size() == 2
    assert ("user1", 1) not in cache._cache
    assert ("user2", 1) in cache._cache
    assert ("user3", 1) in cache._cache


def test_3_cache_falls_back_to_base_when_version_dir_missing(tmp_path: Path):
    # Missing version dir entirely
    cache = ModelCache(capacity=8, data_root=tmp_path)
    model = cache.get("nonexistent_player", 1)
    assert model is not None
    assert isinstance(model, torch.nn.Module)
