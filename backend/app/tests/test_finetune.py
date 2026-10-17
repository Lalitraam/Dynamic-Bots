"""
Unit tests for fine-tuning pipeline (app/ml/finetune.py).
Tests variant selection, parameter freezing, frozen BatchNorm stability,
wall-clock timeout, and end-to-end smoke fine-tuning on synthetic fixtures.
"""
from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset

from app import config
from app.ml.finetune import (
    FineTuneVariant,
    fine_tune,
    freeze_for_variant,
    select_variant,
    set_frozen_bn_eval,
)
from app.ml.model_v2 import ChessPolicyNetV2


class SyntheticDataset(Dataset):
    """Synthetic dataset generating all required tensors for fine_tune()."""

    def __init__(self, n: int = 16):
        torch.manual_seed(42)
        self.n = n
        self.boards = torch.randn(n, 22, 8, 8, dtype=torch.float32)
        self.metas = torch.randn(n, 10, dtype=torch.float32)
        self.targets = torch.randint(0, 64, (n,), dtype=torch.long)
        self.promos = torch.zeros(n, dtype=torch.long)
        self.target_pair_masks = torch.zeros(n, dtype=torch.bool)
        self.weights = torch.ones(n, dtype=torch.float32)

        self.legal_masks = torch.zeros(n, 4096, dtype=torch.bool)
        for i in range(n):
            self.legal_masks[i, self.targets[i]] = True
            for extra in range(5):
                self.legal_masks[i, (self.targets[i].item() + extra + 1) % 4096] = True

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "board": self.boards[idx],
            "meta": self.metas[idx],
            "target_move_idx": self.targets[idx],
            "promo": self.promos[idx],
            "target_pair_mask": self.target_pair_masks[idx],
            "legal_mask": self.legal_masks[idx],
            "sample_weight": self.weights[idx],
        }


def test_1_select_variant_by_game_count():
    """Verify variant selection rules based on total games and validation games."""
    assert select_variant(1200, 30) == FineTuneVariant.FULL
    assert select_variant(800, 30) == FineTuneVariant.FULL
    assert select_variant(799, 30) == FineTuneVariant.FREEZE
    assert select_variant(150, 30) == FineTuneVariant.FREEZE
    assert select_variant(149, 30) == FineTuneVariant.FILM_ONLY
    assert select_variant(50, 30) == FineTuneVariant.FILM_ONLY
    with pytest.raises(ValueError):
        select_variant(49, 30)
    assert select_variant(500, 5) == FineTuneVariant.FILM_ONLY


def test_2_freeze_full_nothing_frozen():
    """Verify FULL variant freezes no parameters."""
    model = ChessPolicyNetV2(channels=32, num_blocks=4, policy_dim=16, meta_dim=10, num_planes=22)
    frozen = freeze_for_variant(model, FineTuneVariant.FULL)
    assert len(frozen) == 0
    for name, param in model.named_parameters():
        assert param.requires_grad is True, f"{name} should be trainable in FULL"


def test_3_freeze_variant_correct_set():
    """Verify FREEZE variant freezes stem, meta_mlp, and blocks[0..2] while keeping later blocks trainable."""
    model = ChessPolicyNetV2(channels=32, num_blocks=6, policy_dim=16, meta_dim=10, num_planes=22)
    frozen = freeze_for_variant(model, FineTuneVariant.FREEZE)

    assert "stem_conv.weight" in frozen
    assert "stem_bn.weight" in frozen
    assert "meta_mlp.0.weight" in frozen

    for i in range(3):
        assert f"blocks.{i}.conv1.weight" in frozen
        assert f"film_generators.{i}.weight" in frozen

    for i in range(3, 6):
        assert f"blocks.{i}.conv1.weight" not in frozen
        assert f"film_generators.{i}.weight" not in frozen

    assert "q_proj.weight" not in frozen
    assert "k_proj.weight" not in frozen
    assert "bias_pair" not in frozen
    assert "promo_head.weight" not in frozen


def test_4_film_only_keeps_only_film_and_heads():
    """Verify FILM_ONLY variant freezes everything except film generators, bias pair, and promo head."""
    model = ChessPolicyNetV2(channels=32, num_blocks=4, policy_dim=16, meta_dim=10, num_planes=22)
    frozen = freeze_for_variant(model, FineTuneVariant.FILM_ONLY)

    for name, param in model.named_parameters():
        if (
            name.startswith("film_generators.")
            or name == "bias_pair"
            or name.startswith("promo_head.")
        ):
            assert param.requires_grad is True, f"{name} should be trainable in FILM_ONLY"
            assert name not in frozen
        else:
            assert param.requires_grad is False, f"{name} should be frozen in FILM_ONLY"
            assert name in frozen


def test_5_frozen_bn_stats_do_not_update():
    """Verify that frozen BatchNorm running statistics do not change during training."""
    model = ChessPolicyNetV2(channels=32, num_blocks=4, policy_dim=16, meta_dim=10, num_planes=22)
    freeze_for_variant(model, FineTuneVariant.FREEZE)

    init_stem_mean = model.stem_bn.running_mean.clone()
    init_stem_var = model.stem_bn.running_var.clone()
    init_b0_mean = model.blocks[0].bn1.running_mean.clone()
    init_b0_var = model.blocks[0].bn1.running_var.clone()

    # In fine-tuning, model.train() is called, immediately followed by set_frozen_bn_eval()
    model.train()
    set_frozen_bn_eval(model, FineTuneVariant.FREEZE)

    # Perform multiple forward passes with non-zero batches
    for _ in range(5):
        b = torch.randn(8, 22, 8, 8) * 10.0 + 5.0
        m = torch.randn(8, 10)
        _ = model(b, m)

    assert torch.equal(model.stem_bn.running_mean, init_stem_mean), "stem_bn running_mean updated!"
    assert torch.equal(model.stem_bn.running_var, init_stem_var), "stem_bn running_var updated!"
    assert torch.equal(model.blocks[0].bn1.running_mean, init_b0_mean), "blocks[0].bn1 running_mean updated!"
    assert torch.equal(model.blocks[0].bn1.running_var, init_b0_var), "blocks[0].bn1 running_var updated!"


def test_6_wall_clock_cap(tmp_path):
    """Verify fine_tune stops with stop_reason='wall_clock' when wall-clock limit is exceeded."""
    out_dir = tmp_path / "timeout_run"
    base_ckpt = tmp_path / "base_timeout.pt"
    dummy = ChessPolicyNetV2(channels=16, num_blocks=1, policy_dim=8, meta_dim=10, num_planes=22)
    torch.save({"state_dict": dummy.state_dict(), "config": dummy.to_config()}, base_ckpt)

    ds_train = SyntheticDataset(n=20)
    ds_val = SyntheticDataset(n=10)

    res = fine_tune(
        base_checkpoint_path=base_ckpt,
        train_parquet=ds_train,
        val_parquet=ds_val,
        out_dir=out_dir,
        variant=FineTuneVariant.FILM_ONLY,
        max_minutes=0.0,
    )
    assert res["stop_reason"] == "wall_clock"


def test_7_smoke_finetune_1_epoch(tmp_path):
    """Verify smoke run of fine_tune generates weights, config, training log and history."""
    out_dir = tmp_path / "smoke_run"
    base_ckpt = tmp_path / "base_smoke.pt"
    dummy = ChessPolicyNetV2(channels=16, num_blocks=1, policy_dim=8, meta_dim=10, num_planes=22)
    torch.save({"state_dict": dummy.state_dict(), "config": dummy.to_config()}, base_ckpt)

    ds_train = SyntheticDataset(n=16)
    ds_val = SyntheticDataset(n=8)

    res = fine_tune(
        base_checkpoint_path=base_ckpt,
        train_parquet=ds_train,
        val_parquet=ds_val,
        out_dir=out_dir,
        variant=FineTuneVariant.FREEZE,
        max_minutes=1,
    )

    assert res["epochs_run"] >= 1
    assert "history" in res
    assert len(res["history"]) >= 1
    assert (out_dir / "weights.pt").exists()
    assert (out_dir / "config.json").exists()
    assert (out_dir / "training_log.json").exists()
