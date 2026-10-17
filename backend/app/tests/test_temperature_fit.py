"""
Unit tests for temperature calibration (app/ml/temperature_fit.py).
Tests NLL calibration improvements and argmin selection on synthetic fixtures.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from app.ml.model_v2 import ChessPolicyNetV2
from app.ml.temperature_fit import fit_temperature


class SyntheticValDataset(Dataset):
    """Synthetic dataset generating deterministic board inputs and legal masks."""

    def __init__(self, n: int = 16):
        torch.manual_seed(123)
        self.n = n
        self.boards = torch.randn(n, 22, 8, 8, dtype=torch.float32)
        self.metas = torch.randn(n, 10, dtype=torch.float32)
        self.targets = torch.randint(0, 64, (n,), dtype=torch.long)
        self.weights = torch.ones(n, dtype=torch.float32)

        # Ensure legal mask has the target move and a few others
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
            "legal_mask": self.legal_masks[idx],
            "sample_weight": self.weights[idx],
        }


def test_1_temperature_improves_nll():
    """Verify that temperature scaling finds an optimal T with lower NLL than uncalibrated default."""
    model = ChessPolicyNetV2(channels=16, num_blocks=1, policy_dim=8, meta_dim=10, num_planes=22)
    # Inject large bias to make default predictions overconfident
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(5.0)

    ds = SyntheticValDataset(n=20)
    loader = DataLoader(ds, batch_size=10, shuffle=False)

    grid = [0.6, 0.8, 1.0, 1.2, 1.4, 1.5]
    res = fit_temperature(model, loader, temperature_grid=grid)

    assert "T_base" in res
    assert "val_nll_before" in res
    assert "val_nll_after" in res
    assert res["val_nll_after"] < res["val_nll_before"], (
        f"Expected NLL improvement: after={res['val_nll_after']:.4f} < before={res['val_nll_before']:.4f}"
    )


def test_2_grid_picks_argmin():
    """Verify that fit_temperature picks the exact argmin across the temperature grid."""
    model = ChessPolicyNetV2(channels=16, num_blocks=1, policy_dim=8, meta_dim=10, num_planes=22)
    ds = SyntheticValDataset(n=10)
    loader = DataLoader(ds, batch_size=10, shuffle=False)

    grid = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    res = fit_temperature(model, loader, temperature_grid=grid)

    grid_dict = res["grid"]
    argmin_t = float(min(grid_dict.keys(), key=lambda k: grid_dict[k]))
    assert res["T_base"] == argmin_t, f"Expected {argmin_t}, got {res['T_base']}"
    assert res["val_nll_after"] == grid_dict[str(argmin_t)]
