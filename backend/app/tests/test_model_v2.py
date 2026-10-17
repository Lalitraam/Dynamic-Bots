"""
Unit tests for ChessPolicyNetV2 (app/ml/model_v2.py).
Tests forward shapes for P=18 and P=22, FiLM metadata sensitivity, determinism,
and Overfit-64 convergence on CPU.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
import torch.optim as optim

from app.ml.model_v2 import ChessPolicyNetV2


def test_forward_shapes_p18_and_p22():
    """
    Forward pass test for P=18 and P=22 input planes with batch=4:
    Assert policy_logits shape is (4, 4096), promo_logits shape is (4, 4),
    and dtypes are float32.
    """
    B = 4
    meta = torch.randn(B, 10, dtype=torch.float32)

    # 1. Test model initialized with num_planes=22
    model22 = ChessPolicyNetV2(channels=32, num_blocks=2, policy_dim=16, meta_dim=10, num_planes=22)
    board22 = torch.randn(B, 22, 8, 8, dtype=torch.float32)
    out22 = model22(board22, meta)
    assert out22["policy_logits"].shape == (B, 4096)
    assert out22["promo_logits"].shape == (B, 4)
    assert out22["policy_logits"].dtype == torch.float32
    assert out22["promo_logits"].dtype == torch.float32

    # Also test passing 18 planes to the 22-plane model (auto zero-padding)
    board18_pad = torch.randn(B, 18, 8, 8, dtype=torch.float32)
    out18_pad = model22(board18_pad, meta)
    assert out18_pad["policy_logits"].shape == (B, 4096)
    assert out18_pad["promo_logits"].shape == (B, 4)

    # 2. Test model initialized with num_planes=18
    model18 = ChessPolicyNetV2(channels=32, num_blocks=2, policy_dim=16, meta_dim=10, num_planes=18)
    board18 = torch.randn(B, 18, 8, 8, dtype=torch.float32)
    out18 = model18(board18, meta)
    assert out18["policy_logits"].shape == (B, 4096)
    assert out18["promo_logits"].shape == (B, 4)


def test_film_metadata_sensitivity():
    """
    FiLM sanity test:
    Given the identical board tensor and two DIFFERENT metadata vectors,
    the model's policy_logits must differ by > 1e-3 somewhere.
    """
    torch.manual_seed(42)
    model = ChessPolicyNetV2(channels=48, num_blocks=3, policy_dim=24, meta_dim=10, num_planes=22)
    # Ensure FiLM has non-zero weights
    for film in model.film_generators:
        nn_init = torch.randn_like(film.weight) * 0.1
        film.weight.data.copy_(nn_init)
        film.bias.data.copy_(torch.randn_like(film.bias) * 0.1)

    board = torch.randn(1, 22, 8, 8)
    meta_a = torch.zeros(1, 10)
    meta_b = torch.ones(1, 10) * 2.0

    model.eval()
    with torch.no_grad():
        out_a = model(board, meta_a)
        out_b = model(board, meta_b)

    diff = (out_a["policy_logits"] - out_b["policy_logits"]).abs().max().item()
    assert diff > 1e-3, f"Expected FiLM differences > 1e-3, got max diff {diff}"


def test_determinism_across_calls():
    """
    Determinism test:
    Given fixed seed and identical inputs, two forward passes produce
    identical outputs.
    """
    torch.manual_seed(0)
    model = ChessPolicyNetV2(channels=32, num_blocks=2, policy_dim=16, meta_dim=10, num_planes=22)
    board = torch.randn(2, 22, 8, 8)
    meta = torch.randn(2, 10)

    model.eval()
    with torch.no_grad():
        out_1 = model(board, meta)
        out_2 = model(board, meta)

    assert torch.equal(out_1["policy_logits"], out_2["policy_logits"])
    assert torch.equal(out_1["promo_logits"], out_2["promo_logits"])


def test_overfit_64_positions_on_cpu():
    """
    Overfit-64 test:
    Random 64 positions, cross-entropy training; assert model reaches
    top-1 >= 0.95 in <= 300 steps on CPU.
    """
    torch.manual_seed(0)
    B = 64
    board = torch.randn(B, 22, 8, 8)
    meta = torch.randn(B, 10)
    targets = torch.randint(0, 4096, (B,))

    model = ChessPolicyNetV2(channels=32, num_blocks=2, policy_dim=16, meta_dim=10, num_planes=22)
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)

    model.train()
    reached_acc = 0.0
    steps_taken = 0

    for step in range(1, 301):
        optimizer.zero_grad()
        out = model(board, meta)
        logits = out["policy_logits"]
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=-1)
        acc = (preds == targets).float().mean().item()
        if acc >= 0.95:
            reached_acc = acc
            steps_taken = step
            break

    assert reached_acc >= 0.95, f"Overfit failed to reach 0.95 top-1 (reached {reached_acc})"
    assert steps_taken <= 300, f"Overfit took too many steps ({steps_taken})"


def test_from_config_and_to_config():
    """
    Config serialization test:
    Verify ChessPolicyNetV2.from_config correctly constructs the module.
    """
    cfg = {
        "channels": 64,
        "num_blocks": 4,
        "policy_dim": 24,
        "meta_dim": 10,
        "num_planes": 18,
    }
    model = ChessPolicyNetV2.from_config(cfg)
    assert model.channels == 64
    assert model.num_blocks == 4
    assert model.policy_dim == 24
    assert model.meta_dim == 10
    assert model.num_planes == 18
    assert model.to_config() == cfg
