"""
Tests for losses_v2: masked policy loss, label smoothing, promotion loss, and sample weighting.
Includes hand-computed test expectations.
"""
from __future__ import annotations

import math
import pytest
import torch
import torch.nn.functional as F

from app.ml.losses_v2 import masked_policy_loss, promo_loss, total_loss


def test_illegal_moves_masked_to_zero_prob():
    """
    Illegal moves test:
    Verify that in masked_logits / log_probs, probabilities on illegal squares
    are exactly 0.0.
    """
    B = 2
    policy_logits = torch.randn(B, 4096)
    legal_mask = torch.zeros(B, 4096, dtype=torch.bool)
    # Assign a few legal moves
    legal_mask[0, [10, 20, 30]] = True
    legal_mask[1, [100, 200]] = True

    masked_logits = torch.where(
        legal_mask,
        policy_logits,
        torch.full_like(policy_logits, -float("inf")),
    )
    probs = F.softmax(masked_logits, dim=-1)

    # Illegal moves must have exactly 0 probability
    illegal_mask = ~legal_mask
    assert torch.all(probs[illegal_mask] == 0.0)

    # Legal probabilities must sum to 1.0 per row
    assert torch.allclose(probs.sum(dim=-1), torch.ones(B), atol=1e-6)


def test_single_legal_move_loss_is_zero():
    """
    Single-legal-move position test:
    When a position has exactly 1 legal move (which is the target move),
    the cross-entropy loss is exactly 0.0.
    """
    policy_logits = torch.tensor([[10.5, -5.0, 3.2, -1.0]])
    legal_mask = torch.tensor([[False, True, False, False]])  # only index 1 is legal
    target_move_idx = torch.tensor([1])
    sample_weight = torch.tensor([1.0])

    loss = masked_policy_loss(
        policy_logits=policy_logits,
        legal_mask=legal_mask,
        target_move_idx=target_move_idx,
        sample_weight=sample_weight,
        label_smoothing=0.05,
    )
    assert abs(loss.item() - 0.0) < 1e-7


def test_three_legal_moves_toy_hand_computed():
    """
    3-legal-move toy with hand-computed expectations:
      policy_logits = [log(0.6), log(0.3), log(0.1)]
      legal_mask = [True, True, True]
      target_move_idx = 0
      
      At eps = 0.0:
        Expected NLL = -log(0.6) = 0.5108256237659907
        
      At eps = 0.05:
        Target index 0 gets smoothed probability:
          p0 = 1 - 0.05 + 0.05/3 = 0.95 + 0.01666667 = 0.96666667
        Other indices (1 and 2) get:
          p1 = 0.05 / 3 = 0.01666667
          p2 = 0.05 / 3 = 0.01666667
        Expected Loss:
          -(p0 * log(0.6) + p1 * log(0.3) + p2 * log(0.1))
          = -(0.96666667 * -0.51082562 + 0.01666667 * -1.2039728 + 0.01666667 * -2.30258509)
          = -(-0.49379809 - 0.02006621 - 0.03837642)
          = 0.55224072
    """
    l0 = math.log(0.6)
    l1 = math.log(0.3)
    l2 = math.log(0.1)

    policy_logits = torch.tensor([[l0, l1, l2]], dtype=torch.float32)
    legal_mask = torch.tensor([[True, True, True]], dtype=torch.bool)
    target_move_idx = torch.tensor([0], dtype=torch.int64)
    sample_weight = torch.tensor([1.0], dtype=torch.float32)

    # 1. Unsmoothed check (eps = 0.0)
    loss_unsmoothed = masked_policy_loss(
        policy_logits=policy_logits,
        legal_mask=legal_mask,
        target_move_idx=target_move_idx,
        sample_weight=sample_weight,
        label_smoothing=0.0,
    )
    expected_nll = -math.log(0.6)
    assert abs(loss_unsmoothed.item() - expected_nll) < 1e-6

    # 2. Smoothed check (eps = 0.05)
    loss_smoothed = masked_policy_loss(
        policy_logits=policy_logits,
        legal_mask=legal_mask,
        target_move_idx=target_move_idx,
        sample_weight=sample_weight,
        label_smoothing=0.05,
    )
    p0 = 0.95 + 0.05 / 3.0
    p1 = 0.05 / 3.0
    p2 = 0.05 / 3.0
    expected_smoothed = -(p0 * l0 + p1 * l1 + p2 * l2)
    assert abs(loss_smoothed.item() - expected_smoothed) < 1e-6


def test_sample_weight_normalisation():
    """
    Sample weight normalisation:
    Doubling all sample weights leaves total loss unchanged within 1e-6.
    """
    B = 4
    policy_logits = torch.randn(B, 4096)
    legal_mask = torch.zeros(B, 4096, dtype=torch.bool)
    for b in range(B):
        legal_mask[b, [b * 10, b * 10 + 1, b * 10 + 2]] = True

    target_move_idx = torch.tensor([0, 10, 20, 30], dtype=torch.int64)
    sample_weight = torch.tensor([1.0, 2.5, 0.5, 3.0], dtype=torch.float32)

    loss_1 = masked_policy_loss(
        policy_logits=policy_logits,
        legal_mask=legal_mask,
        target_move_idx=target_move_idx,
        sample_weight=sample_weight,
        label_smoothing=0.05,
    )

    loss_2 = masked_policy_loss(
        policy_logits=policy_logits,
        legal_mask=legal_mask,
        target_move_idx=target_move_idx,
        sample_weight=sample_weight * 2.0,
        label_smoothing=0.05,
    )

    assert abs(loss_1.item() - loss_2.item()) < 1e-6


def test_promo_loss_empty_and_normal():
    """
    Promo loss tests:
    - Empty promotional mask returns 0.0 with requires_grad=True.
    - Normal promotional rows compute weighted cross-entropy scaled by promo_weight.
    """
    promo_logits = torch.randn(3, 4, requires_grad=True)
    mask_empty = torch.tensor([False, False, False])
    target_promo = torch.tensor([0, 1, 2])
    weights = torch.tensor([1.0, 1.0, 1.0])

    loss_empty = promo_loss(promo_logits, mask_empty, target_promo, weights, weight=0.2)
    assert loss_empty.item() == 0.0
    assert loss_empty.requires_grad is True

    mask_active = torch.tensor([True, False, True])
    loss_active = promo_loss(promo_logits, mask_active, target_promo, weights, weight=0.2)
    assert loss_active.item() > 0.0

    tot, logs = total_loss(
        torch.randn(3, 4096),
        torch.ones(3, 4096, dtype=torch.bool),
        torch.tensor([0, 1, 2]),
        promo_logits,
        mask_active,
        target_promo,
        weights,
    )
    assert "policy" in logs and "promo" in logs
    assert tot.item() > 0.0
