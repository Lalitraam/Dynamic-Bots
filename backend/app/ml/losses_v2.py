"""
Loss functions for Phase B/C chess policy and promotion training (v2).
Implements masked label-smoothed policy cross-entropy and promotion classification loss.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_policy_loss(
    policy_logits: torch.Tensor,
    legal_mask: torch.Tensor,
    target_move_idx: torch.Tensor,
    sample_weight: torch.Tensor,
    label_smoothing: float = 0.05,
) -> torch.Tensor:
    """
    Masked label-smoothed cross-entropy over legal moves.

    Parameters
    ----------
    policy_logits : torch.Tensor
        Raw move logits of shape (B, 4096).
    legal_mask : torch.Tensor
        Boolean mask of shape (B, 4096) indicating legal moves.
    target_move_idx : torch.Tensor
        Ground-truth move index of shape (B,).
    sample_weight : torch.Tensor
        Per-sample weight of shape (B,).
    label_smoothing : float
        Smoothing factor eps spread uniformly over legal moves (default: 0.05).

    Returns
    -------
    torch.Tensor
        Scalar weighted loss.
    """
    # 1. Mask illegal moves to -inf and compute log-softmax over legal actions
    masked_logits = torch.where(
        legal_mask,
        policy_logits,
        torch.full_like(policy_logits, -float("inf")),
    )
    log_probs = F.log_softmax(masked_logits, dim=-1)

    # 2. Label-smoothed CE: target gets (1 - eps + eps/n_legal),
    #    each other legal move gets eps/n_legal.
    n_legal = legal_mask.sum(dim=-1).float().clamp(min=1.0)
    target_logp = log_probs.gather(1, target_move_idx.unsqueeze(1)).squeeze(1)
    legal_logp_sum = torch.where(legal_mask, log_probs, torch.zeros_like(log_probs)).sum(dim=-1)

    row_loss = -(
        (1.0 - label_smoothing) * target_logp
        + (label_smoothing / n_legal) * legal_logp_sum
    )

    # 3. Per-row weights normalised by their sum over the batch
    norm_weights = sample_weight / sample_weight.sum().clamp(min=1e-8)
    return (row_loss * norm_weights).sum()


def promo_loss(
    promo_logits: torch.Tensor,
    target_pair_mask: torch.Tensor,
    target_promo: torch.Tensor,
    sample_weight: torch.Tensor,
    weight: float = 0.2,
) -> torch.Tensor:
    """
    Weighted cross-entropy for pawn promotion piece class.

    Parameters
    ----------
    promo_logits : torch.Tensor
        Promotion piece logits of shape (B, 4).
    target_pair_mask : torch.Tensor
        Boolean mask of shape (B,) indicating which rows correspond to pawn promotions.
    target_promo : torch.Tensor
        Target promotion class index in {0, 1, 2, 3} of shape (B,).
    sample_weight : torch.Tensor
        Sample weights of shape (B,).
    weight : float
        Loss scaling coefficient (default: 0.2).

    Returns
    -------
    torch.Tensor
        Scalar weighted promo loss, or 0.0 (with requires_grad=True) if no promotion rows.
    """
    if not target_pair_mask.any():
        return (promo_logits * 0.0).sum()

    sub_logits = promo_logits[target_pair_mask]
    sub_targets = target_promo[target_pair_mask]
    sub_weights = sample_weight[target_pair_mask]

    norm_weights = sub_weights / sub_weights.sum().clamp(min=1e-8)
    ce = F.cross_entropy(sub_logits, sub_targets, reduction="none")
    return (ce * norm_weights).sum() * weight


def total_loss(
    policy_logits: torch.Tensor,
    legal_mask: torch.Tensor,
    target_move_idx: torch.Tensor,
    promo_logits: torch.Tensor,
    target_pair_mask: torch.Tensor,
    target_promo: torch.Tensor,
    sample_weight: torch.Tensor,
    label_smoothing: float = 0.05,
    promo_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute combined policy and promotion loss.

    Returns
    -------
    tuple[torch.Tensor, dict[str, float]]
        (total_loss, {"policy": policy_val, "promo": promo_val})
    """
    pol_loss = masked_policy_loss(
        policy_logits=policy_logits,
        legal_mask=legal_mask,
        target_move_idx=target_move_idx,
        sample_weight=sample_weight,
        label_smoothing=label_smoothing,
    )
    pro_loss = promo_loss(
        promo_logits=promo_logits,
        target_pair_mask=target_pair_mask,
        target_promo=target_promo,
        sample_weight=sample_weight,
        weight=promo_weight,
    )
    tot = pol_loss + pro_loss
    return tot, {
        "policy": float(pol_loss.detach().item()),
        "promo": float(pro_loss.detach().item()),
    }
