"""
Combined loss function for ChessPolicyNet (Milestone 3).

Function
--------
compute_loss(outputs, batch, config_overrides=None) -> dict

The returned dict has scalar tensors (for logging all components separately):
  {
    "total": weighted mean over the full per-sample loss,
    "from":  weighted mean of the from_sq cross-entropy,
    "to":    weighted mean of the to_sq cross-entropy,
    "promo": weighted mean of the promo loss (after PROMO_LOSS_WEIGHT scaling),
    "aux":   weighted mean of the aux losses (after AUX_LOSS_WEIGHT scaling),
  }

Design notes
------------
* All component losses are computed per-sample (reduction="none") and then
  multiplied by batch["sample_weight"] before taking the mean.  This ensures
  that recency-weighted sampling (not active in Milestone 2, always 1.0) will
  take effect automatically when weights differ in future milestones.
* Aux losses (piece_type, is_capture, is_check, is_castle) are regularizers only;
  they are never used at inference time.
* Label smoothing is applied only to the cross-entropy heads (from_sq, to_sq,
  piece_type) via F.cross_entropy's built-in parameter.
* is_capture / is_check / is_castle targets are float32 (0.0/1.0) as returned by
  PlayerMoveDataset -- this matches BCE's expected dtype; no cast needed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .. import config as _cfg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch:   dict[str, torch.Tensor],
    config_overrides: dict | None = None,
) -> dict[str, torch.Tensor]:
    """
    Compute the combined training loss for one mini-batch.

    Parameters
    ----------
    outputs : dict
        Raw logit tensors as returned by ChessPolicyNet.forward().
        Required keys: from_sq_logits, to_sq_logits, promo_logits,
        piece_type_logits, is_capture_logit, is_check_logit, is_castle_logit.
    batch : dict
        Target tensors as returned by PlayerMoveDataset.__getitem__ / DataLoader.
        Required keys: from_sq (int64), to_sq (int64), promo (int64),
        piece_type (int64), is_capture (float32), is_check (float32),
        is_castle (float32), sample_weight (float32).
    config_overrides : dict, optional
        Override any config constant by name for testing (e.g.
        {"LABEL_SMOOTHING": 0.0, "AUX_LOSS_WEIGHT": 0.0}).

    Returns
    -------
    dict with keys "total", "from", "to", "promo", "aux" — all scalar tensors.
    """
    # Resolve config values (allow test overrides)
    def _cfg_val(name: str):
        if config_overrides and name in config_overrides:
            return config_overrides[name]
        return getattr(_cfg, name)

    label_smoothing  = float(_cfg_val("LABEL_SMOOTHING"))
    promo_weight     = float(_cfg_val("PROMO_LOSS_WEIGHT"))
    aux_weight       = float(_cfg_val("AUX_LOSS_WEIGHT"))

    w = batch["sample_weight"]  # (B,)  float32

    # ------------------------------------------------------------------
    # From-square loss  (B,)
    # ------------------------------------------------------------------
    from_loss = F.cross_entropy(
        outputs["from_sq_logits"],
        batch["from_sq"],
        label_smoothing=label_smoothing,
        reduction="none",
    )

    # ------------------------------------------------------------------
    # To-square loss  (B,)
    # ------------------------------------------------------------------
    to_loss = F.cross_entropy(
        outputs["to_sq_logits"],
        batch["to_sq"],
        label_smoothing=label_smoothing,
        reduction="none",
    )

    # ------------------------------------------------------------------
    # Promotion loss  (B,)  — weighted low (PROMO_LOSS_WEIGHT)
    # ------------------------------------------------------------------
    promo_loss = (
        F.cross_entropy(
            outputs["promo_logits"],
            batch["promo"],
            reduction="none",
        )
        * promo_weight
    )

    # ------------------------------------------------------------------
    # Auxiliary losses  (B,)  — weighted low (AUX_LOSS_WEIGHT)
    # is_capture/is_check/is_castle are float32 targets, matching BCE's
    # expected dtype (no cast needed — confirmed against dataset.py).
    # ------------------------------------------------------------------
    aux_loss = aux_weight * (
        F.cross_entropy(
            outputs["piece_type_logits"],
            batch["piece_type"],
            label_smoothing=label_smoothing,
            reduction="none",
        )
        + F.binary_cross_entropy_with_logits(
            outputs["is_capture_logit"].squeeze(-1),
            batch["is_capture"],
            reduction="none",
        )
        + F.binary_cross_entropy_with_logits(
            outputs["is_check_logit"].squeeze(-1),
            batch["is_check"],
            reduction="none",
        )
        + F.binary_cross_entropy_with_logits(
            outputs["is_castle_logit"].squeeze(-1),
            batch["is_castle"],
            reduction="none",
        )
    )

    # ------------------------------------------------------------------
    # Per-sample total, weighted by sample_weight, then meaned
    # ------------------------------------------------------------------
    per_sample_total = from_loss + to_loss + promo_loss + aux_loss
    weighted_total   = per_sample_total * w

    total = weighted_total.mean()

    # Return each component as a weighted mean for logging consistency
    return {
        "total": total,
        "from":  (from_loss  * w).mean(),
        "to":    (to_loss    * w).mean(),
        "promo": (promo_loss * w).mean(),
        "aux":   (aux_loss   * w).mean(),
    }
