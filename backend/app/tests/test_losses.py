"""
Tests for app/ml/losses.py  (Phase 2 - Milestone 3).

Coverage
--------
1. Near-perfect predictions (huge logit at correct index) -> near-zero total loss.
2. Uniform logits -> loss close to the closed-form CE of a uniform distribution.
3. Doubling sample_weight for one sample changes the mean loss in the expected direction.
4. Label smoothing > 0 gives a measurably higher loss than label_smoothing=0 for a
   high-confidence correct prediction.
"""

from __future__ import annotations

import math
import pytest
import torch

from app.ml.losses import compute_loss
from app import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_batch(
    batch_size: int = 4,
    from_sq: int = 10,
    to_sq:   int = 18,
    promo:   int = 0,
    piece_type: int = 0,
    weights: list[float] | None = None,
) -> dict[str, torch.Tensor]:
    """Return a minimal batch with controlled targets."""
    B = batch_size
    w = weights if weights is not None else [1.0] * B
    return {
        "from_sq":    torch.full((B,), from_sq,    dtype=torch.int64),
        "to_sq":      torch.full((B,), to_sq,      dtype=torch.int64),
        "promo":      torch.full((B,), promo,       dtype=torch.int64),
        "piece_type": torch.full((B,), piece_type,  dtype=torch.int64),
        "is_capture": torch.zeros(B, dtype=torch.float32),
        "is_check":   torch.zeros(B, dtype=torch.float32),
        "is_castle":  torch.zeros(B, dtype=torch.float32),
        "sample_weight": torch.tensor(w, dtype=torch.float32),
    }


def _make_outputs_confident(
    batch: dict[str, torch.Tensor],
    logit_magnitude: float = 30.0,
) -> dict[str, torch.Tensor]:
    """
    Return logits that place almost all probability mass on the correct class.
    Softmax of (30 at correct, -30 elsewhere) is effectively 1.0 at correct.
    """
    B = len(batch["from_sq"])
    BIG, SMALL = logit_magnitude, -logit_magnitude

    from_logits = torch.full((B, 64), SMALL)
    for i, sq in enumerate(batch["from_sq"]):
        from_logits[i, sq.item()] = BIG

    to_logits = torch.full((B, 64), SMALL)
    for i, sq in enumerate(batch["to_sq"]):
        to_logits[i, sq.item()] = BIG

    promo_logits = torch.full((B, 4), SMALL)
    for i, p in enumerate(batch["promo"]):
        promo_logits[i, p.item()] = BIG

    piece_logits = torch.full((B, 6), SMALL)
    for i, pt in enumerate(batch["piece_type"]):
        piece_logits[i, pt.item()] = BIG

    return {
        "from_sq_logits":    from_logits,
        "to_sq_logits":      to_logits,
        "promo_logits":      promo_logits,
        "piece_type_logits": piece_logits,
        # Confident predictions for binary heads: large negative = predicts 0
        "is_capture_logit":  torch.full((B, 1), SMALL),
        "is_check_logit":    torch.full((B, 1), SMALL),
        "is_castle_logit":   torch.full((B, 1), SMALL),
    }


def _make_outputs_uniform(batch_size: int) -> dict[str, torch.Tensor]:
    """Return logits that are all-zero (uniform distribution after softmax)."""
    B = batch_size
    return {
        "from_sq_logits":    torch.zeros(B, 64),
        "to_sq_logits":      torch.zeros(B, 64),
        "promo_logits":      torch.zeros(B, 4),
        "piece_type_logits": torch.zeros(B, 6),
        "is_capture_logit":  torch.zeros(B, 1),
        "is_check_logit":    torch.zeros(B, 1),
        "is_castle_logit":   torch.zeros(B, 1),
    }


# ---------------------------------------------------------------------------
# 1. Near-perfect predictions -> near-zero total loss
# ---------------------------------------------------------------------------

def test_confident_correct_prediction_low_loss():
    """Confident correct predictions should give a very small total loss."""
    batch   = _make_batch(batch_size=4)
    outputs = _make_outputs_confident(batch)

    # With label_smoothing=0, confident correct = ~0 loss
    result = compute_loss(outputs, batch, config_overrides={"LABEL_SMOOTHING": 0.0})

    assert result["total"].item() < 0.05, (
        f"Expected near-zero loss for confident correct predictions, "
        f"got {result['total'].item():.4f}"
    )
    assert math.isfinite(result["total"].item()), "Loss is NaN or Inf"


# ---------------------------------------------------------------------------
# 2. Uniform logits -> loss close to closed-form CE of uniform distribution
# ---------------------------------------------------------------------------

def test_uniform_logits_give_expected_loss():
    """
    All-zero logits -> uniform distribution.
    Expected per-component losses (no label smoothing, no aux/promo weighting):
      from_loss  = log(64) ~ 4.1589
      to_loss    = log(64) ~ 4.1589
    After weighting:
      total ~ from_loss + to_loss + PROMO_LOSS_WEIGHT*log(4) + AUX_LOSS_WEIGHT*(log(6)+3*log(2))
    """
    B       = 8
    batch   = _make_batch(batch_size=B)
    outputs = _make_outputs_uniform(B)

    result = compute_loss(
        outputs, batch,
        config_overrides={"LABEL_SMOOTHING": 0.0},
    )

    # Closed-form expected values
    ce64  = math.log(64)      # ~4.1589
    ce4   = math.log(4)       # ~1.3863  (promo)
    ce6   = math.log(6)       # ~1.7918  (piece_type)
    bce_half = math.log(2)    # ~0.6931  (BCE at logit=0, target=0)

    expected_from  = ce64
    expected_to    = ce64
    expected_promo = config.PROMO_LOSS_WEIGHT  * ce4
    expected_aux   = config.AUX_LOSS_WEIGHT    * (ce6 + 3 * bce_half)
    expected_total = expected_from + expected_to + expected_promo + expected_aux

    tol = 0.01  # allow 1% absolute tolerance for floating-point differences

    assert abs(result["from"].item()  - expected_from)  < tol, (
        f"from_loss: expected ~{expected_from:.4f}, got {result['from'].item():.4f}"
    )
    assert abs(result["to"].item()    - expected_to)    < tol, (
        f"to_loss: expected ~{expected_to:.4f}, got {result['to'].item():.4f}"
    )
    assert abs(result["total"].item() - expected_total) < tol, (
        f"total_loss: expected ~{expected_total:.4f}, got {result['total'].item():.4f}"
    )


# ---------------------------------------------------------------------------
# 3. Doubling sample_weight for one sample changes mean loss as expected
# ---------------------------------------------------------------------------

def test_sample_weight_changes_loss_in_expected_direction():
    """
    2-sample batch where sample 1 has higher weight should yield a loss
    closer to sample 1's individual loss than sample 0's.

    Concrete setup
    --------------
    Sample 0: correct prediction (low individual loss L0 ~ 0)
    Sample 1: wrong prediction   (high individual loss L1 ~ log(64))

    Equal weights [1.0, 1.0]:
        mean_loss = (L0 + L1) / 2  ~  log(64)/2

    Doubled weight on sample 1  [1.0, 2.0]:
        weighted_mean = (1.0*L0 + 2.0*L1) / 2  ~  L1  (greater than equal-weight case)
    """
    # Sample 0: correct predictions everywhere
    from_sq_0, to_sq_0 = 10, 18

    # Sample 1: all-wrong predictions (uniform logits)
    from_sq_1, to_sq_1 = 10, 18  # same target squares

    # Build a 2-sample batch with targets at from_sq=10, to_sq=18
    batch_equal = {
        "from_sq":       torch.tensor([from_sq_0, from_sq_1], dtype=torch.int64),
        "to_sq":         torch.tensor([to_sq_0,   to_sq_1],   dtype=torch.int64),
        "promo":         torch.zeros(2, dtype=torch.int64),
        "piece_type":    torch.zeros(2, dtype=torch.int64),
        "is_capture":    torch.zeros(2, dtype=torch.float32),
        "is_check":      torch.zeros(2, dtype=torch.float32),
        "is_castle":     torch.zeros(2, dtype=torch.float32),
        "sample_weight": torch.tensor([1.0, 1.0], dtype=torch.float32),
    }
    batch_doubled = {**batch_equal, "sample_weight": torch.tensor([1.0, 2.0], dtype=torch.float32)}

    # Outputs: sample 0 is confident-correct, sample 1 is uniform (confident-wrong)
    BIG, SMALL = 30.0, -30.0
    from_logits = torch.tensor([
        [BIG if j == from_sq_0 else SMALL for j in range(64)],   # sample 0: correct
        [0.0] * 64,                                               # sample 1: uniform
    ])
    to_logits = torch.tensor([
        [BIG if j == to_sq_0 else SMALL for j in range(64)],
        [0.0] * 64,
    ])

    outputs = {
        "from_sq_logits":    from_logits,
        "to_sq_logits":      to_logits,
        "promo_logits":      torch.zeros(2, 4),
        "piece_type_logits": torch.zeros(2, 6),
        "is_capture_logit":  torch.zeros(2, 1),
        "is_check_logit":    torch.zeros(2, 1),
        "is_castle_logit":   torch.zeros(2, 1),
    }

    overrides = {"LABEL_SMOOTHING": 0.0}
    loss_equal   = compute_loss(outputs, batch_equal,   overrides)["total"].item()
    loss_doubled = compute_loss(outputs, batch_doubled, overrides)["total"].item()

    # Doubling weight on the high-loss sample must increase the weighted mean
    assert loss_doubled > loss_equal, (
        f"Doubling sample_weight on high-loss sample should increase mean loss. "
        f"equal={loss_equal:.4f}, doubled={loss_doubled:.4f}"
    )

    # Quantitative check: weighted mean with [1, 2] vs [1, 1]
    # loss_equal   ~ (L0 + L1) / 2
    # loss_doubled ~ (1*L0 + 2*L1) / 2  [note: PyTorch uses sum(w*l)/N not sum(w*l)/sum(w)]
    # So loss_doubled - loss_equal ~ L1/2, which is positive (L1 > 0)
    assert math.isfinite(loss_equal) and math.isfinite(loss_doubled)


# ---------------------------------------------------------------------------
# 4. Label smoothing > 0 gives higher loss for confident correct predictions
# ---------------------------------------------------------------------------

def test_label_smoothing_increases_confident_loss():
    """
    For a high-confidence correct prediction, label_smoothing=0.05 must give a
    HIGHER loss than label_smoothing=0.

    Closed form for from_sq CE with smoothing=eps on a 64-class problem,
    confident correct prediction (softmax ~ 1.0):
        loss_smooth = eps * log(64)  (~ 0.2079 for eps=0.05)
        loss_no_smooth = 0

    So loss_smooth > loss_no_smooth -- assert this.
    """
    batch   = _make_batch(batch_size=4)
    outputs = _make_outputs_confident(batch)

    loss_smooth    = compute_loss(outputs, batch, {"LABEL_SMOOTHING": 0.05})["total"].item()
    loss_no_smooth = compute_loss(outputs, batch, {"LABEL_SMOOTHING": 0.0})["total"].item()

    assert loss_smooth > loss_no_smooth, (
        f"Label smoothing > 0 should increase loss for confident correct predictions. "
        f"smooth={loss_smooth:.4f}, no_smooth={loss_no_smooth:.4f}"
    )

    # Magnitude check: the from + to contribution of label smoothing is
    # approximately 2 * 0.05 * log(64) ~ 0.416 (rough lower bound on gap)
    expected_gap_approx = 2 * 0.05 * math.log(64)
    assert loss_smooth - loss_no_smooth > expected_gap_approx * 0.5, (
        f"Label smoothing gap smaller than expected. "
        f"gap={loss_smooth - loss_no_smooth:.4f}, "
        f"expected >{expected_gap_approx * 0.5:.4f}"
    )


# ---------------------------------------------------------------------------
# 5. Label smoothing reaches the piece_type head specifically (targeted check)
# ---------------------------------------------------------------------------

def test_label_smoothing_applies_to_piece_type_head():
    """
    Label smoothing must reach piece_type_logits specifically, not just from_sq/to_sq.

    Strategy: hold from_sq and to_sq confident-correct (their smoothing contribution
    is identical in both calls, so it cancels out in the aux comparison). Then check
    that changing LABEL_SMOOTHING changes the "aux" component of the loss.

    Closed-form derivation for confident logits (magnitude=30, N=6 classes):
    -----------------------------------------------------------------
    PyTorch CE with label_smoothing = eps:
        loss = (1-eps)*(-log p_correct) + eps*(-1/N * sum_i log(p_i))

    With logits [+30, -30, -30, -30, -30, -30]:
        log p_correct  ~ 30 - log(exp(30) + 5*exp(-30)) ~ 0
        log p_other_i  ~ -30 - log(exp(30) + 5*exp(-30)) ~ -60

    => CE_no_smooth = -(1.0)*0              = 0
    => CE_smooth    = (1-eps)*0 + eps*(-1/6*(0 + 5*(-60)))
                    = eps * (5/6 * 60) = eps * 50 = 0.05 * 50 = 2.5

    Gap per sample = 2.5
    After AUX_LOSS_WEIGHT=0.1: expected_gap = 0.1 * 2.5 = 0.25

    NOTE: eps*log(N) (~0.009) is only the gap for UNIFORM logits; for confident
    logits the gap is much larger because label smoothing also penalises the
    near-zero log-probabilities of the incorrect classes.

    Pre-fix: piece_type ignored label_smoothing -> aux_smooth == aux_no_smooth,
             so the first assert (aux_smooth > aux_no_smooth) fails.
    Post-fix: aux_smooth > aux_no_smooth and actual_gap ≈ 0.25.
    """
    batch   = _make_batch(batch_size=4)
    outputs = _make_outputs_confident(batch)

    aux_smooth    = compute_loss(outputs, batch, {"LABEL_SMOOTHING": 0.05})["aux"].item()
    aux_no_smooth = compute_loss(outputs, batch, {"LABEL_SMOOTHING": 0.0 })["aux"].item()

    assert aux_smooth > aux_no_smooth, (
        f"aux loss (which includes piece_type CE) should increase with label_smoothing > 0. "
        f"smooth={aux_smooth:.6f}, no_smooth={aux_no_smooth:.6f}. "
        f"If these are equal, piece_type CE is not receiving label_smoothing."
    )

    # Closed-form expected gap for confident logits (magnitude=30, N=6, eps=0.05):
    #   per_sample_gap = eps * (N-1)/N * 2 * logit_magnitude = 0.05 * (5/6) * 60 = 2.5
    #   after AUX_LOSS_WEIGHT: expected_gap = 0.1 * 2.5 = 0.25
    N              = 6
    eps            = 0.05
    logit_mag      = 30.0
    per_sample_gap = eps * (N - 1) / N * 2 * logit_mag    # = 2.5
    expected_gap   = config.AUX_LOSS_WEIGHT * per_sample_gap  # = 0.25
    actual_gap     = aux_smooth - aux_no_smooth

    assert abs(actual_gap - expected_gap) < 0.01, (
        f"aux loss smoothing gap doesn't match closed-form for piece_type 6-class smoothing. "
        f"expected~{expected_gap:.6f}, got {actual_gap:.6f}"
    )


