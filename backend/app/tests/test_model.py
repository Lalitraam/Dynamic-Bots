"""
Tests for app/ml/model.py  (Phase 1 - Milestone 3).

Coverage
--------
* Forward pass produces correctly-shaped outputs for all 7 heads.
* eval() mode + same seed -> identical outputs (dropout disabled).
* train() mode + same seed -> DIFFERENT outputs (dropout active).
* Parameter count is printed for sanity (no hard bound).
* Model accepts both uint8 and float32 board tensors.
"""

import math
import pytest
import torch

from app.ml.model import ChessPolicyNet
from app import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_batch(batch_size: int = 4, dtype: torch.dtype = torch.uint8):
    """Return (board, meta) tensors with the right shapes and dtypes."""
    board = torch.randint(0, 2, (batch_size, 18, 8, 8), dtype=dtype)
    meta  = torch.randn(batch_size, config.META_DIM, dtype=torch.float32)
    return board, meta


def _make_model(small: bool = True) -> ChessPolicyNet:
    """Create a small model for faster tests (channels=8, res_blocks=1, hidden=32)."""
    if small:
        return ChessPolicyNet(channels=8, num_res_blocks=1, hidden_dim=32, dropout=0.3)
    return ChessPolicyNet()


# ---------------------------------------------------------------------------
# 1. Forward pass shape correctness
# ---------------------------------------------------------------------------

def test_forward_output_shapes():
    model = _make_model()
    board, meta = _make_batch(batch_size=4)

    model.eval()
    with torch.no_grad():
        out = model(board, meta)

    assert out["from_sq_logits"].shape    == (4, 64),  "from_sq_logits wrong shape"
    assert out["to_sq_logits"].shape      == (4, 64),  "to_sq_logits wrong shape"
    assert out["promo_logits"].shape      == (4, config.PROMO_CLASSES), "promo_logits wrong shape"
    assert out["piece_type_logits"].shape == (4, 6),   "piece_type_logits wrong shape"
    assert out["is_capture_logit"].shape  == (4, 1),   "is_capture_logit wrong shape"
    assert out["is_check_logit"].shape    == (4, 1),   "is_check_logit wrong shape"
    assert out["is_castle_logit"].shape   == (4, 1),   "is_castle_logit wrong shape"


# ---------------------------------------------------------------------------
# 2. eval() mode: deterministic across two forward passes
# ---------------------------------------------------------------------------

def test_eval_mode_is_deterministic():
    model = _make_model()
    model.eval()

    board, meta = _make_batch(batch_size=4)

    torch.manual_seed(0)
    with torch.no_grad():
        out1 = model(board, meta)

    torch.manual_seed(0)
    with torch.no_grad():
        out2 = model(board, meta)

    # All outputs must be bitwise identical in eval mode (dropout disabled)
    for key in out1:
        assert torch.equal(out1[key], out2[key]), (
            f"eval() outputs differ for key '{key}' — dropout is active in eval mode!"
        )


# ---------------------------------------------------------------------------
# 3. train() mode: dropout causes different outputs
# ---------------------------------------------------------------------------

def test_train_mode_has_dropout_variation():
    model = _make_model()
    model.train()

    board, meta = _make_batch(batch_size=4)

    torch.manual_seed(0)
    out1 = model(board, meta)

    torch.manual_seed(1)
    out2 = model(board, meta)

    # At least one head should differ due to dropout
    any_different = any(
        not torch.equal(out1[k], out2[k]) for k in out1
    )
    assert any_different, (
        "train() outputs are identical across different seeds — dropout appears disabled!"
    )


# ---------------------------------------------------------------------------
# 4. Parameter count (sanity visibility, no hard bound)
# ---------------------------------------------------------------------------

def test_parameter_count_logged():
    model = ChessPolicyNet()  # full default size
    total = sum(p.numel() for p in model.parameters())
    print(f"\nChessPolicyNet total parameters: {total:,}")
    # Sanity: at least 100 K, at most 100 M
    assert 100_000 <= total <= 100_000_000, f"Unexpected param count: {total:,}"


# ---------------------------------------------------------------------------
# 5. Accepts both uint8 and float32 board tensors
# ---------------------------------------------------------------------------

def test_accepts_uint8_board():
    model = _make_model()
    model.eval()
    board_u8, meta = _make_batch(batch_size=2, dtype=torch.uint8)
    with torch.no_grad():
        out = model(board_u8, meta)
    assert out["from_sq_logits"].shape == (2, 64)


def test_accepts_float32_board():
    model = _make_model()
    model.eval()
    board_f32, meta = _make_batch(batch_size=2, dtype=torch.float32)
    with torch.no_grad():
        out = model(board_f32, meta)
    assert out["from_sq_logits"].shape == (2, 64)


def test_uint8_and_float32_give_same_result():
    """Same board data encoded as uint8 and float32 must produce identical logits."""
    model = _make_model()
    model.eval()

    board_u8  = torch.randint(0, 2, (2, 18, 8, 8), dtype=torch.uint8)
    board_f32 = board_u8.float()
    meta      = torch.randn(2, config.META_DIM)

    with torch.no_grad():
        out_u8  = model(board_u8,  meta)
        out_f32 = model(board_f32, meta)

    for key in out_u8:
        assert torch.allclose(out_u8[key], out_f32[key], atol=1e-6), (
            f"uint8 and float32 inputs give different results for '{key}'"
        )
