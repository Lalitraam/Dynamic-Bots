"""
Inference utilities and model export for chess player-cloning (Milestone 3).

Functions
---------
predict_move(model, board, meta_inputs=None, device="cpu", temperature=1.0,
             opening_book=None, rng=None, fallback_model=None) -> chess.Move
save_model(model, path) -> None
load_model(path, device="cpu", **model_kwargs) -> ChessPolicyNet
"""

from __future__ import annotations

from pathlib import Path
import random
from typing import TYPE_CHECKING, Any

import chess
import numpy as np
import torch
import torch.nn as nn

from .. import config
from . import encoding
from .model import ChessPolicyNet

if TYPE_CHECKING:
    from ..services.opening_book import OpeningBook


def predict_move(
    model: nn.Module,
    board: chess.Board,
    meta_inputs: dict[str, Any] | None = None,
    device: torch.device | str = "cpu",
    temperature: float = config.INFERENCE_TEMPERATURE_DEFAULT,
    opening_book: OpeningBook | None = None,
    rng: random.Random | None = None,
    fallback_model: Any = None,  # Reserved for future shared-base-model fallback; unused in Milestone 3
) -> chess.Move:
    """
    Predict the next move for a given position using opening book or the trained CNN.

    Uses independent from_sq and to_sq heads corrected by legal-move masking to guarantee
    100% legal moves at inference time.

    Parameters
    ----------
    model : nn.Module
        The policy network instance.
    board : chess.Board
        Current board state.
    meta_inputs : dict, optional
        Metadata attributes for encode_meta (player_rating, opponent_rating, etc.).
    device : torch.device or str
        Device for model inference ('cuda' or 'cpu').
    temperature : float
        Sampling temperature (T -> 0 behaves as argmax).
    opening_book : OpeningBook, optional
        If provided and the position matches the book, moves are sampled from the book directly.
    rng : random.Random, optional
        Random generator for reproducible move sampling.
    fallback_model : Any, optional
        Reserved for future shared-base-model fallback; unused in Milestone 3.

    Returns
    -------
    chess.Move
        A legal move in the given position.
    """
    if rng is None:
        rng = random.Random()

    # 1. Opening book check
    if opening_book is not None:
        book_probs = opening_book.query(board)
        if book_probs is not None:
            sampled_move = opening_book.sample(board, rng=rng)
            if sampled_move is not None:
                return sampled_move

    # Check for game termination / no legal moves
    if not board.legal_moves:
        raise ValueError(f"No legal moves available in position {board.fen()}")

    # 2. Board and metadata encoding
    default_meta = {
        "player_rating": 1500,
        "opponent_rating": 1500,
        "base_seconds": 300,
        "increment_seconds": 0,
        "ply": board.ply(),
        "clock_before": None,
        "clock_available": False,
        "halfmove_clock": board.halfmove_clock,
    }
    actual_meta = {**default_meta, **(meta_inputs or {})}

    board_arr = encoding.encode_board(board)
    meta_arr = encoding.encode_meta(**actual_meta)

    board_tensor = torch.from_numpy(board_arr).unsqueeze(0).to(device)
    meta_tensor = torch.from_numpy(meta_arr).unsqueeze(0).to(device)

    # 3. Model forward pass (eval mode)
    model.eval()
    with torch.no_grad():
        outputs = model(board=board_tensor, meta=meta_tensor)
        from_logits = outputs["from_sq_logits"][0].cpu()  # (64,)
        to_logits = outputs["to_sq_logits"][0].cpu()      # (64,)
        promo_logits = outputs["promo_logits"][0].cpu()    # (4,)

    # 4. Legal-move masking
    mask = encoding.legal_move_mask(board)  # (64, 64) bool in mover frame
    from_mask = mask.any(axis=1)           # (64,) bool in mover frame

    if not from_mask.any():
        raise ValueError(f"No legal from-squares found in position {board.fen()}")

    legal_from_indices = np.where(from_mask)[0].tolist()

    # Step 4a: Sample or argmax from_sq
    if temperature < 1e-5:
        # Near-deterministic / argmax
        best_from = legal_from_indices[0]
        best_from_val = from_logits[best_from].item()
        for idx in legal_from_indices[1:]:
            val = from_logits[idx].item()
            if val > best_from_val:
                best_from_val = val
                best_from = idx
        from_sq = best_from
    else:
        legal_from_logits = from_logits[from_mask] / temperature
        legal_from_logits = legal_from_logits - legal_from_logits.max()
        from_probs = torch.softmax(legal_from_logits, dim=-1).tolist()
        from_sq = rng.choices(legal_from_indices, weights=from_probs, k=1)[0]

    # Step 4b: Sample or argmax to_sq conditioned on sampled from_sq
    to_mask = mask[from_sq, :]  # (64,) bool for this specific from_sq
    legal_to_indices = np.where(to_mask)[0].tolist()

    if not legal_to_indices:
        raise ValueError(f"No legal destination squares for from_sq={from_sq} in {board.fen()}")

    if temperature < 1e-5:
        best_to = legal_to_indices[0]
        best_to_val = to_logits[best_to].item()
        for idx in legal_to_indices[1:]:
            val = to_logits[idx].item()
            if val > best_to_val:
                best_to_val = val
                best_to = idx
        to_sq = best_to
    else:
        legal_to_logits = to_logits[to_mask] / temperature
        legal_to_logits = legal_to_logits - legal_to_logits.max()
        to_probs = torch.softmax(legal_to_logits, dim=-1).tolist()
        to_sq = rng.choices(legal_to_indices, weights=to_probs, k=1)[0]

    # 5. Resolve promotion
    legal_promos = encoding.legal_promo_classes(board, from_sq, to_sq)
    if not legal_promos:
        promo = 0
    elif len(legal_promos) == 1:
        promo = legal_promos[0]
    else:
        # Genuine promotion choice: pick argmax restricted to legal promo classes.
        # Queen (class 0) is default on ties.
        best_cls = legal_promos[0]
        best_logit = promo_logits[best_cls].item()
        for cls in legal_promos[1:]:
            cls_logit = promo_logits[cls].item()
            if cls_logit > best_logit:
                best_logit = cls_logit
                best_cls = cls
            elif cls_logit == best_logit:
                if cls == 0 or (best_cls != 0 and cls < best_cls):
                    best_cls = cls
        promo = best_cls

    # 6. Decode into real-board chess.Move (handles mirroring and legality check)
    return encoding.decode_move(board, from_sq, to_sq, promo)


def save_model(model: nn.Module, path: Path | str) -> None:
    """Save model state_dict to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(
    path: Path | str,
    device: torch.device | str = "cpu",
    **model_kwargs: Any,
) -> ChessPolicyNet:
    """Load a ChessPolicyNet model from a state_dict checkpoint."""
    model = ChessPolicyNet(**model_kwargs)
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
