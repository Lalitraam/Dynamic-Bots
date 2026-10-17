"""
Board and move encoding for Phase B chess imitation learning (v3).

Extends v2 encoding with:
- Optional history planes (planes 18-21: prev_move_1 from/to, prev_move_2 from/to in mover frame)
- Compact legal move index generation (sorted unique mover-frame from*64+to)
- 10-dimensional metadata vector with legal move count and berserk flag
- samples_v3 row encoder

Dependencies: numpy, python-chess, config, encoding ONLY (no torch).
"""

from __future__ import annotations

import math
from typing import Any

import chess
import numpy as np

from .. import config
from . import encoding


def sq_to_row_col(sq: int) -> tuple[int, int]:
    """Convert a chess square (0–63) to (row, col) = (sq // 8, sq % 8)."""
    return sq // 8, sq % 8


def _mirror_square(sq: int) -> int:
    """Mirror square vertically (rank flip) for Black's perspective."""
    return sq ^ 56


def _parse_uci_move(m: Any) -> chess.Move | None:
    """Safely parse a UCI move string or chess.Move, returning None on missing/invalid."""
    if m is None:
        return None
    if isinstance(m, float) and math.isnan(m):
        return None
    if isinstance(m, str):
        m_str = m.strip()
        if not m_str:
            return None
        try:
            return chess.Move.from_uci(m_str)
        except (ValueError, chess.InvalidMoveError):
            return None
    if isinstance(m, chess.Move):
        return m
    return None


def encode_board_v3(
    board: chess.Board,
    prev_move_1: str | chess.Move | None = None,
    prev_move_2: str | chess.Move | None = None,
    history: bool = config.HISTORY_PLANES,
) -> np.ndarray:
    """Encode a chess position with optional history planes.

    Planes 0-17: encoding.encode_board(board)
    If history is True:
        Plane 18: from-square of prev_move_1
        Plane 19: to-square of prev_move_1
        Plane 20: from-square of prev_move_2
        Plane 21: to-square of prev_move_2
        Each plane contains a single 1 at (row=sq//8, col=sq%8) in the MOVER's frame
        (squares ^ 56 when Black is to move; prev moves arrive as real-frame UCI strings;
        None or missing -> all-zero plane).
        Total planes P = 22.
    If history is False:
        Planes P = 18.

    Returns
    -------
    np.ndarray
        Shape (P, 8, 8), dtype uint8, values in {0, 1}.
    """
    base = encoding.encode_board(board)
    if not history:
        return base

    planes = np.zeros((22, 8, 8), dtype=np.uint8)
    planes[:18] = base

    is_black = (board.turn == chess.BLACK)

    m1 = _parse_uci_move(prev_move_1)
    if m1 is not None:
        f1 = _mirror_square(m1.from_square) if is_black else m1.from_square
        t1 = _mirror_square(m1.to_square) if is_black else m1.to_square
        r_f1, c_f1 = sq_to_row_col(f1)
        r_t1, c_t1 = sq_to_row_col(t1)
        planes[18, r_f1, c_f1] = 1
        planes[19, r_t1, c_t1] = 1

    m2 = _parse_uci_move(prev_move_2)
    if m2 is not None:
        f2 = _mirror_square(m2.from_square) if is_black else m2.from_square
        t2 = _mirror_square(m2.to_square) if is_black else m2.to_square
        r_f2, c_f2 = sq_to_row_col(f2)
        r_t2, c_t2 = sq_to_row_col(t2)
        planes[20, r_f2, c_f2] = 1
        planes[21, r_t2, c_t2] = 1

    return planes


def legal_indices(board: chess.Board) -> np.ndarray:
    """Return sorted unique legal move indices (from*64 + to) in mover frame.

    When Black is to move, squares are mirrored via sq ^ 56.
    Promotions collapse to a single from*64+to entry (unique).
    Castling is the king's two-square move (e.g. e1g1 / e1c1).

    Returns
    -------
    np.ndarray
        Shape (num_unique_moves,), dtype int16, strictly ascending.
    """
    is_black = (board.turn == chess.BLACK)
    indices: set[int] = set()

    for move in board.legal_moves:
        f = _mirror_square(move.from_square) if is_black else move.from_square
        t = _mirror_square(move.to_square) if is_black else move.to_square
        indices.add(f * 64 + t)

    return np.array(sorted(indices), dtype=np.int16)


def encode_meta_v3(
    player_rating: int,
    opponent_rating: int,
    base_seconds: int,
    increment_seconds: int,
    ply: int,
    clock_before: float | None,
    clock_available: bool,
    halfmove_clock: int,
    num_legal_moves: int,
    berserk: bool,
) -> np.ndarray:
    """Encode game/position metadata as a float32 vector of length META_DIM_V3 = 10.

    Vector elements
    ---------------
    0-7: encoding.encode_meta(...) (8 v2 features)
    8:   min(num_legal_moves / 60.0, 1.0)
    9:   1.0 if berserk else 0.0

    Returns
    -------
    np.ndarray
        Shape (10,), dtype float32.
    """
    meta = np.zeros(config.META_DIM_V3, dtype=np.float32)
    meta[:8] = encoding.encode_meta(
        player_rating=player_rating,
        opponent_rating=opponent_rating,
        base_seconds=base_seconds,
        increment_seconds=increment_seconds,
        ply=ply,
        clock_before=clock_before,
        clock_available=clock_available,
        halfmove_clock=halfmove_clock,
    )
    meta[8] = min(float(num_legal_moves) / 60.0, 1.0)
    meta[9] = 1.0 if berserk else 0.0
    return meta


def encode_row_v3(
    row: Any,
    history: bool = config.HISTORY_PLANES,
) -> dict[str, Any]:
    """Encode a samples_v3 row into tensors and labels.

    Parameters
    ----------
    row : Any
        A dict, pandas Series, or object containing samples_v3 columns:
        fen, uci, prev_move_1, prev_move_2, player_rating, opponent_rating,
        base_s, increment_s, ply, clock_before_move, halfmove_clock,
        num_legal_moves, target_berserk, sample_weight.
    history : bool
        Whether to generate history planes (P=22 if True, P=18 if False).

    Returns
    -------
    dict
        "planes": np.ndarray uint8 (P, 8, 8)
        "meta": np.ndarray float32 (10,)
        "move_idx": np.int16 scalar
        "promo": np.int8 scalar
        "legal_idx": np.ndarray int16 (1D array of legal move indices)
        "weight": np.float32 scalar
        "berserk": bool
    """
    def _get(key: str, default: Any = None) -> Any:
        if isinstance(row, dict):
            return row.get(key, default)
        if hasattr(row, key):
            val = getattr(row, key)
            return val if val is not None else default
        try:
            val = row[key]
            return val if val is not None else default
        except (KeyError, IndexError, TypeError):
            return default

    fen = str(_get("fen"))
    board = chess.Board(fen)

    uci_str = str(_get("uci"))
    move = chess.Move.from_uci(uci_str)
    move_dict = encoding.encode_move(board, move)

    from_sq = move_dict["from_sq"]
    to_sq = move_dict["to_sq"]
    move_idx = np.int16(from_sq * 64 + to_sq)
    promo = np.int8(move_dict["promo"])

    legal_idx = legal_indices(board)
    weight = np.float32(_get("sample_weight", 1.0))
    berserk = bool(_get("target_berserk", False))

    cb = _get("clock_before_move")
    if cb is None or (isinstance(cb, float) and math.isnan(cb)):
        clock_before = None
        clock_available = False
    else:
        clock_before = float(cb)
        clock_available = True

    base_s = int(_get("base_s", 0))
    increment_s = int(_get("increment_s", 0))
    player_rating = int(_get("player_rating", config.DEFAULT_RATING))
    opp_rating = int(_get("opponent_rating", config.DEFAULT_RATING))
    ply = int(_get("ply", 0))
    halfmove_clock = int(_get("halfmove_clock", 0))

    num_legal_val = _get("num_legal_moves")
    if num_legal_val is None or (isinstance(num_legal_val, float) and math.isnan(num_legal_val)):
        num_legal = board.legal_moves.count()
    else:
        num_legal = int(num_legal_val)

    meta = encode_meta_v3(
        player_rating=player_rating,
        opponent_rating=opp_rating,
        base_seconds=base_s,
        increment_seconds=increment_s,
        ply=ply,
        clock_before=clock_before,
        clock_available=clock_available,
        halfmove_clock=halfmove_clock,
        num_legal_moves=num_legal,
        berserk=berserk,
    )

    pm1 = _get("prev_move_1")
    pm2 = _get("prev_move_2")
    planes = encode_board_v3(
        board=board,
        prev_move_1=pm1,
        prev_move_2=pm2,
        history=history,
    )

    return {
        "planes": planes,
        "meta": meta,
        "move_idx": move_idx,
        "promo": promo,
        "legal_idx": legal_idx,
        "weight": weight,
        "berserk": berserk,
    }
