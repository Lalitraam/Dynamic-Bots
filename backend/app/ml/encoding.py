"""
Board and move encoding for chess imitation learning.

Single source of truth for all encoding/decoding logic.
Dependencies: numpy, python-chess ONLY (no torch, no FastAPI).

Coordinate convention
---------------------
All positions are encoded from the **side-to-move's perspective**.
When Black is to move, ``board.mirror()`` (vertical flip + color swap)
is applied so the mover always appears as "White" in the tensor.

Move labels are ALSO mirrored into the same frame (Black's e7e5 → e2e4).
``decode_move`` un-mirrors.  Uses ``chess.square_mirror`` (sq ^ 56).

Square-to-grid mapping (ONE canonical definition)
--------------------------------------------------
``sq_to_row_col(sq)`` → ``(sq // 8, sq % 8)``

    a1 = 0 → (0, 0)    h1 = 7 → (0, 7)
    a2 = 8 → (1, 0)    ...
    a8 = 56 → (7, 0)   h8 = 63 → (7, 7)

Row = rank index 0–7, Col = file index 0–7.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import chess
import numpy as np

from .. import config

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

# Ordered piece types matching plane indices 0–5
_PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

# chess piece_type → plane offset (0–5)
_PIECE_TO_PLANE: dict[int, int] = {pt: i for i, pt in enumerate(_PIECE_TYPES)}

# Promotion class mapping:  0 = none-or-queen, 1 = knight, 2 = bishop, 3 = rook
_PROMO_TO_CLASS: dict[int | None, int] = {
    None: 0,
    chess.QUEEN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
}

_CLASS_TO_PROMO: dict[int, int | None] = {
    0: None,       # resolved to QUEEN at decode time when pawn reaches last rank
    1: chess.KNIGHT,
    2: chess.BISHOP,
    3: chess.ROOK,
}


# ---------------------------------------------------------------------------
# Square mapping
# ---------------------------------------------------------------------------

def sq_to_row_col(sq: int) -> tuple[int, int]:
    """Convert a chess square (0–63) to ``(row, col)``.

    Mapping (single canonical definition):
        row = sq // 8  (rank index, 0 = rank 1)
        col = sq % 8   (file index, 0 = file a)

    Examples::

        a1 (0)  → (0, 0)
        h1 (7)  → (0, 7)
        e4 (28) → (3, 4)
        a8 (56) → (7, 0)
        h8 (63) → (7, 7)
    """
    return sq // 8, sq % 8


def _mirror_square(sq: int) -> int:
    """Mirror a square vertically (rank flip): a1 ↔ a8, b2 ↔ b7, etc."""
    return sq ^ 56  # same as chess.square_mirror(sq)


# ---------------------------------------------------------------------------
# Board encoding
# ---------------------------------------------------------------------------

def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a chess position as an ``(18, 8, 8)`` uint8 tensor.

    The position is always encoded from the side-to-move's perspective.
    If Black is to move, ``board.mirror()`` is applied internally.

    Plane layout
    ------------
    ======  ================================================
    Plane   Content
    ======  ================================================
    0–5     Mover's pieces: P, N, B, R, Q, K
    6–11    Opponent's pieces: P, N, B, R, Q, K
    12      Constant plane — all 1s if the **original** side
            to move was White, else all 0s
    13      Mover's kingside castling right  (constant plane)
    14      Mover's queenside castling right (constant plane)
    15      Opponent's kingside castling right
    16      Opponent's queenside castling right
    17      En passant target square (single 1 in mirrored
            frame, or all zeros)
    ======  ================================================

    Returns
    -------
    np.ndarray
        Shape ``(18, 8, 8)``, dtype ``uint8``, values in ``{0, 1}``.
    """
    tensor = np.zeros((config.BOARD_PLANES, 8, 8), dtype=np.uint8)

    original_is_white = board.turn == chess.WHITE

    # Mirror if Black to move — mover always appears as White
    work = board.mirror() if not original_is_white else board

    # Planes 0–5: mover's pieces (White in the working board)
    for plane_idx, pt in enumerate(_PIECE_TYPES):
        for sq in work.pieces(pt, chess.WHITE):
            r, c = sq_to_row_col(sq)
            tensor[plane_idx, r, c] = 1

    # Planes 6–11: opponent's pieces (Black in the working board)
    for plane_idx, pt in enumerate(_PIECE_TYPES):
        for sq in work.pieces(pt, chess.BLACK):
            r, c = sq_to_row_col(sq)
            tensor[6 + plane_idx, r, c] = 1

    # Plane 12: side-to-move indicator (original frame)
    if original_is_white:
        tensor[12, :, :] = 1

    # Planes 13–14: mover castling rights
    if work.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1
    if work.has_queenside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1

    # Planes 15–16: opponent castling rights
    if work.has_kingside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1
    if work.has_queenside_castling_rights(chess.BLACK):
        tensor[16, :, :] = 1

    # Plane 17: en passant target square (in mirrored frame)
    if work.ep_square is not None:
        r, c = sq_to_row_col(work.ep_square)
        tensor[17, r, c] = 1

    return tensor


# ---------------------------------------------------------------------------
# Move encoding / decoding
# ---------------------------------------------------------------------------

def encode_move(board: chess.Board, move: chess.Move) -> dict:
    """Encode a move into mover-frame labels.

    **Must be called BEFORE the move is pushed** to the board.
    ``is_capture`` and ``is_check`` are evaluated on the pre-move board
    via ``board.is_capture(move)`` and ``board.gives_check(move)``.

    Parameters
    ----------
    board : chess.Board
        Position *before* the move (real frame).
    move : chess.Move
        The move to encode (real frame).

    Returns
    -------
    dict
        ``from_sq``      int 0–63, mover frame
        ``to_sq``        int 0–63, mover frame
        ``promo``        int 0–3 (0 = none/queen, 1 = knight, 2 = bishop, 3 = rook)
        ``piece_type``   int 0–5 (P, N, B, R, Q, K)
        ``is_capture``   bool — includes en passant
        ``is_check``     bool
        ``is_castle``    bool
    """
    is_black = board.turn == chess.BLACK

    # Piece type of the moving piece
    piece = board.piece_at(move.from_square)
    piece_type_idx = _PIECE_TO_PLANE[piece.piece_type] if piece else 0

    # Auxiliary labels — BEFORE pushing the move
    is_capture = board.is_capture(move)
    is_check = board.gives_check(move)
    is_castle = board.is_castling(move)

    # From/to squares in mover frame
    from_sq = _mirror_square(move.from_square) if is_black else move.from_square
    to_sq = _mirror_square(move.to_square) if is_black else move.to_square

    # Promotion class
    promo_class = _PROMO_TO_CLASS.get(move.promotion, 0)

    return {
        "from_sq": from_sq,
        "to_sq": to_sq,
        "promo": promo_class,
        "piece_type": piece_type_idx,
        "is_capture": is_capture,
        "is_check": is_check,
        "is_castle": is_castle,
    }


def decode_move(
    board: chess.Board,
    from_sq: int,
    to_sq: int,
    promo: int,
) -> chess.Move:
    """Decode mover-frame labels back into a ``chess.Move`` (real frame).

    Parameters
    ----------
    board : chess.Board
        Position *before* the move (real frame).
    from_sq : int
        Source square in mover frame (0–63).
    to_sq : int
        Target square in mover frame (0–63).
    promo : int
        Promotion class: 0 = none/queen, 1 = knight, 2 = bishop, 3 = rook.

    Returns
    -------
    chess.Move
        A legal move in the real frame.

    Raises
    ------
    ValueError
        If the decoded move is not legal in the given position.
    """
    is_black = board.turn == chess.BLACK

    # Un-mirror squares
    real_from = _mirror_square(from_sq) if is_black else from_sq
    real_to = _mirror_square(to_sq) if is_black else to_sq

    # Determine promotion piece
    promotion = _CLASS_TO_PROMO.get(promo)

    # For class 0 (none/queen): decide based on whether it's actually a
    # pawn reaching the last rank
    if promo == 0:
        piece = board.piece_at(real_from)
        if piece and piece.piece_type == chess.PAWN:
            to_rank = chess.square_rank(real_to)
            if (board.turn == chess.WHITE and to_rank == 7) or \
               (board.turn == chess.BLACK and to_rank == 0):
                promotion = chess.QUEEN

    move = chess.Move(real_from, real_to, promotion=promotion)

    if move not in board.legal_moves:
        raise ValueError(
            f"Decoded move {move.uci()} is not legal in position "
            f"{board.fen()}"
        )

    return move


# ---------------------------------------------------------------------------
# Legal-move mask
# ---------------------------------------------------------------------------

def legal_move_mask(board: chess.Board) -> np.ndarray:
    """Compute a ``(64, 64)`` boolean from/to mask of legal moves in mover frame.

    For a given ``(from_sq, to_sq)`` pair in the mover frame,
    ``mask[from_sq, to_sq]`` is True if at least one legal move exists
    with that source and destination (covering all promotion types).

    Returns
    -------
    np.ndarray
        Shape ``(64, 64)``, dtype ``bool``.
    """
    mask = np.zeros((64, 64), dtype=bool)
    is_black = board.turn == chess.BLACK

    for move in board.legal_moves:
        f = _mirror_square(move.from_square) if is_black else move.from_square
        t = _mirror_square(move.to_square) if is_black else move.to_square
        mask[f, t] = True

    return mask


def legal_promo_classes(
    board: chess.Board,
    from_sq_mover: int,
    to_sq_mover: int,
) -> list[int]:
    """Return the sorted list of legal promotion classes for a from/to pair.

    Parameters
    ----------
    board : chess.Board
        Position before the move (real frame).
    from_sq_mover : int
        Source square in mover frame.
    to_sq_mover : int
        Destination square in mover frame.

    Returns
    -------
    list[int]
        Sorted unique promotion classes (0–3) that are legal for this
        from/to pair.  Empty if no legal move exists for these squares.
    """
    is_black = board.turn == chess.BLACK

    real_from = _mirror_square(from_sq_mover) if is_black else from_sq_mover
    real_to = _mirror_square(to_sq_mover) if is_black else to_sq_mover

    classes: set[int] = set()
    for move in board.legal_moves:
        if move.from_square == real_from and move.to_square == real_to:
            classes.add(_PROMO_TO_CLASS.get(move.promotion, 0))

    return sorted(classes)


# ---------------------------------------------------------------------------
# Metadata encoding
# ---------------------------------------------------------------------------

def encode_meta(
    player_rating: int,
    opponent_rating: int,
    base_seconds: int,
    increment_seconds: int,
    ply: int,
    clock_before: float | None,
    clock_available: bool,
    halfmove_clock: int,
) -> np.ndarray:
    """Encode game/position metadata as a float32 vector.

    Parameters
    ----------
    player_rating : int
        Player's rating for this game.
    opponent_rating : int
        Opponent's rating for this game.
    base_seconds : int
        Raw base time in seconds (from ``parse_time_control``).
        **NOT** the Lichess estimated time (base + 40*inc).
    increment_seconds : int
        Increment in seconds (from ``parse_time_control``).
    ply : int
        Current ply number (0-indexed).
    clock_before : float or None
        Clock remaining in seconds before this move, or None if
        unavailable.
    clock_available : bool
        Whether clock data is available for this move.
    halfmove_clock : int
        Board's halfmove clock (plies since last capture or pawn push).

    Returns
    -------
    np.ndarray
        Shape ``(META_DIM,)`` = ``(8,)``, dtype ``float32``.

    Vector elements
    ---------------
    ===  ===============================================================
    Idx  Feature
    ===  ===============================================================
    0    ``(player_rating − 1500) / 400``
    1    ``(opponent_rating − player_rating) / 400``
    2    base_seconds, log-scaled & normalized to [0, 1] over [60, 1499]
    3    ``increment_seconds / 30``, clipped to [0, 1]
    4    ``ply / 200``, clipped to [0, 1]
    5    ``clock_remaining / base_time``, clipped to META_CLOCK_CLIP;
         defaults to META_CLOCK_DEFAULT (0.5) if clock unavailable
    6    clock_available flag (0.0 or 1.0)
    7    ``halfmove_clock / 100``, clipped to [0, 1]
    ===  ===============================================================
    """
    meta = np.zeros(config.META_DIM, dtype=np.float32)

    # 0: player rating, centered and scaled
    meta[0] = (player_rating - config.META_RATING_CENTER) / config.META_RATING_SCALE

    # 1: rating gap (positive = stronger opponent)
    meta[1] = (opponent_rating - player_rating) / config.META_RATING_SCALE

    # 2: base time, log-scaled and normalized to [0, 1]
    base_clamped = max(config.META_BASE_TIME_MIN, min(base_seconds, config.META_BASE_TIME_MAX))
    log_min = math.log(config.META_BASE_TIME_MIN)
    log_max = math.log(config.META_BASE_TIME_MAX)
    meta[2] = (math.log(base_clamped) - log_min) / (log_max - log_min)

    # 3: increment, linearly normalized and clipped
    meta[3] = min(increment_seconds / config.META_INCREMENT_CLIP, 1.0)

    # 4: ply, linearly normalized and clipped
    meta[4] = min(ply / config.META_PLY_CLIP, 1.0)

    # 5: clock ratio
    if clock_available and clock_before is not None and base_seconds > 0:
        meta[5] = min(clock_before / base_seconds, config.META_CLOCK_CLIP)
    else:
        meta[5] = config.META_CLOCK_DEFAULT

    # 6: clock available flag
    meta[6] = 1.0 if clock_available else 0.0

    # 7: halfmove clock, normalized and clipped
    meta[7] = min(halfmove_clock / config.META_HALFMOVE_CLIP, 1.0)

    return meta
