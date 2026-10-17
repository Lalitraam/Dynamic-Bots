"""Complexity and position feature extraction for error modeling (Phase P)."""

from __future__ import annotations
import math
import chess
import numpy as np

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

FEATURE_NAMES = [
    "num_legal",
    "forcing",
    "policy_entropy",
    "top_gap",
    "n_plausible",
    "material_imbalance",
    "own_hanging",
    "in_check",
    "phase",
    "clock_ratio",
    "clock_abs",
    "opp_last_capture",
]


def compute_features(
    board: chess.Board,
    clock_before: float | None,
    base_s: int,
    ply: int,
    policy_log_probs: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute 12 complexity and situational features for a chess position.

    1. num_legal: count of legal moves
    2. forcing: count of legal checks and captures
    3. policy_entropy: Shannon entropy from policy_log_probs, or 0.0 if None
    4. top_gap: log p1 - log p2, or 0.0 if None
    5. n_plausible: count of moves with probability >= 0.05, or 0 if None
    6. material_imbalance: mover material minus opponent material (P=1, N=3, B=3, R=5, Q=9)
    7. own_hanging: mover pieces attacked by opponent and not defended by own pieces
    8. in_check: 1.0 if mover is in check, else 0.0
    9. phase: 0=opening, 1=middlegame, 2=endgame based on non-pawn material
    10. clock_ratio: clock_before / base_s, or -1.0 if unknown
    11. clock_abs: log(clock_before + 1), or 0.0 if unknown
    12. opp_last_capture: 1.0 if opponent's previous move was a capture, else 0.0
    """
    # 1. num_legal
    num_legal = float(board.legal_moves.count())

    # 2. forcing (checks + captures)
    forcing_count = 0
    for move in board.legal_moves:
        if board.is_capture(move) or board.gives_check(move):
            forcing_count += 1
    forcing = float(forcing_count)

    # 3. policy_entropy, 4. top_gap, 5. n_plausible
    policy_entropy = 0.0
    top_gap = 0.0
    n_plausible = 0.0

    if policy_log_probs and len(policy_log_probs) > 0:
        log_probs = np.array(list(policy_log_probs.values()), dtype=float)
        # Shift for numerical stability in case of logits
        max_lp = np.max(log_probs)
        exps = np.exp(log_probs - max_lp)
        sum_exps = np.sum(exps)
        if sum_exps > 0:
            probs = exps / sum_exps
            # Shannon entropy: - sum(p * ln(p))
            pos_probs = probs[probs > 0]
            policy_entropy = float(-np.sum(pos_probs * np.log(pos_probs)))
            # n_plausible
            n_plausible = float(np.sum(probs >= 0.05))

        # top_gap: difference between top-1 and top-2
        sorted_lp = np.sort(log_probs)[::-1]
        if len(sorted_lp) >= 2:
            top_gap = float(sorted_lp[0] - sorted_lp[1])

    # 6. material_imbalance
    mover_color = board.turn
    opp_color = not mover_color
    mover_mat = sum(len(board.pieces(pt, mover_color)) * val for pt, val in PIECE_VALUES.items())
    opp_mat = sum(len(board.pieces(pt, opp_color)) * val for pt, val in PIECE_VALUES.items())
    material_imbalance = float(mover_mat - opp_mat)

    # 7. own_hanging (mover pieces attacked by opponent and not defended)
    own_hanging_count = 0
    for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        for sq in board.pieces(pt, mover_color):
            is_attacked = board.is_attacked_by(opp_color, sq)
            is_defended = board.is_attacked_by(mover_color, sq)
            if is_attacked and not is_defended:
                own_hanging_count += 1
    own_hanging = float(own_hanging_count)

    # 8. in_check
    in_check = 1.0 if board.is_check() else 0.0

    # 9. phase (0=opening, 1=middlegame, 2=endgame based on non-pawn material)
    non_pawn_mat = 0
    for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        count = len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK))
        non_pawn_mat += count * PIECE_VALUES[pt]

    if non_pawn_mat >= 48:
        phase = 0.0  # Opening
    elif non_pawn_mat > 16:
        phase = 1.0  # Middlegame
    else:
        phase = 2.0  # Endgame

    # 10. clock_ratio
    if clock_before is not None and base_s > 0 and clock_before >= 0:
        clock_ratio = float(clock_before / base_s)
    else:
        clock_ratio = -1.0

    # 11. clock_abs
    if clock_before is not None and clock_before >= 0:
        clock_abs = float(math.log(clock_before + 1.0))
    else:
        clock_abs = 0.0

    # 12. opp_last_capture
    opp_last_capture = 0.0
    if len(board.move_stack) > 0:
        b_copy = board.copy()
        last_move = b_copy.pop()
        if b_copy.is_capture(last_move):
            opp_last_capture = 1.0

    return {
        "num_legal": num_legal,
        "forcing": forcing,
        "policy_entropy": policy_entropy,
        "top_gap": top_gap,
        "n_plausible": n_plausible,
        "material_imbalance": material_imbalance,
        "own_hanging": own_hanging,
        "in_check": in_check,
        "phase": phase,
        "clock_ratio": clock_ratio,
        "clock_abs": clock_abs,
        "opp_last_capture": opp_last_capture,
    }
