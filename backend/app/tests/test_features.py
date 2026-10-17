"""Unit tests for complexity feature extraction (Phase P)."""

import chess
import numpy as np
import pytest

from app.analysis.features import FEATURE_NAMES, compute_features


def test_feature_dict_keys():
    board = chess.Board()
    feats = compute_features(board, clock_before=120.0, base_s=180, ply=0)
    assert set(feats.keys()) == set(FEATURE_NAMES)
    assert len(feats) == 12


def test_hand_verified_in_check():
    # Position where Black king is in check by White queen
    fen_check = "rnb1kbnr/pppp1ppp/8/4p3/7q/5P2/PPPPP1PP/RNBQKBNR w KQkq - 1 3"
    board = chess.Board(fen_check)
    assert board.is_check()
    feats = compute_features(board, clock_before=60.0, base_s=180, ply=5)
    assert feats["in_check"] == 1.0

    # Starting position: not in check
    feats_start = compute_features(chess.Board(), clock_before=180.0, base_s=180, ply=0)
    assert feats_start["in_check"] == 0.0


def test_hand_verified_num_legal():
    # Starting position has exactly 20 legal moves
    board = chess.Board()
    feats = compute_features(board, clock_before=180.0, base_s=180, ply=0)
    assert feats["num_legal"] == 20.0

    # Fool's mate: 0 legal moves
    board_mate = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    feats_mate = compute_features(board_mate, clock_before=100.0, base_s=180, ply=4)
    assert feats_mate["num_legal"] == 0.0


def test_hand_verified_material_imbalance():
    # Starting position: 0.0
    feats_equal = compute_features(chess.Board(), clock_before=180.0, base_s=180, ply=0)
    assert feats_equal["material_imbalance"] == 0.0

    # White is up a Queen (9 points), White to move
    # White: K + Q (9), Black: K (0) -> +9.0
    board_up_q = chess.Board("8/8/8/8/8/8/4Q3/4K2k w - - 0 1")
    feats_up_q = compute_features(board_up_q, clock_before=100.0, base_s=180, ply=40)
    assert feats_up_q["material_imbalance"] == 9.0

    # Black to move on the same board: Black is down 9 points -> -9.0
    board_down_q = chess.Board("8/8/8/8/8/8/4Q3/4K2k b - - 0 1")
    feats_down_q = compute_features(board_down_q, clock_before=100.0, base_s=180, ply=40)
    assert feats_down_q["material_imbalance"] == -9.0


def test_hand_verified_phase_boundaries():
    # 1. Opening: starting position (non-pawn material = 62 >= 48)
    feats_opening = compute_features(chess.Board(), clock_before=180.0, base_s=180, ply=0)
    assert feats_opening["phase"] == 0.0

    # 2. Middlegame: queens off the board (non-pawn material = 62 - 18 = 44, between 16 and 48)
    fen_queens_off = "r1b1k2r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1B1K2R w KQkq - 0 7"
    feats_mid = compute_features(chess.Board(fen_queens_off), clock_before=120.0, base_s=180, ply=14)
    assert feats_mid["phase"] == 1.0

    # 3. Endgame: kings + rooks only (each side has 1 rook = 5+5 = 10 non-pawn material <= 16)
    fen_endgame = "4k3/pppppppp/8/8/8/8/PPPPPPPP/R3K3 w - - 0 30"
    feats_end = compute_features(chess.Board(fen_endgame), clock_before=30.0, base_s=180, ply=60)
    assert feats_end["phase"] == 2.0


def test_hand_verified_own_hanging():
    # Position where White's e4 pawn is attacked by Black's d5 pawn, and e4 is not defended
    # Board: White e4, Black d5. e4 attacked by d5, nothing defends e4.
    fen_hanging = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    # Actually in that standard position, d1 queen defends e4, f1 bishop doesn't, b1 knight doesn't.
    # Let's verify undefended piece:
    # White knight on e5, Black pawn on d6 attacks e5, no White piece attacks e5:
    fen_knight_hanging = "8/8/3p4/4N3/8/8/8/4K2k w - - 0 1"
    board = chess.Board(fen_knight_hanging)
    assert board.is_attacked_by(chess.BLACK, chess.E5)
    assert not board.is_attacked_by(chess.WHITE, chess.E5)
    feats = compute_features(board, clock_before=50.0, base_s=180, ply=20)
    assert feats["own_hanging"] == 1.0


def test_policy_prob_features():
    board = chess.Board()
    # 3 legal moves with specified log probabilities
    log_probs = {"e2e4": 0.0, "d2d4": -1.0, "g1f3": -3.0}
    feats = compute_features(board, clock_before=60.0, base_s=180, ply=1, policy_log_probs=log_probs)

    assert feats["policy_entropy"] > 0.0
    assert abs(feats["top_gap"] - 1.0) < 1e-4
    assert feats["n_plausible"] >= 2.0
