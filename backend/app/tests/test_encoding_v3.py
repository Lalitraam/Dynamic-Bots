"""
Tests for Phase B v3 encoding (app/ml/encoding_v3.py).

Verifies:
1. planes 0-17 of encode_row_v3 equal encoding.encode_board for 200 random positions.
2. history planes, hand-traced:
   (a) Black to move after 1.e4 d5 2.exd5, prev_move_1="e4d5", prev_move_2="d7d5":
       plane 18 single 1 at row 4 col 4, plane 19 at row 3 col 3,
       plane 20 at row 1 col 3, plane 21 at row 3 col 3 (mirrored).
   (b) White to move after 1.e4 e5, prev_move_1="e7e5", prev_move_2="e2e4":
       plane 18 row 6 col 4, plane 19 row 4 col 4,
       plane 20 row 1 col 4, plane 21 row 3 col 4.
   (c) both None -> planes 18-21 all zero. With history off: P == 18.
3. encode_meta_v3 with player 1900, opp 2000, base 180, inc 2, ply 20, clock 150.0 available, halfmove 4:
   first 8 equal values in test_encode_meta; [8] for num_legal_moves 30 = 0.5 and for 90 = 1.0;
   [9] = 1.0 for berserk True and 0.0 for False; shape (10,), float32.
4. legal_indices for 200 random positions & promotion position hand-computed.
"""

import math
import random
import chess
import numpy as np
import pytest

from app import config
from app.ml import encoding
from app.ml import encoding_v3


def _generate_200_positions() -> list[tuple[chess.Board, chess.Move]]:
    """Deterministically generate 200 valid (board, move) pairs."""
    rng = random.Random(42)
    positions: list[tuple[chess.Board, chess.Move]] = []
    seen_fens: set[str] = set()

    while len(positions) < 200:
        board = chess.Board()
        game_length = rng.randint(5, 60)
        for _ in range(game_length):
            if board.is_game_over():
                break
            moves = list(board.legal_moves)
            if not moves:
                break
            move = rng.choice(moves)
            fen = board.fen()
            if fen not in seen_fens:
                seen_fens.add(fen)
                positions.append((board.copy(), move))
                if len(positions) >= 200:
                    break
            board.push(move)

    return positions


def test_1_planes_0_to_17_equal_encode_board_for_200_positions():
    """
    Test 1: Assert that planes 0-17 of encode_row_v3 equal encoding.encode_board
    for 200 distinct random chess positions.

    Hand-computed invariant:
    v3 extends the 18 board planes with 4 optional history planes (P=22).
    Planes 0 through 17 represent the identical piece placement, side-to-move,
    castling rights, and en-passant square as computed by v2 encoding.encode_board.
    For every position, np.array_equal(row_dict['planes'][:18], encode_board(board))
    must evaluate to True.
    """
    pairs = _generate_200_positions()
    assert len(pairs) == 200

    for raw_board, move in pairs:
        fen = raw_board.fen()
        board = chess.Board(fen)
        v2_planes = encoding.encode_board(board)

        row = {
            "fen": fen,
            "uci": move.uci(),
            "prev_move_1": None,
            "prev_move_2": None,
            "player_rating": 1500,
            "opponent_rating": 1500,
            "base_s": 300,
            "increment_s": 0,
            "ply": board.ply(),
            "clock_before_move": 300.0,
            "halfmove_clock": board.halfmove_clock,
            "num_legal_moves": board.legal_moves.count(),
            "target_berserk": False,
            "sample_weight": 1.0,
        }

        row_dict = encoding_v3.encode_row_v3(row, history=True)
        v3_planes = row_dict["planes"]

        assert v3_planes.shape == (22, 8, 8)
        assert np.array_equal(v3_planes[:18], v2_planes)


def test_2_history_planes_hand_traced():
    """
    Test 2: History planes hand-traced for White and Black mover frames.

    Hand computation:
    (a) Black to move after 1.e4 d5 2.exd5:
        Real squares:
        - prev_move_1 = "e4d5":
          from_sq = e4 = file e (4) + rank 4 (3*8) = 28.
          to_sq = d5 = file d (3) + rank 5 (4*8) = 35.
          Black is mover -> mirrored via sq ^ 56:
          from_sq_mirrored = 28 ^ 56 = 36 -> row = 36//8 = 4, col = 36%8 = 4.
          to_sq_mirrored = 35 ^ 56 = 27 -> row = 27//8 = 3, col = 27%8 = 3.
        - prev_move_2 = "d7d5":
          from_sq = d7 = file d (3) + rank 7 (6*8) = 51.
          to_sq = d5 = file d (3) + rank 5 (4*8) = 35.
          Mirrored via sq ^ 56:
          from_sq_mirrored = 51 ^ 56 = 11 -> row = 11//8 = 1, col = 11%8 = 3.
          to_sq_mirrored = 35 ^ 56 = 27 -> row = 27//8 = 3, col = 27%8 = 3.
        Expected:
          plane 18: single 1 at row 4 col 4 (sum=1)
          plane 19: single 1 at row 3 col 3 (sum=1)
          plane 20: single 1 at row 1 col 3 (sum=1)
          plane 21: single 1 at row 3 col 3 (sum=1)

    (b) White to move after 1.e4 e5:
        Real squares (White is mover, no mirror):
        - prev_move_1 = "e7e5":
          from_sq = e7 = 52 -> row = 52//8 = 6, col = 52%8 = 4.
          to_sq = e5 = 36 -> row = 36//8 = 4, col = 36%8 = 4.
        - prev_move_2 = "e2e4":
          from_sq = e2 = 12 -> row = 12//8 = 1, col = 12%8 = 4.
          to_sq = e4 = 28 -> row = 28//8 = 3, col = 28%8 = 4.
        Expected:
          plane 18: single 1 at row 6 col 4
          plane 19: single 1 at row 4 col 4
          plane 20: single 1 at row 1 col 4
          plane 21: single 1 at row 3 col 4

    (c) Both moves None:
        Planes 18-21 all zero (sum=0).
        With history off (history=False): P == 18.
    """
    # (a) Black to move after 1.e4 d5 2.exd5
    board_a = chess.Board()
    board_a.push_san("e4")
    board_a.push_san("d5")
    board_a.push_san("exd5")
    assert board_a.turn == chess.BLACK

    planes_a = encoding_v3.encode_board_v3(
        board_a,
        prev_move_1="e4d5",
        prev_move_2="d7d5",
        history=True,
    )
    assert planes_a.shape == (22, 8, 8)

    # Plane 18: row 4, col 4
    assert planes_a[18, 4, 4] == 1
    assert planes_a[18].sum() == 1

    # Plane 19: row 3, col 3
    assert planes_a[19, 3, 3] == 1
    assert planes_a[19].sum() == 1

    # Plane 20: row 1, col 3
    assert planes_a[20, 1, 3] == 1
    assert planes_a[20].sum() == 1

    # Plane 21: row 3, col 3
    assert planes_a[21, 3, 3] == 1
    assert planes_a[21].sum() == 1

    # (b) White to move after 1.e4 e5
    board_b = chess.Board()
    board_b.push_san("e4")
    board_b.push_san("e5")
    assert board_b.turn == chess.WHITE

    planes_b = encoding_v3.encode_board_v3(
        board_b,
        prev_move_1="e7e5",
        prev_move_2="e2e4",
        history=True,
    )
    assert planes_b.shape == (22, 8, 8)

    # Plane 18: row 6, col 4
    assert planes_b[18, 6, 4] == 1
    assert planes_b[18].sum() == 1

    # Plane 19: row 4, col 4
    assert planes_b[19, 4, 4] == 1
    assert planes_b[19].sum() == 1

    # Plane 20: row 1, col 4
    assert planes_b[20, 1, 4] == 1
    assert planes_b[20].sum() == 1

    # Plane 21: row 3, col 4
    assert planes_b[21, 3, 4] == 1
    assert planes_b[21].sum() == 1

    # (c) Both None
    planes_c = encoding_v3.encode_board_v3(
        board_a,
        prev_move_1=None,
        prev_move_2=None,
        history=True,
    )
    assert planes_c[18:22].sum() == 0

    # History off -> P == 18
    planes_no_hist = encoding_v3.encode_board_v3(
        board_a,
        prev_move_1="e4d5",
        prev_move_2="d7d5",
        history=False,
    )
    assert planes_no_hist.shape == (18, 8, 8)


def test_3_encode_meta_v3_hand_computed():
    """
    Test 3: Metadata encoding vector of length META_DIM_V3 = 10.

    Setup:
      player_rating = 1900
      opponent_rating = 2000
      base_seconds = 180
      increment_seconds = 2
      ply = 20
      clock_before = 150.0 (available)
      halfmove_clock = 4

    Hand computation:
      [0] (1900 - 1500) / 400 = 1.0
      [1] (2000 - 1900) / 400 = 0.25
      [2] (ln(180) - ln(60)) / (ln(1499) - ln(60))
          ln(180/60) / ln(1499/60) = ln(3) / ln(24.983333) ~ 1.0986123 / 3.2182093 ~ 0.3413738
      [3] 2 / 30 = 0.0666667
      [4] 20 / 200 = 0.1
      [5] 150.0 / 180.0 = 0.8333333
      [6] 1.0 (clock available)
      [7] 4 / 100 = 0.04
      Features 0-7 are identical to v2 encoding.encode_meta(...).
      [8] Legal move feature: min(num_legal_moves / 60.0, 1.0)
          For num_legal_moves = 30: min(30/60, 1.0) = 0.5
          For num_legal_moves = 90: min(90/60, 1.0) = 1.0
      [9] Berserk feature:
          For berserk = True: 1.0
          For berserk = False: 0.0
      Output shape: (10,), dtype: float32.
    """
    # 1. Check with num_legal_moves=30 and berserk=True
    meta_30_true = encoding_v3.encode_meta_v3(
        player_rating=1900,
        opponent_rating=2000,
        base_seconds=180,
        increment_seconds=2,
        ply=20,
        clock_before=150.0,
        clock_available=True,
        halfmove_clock=4,
        num_legal_moves=30,
        berserk=True,
    )
    assert meta_30_true.shape == (10,)
    assert meta_30_true.dtype == np.float32

    # Check v2 equality for first 8 features
    v2_meta = encoding.encode_meta(
        player_rating=1900,
        opponent_rating=2000,
        base_seconds=180,
        increment_seconds=2,
        ply=20,
        clock_before=150.0,
        clock_available=True,
        halfmove_clock=4,
    )
    assert np.allclose(meta_30_true[:8], v2_meta, atol=1e-6)

    # Hand-computed checks on first 8
    assert pytest.approx(meta_30_true[0], abs=1e-5) == 1.0
    assert pytest.approx(meta_30_true[1], abs=1e-5) == 0.25
    assert pytest.approx(meta_30_true[3], abs=1e-5) == 2.0 / 30.0
    assert pytest.approx(meta_30_true[4], abs=1e-5) == 0.1
    assert pytest.approx(meta_30_true[5], abs=1e-5) == 150.0 / 180.0
    assert pytest.approx(meta_30_true[6], abs=1e-5) == 1.0
    assert pytest.approx(meta_30_true[7], abs=1e-5) == 0.04

    # Feature 8 and 9
    assert pytest.approx(meta_30_true[8], abs=1e-5) == 0.5
    assert pytest.approx(meta_30_true[9], abs=1e-5) == 1.0

    # 2. Check with num_legal_moves=90 and berserk=False
    meta_90_false = encoding_v3.encode_meta_v3(
        player_rating=1900,
        opponent_rating=2000,
        base_seconds=180,
        increment_seconds=2,
        ply=20,
        clock_before=150.0,
        clock_available=True,
        halfmove_clock=4,
        num_legal_moves=90,
        berserk=False,
    )
    assert pytest.approx(meta_90_false[8], abs=1e-5) == 1.0
    assert pytest.approx(meta_90_false[9], abs=1e-5) == 0.0


def test_4_legal_indices_coverage_and_hand_computed_promotion():
    """
    Test 4: legal_indices coverage and hand-computed promotion count.

    Hand computation:
    Promotion position FEN: "7k/4P3/8/8/8/8/8/4K3 w - - 0 1"
    White to move:
    - Pawn on e7 (square 52):
      Can promote to e8 (square 60).
      Moves: e7e8q, e7e8r, e7e8b, e7e8n (4 promotion moves).
      In mover frame (White, real==mover):
      from_sq = 52, to_sq = 60.
      Index = 52 * 64 + 60 = 3328 + 60 = 3388.
      All 4 promotions collapse into 1 unique from*64+to index.
    - King on e1 (square 4):
      King moves:
      e1d1: 4*64 + 3 = 259
      e1f1: 4*64 + 5 = 261
      e1d2: 4*64 + 11 = 267
      e1e2: 4*64 + 12 = 268
      e1f2: 4*64 + 13 = 269
      (5 king moves).
    Total legal moves in python-chess = 4 + 5 = 9.
    Total unique legal indices = 1 (pawn) + 5 (king) = 6.
    Expected sorted unique array:
    [259, 261, 267, 268, 269, 3388] (length 6, dtype int16).
    """
    # 1. Test promotion position hand-computed values
    promo_board = chess.Board("7k/4P3/8/8/8/8/8/4K3 w - - 0 1")
    assert len(list(promo_board.legal_moves)) == 9

    promo_indices = encoding_v3.legal_indices(promo_board)
    assert len(promo_indices) == 6
    expected_indices = np.array([259, 261, 267, 268, 269, 3388], dtype=np.int16)
    np.testing.assert_array_equal(promo_indices, expected_indices)

    # 2. Test 200 random positions
    pairs = _generate_200_positions()
    for board, _ in pairs:
        indices = encoding_v3.legal_indices(board)

        # Sorted unique (strictly ascending)
        assert indices.dtype == np.int16
        assert len(indices) > 0
        if len(indices) > 1:
            assert np.all(np.diff(indices) > 0)

        # Every legal move's from*64+to in mover frame is present
        is_black = (board.turn == chess.BLACK)
        for move in board.legal_moves:
            f = move.from_square ^ 56 if is_black else move.from_square
            t = move.to_square ^ 56 if is_black else move.to_square
            expected_idx = f * 64 + t
            assert expected_idx in indices

        # Each index decodes back via encoding.decode_move for some promo class
        for idx in indices:
            from_sq = int(idx) // 64
            to_sq = int(idx) % 64
            found_legal = False
            for promo_class in (0, 1, 2, 3):
                try:
                    dec_move = encoding.decode_move(board, from_sq, to_sq, promo_class)
                    if dec_move in board.legal_moves:
                        found_legal = True
                        break
                except ValueError:
                    pass
            assert found_legal, f"Index {idx} failed to decode to legal move in {board.fen()}"
