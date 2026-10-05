import chess
import numpy as np
import pytest

from app import config
from app.ml.encoding import (
    sq_to_row_col,
    encode_board,
    encode_move,
    decode_move,
    legal_move_mask,
    legal_promo_classes,
    encode_meta,
)

# ---------------------------------------------------------------------------
# Basic encoding / decoding tests
# ---------------------------------------------------------------------------

def test_sq_to_row_col():
    assert sq_to_row_col(chess.A1) == (0, 0)
    assert sq_to_row_col(chess.H1) == (0, 7)
    assert sq_to_row_col(chess.A8) == (7, 0)
    assert sq_to_row_col(chess.H8) == (7, 7)
    assert sq_to_row_col(chess.E4) == (3, 4)

def test_tensor_basics():
    board = chess.Board()
    tensor = encode_board(board)

    # Shape and dtype
    assert tensor.shape == (config.BOARD_PLANES, 8, 8)
    assert tensor.dtype == np.uint8

    # Values in {0, 1}
    assert np.all(np.logical_or(tensor == 0, tensor == 1))

    # Initial position checks (White to move)
    # Plane 12 (original side to move is White) -> all 1s
    assert np.all(tensor[12] == 1)

    # Castling rights
    assert np.all(tensor[13] == 1)  # White K
    assert np.all(tensor[14] == 1)  # White Q
    assert np.all(tensor[15] == 1)  # Black K
    assert np.all(tensor[16] == 1)  # Black Q

    # No en passant
    assert np.all(tensor[17] == 0)

    # Kings
    # Plane 5: mover (White) King -> e1 (row 0, col 4)
    assert np.sum(tensor[5]) == 1
    assert tensor[5, 0, 4] == 1

    # Plane 11: opponent (Black) King -> e8 (row 7, col 4)
    assert np.sum(tensor[11]) == 1
    assert tensor[11, 7, 4] == 1

    # Non-empty piece planes
    assert np.sum(tensor[0]) == 8  # White pawns
    assert np.sum(tensor[6]) == 8  # Black pawns

def test_symmetry():
    # Setup a random position where it's White to move
    fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5"
    board = chess.Board(fen)
    tensor_w = encode_board(board)

    # Mirror it (now it's Black to move, with pieces flipped)
    board_mirrored = board.mirror()
    tensor_b = encode_board(board_mirrored)

    # They should be identical, except plane 12
    for i in range(config.BOARD_PLANES):
        if i == 12:
            assert np.all(tensor_w[12] == 1)
            assert np.all(tensor_b[12] == 0)
        else:
            np.testing.assert_array_equal(tensor_w[i], tensor_b[i], err_msg=f"Mismatch at plane {i}")

# ---------------------------------------------------------------------------
# Move encoding / decoding (round trip)
# ---------------------------------------------------------------------------

def _assert_round_trip(board: chess.Board, move: chess.Move):
    labels = encode_move(board, move)
    decoded = decode_move(board, labels["from_sq"], labels["to_sq"], labels["promo"])
    assert decoded == move, f"Round trip failed for {move.uci()} in {board.fen()}"

    # Also check that the move is in the legal move mask
    mask = legal_move_mask(board)
    assert mask[labels["from_sq"], labels["to_sq"]] == True, "Move not in legal mask"

def test_move_round_trip_white():
    board = chess.Board()
    # Pawn push
    _assert_round_trip(board, chess.Move.from_uci("e2e4"))
    # Knight move
    _assert_round_trip(board, chess.Move.from_uci("g1f3"))

def test_move_round_trip_black():
    board = chess.Board()
    board.push_uci("e2e4")
    # Pawn push
    _assert_round_trip(board, chess.Move.from_uci("e7e5"))
    # Knight move
    _assert_round_trip(board, chess.Move.from_uci("g8f6"))

def test_castling():
    # White kingside
    board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5")
    _assert_round_trip(board, chess.Move.from_uci("e1g1"))

    # White queenside
    board = chess.Board("r3k2r/ppppqppp/2n2n2/2b1p3/2B1P1b1/2NP1N2/PPP1QPPP/R3K2R w KQkq - 6 5")
    _assert_round_trip(board, chess.Move.from_uci("e1c1"))

    # Black kingside
    board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R b KQkq - 6 5")
    _assert_round_trip(board, chess.Move.from_uci("e8g8"))

    # Black queenside
    board = chess.Board("r3k2r/ppppqppp/2n2n2/2b1p3/2B1P1b1/2NP1N2/PPP1QPPP/R3K2R b KQkq - 6 5")
    _assert_round_trip(board, chess.Move.from_uci("e8c8"))

def test_en_passant():
    # White en passant
    board = chess.Board("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
    ep_move = chess.Move.from_uci("e5d6")
    _assert_round_trip(board, ep_move)
    labels = encode_move(board, ep_move)
    assert labels["is_capture"] == True, "En passant should count as a capture"

    # Black en passant
    board = chess.Board("rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 3")
    ep_move = chess.Move.from_uci("e4d3")
    _assert_round_trip(board, ep_move)
    labels = encode_move(board, ep_move)
    assert labels["is_capture"] == True, "En passant should count as a capture"

def test_promotion():
    # White promotions
    board = chess.Board("8/3P4/8/8/8/8/8/K6k w - - 0 1")
    for piece in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
        _assert_round_trip(board, chess.Move.from_uci(f"d7d8{chess.piece_symbol(piece)}"))

    # Black promotions
    board = chess.Board("8/8/8/8/8/8/3p4/K6k b - - 0 1")
    for piece in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
        _assert_round_trip(board, chess.Move.from_uci(f"d2d1{chess.piece_symbol(piece)}"))

def test_legal_promo_classes():
    # Pawn on 7th rank
    board = chess.Board("8/3P4/8/8/8/8/8/K6k w - - 0 1")
    # Real frame from/to: d7 to d8 -> 51 to 59
    # Mover frame (White) from/to: 51 to 59
    classes = legal_promo_classes(board, 51, 59)
    assert classes == [0, 1, 2, 3]

    # Non-promoting move (King move a1 to a2)
    # Real frame: a1 to a2 -> 0 to 8
    # Mover frame (White): 0 to 8
    classes = legal_promo_classes(board, 0, 8)
    assert classes == [0]

    # Impossible non-promoting move for that pawn (e.g. d7 to e8 capture where there is no piece)
    classes = legal_promo_classes(board, 51, 60)
    assert classes == []

# ---------------------------------------------------------------------------
# Metadata tests
# ---------------------------------------------------------------------------

def test_encode_meta():
    meta = encode_meta(
        player_rating=1900,
        opponent_rating=2000,
        base_seconds=180,
        increment_seconds=2,
        ply=20,
        clock_before=150.0,
        clock_available=True,
        halfmove_clock=4,
    )

    assert meta.shape == (config.META_DIM,)
    assert meta.dtype == np.float32

    assert np.isclose(meta[0], 400 / 400) # (1900 - 1500) / 400 = 1.0
    assert np.isclose(meta[1], 100 / 400) # (2000 - 1900) / 400 = 0.25

    # Base time: 180s. Min=60, Max=1499
    import math
    expected_base = (math.log(180) - math.log(60)) / (math.log(1499) - math.log(60))
    assert np.isclose(meta[2], expected_base)

    assert np.isclose(meta[3], 2 / 30) # inc / 30
    assert np.isclose(meta[4], 20 / 200) # ply / 200
    assert np.isclose(meta[5], 150.0 / 180) # clock_before / base_seconds
    assert np.isclose(meta[6], 1.0) # clock_available
    assert np.isclose(meta[7], 4 / 100) # halfmove_clock / 100

def test_encode_meta_no_clock():
    meta = encode_meta(
        player_rating=1500,
        opponent_rating=1500,
        base_seconds=600,
        increment_seconds=0,
        ply=10,
        clock_before=None,
        clock_available=False,
        halfmove_clock=0,
    )
    assert np.isclose(meta[5], config.META_CLOCK_DEFAULT)
    assert np.isclose(meta[6], 0.0)

def test_encode_meta_clipping():
    meta = encode_meta(
        player_rating=3000,
        opponent_rating=1000,
        base_seconds=30, # below min
        increment_seconds=60, # above max (30)
        ply=300, # above max (200)
        clock_before=40.0, # 40 / 10 (base_seconds bounded) -> 4.0, clip at 3.0
        clock_available=True,
        halfmove_clock=150, # above max (100)
    )

    assert np.isclose(meta[2], 0.0) # base_seconds clamped to 60 -> log(60) -> 0
    assert np.isclose(meta[3], 1.0) # inc clipped to 1.0
    assert np.isclose(meta[4], 1.0) # ply clipped to 1.0
    # Wait, the meta[5] (clock ratio) uses the raw base_seconds for ratio!
    # Let's check encoding.py logic: meta[5] uses base_seconds=30
    assert np.isclose(meta[5], min(40.0 / 30.0, 3.0)) 
    assert np.isclose(meta[7], 1.0) # halfmove clipped to 1.0

def test_is_capture_and_check_timing():
    board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5")
    # Ng5 is not a capture, but doesn't give check
    # Let's find a move that gives check or capture
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    move = chess.Move.from_uci("d1h5") # Qh5, no check, no capture
    labels = encode_move(board, move)
    assert not labels["is_check"]
    assert not labels["is_capture"]

    board.push(move)
    board.push_uci("g7g6") # ...g6
    
    # Qxe5+, capture and check!
    move2 = chess.Move.from_uci("h5e5")
    labels2 = encode_move(board, move2)
    assert labels2["is_check"]
    assert labels2["is_capture"]
