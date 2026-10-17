"""
Tests for Opening Book V2 service (app/services/opening_book_v2.py).

Verifies:
1. Ply cutoff: ply 17 excluded, ply 16 included; query returns None if board.ply() > 16.
2. Raw visits threshold: 4 raw rows (any weights) -> None; 5 raw rows -> distribution.
3. Weights: e2e4 raw 3 weights [1,1,1] (w=3.0) and d2d4 raw 2 weights [2,2] (w=4.0) ->
   prob(e4) = 3/7 ~= 0.428571, prob(d4) = 4/7 ~= 0.571429 (differs from raw 0.6 / 0.4).
4. Mode 'eval' ignores val/test rows; mode 'release' includes them; bad mode -> ValueError.
5. Newer games dominate: moves with weights 1.6 vs 0.4 yield 4:1 prob ratio (0.8 vs 0.2).
6. JSON round-trip: save/load gives equal dict and identical query results.
7. total_visits returns raw count n.
"""

from pathlib import Path
import chess
import pytest

from app.services.opening_book_v2 import BookV2, build_book_v2

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_1_ply_cutoff():
    """
    Test 1: Ply cutoff at 16 plies.

    A row with ply 17 is excluded from the build; ply 16 is included.
    Query on a board with board.ply() == 17 returns None even if the position key exists.
    """
    rows = [
        {"fen": START_FEN, "uci": "e2e4", "ply": 16, "split": "train", "sample_weight": 1.0}
        for _ in range(5)
    ]
    # Add rows at ply 17
    rows.append({"fen": START_FEN, "uci": "g1f3", "ply": 17, "split": "train", "sample_weight": 1.0})

    book_dict = build_book_v2(rows, mode="eval")
    book = BookV2.from_dict(book_dict)

    # Move g1f3 at ply 17 must be excluded
    key = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
    assert "g1f3" not in book.positions[key]
    assert "e2e4" in book.positions[key]

    # Query on normal starting board (ply 0 <= 16)
    board_0 = chess.Board()
    assert board_0.ply() == 0
    res_0 = book.query(board_0)
    assert res_0 is not None
    assert "e2e4" in res_0

    # Create a board at ply 17 with the same starting piece placement
    board_17 = chess.Board(START_FEN)
    # Manually push 17 dummy reversible moves or set halfmove/turn, but chess.Board calculates ply:
    # A cleaner way: push 17 moves to reach board.ply() == 17
    b = chess.Board()
    # 8 full moves back and forth knight moves + 1 move = 17 plies
    moves = ["g1f3", "g8f6", "f3g1", "f6g8"] * 4 + ["e2e4"]
    for m in moves:
        b.push_san(m)
    assert b.ply() == 17

    # Manually put the key of board 'b' into the book with >=5 visits
    b_key = " ".join(b.fen().split()[:4])
    book.positions[b_key] = {"e7e5": {"w": 5.0, "n": 5}}

    # Query on board with ply == 17 must return None due to cutoff
    assert book.query(b) is None


def test_2_threshold_uses_raw_counts():
    """
    Test 2: Thresholding uses RAW visits, not sum of weights.

    4 raw rows with high weights (total w = 40.0) -> None (< 5 raw visits).
    Adding a 5th raw row (total raw n = 5) -> distribution returned.
    """
    rows = [
        {"fen": START_FEN, "uci": "e2e4", "ply": 0, "split": "train", "sample_weight": 10.0}
        for _ in range(4)
    ]
    book_dict = build_book_v2(rows, mode="eval")
    book = BookV2.from_dict(book_dict)

    board = chess.Board()
    # 4 visits < 5 -> None
    assert book.query(board) is None

    # Add 5th visit with small weight 1.0
    rows.append({"fen": START_FEN, "uci": "e2e4", "ply": 0, "split": "train", "sample_weight": 1.0})
    book_5 = BookV2.from_dict(build_book_v2(rows, mode="eval"))
    res = book_5.query(board)
    assert res is not None
    assert res["e2e4"] == 1.0


def test_3_weights_diverge_from_raw_probabilities():
    """
    Test 3: Weighted probability calculation diverges from raw visit ratios.

    Position has:
    - e2e4: raw count 3, weights [1.0, 1.0, 1.0] -> w = 3.0
    - d2d4: raw count 2, weights [2.0, 2.0] -> w = 4.0
    Total raw visits n = 5.
    Raw ratios: e4 = 3/5 = 0.60, d4 = 2/5 = 0.40.
    Weighted probabilities:
    - e4 = 3.0 / (3.0 + 4.0) = 3/7 = 0.428571...
    - d4 = 4.0 / (3.0 + 4.0) = 4/7 = 0.571429...
    """
    rows = [
        {"fen": START_FEN, "uci": "e2e4", "ply": 0, "split": "train", "sample_weight": 1.0},
        {"fen": START_FEN, "uci": "e2e4", "ply": 0, "split": "train", "sample_weight": 1.0},
        {"fen": START_FEN, "uci": "e2e4", "ply": 0, "split": "train", "sample_weight": 1.0},
        {"fen": START_FEN, "uci": "d2d4", "ply": 0, "split": "train", "sample_weight": 2.0},
        {"fen": START_FEN, "uci": "d2d4", "ply": 0, "split": "train", "sample_weight": 2.0},
    ]
    book = BookV2.from_dict(build_book_v2(rows, mode="eval"))
    probs = book.query(chess.Board())
    assert probs is not None

    expected_e4 = 3.0 / 7.0
    expected_d4 = 4.0 / 7.0

    assert probs["e2e4"] == pytest.approx(expected_e4, rel=1e-5)
    assert probs["d2d4"] == pytest.approx(expected_d4, rel=1e-5)
    assert abs(probs["e2e4"] - 0.6) > 0.1  # Confirms it diverges from raw 0.6


def test_4_mode_eval_vs_release_and_invalid():
    """
    Test 4:
    - mode 'eval' includes only split == 'train' rows.
    - mode 'release' includes train, val, and test rows.
    - invalid mode raises ValueError.
    """
    rows = [
        {"fen": START_FEN, "uci": "e2e4", "ply": 0, "split": "train", "sample_weight": 1.0}
        for _ in range(5)
    ]
    rows += [
        {"fen": START_FEN, "uci": "d2d4", "ply": 0, "split": "val", "sample_weight": 1.0}
        for _ in range(5)
    ]
    rows += [
        {"fen": START_FEN, "uci": "c2c4", "ply": 0, "split": "test", "sample_weight": 1.0}
        for _ in range(5)
    ]

    # Eval mode: only e2e4
    book_eval = BookV2.from_dict(build_book_v2(rows, mode="eval"))
    probs_eval = book_eval.query(chess.Board())
    assert probs_eval is not None
    assert set(probs_eval.keys()) == {"e2e4"}

    # Release mode: all three moves included
    book_release = BookV2.from_dict(build_book_v2(rows, mode="release"))
    probs_release = book_release.query(chess.Board())
    assert probs_release is not None
    assert set(probs_release.keys()) == {"e2e4", "d2d4", "c2c4"}

    # Bad mode raises ValueError
    with pytest.raises(ValueError, match="Invalid mode"):
        build_book_v2(rows, mode="unknown_mode")


def test_5_newer_games_dominate_weight():
    """
    Test 5: Newer games dominate via higher weight.

    Move A has 1 visit with newer weight 1.6.
    Move B has 4 visits with older weight 0.1 each (w = 0.4).
    Total raw visits n = 5 >= 5.
    Expected prob ratio:
    prob(A) = 1.6 / (1.6 + 0.4) = 0.8
    prob(B) = 0.4 / (1.6 + 0.4) = 0.2
    Ratio prob(A) / prob(B) = 0.8 / 0.2 = 4.0 (4:1 ratio hand-computed).
    """
    rows = [
        {"fen": START_FEN, "uci": "e2e4", "ply": 0, "split": "train", "sample_weight": 1.6},
        {"fen": START_FEN, "uci": "d2d4", "ply": 0, "split": "train", "sample_weight": 0.1},
        {"fen": START_FEN, "uci": "d2d4", "ply": 0, "split": "train", "sample_weight": 0.1},
        {"fen": START_FEN, "uci": "d2d4", "ply": 0, "split": "train", "sample_weight": 0.1},
        {"fen": START_FEN, "uci": "d2d4", "ply": 0, "split": "train", "sample_weight": 0.1},
    ]
    book = BookV2.from_dict(build_book_v2(rows, mode="eval"))
    probs = book.query(chess.Board())
    assert probs is not None

    assert probs["e2e4"] == pytest.approx(0.8, rel=1e-5)
    assert probs["d2d4"] == pytest.approx(0.2, rel=1e-5)
    assert probs["e2e4"] / probs["d2d4"] == pytest.approx(4.0, rel=1e-5)


def test_6_json_round_trip(tmp_path: Path):
    """
    Test 6: JSON serialization round-trip: save/load gives equal dict and identical query.
    """
    rows = [
        {"fen": START_FEN, "uci": "e2e4", "ply": 0, "split": "train", "sample_weight": 1.0}
        for _ in range(5)
    ]
    book = BookV2.from_dict(build_book_v2(rows, mode="eval"))

    save_path = tmp_path / "book_v2.json"
    book.save(save_path)

    loaded = BookV2.load(save_path)
    assert loaded.to_dict() == book.to_dict()

    board = chess.Board()
    assert loaded.query(board) == book.query(board)


def test_7_total_visits_returns_raw_n():
    """
    Test 7: total_visits returns raw count n (0 if missing).
    """
    rows = [
        {"fen": START_FEN, "uci": "e2e4", "ply": 0, "split": "train", "sample_weight": 3.5},
        {"fen": START_FEN, "uci": "e2e4", "ply": 0, "split": "train", "sample_weight": 2.5},
        {"fen": START_FEN, "uci": "d2d4", "ply": 0, "split": "train", "sample_weight": 1.2},
    ]
    book = BookV2.from_dict(build_book_v2(rows, mode="eval"))

    board = chess.Board()
    # Even if total visits 3 < 5, total_visits() returns raw visit count 3
    assert book.total_visits(board) == 3

    # For an unseen board position, returns 0
    unseen_board = chess.Board("8/8/8/8/8/8/8/4K2k w - - 0 1")
    assert book.total_visits(unseen_board) == 0
