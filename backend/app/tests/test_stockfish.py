"""Unit tests for Stockfish analysis and centipawn loss computation (Phase P)."""

from unittest.mock import MagicMock, patch
import chess
import chess.engine
import pytest

from app import config
from app.analysis.stockfish import StockfishAnalyser, compute_cp_loss


def test_missing_stockfish_path_raises():
    with pytest.raises(FileNotFoundError, match="Stockfish executable not found"):
        StockfishAnalyser(path="non_existent_stockfish_binary.exe")


def test_stockfish_mock_mate_score_conversion():
    # Test mate score conversion (+15000 and -15000)
    mock_engine = MagicMock()

    # White is winning mate in 2
    pov_white_mate = chess.engine.PovScore(chess.engine.Mate(2), chess.WHITE)
    mock_engine.analyse.return_value = {"score": pov_white_mate}

    with patch("chess.engine.SimpleEngine.popen_uci", return_value=mock_engine):
        with patch("pathlib.Path.exists", return_value=True):
            analyser = StockfishAnalyser(path="dummy_sf.exe")
            board = chess.Board()
            eval_cp, mate_in = analyser.analyse_position(board)
            assert eval_cp == 15000.0
            assert mate_in == "2"

            # White is losing mate in 3 (Black has mate)
            pov_black_mate = chess.engine.PovScore(chess.engine.Mate(-3), chess.WHITE)
            mock_engine.analyse.return_value = {"score": pov_black_mate}
            eval_cp_loss, mate_in_loss = analyser.analyse_position(board)
            assert eval_cp_loss == -15000.0
            assert mate_in_loss == "3"


def test_stockfish_mock_cp_sign_flipping_for_black():
    # Test cp sign flipping for black:
    # When board.turn == BLACK, a relative score of +120 means Black is +120 (White is -120).
    mock_engine = MagicMock()
    # Engine returns +120 relative to turn
    pov_score = chess.engine.PovScore(chess.engine.Cp(120), chess.BLACK)
    mock_engine.analyse.return_value = {"score": pov_score}

    with patch("chess.engine.SimpleEngine.popen_uci", return_value=mock_engine):
        with patch("pathlib.Path.exists", return_value=True):
            analyser = StockfishAnalyser(path="dummy_sf.exe")

            board_black = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
            eval_white, mate = analyser.analyse_position(board_black)

            # pov_score.white() for Cp(120) relative to BLACK is -120
            assert eval_white == -120.0
            assert mate is None


def test_compute_cp_loss_scenarios():
    mock_analyser = MagicMock(spec=StockfishAnalyser)

    # 1. Normal good move (White eval before +50, after +40 -> loss 10 cp)
    mock_analyser.analyse_position.side_effect = [(50.0, None), (40.0, None)]
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    loss, is_err = compute_cp_loss(board, move, mock_analyser)
    assert loss == 10.0
    assert is_err is False

    # 2. Blunder (White eval before +50, after -100 -> loss 150 cp >= 100 cp)
    mock_analyser.analyse_position.side_effect = [(50.0, None), (-100.0, None)]
    loss, is_err = compute_cp_loss(board, move, mock_analyser)
    assert loss == 150.0
    assert is_err is True

    # 3. Already lost mover (< -2000 cp) -> returns (0.0, False)
    mock_analyser.analyse_position.side_effect = [(-2500.0, None)]
    loss, is_err = compute_cp_loss(board, move, mock_analyser)
    assert loss == 0.0
    assert is_err is False

    # 4. Clipping at ERROR_MODEL_CP_CLIP (1000)
    mock_analyser.analyse_position.side_effect = [(500.0, None), (-800.0, None)]
    loss, is_err = compute_cp_loss(board, move, mock_analyser)
    assert loss == 1000.0  # 1300 clipped to 1000
    assert is_err is True


def test_checkmate_position_evaluation():
    mock_engine = MagicMock()
    with patch("chess.engine.SimpleEngine.popen_uci", return_value=mock_engine):
        with patch("pathlib.Path.exists", return_value=True):
            analyser = StockfishAnalyser(path="dummy_sf.exe")
            # White checkmated (Fool's Mate)
            board_mated = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
            assert board_mated.is_checkmate()
            eval_cp, mate_in = analyser.analyse_position(board_mated)
            assert eval_cp == -15000.0
            assert mate_in == "0"
