"""Stockfish analysis wrapper for position evaluation and centipawn loss computation (Phase P)."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import chess
import chess.engine

from app import config


class StockfishAnalyser:
    """Wrapper around python-chess engine module for position evaluation."""

    def __init__(
        self,
        path: Path | str | None = None,
        depth: int | None = None,
    ):
        self.engine: chess.engine.SimpleEngine | None = None
        target_path = Path(path) if path is not None else Path(config.STOCKFISH_PATH)
        if not target_path.exists():
            raise FileNotFoundError(
                f"Stockfish executable not found at specified path: {target_path}"
            )
        self.path = target_path
        self.depth = depth if depth is not None else getattr(config, "STOCKFISH_DEPTH", 12)

        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(str(self.path))
            threads = getattr(config, "STOCKFISH_THREADS", 1)
            hash_mb = getattr(config, "STOCKFISH_HASH_MB", 64)
            self.engine.configure({"Threads": threads, "Hash": hash_mb})
        except Exception as e:
            self.close()
            raise RuntimeError(f"Failed to initialize Stockfish engine: {e}") from e

    def analyse_position(self, board: chess.Board) -> tuple[float, str | None]:
        """Evaluate board position from White's perspective.

        Returns (eval_cp, mate_in).
        eval_cp is capped to ±1500 for normal scores, and ±15000 for mate.
        mate_in is string representing distance to mate or None.
        Exceptions close the engine and raise a RuntimeError.
        """
        if self.engine is None:
            raise RuntimeError("StockfishAnalyser engine is closed.")

        if board.is_checkmate():
            # If White's turn and checkmate, Black won: White is -15000
            eval_cp = -15000.0 if board.turn == chess.WHITE else 15000.0
            return eval_cp, "0"

        try:
            info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
            score: chess.engine.PovScore = info["score"]
            pov_white = score.white()

            if pov_white.is_mate():
                mate_val = pov_white.mate()
                eval_cp = 15000.0 if (mate_val is not None and mate_val > 0) else -15000.0
                mate_in = str(abs(mate_val)) if mate_val is not None else "1"
                return eval_cp, mate_in
            else:
                cp = pov_white.score()
                if cp is None:
                    rel = score.relative.score() or 0
                    cp = rel if board.turn == chess.WHITE else -rel
                eval_cp = float(max(-1500.0, min(1500.0, float(cp))))
                return eval_cp, None
        except Exception as e:
            self.close()
            raise RuntimeError(f"Stockfish analysis failed: {e}") from e

    def close(self):
        """Shut down the UCI engine process."""
        if self.engine is not None:
            try:
                self.engine.quit()
            except Exception:
                pass
            self.engine = None

    def __enter__(self) -> StockfishAnalyser:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self):
        self.close()


def compute_cp_loss(
    board_before: chess.Board,
    played_move: chess.Move,
    analyser: StockfishAnalyser,
) -> tuple[float, bool]:
    """Calculate centipawn loss and binary error indicator for played_move.

    cp_loss = max(0, eval_before_mover - eval_after_mover).
    Capped at config.ERROR_MODEL_CP_CLIP.
    If the mover was already lost (eval_before < -2000), return (0.0, False).
    """
    eval_before_white, _ = analyser.analyse_position(board_before)
    is_white_mover = (board_before.turn == chess.WHITE)
    eval_before_mover = eval_before_white if is_white_mover else -eval_before_white

    if eval_before_mover < -2000.0:
        return 0.0, False

    board_after = board_before.copy()
    board_after.push(played_move)
    eval_after_white, _ = analyser.analyse_position(board_after)
    eval_after_mover = eval_after_white if is_white_mover else -eval_after_white

    raw_loss = eval_before_mover - eval_after_mover
    clip_limit = float(getattr(config, "ERROR_MODEL_CP_CLIP", 1000.0))
    cp_loss = float(max(0.0, min(clip_limit, raw_loss)))

    min_cp_error = float(getattr(config, "ERROR_MODEL_MIN_CP_LOSS", 100.0))
    is_error = bool(cp_loss >= min_cp_error)

    return cp_loss, is_error
