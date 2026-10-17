"""In-memory active game state representation (Phase F)."""

from __future__ import annotations
import time
from typing import Any
import chess
from app.game.clock import GameClock


class GameState:
    """Encapsulates in-memory chess board, authoritative clock, and match metadata."""

    def __init__(
        self,
        game_id: str,
        player_username: str,
        human_color: str,
        base_s: int,
        inc_s: int,
        secret_hash: str | None = None,
        owner_id: int | None = None,
    ):
        self.game_id = game_id
        self.player_username = player_username
        self.human_color = human_color.lower().strip()
        self.bot_color = "black" if self.human_color == "white" else "white"
        self.base_s = int(base_s)
        self.inc_s = int(inc_s)
        self.secret_hash = secret_hash
        self.owner_id = owner_id

        self.board = chess.Board()
        self.clock = GameClock(base_s=self.base_s, inc_s=self.inc_s)
        self.moves: list[dict[str, Any]] = []
        self.status: str = "active"  # active, completed, aborted
        self.result: str | None = None
        self.termination: str | None = None

    def to_snapshot(self, now: float | None = None) -> dict[str, Any]:
        """Generate client-facing game state snapshot."""
        cur_now = now if now is not None else time.time()
        clocks = self.clock.get_remaining(cur_now) if self.clock.last_tick is not None else self.clock.remaining_ms
        return {
            "game_id": self.game_id,
            "fen": self.board.fen(),
            "status": self.status,
            "result": self.result,
            "termination": self.termination,
            "human_color": self.human_color,
            "bot_username": self.player_username,
            "moves": self.moves,
            "clocks": clocks,
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "ply": len(self.moves),
        }
