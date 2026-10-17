from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Protocol
import random
import chess

@dataclass
class MoveContext:
    board: chess.Board
    prev_moves: list[str]        # UCI, most recent last
    bot_rating: int
    opp_rating: int
    base_seconds: int
    increment_seconds: int
    bot_clock_ms: int
    opp_clock_ms: int
    ply: int

@dataclass
class MoveDecision:
    move: chess.Move
    source: Literal["book", "policy", "guarded"]
    think_ms: int
    top5: list[tuple[str, float]]
    p_err: float | None
    model_version: str

class MoveProvider(Protocol):
    def decide(self, ctx: MoveContext, rng: random.Random) -> MoveDecision: ...
