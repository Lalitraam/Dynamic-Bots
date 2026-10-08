"""
Game sessions: human vs bot (Milestone 4, Phase 3).

A GameSession owns the authoritative server-side board for one game. The
client only ever sends a move string; the server validates it against ITS board
before applying anything.

Storage
-------
Sessions live in memory (a dict inside SessionManager), like the ingest job
dicts. Games are short-lived and a server restart ending them is acceptable.
Idle sessions expire after config.SESSION_IDLE_TTL_SECONDS and the total number
of live sessions is capped by config.SESSION_MAX_COUNT.

A session holds its own reference to the LoadedBot it started with, so a game
in progress keeps its model even if the model cache evicts it or the player is
retrained mid-game.

Concurrency
-----------
Each session has its own lock. play_turn() (human move + bot reply) runs under
that lock, so two simultaneous requests for the same game are serialised: the
second one is validated against the board the first one left behind.

Atomic turns
------------
If the bot fails after the human's move was applied (model error, illegal bot
move), the human's move is rolled back, so the client can simply retry.
"""
from __future__ import annotations

import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional

import chess

from .. import config
from .model_cache import LoadedBot

# mover(bot, board, meta_inputs, temperature) -> chess.Move
Mover = Callable[[LoadedBot, chess.Board, dict, float], chess.Move]

_UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")


# ---------------------------------------------------------------------------
# Errors (the router maps these to HTTP status codes in Phase 3/4)
# ---------------------------------------------------------------------------

class SessionError(Exception):
    """Base class for session errors."""


class SessionNotFoundError(SessionError):
    """Unknown or expired session id."""


class SessionLimitError(SessionError):
    """Too many live sessions."""


class GameOverError(SessionError):
    """The game is already finished."""


class NotYourTurnError(SessionError):
    """The human tried to move while it is the bot's turn."""


class InvalidMoveError(SessionError):
    """Base class for rejected human moves."""


class InvalidMoveFormatError(InvalidMoveError):
    """The move string is not valid UCI (e.g. 'e2e4', 'e7e8q')."""


class IllegalMoveError(InvalidMoveError):
    """Well-formed UCI, but not legal in the current position."""


class BotMoveError(SessionError):
    """The bot failed to produce a legal move."""


# ---------------------------------------------------------------------------
# Default mover (wraps ml.export.predict_move; torch imported lazily)
# ---------------------------------------------------------------------------

def predict_with_model(
    bot: LoadedBot, board: chess.Board, meta_inputs: dict, temperature: float
) -> chess.Move:
    from ..ml import export

    return export.predict_move(
        bot.model,
        board,
        meta_inputs=meta_inputs,
        device="cpu",
        temperature=temperature,
        opening_book=bot.opening_book,
    )


# ---------------------------------------------------------------------------
# GameSession
# ---------------------------------------------------------------------------

def _color_name(color: bool) -> str:
    return "white" if color == chess.WHITE else "black"


@dataclass
class _MoveRecord:
    ply: int
    san: str
    uci: str
    by: str  # "human" | "bot"

    def as_dict(self) -> dict[str, Any]:
        return {"ply": self.ply, "san": self.san, "uci": self.uci, "by": self.by}


class GameSession:
    def __init__(
        self,
        session_id: str,
        username: str,
        bot: LoadedBot,
        human_color: bool,
        temperature: float,
        human_rating: int,
        bot_rating: int,
        clock: Callable[[], float] = time.monotonic,
        board: Optional[chess.Board] = None,
    ) -> None:
        self.session_id = session_id
        self.username = username
        self.bot = bot
        self.human_color = human_color
        self.bot_color = not human_color
        self.temperature = temperature
        self.human_rating = human_rating
        self.bot_rating = bot_rating
        self.board = board if board is not None else chess.Board()
        self.moves: list[_MoveRecord] = []
        self.result: Optional[str] = None
        self.winner: Optional[str] = None
        self.termination: Optional[str] = None
        self.lock = threading.RLock()
        self._clock = clock
        self.last_active = clock()
        self._update_game_over()

    # ------------------------------------------------------------- properties
    @property
    def finished(self) -> bool:
        return self.result is not None

    def touch(self) -> None:
        self.last_active = self._clock()

    # --------------------------------------------------------------- internals
    def _update_game_over(self) -> None:
        # claim_draw=True also ends the game on threefold repetition and the
        # fifty-move rule, so a game can never loop forever.
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None:
            self.result = self.winner = self.termination = None
            return
        self.result = outcome.result()
        self.winner = None if outcome.winner is None else _color_name(outcome.winner)
        self.termination = outcome.termination.name.lower()

    def _push(self, move: chess.Move, by: str) -> _MoveRecord:
        san = self.board.san(move)           # SAN must be computed BEFORE the push
        self.board.push(move)
        record = _MoveRecord(ply=len(self.moves) + 1, san=san, uci=move.uci(), by=by)
        self.moves.append(record)
        self._update_game_over()
        return record

    def _pop_last(self) -> None:
        self.board.pop()
        self.moves.pop()
        self._update_game_over()

    def _parse_human_move(self, uci: Any) -> chess.Move:
        if not isinstance(uci, str):
            raise InvalidMoveFormatError("Move must be a string like 'e2e4'.")
        text = uci.strip().lower()
        if not _UCI_RE.match(text):
            raise InvalidMoveFormatError(
                f"'{uci}' is not a valid move. Use UCI like 'e2e4' or 'e7e8q'."
            )
        move = chess.Move.from_uci(text)
        if move in self.board.legal_moves:
            return move
        if move.promotion is None:
            queen_version = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
            if queen_version in self.board.legal_moves:
                raise IllegalMoveError(
                    f"Promotion piece required: use '{text}q', '{text}r', "
                    f"'{text}b' or '{text}n'."
                )
        raise IllegalMoveError(f"Illegal move '{text}' in this position.")

    def _bot_reply(self, mover: Mover) -> _MoveRecord:
        meta = {
            "player_rating": self.bot_rating,       # the bot is "the player"
            "opponent_rating": self.human_rating,
        }
        try:
            move = mover(
                self.bot, self.board.copy(stack=False), meta, self.temperature
            )
        except Exception as exc:  # noqa: BLE001
            raise BotMoveError(f"The bot failed to choose a move: {exc}") from exc
        # Safety net: never trust the bot (or a corrupt opening book) blindly.
        if not isinstance(move, chess.Move) or move not in self.board.legal_moves:
            raise BotMoveError("The bot produced an illegal move.")
        return self._push(move, by="bot")

    # --------------------------------------------------------------- public API
    def play_bot_move(self, mover: Mover) -> dict[str, Any]:
        """Let the bot move now (used when the bot has the first move)."""
        with self.lock:
            self.touch()
            if self.finished:
                raise GameOverError("The game is already finished.")
            if self.board.turn != self.bot_color:
                raise NotYourTurnError("It is not the bot's turn.")
            self._bot_reply(mover)
            return self.snapshot()

    def play_turn(self, uci: Any, mover: Mover) -> dict[str, Any]:
        """
        Atomic turn: validate + apply the human's move, then (if the game is
        not over) generate + apply the bot's reply.

        Raises GameOverError, NotYourTurnError, InvalidMoveFormatError,
        IllegalMoveError, BotMoveError. On any error the board is unchanged.
        """
        with self.lock:
            self.touch()
            if self.finished:
                raise GameOverError("The game is already finished.")
            if self.board.turn != self.human_color:
                raise NotYourTurnError("It is not your turn.")

            move = self._parse_human_move(uci)
            self._push(move, by="human")

            if not self.finished:
                try:
                    self._bot_reply(mover)
                except BotMoveError:
                    self._pop_last()          # roll back so the client can retry
                    raise
            return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        """JSON-friendly view of the game (what the API returns)."""
        with self.lock:
            human_to_move = (not self.finished) and self.board.turn == self.human_color
            last_bot = next((m for m in reversed(self.moves) if m.by == "bot"), None)
            return {
                "session_id": self.session_id,
                "username": self.username,
                "human_color": _color_name(self.human_color),
                "bot_color": _color_name(self.bot_color),
                "temperature": self.temperature,
                "fen": self.board.fen(),
                "turn": _color_name(self.board.turn),
                "human_to_move": human_to_move,
                "status": "finished" if self.finished else "active",
                "result": self.result,
                "winner": self.winner,
                "termination": self.termination,
                "in_check": self.board.is_check(),
                "legal_moves": (
                    sorted(m.uci() for m in self.board.legal_moves) if human_to_move else []
                ),
                "moves": [m.as_dict() for m in self.moves],
                "last_bot_move": last_bot.as_dict() if last_bot else None,
            }


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------

class SessionManager:
    """Thread-safe in-memory store of GameSession objects."""

    def __init__(
        self,
        mover: Optional[Mover] = None,
        clock: Callable[[], float] = time.monotonic,
        ttl_seconds: Optional[float] = None,
        max_sessions: Optional[int] = None,
    ) -> None:
        self.mover: Mover = mover or predict_with_model
        self._clock = clock
        self._ttl = ttl_seconds
        self._max = max_sessions
        self._sessions: dict[str, GameSession] = {}
        self._lock = threading.Lock()

    @property
    def ttl_seconds(self) -> float:
        return self._ttl if self._ttl is not None else config.SESSION_IDLE_TTL_SECONDS

    @property
    def max_sessions(self) -> int:
        return self._max if self._max is not None else config.SESSION_MAX_COUNT

    def _is_expired(self, session: GameSession) -> bool:
        return (self._clock() - session.last_active) > self.ttl_seconds

    def _purge_expired_locked(self) -> int:
        stale = [sid for sid, s in self._sessions.items() if self._is_expired(s)]
        for sid in stale:
            del self._sessions[sid]
        return len(stale)

    def purge_expired(self) -> int:
        with self._lock:
            return self._purge_expired_locked()

    def create(
        self,
        username: str,
        bot: LoadedBot,
        human_color: bool,
        temperature: float,
        human_rating: int = config.DEFAULT_RATING,
        bot_rating: int = config.DEFAULT_RATING,
        board: Optional[chess.Board] = None,
    ) -> GameSession:
        """
        Start a game. If the bot has the first move (human plays Black), the
        bot moves immediately. If that fails, nothing is registered.

        Raises SessionLimitError, BotMoveError.
        """
        with self._lock:
            self._purge_expired_locked()
            if len(self._sessions) >= self.max_sessions:
                finished = [s for s in self._sessions.values() if s.finished]
                if not finished:
                    raise SessionLimitError("Too many active games right now. Try again later.")
                oldest = min(finished, key=lambda s: s.last_active)
                del self._sessions[oldest.session_id]

        session = GameSession(
            session_id=uuid.uuid4().hex,
            username=username,
            bot=bot,
            human_color=human_color,
            temperature=temperature,
            human_rating=human_rating,
            bot_rating=bot_rating,
            clock=self._clock,
            board=board,
        )
        if not session.finished and session.board.turn == session.bot_color:
            session.play_bot_move(self.mover)    # may raise BotMoveError -> not registered

        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> GameSession:
        """Return the live session, refreshing its idle timer."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or self._is_expired(session):
                self._sessions.pop(session_id, None)
                raise SessionNotFoundError("Game not found or expired.")
            session.touch()
            return session

    def play_turn(self, session_id: str, uci: Any) -> dict[str, Any]:
        """Convenience: look up the session and play a full turn."""
        return self.get(session_id).play_turn(uci, self.mover)

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)


# Process-wide manager used by the routers. Tests replace this attribute.
default_manager = SessionManager()
