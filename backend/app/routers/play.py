"""
Play router (Milestone 4).

Phase 1-2:
  GET  /api/play/{username}/ready      -> is this player's bot ready to play?
                                          (also verifies the model loads and
                                          warms the model cache)
Phase 3:
  POST /api/play/sessions              -> start a human-vs-bot game
  GET  /api/play/sessions/{session_id} -> current game state
Phase 4:
  POST /api/play/sessions/{session_id}/moves
                                       -> submit the human's move; the server
                                          validates it, applies it, and replies
                                          with the bot's move in one response

Later phases add session creation and the move endpoint to this router.

Status codes
------------
200  state returned / move played
201  game created
400  malformed username, or a malformed / illegal move (board unchanged)
404  no trained model for that player / unknown or expired game
409  game already finished
422  invalid request body (e.g. temperature out of range)
500  the model or the bot failed (board unchanged, the same move can be retried)
503  too many live games
"""
import random

import chess
from fastapi import APIRouter, HTTPException

from ..schemas.play import (
    CreateSessionRequest,
    MoveRequest,
    ReadinessResponse,
    SessionResponse,
)
from ..services import game_sessions, model_cache, readiness
from ..services.game_sessions import (
    BotMoveError,
    GameOverError,
    InvalidMoveError,
    NotYourTurnError,
    SessionLimitError,
    SessionNotFoundError,
)
from ..services.model_cache import ModelLoadError, ModelNotFoundError
from ..services.player_paths import InvalidUsernameError, normalize_username
from .ingest import _jobs

router = APIRouter()


@router.get("/play/{username}/ready", response_model=ReadinessResponse)
def get_ready(username: str):
    """
    Report whether *username* has a trained model ready to play.

    Plain `def` on purpose: FastAPI runs it in a worker thread, so loading a
    model from disk never blocks the event loop.
    """
    try:
        key = normalize_username(username)
    except InvalidUsernameError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return readiness.get_readiness(
        key,
        ingest_job=_jobs.get(key),
        cache=model_cache.default_cache,   # looked up at call time (tests swap it)
    )


@router.post("/play/sessions", response_model=SessionResponse, status_code=201)
def create_session(req: CreateSessionRequest):
    """
    Start a human-vs-bot game against *username*'s bot.

    If the human plays Black, the bot (White) opens immediately and its move is
    already in the returned state.
    """
    try:
        key = normalize_username(req.username)
    except InvalidUsernameError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        bot = model_cache.default_cache.get(key)
    except ModelNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"No trained model for '{key}'. Check /api/play/{key}/ready.",
        )
    except ModelLoadError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if req.human_color == "random":
        human_color = random.choice([chess.WHITE, chess.BLACK])
    else:
        human_color = chess.WHITE if req.human_color == "white" else chess.BLACK

    try:
        session = game_sessions.default_manager.create(
            username=key,
            bot=bot,
            human_color=human_color,
            temperature=req.temperature,
            human_rating=req.human_rating,
            bot_rating=req.bot_rating,
        )
    except SessionLimitError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except BotMoveError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return session.snapshot()


@router.get("/play/sessions/{session_id}", response_model=SessionResponse)
def get_session(session_id: str):
    """Return the current state of a game (also keeps it from expiring)."""
    try:
        session = game_sessions.default_manager.get(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return session.snapshot()


@router.post("/play/sessions/{session_id}/moves", response_model=SessionResponse)
def make_move(session_id: str, req: MoveRequest):
    """
    Play one full turn: validate and apply the human's move, then the bot's reply.

    Plain `def` on purpose: the bot's model inference runs in a worker thread,
    so it never blocks the event loop. The whole turn is atomic per game.
    """
    try:
        return game_sessions.default_manager.play_turn(
            session_id, req.move, temperature=req.temperature
        )
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except InvalidMoveError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except (GameOverError, NotYourTurnError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except BotMoveError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
