"""Games API router using GameManager (Phase F/G)."""

from __future__ import annotations
import secrets
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.auth import get_current_user
from app.game.manager import game_manager

router = APIRouter(prefix="/api/games", tags=["games"])


class CreateGameRequest(BaseModel):
    username: str
    human_color: str = "white"
    base_s: int = 180
    inc_s: int = 2


class CreateGameResponse(BaseModel):
    game_id: str
    secret: str
    human_color: str
    fen: str
    status: str


class MoveRequest(BaseModel):
    move: str  # UCI string (e.g. 'e2e4')
    secret: str


class MoveResponse(BaseModel):
    game_id: str
    fen: str
    moves: list[dict[str, Any]]
    status: str
    result: str | None = None
    bot_think_ms: float | None = None


@router.post("", response_model=CreateGameResponse, status_code=status.HTTP_201_CREATED)
async def start_game(
    req: CreateGameRequest,
    user: dict[str, Any] = Depends(get_current_user),
) -> CreateGameResponse:
    """Initialize a new chess game session against a cloned bot."""
    game_id = secrets.token_hex(12)
    raw_secret = secrets.token_hex(32)

    color = req.human_color.lower().strip()
    if color not in ("white", "black"):
        color = "white"

    game_state = await game_manager.create_game(
        game_id=game_id,
        player_username=req.username.strip(),
        human_color=color,
        base_s=req.base_s,
        inc_s=req.inc_s,
        secret=raw_secret,
        owner_id=user["id"],
    )

    return CreateGameResponse(
        game_id=game_id,
        secret=raw_secret,
        human_color=color,
        fen=game_state.board.fen(),
        status=game_state.status,
    )


@router.post("/{game_id}/move", response_model=MoveResponse)
async def play_move(
    game_id: str,
    req: MoveRequest,
    user: dict[str, Any] = Depends(get_current_user),
) -> MoveResponse:
    """Submit a human move and receive the bot's counter-move (REST compatibility)."""
    try:
        game_state = await game_manager.apply_move(
            game_id=game_id,
            uci=req.move,
            secret=req.secret,
        )
    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid game secret",
        )
    except ValueError as e:
        err_msg = str(e)
        if f"Game '{game_id}' not found" in err_msg:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=err_msg,
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=err_msg,
        )

    last_bot_think = None
    if game_state.moves and game_state.moves[-1].get("by") == "bot":
        last_bot_think = game_state.moves[-1].get("elapsed_ms")

    return MoveResponse(
        game_id=game_id,
        fen=game_state.board.fen(),
        moves=game_state.moves,
        status=game_state.status,
        result=game_state.result,
        bot_think_ms=last_bot_think,
    )
