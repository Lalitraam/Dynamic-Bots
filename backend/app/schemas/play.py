"""
Request / response models for the serving API (Milestone 4).
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .. import config

# Stable strings the frontend can switch on.
ReadinessState = Literal[
    "never_ingested",        # nothing on disk, no ingest job known
    "ingesting",             # an ingest job is currently running
    "ingest_failed",         # last ingest job failed (see `detail`)
    "rejected",              # ingested, but too few games to train a model
    "ingested_not_trained",  # dataset exists, no usable model.pt yet
    "model_unreadable",      # model.pt exists but could not be loaded
    "ready",                 # model loads fine: the bot can play
]


class ReadinessResponse(BaseModel):
    username: str
    state: ReadinessState
    ready: bool
    model_tier: Optional[str] = None      # small / standard / full / rejected
    ingest_status: Optional[str] = None   # current in-memory ingest job status, if any
    detail: Optional[str] = None          # human-readable explanation


# ---------------------------------------------------------------------------
# Game sessions (Phase 3)
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    username: str                                   # whose bot to play against
    human_color: Literal["white", "black", "random"] = "white"
    temperature: float = Field(
        default=config.INFERENCE_TEMPERATURE_DEFAULT,
        ge=config.TEMPERATURE_MIN,
        le=config.TEMPERATURE_MAX,
    )
    human_rating: int = Field(
        default=config.DEFAULT_RATING, ge=config.RATING_MIN, le=config.RATING_MAX
    )
    bot_rating: int = Field(
        default=config.DEFAULT_RATING, ge=config.RATING_MIN, le=config.RATING_MAX
    )


class MoveRecordResponse(BaseModel):
    ply: int                       # 1-based
    san: str
    uci: str
    by: Literal["human", "bot"]


class SessionResponse(BaseModel):
    session_id: str
    username: str
    human_color: Literal["white", "black"]
    bot_color: Literal["white", "black"]
    temperature: float
    fen: str
    turn: Literal["white", "black"]
    human_to_move: bool
    status: Literal["active", "finished"]
    result: Optional[str] = None            # "1-0", "0-1", "1/2-1/2"
    winner: Optional[Literal["white", "black"]] = None
    termination: Optional[str] = None       # checkmate, stalemate, threefold_repetition, ...
    in_check: bool
    legal_moves: list[str]                  # UCI; only filled when it is the human's turn
    moves: list[MoveRecordResponse]
    last_bot_move: Optional[MoveRecordResponse] = None
