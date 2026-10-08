"""
Play router (Milestone 4).

Phase 1-2:
  GET /api/play/{username}/ready   -> is this player's bot ready to play?
                                      (also verifies the model loads and
                                      warms the model cache)

Later phases add session creation and the move endpoint to this router.

Status codes
------------
200  always, for a well-formed username (the `state` field says what is going on)
400  malformed username
"""
from fastapi import APIRouter, HTTPException

from ..schemas.play import ReadinessResponse
from ..services import model_cache, readiness
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
