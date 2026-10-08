"""
Request / response models for the serving API (Milestone 4).
"""
from typing import Literal, Optional

from pydantic import BaseModel

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
