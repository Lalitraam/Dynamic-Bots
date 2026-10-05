"""
Chess Bot Clone Factory — FastAPI application entry point.
"""

from fastapi import FastAPI
from .routers import ingest

app = FastAPI(
    title="Chess Bot Clone Factory",
    description="Build a chess bot that imitates a Lichess player.",
    version="0.1.0",
)

app.include_router(ingest.router, prefix="/api")
