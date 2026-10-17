"""
Chess Bot Clone Factory — FastAPI application entry point.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from .db.migrate import run_migrations
from .routers import auth, bots, dashboard, games, ingest, train, ws


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Run SQLite schema migrations on startup
    run_migrations()
    yield


app = FastAPI(
    title="Chess Bot Clone Factory",
    description="Build a chess bot that imitates a Lichess player.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(ingest.router, prefix="/api")
app.include_router(train.router, prefix="/api")
app.include_router(auth.router)
app.include_router(bots.router)
app.include_router(dashboard.router)
app.include_router(games.router)
app.include_router(ws.router)

