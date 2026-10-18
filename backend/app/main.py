"""
Chess Bot Clone Factory — FastAPI application entry point.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
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

app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("SESSION_SECRET", "dev-insecure-secret-change-me"),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router, prefix="/api")
app.include_router(train.router, prefix="/api")
app.include_router(auth.router)
app.include_router(bots.router)
app.include_router(dashboard.router)
app.include_router(games.router)
app.include_router(ws.router)

