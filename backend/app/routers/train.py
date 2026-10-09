"""
Train router: POST /api/train/{username} and GET /api/train_status/{username}.
"""
from __future__ import annotations

import asyncio
import json
from fastapi import APIRouter, BackgroundTasks, HTTPException

from ..services.storage import player_dir, validate_username
from ..services.training_runner import train_player_model
from .ingest import _ACTIVE_STATUSES as _INGEST_ACTIVE_STATUSES, _jobs as _ingest_jobs

router = APIRouter()

# In-memory training jobs: username (lowercase) -> dict
_train_jobs: dict[str, dict] = {}

_ACTIVE_STATUSES = {"queued", "running"}


async def _run_train(username: str) -> None:
    key = username.lower()
    _train_jobs[key]["status"] = "running"
    try:
        result = await asyncio.to_thread(train_player_model, username)
        _train_jobs[key].update({
            "status": "completed",
            "epochs_run": result.get("epochs_run"),
            "best_val_loss": result.get("best_val_loss"),
            "device": result.get("device"),
            "model_tier": result.get("model_tier"),
            "created_at": result.get("created_at"),
        })
    except Exception as exc:
        _train_jobs[key].update({
            "status": "failed",
            "error": str(exc),
        })


@router.post("/train/{username}", status_code=202)
async def start_train(username: str, background_tasks: BackgroundTasks):
    try:
        key = validate_username(username)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    pdir = player_dir(key, create=False)
    info_path = pdir / "info.json"
    samples_path = pdir / "samples.parquet"
    if not info_path.exists() or not samples_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Player '{key}' has never been ingested (missing info.json or samples.parquet).",
        )

    try:
        with open(info_path, "r", encoding="utf-8") as f:
            player_info = json.load(f)
    except Exception:
        player_info = {}

    if player_info.get("model_tier") == "rejected":
        raise HTTPException(
            status_code=400,
            detail=f"Player '{key}' has model_tier 'rejected'. Training is not allowed.",
        )

    # 409 if an ingest job for that user is in ingest._ACTIVE_STATUSES
    ingest_job = _ingest_jobs.get(key, {})
    if ingest_job.get("status") in _INGEST_ACTIVE_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=f"Ingest job for '{key}' is active ({ingest_job['status']}).",
        )

    # 409 if a train job is active
    train_job = _train_jobs.get(key, {})
    if train_job.get("status") in _ACTIVE_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=f"Train job for '{key}' is already {train_job['status']}.",
        )

    _train_jobs[key] = {
        "username": key,
        "status": "queued",
    }
    background_tasks.add_task(_run_train, key)
    return {"username": key, "status": "queued"}


@router.get("/train_status/{username}")
async def get_train_status(username: str):
    try:
        key = validate_username(username)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # In-memory job first
    if key in _train_jobs:
        return _train_jobs[key]

    pdir = player_dir(key, create=False)
    t_info_path = pdir / "training_info.json"
    info_path = pdir / "info.json"

    if t_info_path.exists():
        try:
            with open(t_info_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "username": key,
                "status": "completed",
                "epochs_run": data.get("epochs_run"),
                "best_val_loss": data.get("best_val_loss"),
                "device": data.get("device"),
                "model_tier": data.get("model_tier"),
                "created_at": data.get("created_at"),
                "source": "disk",
            }
        except Exception:
            pass

    if info_path.exists():
        return {"status": "not_trained"}

    raise HTTPException(
        status_code=404,
        detail=f"No training data found for '{key}'.",
    )
