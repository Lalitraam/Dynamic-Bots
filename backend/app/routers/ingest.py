"""
Ingest router: POST /api/ingest/{username} and GET /api/status/{username}.

Status flow
-----------
POST /api/ingest/{username}
  - If a job for this username is already "queued" or "running", return 409.
  - Otherwise set status to "queued" synchronously, then enqueue the
    background task.

GET /api/status/{username}
  - Returns the current job status for the username.

Completed payload includes:
  total_games, sampled_games, model_tier, quota_used,
  bullet_count, blitz_count, rapid_count, drop_stats
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException
from .. import config
from ..services.downloader import stream_games
from ..services.parser import parse_stream_into_buckets
from ..services.sampler import sample_games
from ..services.storage import save_games

router = APIRouter()

# ---------------------------------------------------------------------------
# In-memory job state (sufficient for Milestone 1 single-process usage).
# ---------------------------------------------------------------------------

# Maps username (lower) -> status dict
_jobs: dict[str, dict] = {}

_ACTIVE_STATUSES = {
    "queued",
    "connecting",
    "downloading",
    "parsing_and_filtering",
    "sampling",
    "saving",
    "extracting",
    "building_dataset",
    "running",
}


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------

async def _run_ingest(username: str) -> None:
    """Full ingest pipeline: download → parse → sample → save."""
    key = username.lower()
    _jobs[key]["status"] = "connecting"
    _jobs[key]["progress"] = 0.0

    def on_progress(count: int):
        pct = min(100.0, round((count / config.DEFAULT_MAX_GAMES) * 100, 1))
        _jobs[key]["status"] = "downloading"
        _jobs[key]["progress"] = pct

    try:
        # 1. Stream and parse
        stream = stream_games(username)
        buckets, total_valid, drop_stats = await parse_stream_into_buckets(
            stream, username, on_progress=on_progress
        )

        _jobs[key]["status"] = "parsing_and_filtering"

        if total_valid < config.MIN_GAMES_REQUIRED:
            _jobs[key].update({
                "status": "failed",
                "error": (
                    f"Not enough valid games: {total_valid} "
                    f"(minimum {config.MIN_GAMES_REQUIRED})"
                ),
                "drop_stats": drop_stats,
            })
            return

        # 2. Sample
        _jobs[key]["status"] = "sampling"
        result = sample_games(buckets)
        games = result["games"]

        # 3. Persist
        _jobs[key]["status"] = "saving"
        save_games(username, games)

        # 4. Build dataset (extract, split, parquet, book, report)
        _jobs[key]["status"] = "building_dataset"
        from ..services.dataset_builder import build_dataset
        import asyncio
        info = await asyncio.to_thread(build_dataset, username)

        # 5. Update status
        tier = config.model_tier(len(games))
        _jobs[key].update({
            "status": "completed",
            "progress": 100.0,
            "total_games": total_valid,
            "sampled_games": len(games),
            "model_tier": tier,
            "quota_used": result["quota_used"],
            "bullet_count": result["bullet_count"],
            "blitz_count": result["blitz_count"],
            "rapid_count": result["rapid_count"],
            "drop_stats": drop_stats,
            "dataset_info": info,
        })

    except Exception as exc:  # noqa: BLE001
        _jobs[key].update({
            "status": "failed",
            "error": str(exc),
        })


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/ingest/{username}", status_code=202)
async def start_ingest(username: str, background_tasks: BackgroundTasks):
    """
    Start a game-ingestion job for *username*.

    Returns 409 if a job is already active.
    Returns 202 Accepted with the initial status payload.
    """
    key = username.lower()
    existing = _jobs.get(key, {})
    if existing.get("status") in _ACTIVE_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=f"Job for '{key}' is already {existing['status']}.",
        )

    # Set queued synchronously BEFORE adding the background task.
    _jobs[key] = {"status": "queued", "username": key, "progress": 0.0}
    background_tasks.add_task(_run_ingest, username)

    return {"username": key, "status": "queued"}


@router.get("/status/{username}")
async def get_status(username: str):
    """Return the current job status for *username*."""
    key = username.lower()
    job = _jobs.get(key)
    if job is not None:
        return job
        
    # Fallback to info.json on disk
    from ..services.storage import player_dir
    import json
    info_path = player_dir(key) / "info.json"
    if info_path.exists():
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            return {
                "username": key,
                "status": "completed",
                "progress": 100.0,
                "dataset_info": info
            }
        except json.JSONDecodeError:
            pass

    raise HTTPException(status_code=404, detail=f"No job found for '{key}'.")
