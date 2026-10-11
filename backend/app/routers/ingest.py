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

from datetime import datetime, timezone
from fastapi import APIRouter, BackgroundTasks, HTTPException
from .. import config
from ..services.downloader import stream_games
from ..services.parser import parse_stream_into_buckets
from ..services.sampler import sample_games
from ..services.storage import save_games, validate_username
from ..services.ingest_stats import IngestStats

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

    stream = None
    drop_stats = None
    stats = IngestStats()
    try:
        # 1. Stream and parse
        now = datetime.now(timezone.utc)
        stream = stream_games(username, overlap_ms=config.OVERLAP_MS, stats=stats, now=now)
        buckets, total_valid, drop_stats = await parse_stream_into_buckets(
            stream, username, on_progress=on_progress, stats=stats, now=now
        )

        _jobs[key]["status"] = "parsing_and_filtering"

        retries_used = getattr(stream, "retries_used", 0)
        if stats.retries_used:
            retries_used = stats.retries_used

        if total_valid < config.MIN_GAMES_REQUIRED:
            _jobs[key].update({
                "status": "failed",
                "error": (
                    f"Not enough valid games: {total_valid} "
                    f"(minimum {config.MIN_GAMES_REQUIRED})"
                ),
                "drop_stats": drop_stats,
                "retries_used": retries_used,
                "skipped_duplicates": stats.skipped_duplicates,
                "request_log": stats.request_log,
            })
            return

        # 2. Sample
        _jobs[key]["status"] = "sampling"
        result = sample_games(buckets)
        games = result["games"]

        # Write ingest_info.json into player_dir after sampling
        from ..services.storage import player_dir
        import json
        pdir = player_dir(username)
        pdir.mkdir(parents=True, exist_ok=True)
        stats.retries_used = retries_used
        stats.requests = getattr(stream, "requests", stats.requests)
        stats.quota_used = result["quota_used"]
        stats.bucket_counts = {
            "bullet": result["bullet_count"],
            "blitz": result["blitz_count"],
            "rapid": result["rapid_count"],
        }
        with open(pdir / "ingest_info.json", "w", encoding="utf-8") as f:
            json.dump(stats.to_dict(), f, indent=2)

        # 3. Persist
        _jobs[key]["status"] = "saving"
        save_games(username, games)

        # 4. Build dataset (extract, split, parquet, book, report)
        _jobs[key]["status"] = "building_dataset"
        from ..services.dataset_builder import build_dataset
        import asyncio
        info = await asyncio.to_thread(build_dataset, username, retries_used=retries_used)

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
            "retries_used": retries_used,
            "skipped_duplicates": stats.skipped_duplicates,
            "request_log": stats.request_log,
            "dataset_info": info,
        })

    except Exception as exc:  # noqa: BLE001
        retries_used = getattr(stream, "retries_used", 0) if stream is not None else 0
        if stats and stats.retries_used:
            retries_used = stats.retries_used
        collected_drops = stats.drop_stats if (stats and stats.drop_stats) else drop_stats
        _jobs[key].update({
            "status": "failed",
            "error": str(exc),
            "retries_used": retries_used,
            "skipped_duplicates": stats.skipped_duplicates if stats else 0,
            "request_log": stats.request_log if stats else [],
            "drop_stats": collected_drops,
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
    try:
        key = validate_username(username)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    existing = _jobs.get(key, {})
    if existing.get("status") in _ACTIVE_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=f"Job for '{key}' is already {existing['status']}.",
        )

    from .train import _train_jobs, _ACTIVE_STATUSES as _TRAIN_ACTIVE
    if _train_jobs.get(key, {}).get("status") in _TRAIN_ACTIVE:
        raise HTTPException(
            status_code=409,
            detail=f"Training job for '{key}' is active.",
        )

    # Set queued synchronously BEFORE adding the background task.
    _jobs[key] = {"status": "queued", "username": key, "progress": 0.0}
    background_tasks.add_task(_run_ingest, username)

    return {"username": key, "status": "queued"}


@router.get("/status/{username}")
async def get_status(username: str):
    """Return the current job status for *username*."""
    try:
        key = validate_username(username)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    job = _jobs.get(key)
    if job is not None:
        return job

    # Fallback to ingest_info.json and info.json on disk
    from ..services.storage import player_dir
    import json
    pdir = player_dir(key, create=False)
    ingest_info_path = pdir / "ingest_info.json"
    info_path = pdir / "info.json"

    if ingest_info_path.exists() or info_path.exists():
        status = "completed" if info_path.exists() else "incomplete"
        resp = {
            "username": key,
            "status": status,
            "progress": 100.0,
            "retries_used": 0,
            "requests": 0,
            "skipped_duplicates": 0,
            "request_log": [],
            "raw_lines": 0,
            "drop_stats": {},
            "dataset_info": {},
        }
        if ingest_info_path.exists():
            try:
                with open(ingest_info_path, "r", encoding="utf-8") as f:
                    ingest_data = json.load(f)
                resp.update({
                    "retries_used": ingest_data.get("retries_used", 0),
                    "requests": ingest_data.get("requests", 0),
                    "skipped_duplicates": ingest_data.get("skipped_duplicates", 0),
                    "request_log": ingest_data.get("request_log", []),
                    "raw_lines": ingest_data.get("raw_lines", 0),
                    "drop_stats": ingest_data.get("drop_stats", {}),
                })
            except json.JSONDecodeError:
                pass

        if info_path.exists():
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    info = json.load(f)
                resp["dataset_info"] = info
                if not ingest_info_path.exists():
                    resp["retries_used"] = info.get("retries_used", 0)
            except json.JSONDecodeError:
                pass

        return resp

    raise HTTPException(status_code=404, detail=f"No job found for '{key}'.")
