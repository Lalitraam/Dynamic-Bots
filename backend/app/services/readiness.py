"""
Readiness check for the serving API (Milestone 4, Phase 1).

Answers one question for the frontend: "can I play against this player's bot?"

The check is purely read-only: it looks at files under the player's folder
and at the in-memory ingest job (passed in by the router). It never creates
folders and does not load the model (loadability is verified by the model
cache in Phase 2).

Decision order
--------------
1. model.pt exists and is non-empty                 -> "ready"
2. an ingest job is active                          -> "ingesting"
3. info.json exists                                 -> "rejected" or
                                                       "ingested_not_trained"
4. the last ingest job failed                       -> "ingest_failed"
5. otherwise                                        -> "never_ingested"

Notes
-----
* A usable model wins over everything else, so a player who is being
  re-ingested can still be played against.
* Players with too few games are rejected during ingestion before anything is
  written to disk, so that case shows up as "ingest_failed" (with the reason in
  `detail`) while the server is running, and as "never_ingested" after a
  restart. "rejected" is reported when info.json says so.
* There is no "training in progress" state yet: this repo has no training
  endpoint or training job tracker to read it from.
"""
import json
from typing import Any, Optional

from . import player_paths

_NON_ACTIVE_JOB_STATUSES = {None, "failed", "completed"}


def _read_json(path) -> Optional[dict]:
    """Return parsed JSON dict, or None if missing / unreadable / not a dict."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _model_file_usable(username: str) -> bool:
    path = player_paths.model_path(username)
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def get_readiness(
    username: str,
    ingest_job: Optional[dict] = None,
) -> dict[str, Any]:
    """
    Report whether *username*'s bot can play.

    Raises player_paths.InvalidUsernameError for malformed usernames.
    *ingest_job* is the in-memory job dict from routers/ingest.py (or None).
    """
    key = player_paths.normalize_username(username)
    job_status = (ingest_job or {}).get("status")

    # Tier: prefer training_info.json (written with the model), then info.json.
    info = _read_json(player_paths.info_path(key))
    training_info = _read_json(player_paths.training_info_path(key))
    tier = (training_info or {}).get("model_tier") or (info or {}).get("model_tier")

    def result(state: str, detail: str) -> dict[str, Any]:
        return {
            "username": key,
            "state": state,
            "ready": state == "ready",
            "model_tier": tier,
            "ingest_status": job_status,
            "detail": detail,
        }

    # 1. A usable model beats everything.
    if _model_file_usable(key):
        return result("ready", "Trained model found.")

    # 2. Ingestion in progress.
    if job_status not in _NON_ACTIVE_JOB_STATUSES:
        return result("ingesting", f"Ingestion is {job_status}.")

    # 3. Dataset on disk but no model.
    if player_paths.info_path(key).is_file():
        if tier == "rejected":
            return result(
                "rejected",
                "Not enough games to train a model for this player.",
            )
        return result(
            "ingested_not_trained",
            "Games are ingested, but no trained model exists yet.",
        )

    # 4. Last ingest attempt failed.
    if job_status == "failed":
        error = (ingest_job or {}).get("error") or "Ingestion failed."
        return result("ingest_failed", str(error))

    # 5. Nothing known.
    return result("never_ingested", "This player has not been ingested yet.")
