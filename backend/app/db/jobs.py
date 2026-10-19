"""Job queue database helpers (Phase G)."""

from __future__ import annotations
import json
from pathlib import Path
import sqlite3
from typing import Any
from app.db.migrate import get_db


def create_job(
    kind: str,
    username: str,
    requested_by: int | None = None,
    payload: dict[str, Any] | None = None,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """Create a new asynchronous background job."""
    close_after = False
    if conn is None:
        conn = get_db()
        close_after = True
    payload_str = json.dumps(payload) if payload else None
    try:
        with conn:
            cur = conn.execute(
                """
                INSERT INTO jobs (kind, username, requested_by, status, step, progress, payload_json)
                VALUES (?, ?, ?, 'queued', 'queued', 0.0, ?);
                """,
                (kind, username, requested_by, payload_str),
            )
            job_id = cur.lastrowid
        row = conn.execute("SELECT * FROM jobs WHERE id = ?;", (job_id,)).fetchone()
        return dict(row)
    finally:
        if close_after:
            conn.close()


def get_job(job_id: int, conn: sqlite3.Connection | None = None) -> dict[str, Any] | None:
    """Fetch job by primary key id."""
    close_after = False
    if conn is None:
        conn = get_db()
        close_after = True
    try:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?;", (job_id,)).fetchone()
        return dict(row) if row else None
    finally:
        if close_after:
            conn.close()


def list_jobs_for_user(user_id: int, conn: sqlite3.Connection | None = None) -> list[dict[str, Any]]:
    """List all jobs requested by a specific user."""
    close_after = False
    if conn is None:
        conn = get_db()
        close_after = True
    try:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE requested_by = ? ORDER BY created_at DESC;",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        if close_after:
            conn.close()


def claim_next_job(conn: sqlite3.Connection | None = None) -> dict[str, Any] | None:
    """Atomically pick oldest queued job, set to 'running', return it."""
    close_after = False
    if conn is None:
        conn = get_db()
        close_after = True
    try:
        cur = conn.execute(
            """
            UPDATE jobs SET status = 'running', started_at = CURRENT_TIMESTAMP
            WHERE id = (
                SELECT id FROM jobs WHERE status = 'queued' ORDER BY created_at ASC LIMIT 1
            ) AND status = 'queued'
            RETURNING *;
            """
        )
        row = cur.fetchone()
        conn.commit()
        return dict(row) if row else None
    finally:
        if close_after:
            conn.close()


def update_job(
    conn: sqlite3.Connection | None = None,
    job_id: int = 0,
    status: str | None = None,
    step: str | None = None,
    progress: float | None = None,
    error: str | None = None,
) -> None:
    """Update specific fields of a job."""
    close_after = False
    if conn is None:
        conn = get_db()
        close_after = True
    try:
        updates = []
        params = []
        if status is not None:
            updates.append("status = ?")
            params.append(status)
            if status in ("completed", "failed"):
                updates.append("finished_at = CURRENT_TIMESTAMP")
        if step is not None:
            updates.append("step = ?")
            params.append(step)
        if progress is not None:
            updates.append("progress = ?")
            params.append(progress)
        if error is not None:
            updates.append("error = ?")
            params.append(error)

        if updates:
            params.append(job_id)
            query = f"UPDATE jobs SET {', '.join(updates)} WHERE id = ?;"
            conn.execute(query, tuple(params))
            conn.commit()
    finally:
        if close_after:
            conn.close()


def update_player(
    conn: sqlite3.Connection | None = None,
    username: str = "",
    status: str = "ready",
    error: str | None = None,
) -> None:
    """Update player status (queued, building, ready, failed)."""
    close_after = False
    if conn is None:
        conn = get_db()
        close_after = True
    try:
        conn.execute(
            "UPDATE players SET status = ?, error = ?, updated_at = CURRENT_TIMESTAMP WHERE username = ?;",
            (status, error, username),
        )
        conn.commit()
    finally:
        if close_after:
            conn.close()
