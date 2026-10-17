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
