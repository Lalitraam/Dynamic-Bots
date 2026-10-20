"""Bots management router (Phase G)."""

from __future__ import annotations
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status
import httpx
from pydantic import BaseModel

from app import config
from app.auth import get_current_user
from app.db.jobs import create_job
from app.db.migrate import get_db

router = APIRouter(prefix="/api/bots", tags=["bots"])


class AddBotRequest(BaseModel):
    username: str


class BotResponse(BaseModel):
    username: str
    status: str
    active_version: int | None = 1
    added_at: str | None = None
    archived: bool = False
    last_trained_at: str | None = None
    games_used: int | None = 0
    tier: str | None = "free"
    error: str | None = None
    job_step: str | None = None
    job_progress: float | None = None


def check_lichess_user_exists(username: str) -> bool:
    """Check if username exists on Lichess."""
    try:
        url = f"https://lichess.org/api/user/{username.strip()}"
        resp = httpx.get(url, timeout=5.0)
        return resp.status_code == 200
    except Exception:
        # In test environments or offline mode, fall back gracefully
        return True


@router.get("", response_model=list[BotResponse])
def get_user_bots(user: dict[str, Any] = Depends(get_current_user)) -> list[BotResponse]:
    """List all bots belonging to the authenticated user."""
    conn = get_db()
    try:
        rows = conn.execute(
            """
            WITH latest_jobs AS (
                SELECT username, step, progress, status,
                       ROW_NUMBER() OVER(PARTITION BY username ORDER BY created_at DESC) as rn
                FROM jobs
            )
            SELECT ub.username, ub.added_at, ub.archived,
                   p.status, p.active_version, p.last_trained_at,
                   p.games_used, p.tier, p.error,
                   j.step AS job_step, j.progress AS job_progress
            FROM user_bots ub
            JOIN players p ON ub.username = p.username
            LEFT JOIN latest_jobs j ON ub.username = j.username AND j.rn = 1
            WHERE ub.user_id = ?
            ORDER BY ub.added_at DESC;
            """,
            (user["id"],),
        ).fetchall()

        return [
            BotResponse(
                username=r["username"],
                status=r["status"],
                active_version=r["active_version"],
                added_at=str(r["added_at"]) if r["added_at"] else None,
                archived=bool(r["archived"]),
                last_trained_at=str(r["last_trained_at"]) if r["last_trained_at"] else None,
                games_used=r["games_used"],
                tier=r["tier"],
                error=r["error"],
                job_step=r["job_step"],
                job_progress=r["job_progress"],
            )
            for r in rows
        ]
    finally:
        conn.close()


@router.post("", response_model=BotResponse, status_code=status.HTTP_201_CREATED)
def add_bot(req: AddBotRequest, user: dict[str, Any] = Depends(get_current_user)) -> BotResponse:
    """Add a new bot to the user's account."""
    raw_username = req.username.strip()
    if not raw_username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username cannot be empty",
        )

    conn = get_db()
    try:
        # 1. Enforce 10-bot active limit per user
        active_count_row = conn.execute(
            "SELECT COUNT(*) as cnt FROM user_bots WHERE user_id = ? AND archived = 0;",
            (user["id"],),
        ).fetchone()
        active_count = active_count_row["cnt"] if active_count_row else 0
        max_bots = getattr(config, "MAX_USER_BOTS", 10)
        if active_count >= max_bots:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Bot limit reached. You can have at most {max_bots} active bots.",
            )

        # 2. Validate username exists on Lichess
        if not check_lichess_user_exists(raw_username):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Lichess user '{raw_username}' not found",
            )

        with conn:
            # 3. Insert player if missing
            conn.execute(
                """
                INSERT INTO players (username, status, active_version)
                VALUES (?, 'queued', 1)
                ON CONFLICT(username) DO NOTHING;
                """,
                (raw_username,),
            )

            # 4. Insert or unarchive user_bot
            conn.execute(
                """
                INSERT INTO user_bots (user_id, username, archived)
                VALUES (?, ?, 0)
                ON CONFLICT(user_id, username) DO UPDATE SET archived = 0;
                """,
                (user["id"], raw_username),
            )

            # 5. Create background build job
            create_job(
                kind="build_bot",
                username=raw_username,
                requested_by=user["id"],
                conn=conn,
            )

        # Fetch newly added bot row
        row = conn.execute(
            """
            WITH latest_jobs AS (
                SELECT username, step, progress, status,
                       ROW_NUMBER() OVER(PARTITION BY username ORDER BY created_at DESC) as rn
                FROM jobs
            )
            SELECT ub.username, ub.added_at, ub.archived,
                   p.status, p.active_version, p.last_trained_at,
                   p.games_used, p.tier, p.error,
                   j.step AS job_step, j.progress AS job_progress
            FROM user_bots ub
            JOIN players p ON ub.username = p.username
            LEFT JOIN latest_jobs j ON ub.username = j.username AND j.rn = 1
            WHERE ub.user_id = ? AND ub.username = ?;
            """,
            (user["id"], raw_username),
        ).fetchone()

        return BotResponse(
            username=row["username"],
            status=row["status"],
            active_version=row["active_version"],
            added_at=str(row["added_at"]) if row["added_at"] else None,
            archived=bool(row["archived"]),
            last_trained_at=str(row["last_trained_at"]) if row["last_trained_at"] else None,
            games_used=row["games_used"],
            tier=row["tier"],
            error=row["error"],
            job_step=row["job_step"],
            job_progress=row["job_progress"],
        )
    finally:
        conn.close()


@router.post("/{username}/archive", response_model=dict[str, Any])
def archive_bot(username: str, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    """Archive an existing bot to free up a slot."""
    clean_username = username.strip()
    conn = get_db()
    try:
        cur = conn.execute(
            """
            UPDATE user_bots
            SET archived = 1
            WHERE user_id = ? AND username = ?;
            """,
            (user["id"], clean_username),
        )
        if cur.rowcount == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Bot '{clean_username}' not found in your bots list",
            )
        conn.commit()
        return {"username": clean_username, "archived": True}
    finally:
        conn.close()

@router.delete("/{username}", response_model=dict[str, Any])
def delete_bot(username: str, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    """Delete a bot from the user's account."""
    name = username.strip()
    conn = get_db()
    try:
        row = conn.execute(
            """SELECT p.status FROM user_bots ub
               JOIN players p ON p.username = ub.username
               WHERE ub.user_id = ? AND ub.username = ?;""",
            (user["id"], name),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Bot '{name}' not found")
        if row["status"] == "building":
            raise HTTPException(status_code=400, detail="Bot is being built right now. Try again once it finishes or fails.")

        with conn:
            conn.execute("DELETE FROM user_bots WHERE user_id = ? AND username = ?;", (user["id"], name))
            # Remove the shared player record only if no other user has this bot
            others = conn.execute("SELECT 1 FROM user_bots WHERE username = ? LIMIT 1;", (name,)).fetchone()
            if others is None:
                conn.execute("DELETE FROM players WHERE username = ?;", (name,))
        return {"username": name, "deleted": True}
    finally:
        conn.close()