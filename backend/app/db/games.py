"""Game and move persistence helpers (Phase G)."""

from __future__ import annotations
from datetime import datetime, timezone
import sqlite3
from typing import Any
from app.db.migrate import get_db


def create_game(
    game_id: str,
    owner_id: int,
    player_username: str,
    model_version: str,
    human_color: str,
    base_s: int,
    inc_s: int,
    secret_hash: str,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """Insert a new chess game session."""
    close_after = False
    if conn is None:
        conn = get_db()
        close_after = True
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO games (
                    id, owner_id, player_username, model_version, human_color,
                    base_s, inc_s, status, secret_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?);
                """,
                (
                    game_id,
                    owner_id,
                    player_username,
                    model_version,
                    human_color.lower(),
                    base_s,
                    inc_s,
                    secret_hash,
                ),
            )
        row = conn.execute("SELECT * FROM games WHERE id = ?;", (game_id,)).fetchone()
        return dict(row)
    finally:
        if close_after:
            conn.close()


def get_game(game_id: str, conn: sqlite3.Connection | None = None) -> dict[str, Any] | None:
    """Fetch game by game_id."""
    close_after = False
    if conn is None:
        conn = get_db()
        close_after = True
    try:
        row = conn.execute("SELECT * FROM games WHERE id = ?;", (game_id,)).fetchone()
        return dict(row) if row else None
    finally:
        if close_after:
            conn.close()


def update_game(
    game_id: str,
    status: str | None = None,
    result: str | None = None,
    termination: str | None = None,
    pgn: str | None = None,
    ended_at: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> None:
    """Update game outcome."""
    close_after = False
    if conn is None:
        conn = get_db()
        close_after = True
    fields = []
    params = []
    if status is not None:
        fields.append("status = ?")
        params.append(status)
    if result is not None:
        fields.append("result = ?")
        params.append(result)
    if termination is not None:
        fields.append("termination = ?")
        params.append(termination)
    if pgn is not None:
        fields.append("pgn = ?")
        params.append(pgn)
    if ended_at is not None:
        fields.append("ended_at = ?")
        params.append(ended_at)

    if not fields:
        return

    params.append(game_id)
    sql = f"UPDATE games SET {', '.join(fields)} WHERE id = ?;"
    try:
        with conn:
            conn.execute(sql, tuple(params))
    finally:
        if close_after:
            conn.close()


def add_move(
    game_id: str,
    ply: int,
    uci: str,
    san: str,
    by: str,
    clock_after_ms: int | None = None,
    elapsed_ms: int | None = None,
    conn: sqlite3.Connection | None = None,
) -> None:
    """Record an individual game move."""
    close_after = False
    if conn is None:
        conn = get_db()
        close_after = True
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO game_moves (game_id, ply, uci, san, by, clock_after_ms, elapsed_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (game_id, ply, uci, san, by, clock_after_ms, elapsed_ms),
            )
    finally:
        if close_after:
            conn.close()


def get_moves(game_id: str, conn: sqlite3.Connection | None = None) -> list[dict[str, Any]]:
    """Fetch ordered move history for a game."""
    close_after = False
    if conn is None:
        conn = get_db()
        close_after = True
    try:
        rows = conn.execute(
            "SELECT * FROM game_moves WHERE game_id = ? ORDER BY ply ASC;",
            (game_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        if close_after:
            conn.close()


def add_telemetry(
    game_id: str,
    ply: int,
    source: str,
    top1_prob: float | None = None,
    entropy: float | None = None,
    p_err: float | None = None,
    guarded: int = 0,
    infer_ms: float | None = None,
    think_ms: float | None = None,
    model_version: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> None:
    """Record move inference and error telemetry."""
    close_after = False
    if conn is None:
        conn = get_db()
        close_after = True
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO move_telemetry (
                    game_id, ply, source, top1_prob, entropy, p_err,
                    guarded, infer_ms, think_ms, model_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    game_id,
                    ply,
                    source,
                    top1_prob,
                    entropy,
                    p_err,
                    guarded,
                    infer_ms,
                    think_ms,
                    model_version,
                ),
            )
    finally:
        if close_after:
            conn.close()
