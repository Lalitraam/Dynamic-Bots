"""Move telemetry database tracking (Phase E)."""

from __future__ import annotations
import re
import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.engine.provider import MoveDecision


def _extract_version_int(model_ver: str | int | None) -> int:
    """Extract integer version from model_version string (e.g. 'user:v1' -> 1)."""
    if model_ver is None:
        return 0
    if isinstance(model_ver, int):
        return model_ver
    match = re.search(r"v(\d+)", str(model_ver))
    if match:
        return int(match.group(1))
    try:
        return int(model_ver)
    except ValueError:
        return 0


def init_telemetry(conn: sqlite3.Connection) -> None:
    """Create the move_telemetry table if it doesn't exist.
    Schema (from Appendix A):
      move_telemetry(
        game_id TEXT, ply INTEGER, source TEXT, top1_prob REAL, entropy REAL,
        p_err REAL, guarded INTEGER, infer_ms REAL, think_ms INTEGER,
        model_version INTEGER, PRIMARY KEY(game_id, ply)
      )
    """
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS move_telemetry (
                game_id TEXT,
                ply INTEGER,
                source TEXT,
                top1_prob REAL,
                entropy REAL,
                p_err REAL,
                guarded INTEGER,
                infer_ms REAL,
                think_ms INTEGER,
                model_version INTEGER,
                PRIMARY KEY(game_id, ply)
            )
            """
        )


def record_move(
    conn: sqlite3.Connection,
    game_id: str,
    ply: int,
    decision: MoveDecision,
    infer_ms: float,
    entropy: float,
    top1_prob: float,
) -> None:
    """Insert one telemetry row. Uses INSERT OR REPLACE."""
    guarded = 1 if decision.source == "guarded" else 0
    version_int = _extract_version_int(decision.model_version)

    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO move_telemetry (
                game_id, ply, source, top1_prob, entropy,
                p_err, guarded, infer_ms, think_ms, model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(game_id),
                int(ply),
                str(decision.source),
                float(top1_prob),
                float(entropy),
                float(decision.p_err) if decision.p_err is not None else None,
                guarded,
                float(infer_ms),
                int(decision.think_ms),
                version_int,
            ),
        )


def init_telemetry_in_memory() -> sqlite3.Connection:
    """Return an in-memory SQLite connection with the telemetry table.
    For tests and pre-DB-phase use.
    """
    conn = sqlite3.connect(":memory:")
    init_telemetry(conn)
    return conn
