"""Tests for database migrations (Phase G)."""

import sqlite3
from pathlib import Path
import pytest
from app.db.migrate import get_db, run_migrations


def test_migration_runs_cleanly(tmp_path: Path):
    db_file = tmp_path / "test_app.db"
    applied = run_migrations(db_file)
    assert "001_init" in applied

    conn = get_db(db_file)
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    table_names = {r["name"] for r in rows}
    conn.close()

    expected_tables = {
        "schema_migrations",
        "users",
        "sessions",
        "invite_codes",
        "players",
        "user_bots",
        "jobs",
        "games",
        "game_moves",
        "move_telemetry",
    }
    for table in expected_tables:
        assert table in table_names, f"Expected table '{table}' not found in database."


def test_rerunning_migration_is_noop(tmp_path: Path):
    db_file = tmp_path / "test_app.db"
    applied_first = run_migrations(db_file)
    assert len(applied_first) >= 1

    applied_second = run_migrations(db_file)
    assert applied_second == []
