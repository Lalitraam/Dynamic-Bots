"""Database migration runner (Phase G)."""

from __future__ import annotations
from pathlib import Path
import sqlite3
from typing import Generator
from app import config


def get_db(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Create and return a configured SQLite connection."""
    target_path = Path(db_path) if db_path is not None else Path(config.DB_PATH)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(target_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def run_migrations(db_path: Path | str | None = None) -> list[str]:
    """Execute all pending schema SQL migrations in order."""
    conn = get_db(db_path)
    schema_dir = Path(__file__).resolve().parent / "schema"

    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

    applied_rows = conn.execute("SELECT version FROM schema_migrations;").fetchall()
    applied_versions = {row["version"] for row in applied_rows}

    sql_files = sorted(schema_dir.glob("*.sql"))
    newly_applied: list[str] = []

    for sql_file in sql_files:
        version = sql_file.stem
        if version not in applied_versions:
            sql_content = sql_file.read_text(encoding="utf-8")
            with conn:
                try:
                    conn.executescript(sql_content)
                except sqlite3.OperationalError as e:
                    if "duplicate column name" in str(e).lower():
                        for stmt in sql_content.split(";"):
                            stmt_clean = stmt.strip()
                            if stmt_clean:
                                try:
                                    conn.execute(stmt_clean)
                                except sqlite3.OperationalError as stmt_err:
                                    if "duplicate column name" not in str(stmt_err).lower():
                                        raise stmt_err
                    else:
                        raise e
                conn.execute(
                    "INSERT INTO schema_migrations (version) VALUES (?);",
                    (version,),
                )
            newly_applied.append(version)

    conn.close()
    return newly_applied


if __name__ == "__main__":
    applied = run_migrations()
    if applied:
        print(f"Applied migrations: {applied}")
    else:
        print("No new migrations to apply.")
