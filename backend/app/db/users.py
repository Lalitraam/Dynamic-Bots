"""User, Session, and Invite Code database helpers (Phase G)."""

from __future__ import annotations
from datetime import datetime, timezone
import sqlite3
from typing import Any
from app.db.migrate import get_db


class UserId(int):
    """An int subclass representing user_id that also allows dictionary-style lookups for backwards-compatibility."""

    def __new__(cls, val: int, user_dict: dict[str, Any] | None = None):
        obj = super().__new__(cls, val)
        obj.user_dict = user_dict or {}
        return obj

    def __getitem__(self, item: str) -> Any:
        return self.user_dict[item]

    def get(self, item: str, default: Any = None) -> Any:
        return self.user_dict.get(item, default)


def get_user_by_id(user_id: int, conn: sqlite3.Connection | None = None) -> dict[str, Any] | None:
    """Fetch user by primary key id."""
    close_after = False
    if conn is None:
        conn = get_db()
        close_after = True
    try:
        row = conn.execute("SELECT * FROM users WHERE id = ?;", (user_id,)).fetchone()
        return dict(row) if row else None
    finally:
        if close_after:
            conn.close()


def get_user_by_email(email: str, conn: sqlite3.Connection | None = None) -> dict[str, Any] | None:
    """Fetch user by unique email."""
    close_after = False
    if conn is None:
        conn = get_db()
        close_after = True
    try:
        row = conn.execute("SELECT * FROM users WHERE LOWER(email) = LOWER(?);", (email.strip(),)).fetchone()
        return dict(row) if row else None
    finally:
        if close_after:
            conn.close()


def get_user_by_google_id(google_id: str, conn: sqlite3.Connection | None = None) -> dict[str, Any] | None:
    """Return the user row by Google ID, or None."""
    close_after = False
    if conn is None:
        conn = get_db()
        close_after = True
    try:
        row = conn.execute("SELECT * FROM users WHERE google_id = ?;", (google_id,)).fetchone()
        return dict(row) if row else None
    finally:
        if close_after:
            conn.close()


def create_user(
    email: str,
    password_hash: str | None,
    auth_provider: str = "local",
    google_id: str | None = None,
    is_verified: int = 0,
    role: str = "user",
    conn: sqlite3.Connection | None = None,
) -> int:
    """Insert a user and return their id."""
    close_after = False
    if conn is None:
        conn = get_db()
        close_after = True
    try:
        with conn:
            cur = conn.execute(
                """
                INSERT INTO users (email, password_hash, role, auth_provider, google_id, is_verified)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (
                    email.strip().lower(),
                    password_hash,
                    role,
                    auth_provider,
                    google_id,
                    is_verified,
                ),
            )
            user_id = cur.lastrowid
        user_row = get_user_by_id(user_id, conn=conn) if user_id else {}
        return UserId(user_id, user_row)  # type: ignore[return-value]
    finally:
        if close_after:
            conn.close()


def save_otp(email: str, otp_code: str, expires_at: str, conn: sqlite3.Connection | None = None) -> None:
    """Insert or replace an OTP code for the given email."""
    close_after = False
    if conn is None:
        conn = get_db()
        close_after = True
    try:
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO email_verifications (email, otp_code, expires_at)
                VALUES (?, ?, ?);
                """,
                (email.strip().lower(), otp_code.strip(), expires_at),
            )
    finally:
        if close_after:
            conn.close()


def get_otp(email: str, conn: sqlite3.Connection | None = None) -> dict[str, Any] | None:
    """Return {'email', 'otp_code', 'expires_at'} or None."""
    close_after = False
    if conn is None:
        conn = get_db()
        close_after = True
    try:
        row = conn.execute(
            "SELECT email, otp_code, expires_at FROM email_verifications WHERE LOWER(email) = LOWER(?);",
            (email.strip(),),
        ).fetchone()
        return dict(row) if row else None
    finally:
        if close_after:
            conn.close()


def delete_otp(email: str, conn: sqlite3.Connection | None = None) -> None:
    """Remove the OTP entry."""
    close_after = False
    if conn is None:
        conn = get_db()
        close_after = True
    try:
        with conn:
            conn.execute(
                "DELETE FROM email_verifications WHERE LOWER(email) = LOWER(?);",
                (email.strip(),),
            )
    finally:
        if close_after:
            conn.close()


def verify_user_email(email: str, conn: sqlite3.Connection | None = None) -> None:
    """Set is_verified=1 for the given email."""
    close_after = False
    if conn is None:
        conn = get_db()
        close_after = True
    try:
        with conn:
            conn.execute(
                "UPDATE users SET is_verified = 1 WHERE LOWER(email) = LOWER(?);",
                (email.strip(),),
            )
    finally:
        if close_after:
            conn.close()


def get_invite_code(code: str, conn: sqlite3.Connection | None = None) -> dict[str, Any] | None:
    """Fetch invite code record."""
    close_after = False
    if conn is None:
        conn = get_db()
        close_after = True
    try:
        row = conn.execute("SELECT * FROM invite_codes WHERE code = ?;", (code.strip(),)).fetchone()
        return dict(row) if row else None
    except sqlite3.OperationalError:
        return None
    finally:
        if close_after:
            conn.close()


def create_invite_code(
    code: str,
    created_by: int | None = None,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """Create an invite code."""
    close_after = False
    if conn is None:
        conn = get_db()
        close_after = True
    try:
        with conn:
            conn.execute(
                "INSERT INTO invite_codes (code, created_by) VALUES (?, ?);",
                (code.strip(), created_by),
            )
        return get_invite_code(code, conn=conn)  # type: ignore[return-value]
    finally:
        if close_after:
            conn.close()


def mark_invite_code_used(
    code: str,
    used_by_id: int,
    conn: sqlite3.Connection | None = None,
) -> None:
    """Mark an invite code as used by a specific user."""
    close_after = False
    if conn is None:
        conn = get_db()
        close_after = True
    now_str = datetime.now(timezone.utc).isoformat()
    try:
        with conn:
            conn.execute(
                "UPDATE invite_codes SET used_by = ?, used_at = ? WHERE code = ?;",
                (used_by_id, now_str, code.strip()),
            )
    finally:
        if close_after:
            conn.close()


def create_session(
    token_hash: str,
    user_id: int,
    expires_at: str,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """Store session token hash."""
    close_after = False
    if conn is None:
        conn = get_db()
        close_after = True
    try:
        with conn:
            conn.execute(
                "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?);",
                (token_hash, user_id, expires_at),
            )
        return {"token_hash": token_hash, "user_id": user_id, "expires_at": expires_at}
    finally:
        if close_after:
            conn.close()


def get_session(token_hash: str, conn: sqlite3.Connection | None = None) -> dict[str, Any] | None:
    """Fetch session if not expired."""
    close_after = False
    if conn is None:
        conn = get_db()
        close_after = True
    try:
        row = conn.execute(
            """
            SELECT s.*, u.email, u.role
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.token_hash = ?;
            """,
            (token_hash,),
        ).fetchone()
        if not row:
            return None
        session = dict(row)
        # Check expiry
        now_iso = datetime.now(timezone.utc).isoformat()
        if session["expires_at"] < now_iso:
            return None
        return session
    finally:
        if close_after:
            conn.close()


def delete_session(token_hash: str, conn: sqlite3.Connection | None = None) -> None:
    """Delete a session by token hash."""
    close_after = False
    if conn is None:
        conn = get_db()
        close_after = True
    try:
        with conn:
            conn.execute("DELETE FROM sessions WHERE token_hash = ?;", (token_hash,))
    finally:
        if close_after:
            conn.close()
