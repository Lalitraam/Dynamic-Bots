"""Authentication utilities: password hashing, session tokens, and FastAPI dependency (Phase G)."""

from __future__ import annotations
import hashlib
from pathlib import Path
import secrets
from typing import Any
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.db.migrate import get_db
from app.db.users import get_session, get_user_by_id

_ph = PasswordHasher()
security_bearer = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    """Hash a password using Argon2id."""
    return _ph.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    """Verify a plain password against an Argon2 hash."""
    try:
        return _ph.verify(hashed, password)
    except (VerifyMismatchError, Exception):
        return False


def hash_token(token: str) -> str:
    """Hash a raw token with SHA-256 for secure DB lookup."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def generate_session_token() -> tuple[str, str]:
    """Generate (raw_token, token_hash) pair."""
    raw_token = secrets.token_urlsafe(32)
    token_hash = hash_token(raw_token)
    return raw_token, token_hash


def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(security_bearer),
) -> dict[str, Any]:
    """FastAPI dependency to authenticate and return the current user."""
    token: str | None = None
    if credentials is not None:
        token = credentials.credentials
    else:
        # Check Authorization header manually as fallback
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_hash = hash_token(token)
    session = get_session(token_hash)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = get_user_by_id(session["user_id"])
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user
