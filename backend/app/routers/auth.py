"""Auth router: register, login, me endpoints (Phase G)."""

from __future__ import annotations
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app import config
from app.auth import generate_session_token, get_current_user, hash_password, verify_password
from app.db.migrate import get_db
from app.db.users import (
    create_session,
    create_user,
    get_invite_code,
    get_user_by_email,
    mark_invite_code_used,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    email: str
    password: str
    invite_code: str


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserProfileResponse(BaseModel):
    id: int
    email: str
    role: str
    created_at: str | None = None


@router.post("/register", response_model=UserProfileResponse, status_code=status.HTTP_201_CREATED)
def register(req: RegisterRequest) -> UserProfileResponse:
    """Register a new user with a valid invite code."""
    # 1. Validate invite code
    invite = get_invite_code(req.invite_code)
    if not invite:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid invite code",
        )
    if invite.get("used_by") is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invite code has already been used",
        )

    # 2. Check if user already exists
    existing = get_user_by_email(req.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email is already registered",
        )

    # 3. Create user and mark invite code used
    pwd_hash = hash_password(req.password)
    conn = get_db()
    try:
        user = create_user(email=req.email, password_hash=pwd_hash, role="user", conn=conn)
        mark_invite_code_used(code=req.invite_code, used_by_id=user["id"], conn=conn)
    finally:
        conn.close()

    return UserProfileResponse(
        id=user["id"],
        email=user["email"],
        role=user["role"],
        created_at=str(user.get("created_at")),
    )


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest) -> TokenResponse:
    """Authenticate with email and password to receive a bearer session token."""
    user = get_user_by_email(req.email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    if not verify_password(req.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    # Generate session token and store in DB
    raw_token, token_hash = generate_session_token()
    expire_days = getattr(config, "SESSION_EXPIRE_DAYS", 30)
    expires_at = (datetime.now(timezone.utc) + timedelta(days=expire_days)).isoformat()

    create_session(token_hash=token_hash, user_id=user["id"], expires_at=expires_at)

    return TokenResponse(access_token=raw_token, token_type="bearer")


@router.get("/me", response_model=UserProfileResponse)
def get_me(user: dict = Depends(get_current_user)) -> UserProfileResponse:
    """Get the authenticated user's profile."""
    return UserProfileResponse(
        id=user["id"],
        email=user["email"],
        role=user["role"],
        created_at=str(user.get("created_at")),
    )
