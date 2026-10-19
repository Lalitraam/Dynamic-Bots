from __future__ import annotations
import datetime
import os
import random
import secrets
from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr
from starlette.config import Config

from app import config
from app.auth import generate_session_token, get_current_user, hash_password, hash_token, verify_password
from app.db.migrate import get_db
from app.db.users import (
    create_user,
    delete_otp,
    get_otp,
    get_user_by_email,
    get_user_by_google_id,
    save_otp,
    verify_user_email,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _parse_iso(ts_str: str) -> datetime.datetime:
    dt = datetime.datetime.fromisoformat(ts_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt


# --- Google OAuth ---
oauth = OAuth(Config(environ=os.environ))
if config.GOOGLE_CLIENT_ID and config.GOOGLE_CLIENT_SECRET:
    oauth.register(
        name="google",
        client_id=config.GOOGLE_CLIENT_ID,
        client_secret=config.GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )


from app.services.email import send_otp_email

# --- Schemas ---
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    invite_code: str | None = None


class VerifyRequest(BaseModel):
    email: EmailStr
    otp_code: str


class SendOtpRequest(BaseModel):
    email: EmailStr


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


# --- Endpoints ---

@router.post("/register", status_code=201)
def register(req: RegisterRequest) -> dict:
    """Create user with is_verified=0, generate OTP, and send email via SMTP (with dev fallback)."""
    existing = get_user_by_email(req.email)
    if existing:
        if existing.get("is_verified"):
            raise HTTPException(400, "Email already registered. Please sign in.")
        # If existing but unverified, generate a fresh OTP for them
        otp = f"{random.randint(0, 999999):06d}"
        expires = (_utc_now() + datetime.timedelta(minutes=config.OTP_EXPIRY_MINUTES)).isoformat()
        save_otp(req.email, otp, expires)
        sent = send_otp_email(req.email, otp)
        return {
            "message": "Verification code sent to your email." if sent else f"Verification code generated: {otp}",
            "email": req.email,
            "dev_otp": otp,
            "sent_via_email": sent,
        }

    pw_hash = hash_password(req.password)
    is_ver = 1 if req.invite_code else 0
    create_user(req.email, pw_hash, auth_provider="local", is_verified=is_ver)

    otp = f"{random.randint(0, 999999):06d}"
    expires = (_utc_now() + datetime.timedelta(
        minutes=config.OTP_EXPIRY_MINUTES
    )).isoformat()
    save_otp(req.email, otp, expires)

    sent = send_otp_email(req.email, otp)

    return {
        "message": "Verification code sent to your email." if sent else "Verification code sent. Check your email or terminal.",
        "email": req.email,
        "dev_otp": otp,
        "sent_via_email": sent,
    }


@router.post("/send-otp")
def send_otp(req: SendOtpRequest) -> dict:
    """Generate and send an OTP code for passwordless login or verification."""
    email = req.email.strip().lower()
    user = get_user_by_email(email)
    if not user:
        # Create an account automatically so they can log in seamlessly with OTP
        temp_pw_hash = hash_password(secrets.token_urlsafe(16))
        create_user(email, temp_pw_hash, auth_provider="local", is_verified=0)

    otp = f"{random.randint(0, 999999):06d}"
    expires = (_utc_now() + datetime.timedelta(minutes=config.OTP_EXPIRY_MINUTES)).isoformat()
    save_otp(email, otp, expires)
    sent = send_otp_email(email, otp)

    return {
        "message": f"Verification code sent to {email}." if sent else "Verification code generated.",
        "email": email,
        "dev_otp": otp,
        "sent_via_email": sent,
    }


@router.post("/verify-email")
@router.post("/login-otp")
def verify_email(req: VerifyRequest) -> dict:
    """Verify OTP, mark user as verified, and return access token."""
    otp_row = get_otp(req.email)
    if not otp_row:
        raise HTTPException(400, "No pending verification for this email. Please request a new code.")
    if otp_row["otp_code"] != req.otp_code.strip():
        raise HTTPException(400, "Invalid verification code. Please check and try again.")
    if _parse_iso(otp_row["expires_at"]) < _utc_now():
        delete_otp(req.email)
        raise HTTPException(400, "Verification code expired. Please request a new code.")

    verify_user_email(req.email)
    delete_otp(req.email)

    user = get_user_by_email(req.email)
    raw_token, token_hash = generate_session_token()
    expires_at = (_utc_now() + datetime.timedelta(days=30)).isoformat()
    conn = get_db()
    try:
        with conn:
            conn.execute(
                "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
                (token_hash, user["id"], expires_at),
            )
    finally:
        conn.close()

    return {"access_token": raw_token, "token_type": "bearer"}


@router.post("/login")
def login(req: LoginRequest) -> dict:
    user = get_user_by_email(req.email)
    if not user or user.get("password_hash") is None:
        raise HTTPException(401, "Invalid email or password")
    if not verify_password(req.password, user["password_hash"]):
        raise HTTPException(401, "Invalid email or password")
    if user.get("auth_provider") == "local" and not user.get("is_verified"):
        # Auto-dispatch a new OTP to help the user complete verification
        otp = f"{random.randint(0, 999999):06d}"
        expires = (_utc_now() + datetime.timedelta(minutes=config.OTP_EXPIRY_MINUTES)).isoformat()
        save_otp(req.email, otp, expires)
        send_otp_email(req.email, otp)
        raise HTTPException(
            403,
            "Email not verified. A 6-digit verification code has been sent to your email.",
        )

    raw_token, token_hash = generate_session_token()
    expires_at = (_utc_now() + datetime.timedelta(days=30)).isoformat()
    conn = get_db()
    try:
        with conn:
            conn.execute(
                "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
                (token_hash, user["id"], expires_at),
            )
    finally:
        conn.close()
    return {"access_token": raw_token, "token_type": "bearer"}



@router.get("/me")
def me(user: dict = Depends(get_current_user)) -> dict:
    return {"id": user["id"], "email": user["email"], "role": user["role"]}


@router.get("/google/login")
async def google_login(request: Request):
    if not config.GOOGLE_CLIENT_ID:
        raise HTTPException(500, "Google OAuth not configured")
    redirect_uri = request.url_for("google_callback")
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/google/callback", name="google_callback")
async def google_callback(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception:
        return RedirectResponse(url=f"{config.FRONTEND_URL}/login?error=oauth_failed")

    userinfo = token.get("userinfo") or {}
    email = userinfo.get("email")
    google_id = userinfo.get("sub")

    if not email or not google_id:
        return RedirectResponse(url=f"{config.FRONTEND_URL}/login?error=missing_userinfo")

    user = get_user_by_google_id(google_id) or get_user_by_email(email)
    if not user:
        create_user(
            email=email,
            password_hash=None,
            auth_provider="google",
            google_id=google_id,
            is_verified=1,
        )
        user = get_user_by_email(email)
    else:
        if not user.get("google_id"):
            conn = get_db()
            try:
                with conn:
                    conn.execute(
                        "UPDATE users SET google_id = ?, auth_provider = 'google', is_verified = 1 WHERE id = ?",
                        (google_id, user["id"]),
                    )
            finally:
                conn.close()
            user = get_user_by_email(email)

    # Issue our own session token
    raw_token, token_hash = generate_session_token()
    expires_at = (_utc_now() + datetime.timedelta(days=30)).isoformat()
    conn = get_db()
    try:
        with conn:
            conn.execute(
                "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
                (token_hash, user["id"], expires_at),
            )
    finally:
        conn.close()

    return RedirectResponse(url=f"{config.FRONTEND_URL}/login?token={raw_token}")
