"""Tests for authentication and user management with OTP verification and OAuth."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app import config
from app.auth import hash_password, verify_password
from app.db.migrate import run_migrations
from app.db.users import get_otp, get_user_by_email
from app.main import app


@pytest.fixture(autouse=True)
def test_db_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolate tests into a temporary database."""
    test_db = tmp_path / "test_auth.db"
    monkeypatch.setattr(config, "DB_PATH", test_db)
    run_migrations(test_db)


@pytest.fixture
def client():
    return TestClient(app)


def test_password_hashing_and_verification():
    """Test Argon2 password hashing and verification."""
    password = "SuperSecretPassword123!"
    pwd_hash = hash_password(password)

    assert pwd_hash != password
    assert verify_password(password, pwd_hash) is True
    assert verify_password("IncorrectPassword", pwd_hash) is False
    assert verify_password("", pwd_hash) is False


def test_register_creates_unverified_user(client: TestClient):
    """Registering creates a user with is_verified=0."""
    resp = client.post(
        "/api/auth/register",
        json={"email": "unverified@example.com", "password": "Password123!"},
    )
    assert resp.status_code == 201

    user = get_user_by_email("unverified@example.com")
    assert user is not None
    assert user["is_verified"] == 0
    assert user["auth_provider"] == "local"


def test_register_returns_otp_flow(client: TestClient):
    """Registering generates an OTP in the database and prints notification."""
    resp = client.post(
        "/api/auth/register",
        json={"email": "otpuser@example.com", "password": "Password123!"},
    )
    assert resp.status_code == 201
    assert "Verification code sent" in resp.json()["message"]

    otp_row = get_otp("otpuser@example.com")
    assert otp_row is not None
    assert len(otp_row["otp_code"]) == 6
    assert otp_row["otp_code"].isdigit()
    assert "expires_at" in otp_row


def test_verify_email_with_wrong_otp_fails(client: TestClient):
    """Verifying with incorrect OTP returns 400 error."""
    client.post(
        "/api/auth/register",
        json={"email": "wrongotp@example.com", "password": "Password123!"},
    )
    resp = client.post(
        "/api/auth/verify-email",
        json={"email": "wrongotp@example.com", "otp_code": "000000"},
    )
    assert resp.status_code == 400
    assert "Invalid verification code" in resp.json()["detail"]


def test_verify_email_with_correct_otp_returns_token(client: TestClient):
    """Verifying with correct OTP verifies user, deletes OTP, and returns token."""
    client.post(
        "/api/auth/register",
        json={"email": "correctotp@example.com", "password": "Password123!"},
    )
    otp_row = get_otp("correctotp@example.com")
    assert otp_row is not None

    resp = client.post(
        "/api/auth/verify-email",
        json={"email": "correctotp@example.com", "otp_code": otp_row["otp_code"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

    user = get_user_by_email("correctotp@example.com")
    assert user["is_verified"] == 1

    # OTP row cleaned up
    assert get_otp("correctotp@example.com") is None


def test_login_rejects_unverified_user(client: TestClient):
    """Login returns 403 when email is unverified."""
    client.post(
        "/api/auth/register",
        json={"email": "unverified_login@example.com", "password": "Password123!"},
    )
    login_resp = client.post(
        "/api/auth/login",
        json={"email": "unverified_login@example.com", "password": "Password123!"},
    )
    assert login_resp.status_code == 403
    assert "Email not verified" in login_resp.json()["detail"]


def test_login_accepts_verified_user(client: TestClient):
    """Login succeeds once user is verified."""
    client.post(
        "/api/auth/register",
        json={"email": "verified_login@example.com", "password": "Password123!"},
    )
    otp_row = get_otp("verified_login@example.com")
    client.post(
        "/api/auth/verify-email",
        json={"email": "verified_login@example.com", "otp_code": otp_row["otp_code"]},
    )

    login_resp = client.post(
        "/api/auth/login",
        json={"email": "verified_login@example.com", "password": "Password123!"},
    )
    assert login_resp.status_code == 200
    assert "access_token" in login_resp.json()


def test_google_oauth_endpoints_exist():
    """Verify Google OAuth routes are registered in FastAPI."""
    assert str(app.url_path_for("google_login")) == "/api/auth/google/login"
    assert str(app.url_path_for("google_callback")) == "/api/auth/google/callback"
