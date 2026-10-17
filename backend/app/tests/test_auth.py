"""Tests for authentication and user management (Phase G)."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app import config
from app.auth import hash_password, verify_password
from app.db.migrate import run_migrations
from app.db.users import create_invite_code, get_invite_code
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


def test_registration_with_valid_invite_code(client: TestClient):
    """Test user registration with valid and invalid invite codes."""
    create_invite_code("CHESS-2026")

    # Successful registration
    resp = client.post(
        "/api/auth/register",
        json={"email": "newuser@example.com", "password": "Password123!", "invite_code": "CHESS-2026"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["email"] == "newuser@example.com"
    assert data["role"] == "user"

    # Verify invite code is marked used
    invite = get_invite_code("CHESS-2026")
    assert invite is not None
    assert invite["used_by"] == data["id"]

    # Re-use of invite code rejected
    resp_reuse = client.post(
        "/api/auth/register",
        json={"email": "other@example.com", "password": "Password123!", "invite_code": "CHESS-2026"},
    )
    assert resp_reuse.status_code == 400
    assert "already been used" in resp_reuse.json()["detail"]

    # Non-existent invite code rejected
    resp_bad = client.post(
        "/api/auth/register",
        json={"email": "other@example.com", "password": "Password123!", "invite_code": "INVALID-CODE"},
    )
    assert resp_bad.status_code == 400
    assert "Invalid invite code" in resp_bad.json()["detail"]


def test_login_returns_token(client: TestClient):
    """Test login credentials validation and session token generation."""
    create_invite_code("LOGIN-TEST")
    reg_resp = client.post(
        "/api/auth/register",
        json={"email": "logintest@example.com", "password": "Password123!", "invite_code": "LOGIN-TEST"},
    )
    assert reg_resp.status_code == 201

    # Valid login
    login_resp = client.post(
        "/api/auth/login",
        json={"email": "logintest@example.com", "password": "Password123!"},
    )
    assert login_resp.status_code == 200
    login_data = login_resp.json()
    assert "access_token" in login_data
    assert login_data["token_type"] == "bearer"
    assert len(login_data["access_token"]) > 20

    # Invalid password
    bad_resp = client.post(
        "/api/auth/login",
        json={"email": "logintest@example.com", "password": "WrongPassword"},
    )
    assert bad_resp.status_code == 401
    assert "Incorrect email or password" in bad_resp.json()["detail"]


def test_get_current_user_and_bad_tokens(client: TestClient):
    """Test get_current_user endpoint protection against missing/invalid tokens."""
    create_invite_code("ME-TEST")
    client.post(
        "/api/auth/register",
        json={"email": "metest@example.com", "password": "Password123!", "invite_code": "ME-TEST"},
    )
    login_resp = client.post(
        "/api/auth/login",
        json={"email": "metest@example.com", "password": "Password123!"},
    )
    token = login_resp.json()["access_token"]

    # Valid token
    me_resp = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 200
    assert me_resp.json()["email"] == "metest@example.com"

    # Invalid token
    bad_token_resp = client.get("/api/auth/me", headers={"Authorization": "Bearer totally_fake_token_12345"})
    assert bad_token_resp.status_code == 401
    assert "Invalid or expired session token" in bad_token_resp.json()["detail"]

    # Missing Authorization header
    missing_resp = client.get("/api/auth/me")
    assert missing_resp.status_code == 401
