"""Tests for bots API endpoints (Phase G)."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app import config
from app.db.migrate import run_migrations
from app.db.users import create_invite_code
from app.main import app
from app.routers import bots


@pytest.fixture(autouse=True)
def test_db_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolate tests into a temporary database."""
    test_db = tmp_path / "test_bots.db"
    monkeypatch.setattr(config, "DB_PATH", test_db)
    run_migrations(test_db)


@pytest.fixture
def auth_client():
    client = TestClient(app)
    create_invite_code("BOT-TEST-INVITE")
    client.post(
        "/api/auth/register",
        json={"email": "botmaster@example.com", "password": "Password123!", "invite_code": "BOT-TEST-INVITE"},
    )
    login_resp = client.post(
        "/api/auth/login",
        json={"email": "botmaster@example.com", "password": "Password123!"},
    )
    token = login_resp.json()["access_token"]
    client.headers.update({"Authorization": f"Bearer {token}"})
    return client


def test_add_bot_and_mocked_lichess(auth_client: TestClient, monkeypatch: pytest.MonkeyPatch):
    """Test adding a bot with mocked Lichess user validation."""
    # Mock lichess check
    def mock_check(username: str) -> bool:
        return username != "nonexistent_player"

    monkeypatch.setattr(bots, "check_lichess_user_exists", mock_check)

    # Valid bot addition
    resp = auth_client.post("/api/bots", json={"username": "hikaru"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["username"] == "hikaru"
    assert data["status"] == "queued"
    assert data["archived"] is False

    # List bots
    list_resp = auth_client.get("/api/bots")
    assert list_resp.status_code == 200
    bots_list = list_resp.json()
    assert len(bots_list) == 1
    assert bots_list[0]["username"] == "hikaru"

    # Non-existent lichess player
    bad_resp = auth_client.post("/api/bots", json={"username": "nonexistent_player"})
    assert bad_resp.status_code == 404
    assert "not found" in bad_resp.json()["detail"]


def test_10_bot_limit(auth_client: TestClient, monkeypatch: pytest.MonkeyPatch):
    """Test that users cannot exceed the maximum active bot limit."""
    monkeypatch.setattr(bots, "check_lichess_user_exists", lambda u: True)

    # Add 10 bots successfully
    for i in range(10):
        resp = auth_client.post("/api/bots", json={"username": f"bot_player_{i}"})
        assert resp.status_code == 201

    # 11th bot rejected
    overflow_resp = auth_client.post("/api/bots", json={"username": "bot_player_10"})
    assert overflow_resp.status_code == 400
    assert "Bot limit reached" in overflow_resp.json()["detail"]


def test_archive_frees_slot(auth_client: TestClient, monkeypatch: pytest.MonkeyPatch):
    """Test that archiving an active bot frees up a slot to add another bot."""
    monkeypatch.setattr(bots, "check_lichess_user_exists", lambda u: True)

    # Fill 10 slots
    for i in range(10):
        resp = auth_client.post("/api/bots", json={"username": f"slot_bot_{i}"})
        assert resp.status_code == 201

    # Archive one bot
    arch_resp = auth_client.post("/api/bots/slot_bot_0/archive")
    assert arch_resp.status_code == 200
    assert arch_resp.json()["archived"] is True

    # Now adding a new bot succeeds
    new_bot_resp = auth_client.post("/api/bots", json={"username": "slot_bot_10"})
    assert new_bot_resp.status_code == 201
    assert new_bot_resp.json()["username"] == "slot_bot_10"


def test_create_and_play_game(auth_client: TestClient, monkeypatch: pytest.MonkeyPatch):
    """Test creating a game session and playing moves against V3Provider."""
    monkeypatch.setattr(bots, "check_lichess_user_exists", lambda u: True)
    auth_client.post("/api/bots", json={"username": "the_king_crusher"})

    # Create game as White
    game_resp = auth_client.post(
        "/api/games",
        json={"username": "the_king_crusher", "human_color": "white", "base_s": 180, "inc_s": 2},
    )
    assert game_resp.status_code == 201
    game_data = game_resp.json()
    game_id = game_data["game_id"]
    secret = game_data["secret"]
    assert game_data["status"] == "active"

    # Play e2e4
    move_resp = auth_client.post(
        f"/api/games/{game_id}/move",
        json={"move": "e2e4", "secret": secret},
    )
    assert move_resp.status_code == 200
    move_data = move_resp.json()
    assert len(move_data["moves"]) == 2
    assert move_data["moves"][0]["by"] == "human"
    assert move_data["moves"][0]["uci"] == "e2e4"
    assert move_data["moves"][1]["by"] == "bot"
    assert move_data["status"] == "active"

