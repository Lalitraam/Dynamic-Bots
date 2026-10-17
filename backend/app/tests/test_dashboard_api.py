"""Tests for dashboard API endpoint (Phase J)."""

from fastapi.testclient import TestClient
import pytest

from app.auth import get_current_user
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_dashboard_api_unauthorized(client: TestClient):
    """Verify GET /api/bots/{username}/dashboard returns 401 without auth."""
    resp = client.get("/api/bots/the_king_crusher/dashboard")
    assert resp.status_code == 401


def test_dashboard_api_authorized(client: TestClient):
    """Verify GET /api/bots/{username}/dashboard returns 200 with valid mocked auth."""
    app.dependency_overrides[get_current_user] = lambda: {
        "id": 1,
        "email": "tester@example.com",
        "role": "user",
    }
    try:
        resp = client.get("/api/bots/the_king_crusher/dashboard")
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "the_king_crusher"
        assert isinstance(data["rating"], int)
        assert data["rating"] == 2180

        # model_card
        assert "model_card" in data
        assert data["model_card"]["version"] == "v1"
        assert "top1" in data["model_card"]
        assert "nll" in data["model_card"]
        assert "personalisation_status" in data["model_card"]
        assert data["model_card"]["personalized"] is True

        # openings
        assert "openings" in data
        assert "white" in data["openings"]
        assert "black" in data["openings"]
        assert len(data["openings"]["white"]) <= 5
        assert len(data["openings"]["black"]) <= 5
        if data["openings"]["white"]:
            assert "move" in data["openings"]["white"][0]
            assert "probability" in data["openings"]["white"][0]

        # time_management
        assert "time_management" in data
        assert isinstance(data["time_management"], dict)
        assert "opening" in data["time_management"]
        assert "middlegame" in data["time_management"]

        # errors
        assert "errors" in data
        assert isinstance(data["errors"], list)
        if len(data["errors"]) > 0:
            assert "bucket" in data["errors"][0]
            assert "error_rate" in data["errors"][0]

    finally:
        app.dependency_overrides.clear()


def test_dashboard_api_bot_not_found(client: TestClient):
    """Verify 404 is returned when bot does not exist."""
    app.dependency_overrides[get_current_user] = lambda: {
        "id": 1,
        "email": "tester@example.com",
        "role": "user",
    }
    try:
        resp = client.get("/api/bots/nonexistent_bot_xyz/dashboard")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()
    finally:
        app.dependency_overrides.clear()
