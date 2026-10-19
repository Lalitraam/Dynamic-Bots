"""Tests for dashboard API endpoint (Phase J)."""

import json
from pathlib import Path
import pandas as pd
from fastapi.testclient import TestClient
import pytest

from app import config
from app.auth import get_current_user
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def _ensure_king_crusher_fixture():
    p_dir = Path(config.DATA_ROOT) / "players" / "the_king_crusher"
    if (p_dir / "samples_v3.parquet").exists() and (p_dir / "models" / "v1" / "eval.json").exists():
        return
    p_dir.mkdir(parents=True, exist_ok=True)
    models_dir = p_dir / "models" / "v1"
    models_dir.mkdir(parents=True, exist_ok=True)

    with open(models_dir / "eval.json", "w", encoding="utf-8") as f:
        json.dump({"decision": "promote_ft", "ft_metrics": {"top1": 0.42, "nll": 1.25}}, f)

    with open(p_dir / "opening_book.json", "w", encoding="utf-8") as f:
        json.dump({
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -": {"e2e4": 10},
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -": {"e7e5": 8},
        }, f)

    df_samples = pd.DataFrame([
        {
            "game_id": f"g_{i}",
            "ply": 5 if i % 2 == 0 else 25,
            "category": "blitz",
            "color": "white" if i % 2 == 0 else "black",
            "game_result_for_player": 1.0,
            "uci": "e2e4",
            "prev_move_1": None,
            "time_spent_s": 2.5,
            "player_rating": 2180,
        }
        for i in range(20)
    ])
    df_samples.to_parquet(p_dir / "samples_v3.parquet")

    df_analysis = pd.DataFrame([
        {
            "game_id": f"g_{i}",
            "cp_loss": 120 if i % 3 == 0 else 20,
            "clock_ratio": 0.5,
            "clock_before_move": 90.0,
            "base_s": 180,
            "phase": 0.0 if i % 2 == 0 else 1.0,
        }
        for i in range(20)
    ])
    df_analysis.to_parquet(p_dir / "analysis_train.parquet")


def test_dashboard_api_unauthorized(client: TestClient):
    """Verify GET /api/bots/{username}/dashboard returns 401 without auth."""
    resp = client.get("/api/bots/the_king_crusher/dashboard")
    assert resp.status_code == 401


def test_dashboard_api_authorized(client: TestClient):
    """Verify GET /api/bots/{username}/dashboard returns 200 with valid mocked auth."""
    _ensure_king_crusher_fixture()
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
