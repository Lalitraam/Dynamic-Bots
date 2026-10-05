import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app
from app import config

client = TestClient(app)

def test_ingest_status_flow():
    """Test start ingest and status retrieval."""
    username = "test_player_ingest"

    async def mock_stream_games(uname, max_games=config.DEFAULT_MAX_GAMES):
        yield '{"id": "g1", "pgn": "[White \\"test_player_ingest\\"]\\n[Black \\"opponent\\"]\\n[UTCDate \\"2024.01.01\\"]\\n[UTCTime \\"12:00:00\\"]\\n[TimeControl \\"180+2\\"]\\n\\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0"}'

    with patch("app.routers.ingest.stream_games", side_effect=mock_stream_games):
        # Start job
        resp = client.post(f"/api/ingest/{username}")
        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "queued"

        # Check status endpoint
        status_resp = client.get(f"/api/status/{username}")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert status_data["username"] == username
        assert "status" in status_data

def test_ingest_conflict_409():
    """Test POST /api/ingest returns 409 if job is active."""
    username = "active_player"
    from app.routers.ingest import _jobs
    _jobs[username] = {"status": "downloading", "username": username}

    resp = client.post(f"/api/ingest/{username}")
    assert resp.status_code == 409
