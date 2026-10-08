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


@pytest.mark.asyncio
async def test_retries_used_in_status_and_info_json(tmp_path):
    """T9 retries_used exposure: appears in status payload for completed and failed path,
    and in info.json after build_dataset."""
    import httpx
    import json
    from app.routers.ingest import _run_ingest, _jobs
    import app.config as config_module

    orig_data_root = config_module.DATA_ROOT
    config_module.DATA_ROOT = tmp_path

    try:
        # 1. Failed path (network error mid-stream)
        class FailingStream:
            def __init__(self):
                self.retries_used = 3

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise httpx.ReadError("Connection lost after 3 retries")

        with patch("app.routers.ingest.stream_games", return_value=FailingStream()):
            _jobs["user_fail"] = {"status": "queued"}
            await _run_ingest("user_fail")

        assert _jobs["user_fail"]["status"] == "failed"
        assert _jobs["user_fail"]["retries_used"] == 3
        assert _jobs["user_fail"]["drop_stats"] is None

        # 2. Completed path
        valid_pgn = (
            '[Event "Test"]\n[White "user_ok"]\n[Black "opp"]\n'
            '[UTCDate "2024.01.01"]\n[UTCTime "12:00:00"]\n[TimeControl "180+2"]\n\n'
            '1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 1-0'
        )
        lines = [json.dumps({"id": f"g_{i}", "pgn": valid_pgn}) for i in range(200)]

        class SuccessStream:
            def __init__(self):
                self.retries_used = 1
                self.idx = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.idx < len(lines):
                    l = lines[self.idx]
                    self.idx += 1
                    return l
                raise StopAsyncIteration

        with patch("app.routers.ingest.stream_games", return_value=SuccessStream()):
            _jobs["user_ok"] = {"status": "queued"}
            await _run_ingest("user_ok")

        assert _jobs["user_ok"]["status"] == "completed"
        assert _jobs["user_ok"]["retries_used"] == 1

        # Check info.json on disk
        from app.services.storage import player_dir
        info_path = player_dir("user_ok") / "info.json"
        assert info_path.exists()
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        assert info["retries_used"] == 1

        # Check GET /api/status endpoint returns retries_used
        resp = client.get("/api/status/user_ok")
        assert resp.status_code == 200
        assert resp.json()["retries_used"] == 1

    finally:
        config_module.DATA_ROOT = orig_data_root


@pytest.mark.asyncio
async def test_ingest_stats_fallback_and_drop_stats_retention(tmp_path):
    """T14 IngestStats & fallback: failure path retains drop_stats,
    and GET /api/status reads ingest_info.json returning raw_lines, requests, etc."""
    from app.routers.ingest import _run_ingest, _jobs
    from app.services.ingest_stats import IngestStats
    import app.config as config_module
    from app.services.storage import player_dir
    import json

    orig_data_root = config_module.DATA_ROOT
    config_module.DATA_ROOT = tmp_path

    try:
        # Failure path with some drop_stats collected
        class DropStream:
            def __init__(self):
                self.requests = 1
                self.retries_used = 0
                self.yielded = False
            def __aiter__(self):
                return self
            async def __anext__(self):
                if not self.yielded:
                    self.yielded = True
                    return '{"id": "bad", "pgn": ""}'
                raise httpx.ReadError("Broken mid stream")

        with patch("app.routers.ingest.stream_games", return_value=DropStream()):
            _jobs["fail_user"] = {"status": "queued"}
            await _run_ingest("fail_user")

        assert _jobs["fail_user"]["status"] == "failed"
        assert _jobs["fail_user"]["drop_stats"] is not None
        assert _jobs["fail_user"]["drop_stats"]["no_pgn"] == 1

        # Fallback reading ingest_info.json after clearing _jobs
        user = "saved_user"
        pdir = player_dir(user)
        pdir.mkdir(parents=True, exist_ok=True)
        stats = IngestStats(raw_lines=150, requests=2, retries_used=1, drop_stats={"too_short": 5})
        with open(pdir / "ingest_info.json", "w", encoding="utf-8") as f:
            json.dump(stats.to_dict(), f)

        _jobs.pop(user, None)
        resp = client.get(f"/api/status/{user}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["raw_lines"] == 150
        assert data["requests"] == 2
        assert data["retries_used"] == 1
        assert data["drop_stats"] == {"too_short": 5}
    finally:
        config_module.DATA_ROOT = orig_data_root


@pytest.mark.asyncio
async def test_router_resume_overlap_and_slack(tmp_path):
    """T_A2: Router-level test through real _run_ingest with MockTransport.
    ReadError after N games -> resume request has until == last_created_at + OVERLAP_MS
    and max == remaining + RESUME_SLACK.
    """
    import httpx
    import json
    from unittest.mock import AsyncMock, patch
    from app.routers.ingest import _run_ingest, _jobs
    import app.config as config_module

    orig_data_root = config_module.DATA_ROOT
    config_module.DATA_ROOT = tmp_path

    username = "router_resume_user"
    _jobs[username] = {"status": "queued"}

    n_first = 5
    last_created_at = 1700000050000
    captured_requests = []
    attempt = 0

    valid_pgn = (
        f'[Event "Test"]\n[White "{username}"]\n[Black "opp"]\n'
        '[UTCDate "2024.01.01"]\n[UTCTime "12:00:00"]\n[TimeControl "180+2"]\n\n'
        '1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 1-0'
    )

    class PartialStream(httpx.AsyncByteStream):
        def __init__(self, lines):
            self._lines = lines

        async def __aiter__(self):
            for l in self._lines:
                yield (l + "\n").encode()
            raise httpx.ReadError("Drop after N games")

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempt
        attempt += 1
        captured_requests.append(request)
        if attempt == 1:
            first_lines = [
                json.dumps({"id": f"g_first_{i}", "createdAt": last_created_at - (n_first - 1 - i) * 1000, "pgn": valid_pgn})
                for i in range(n_first)
            ]
            return httpx.Response(200, headers={"content-type": "application/x-ndjson"}, stream=PartialStream(first_lines))
        else:
            second_lines = [
                json.dumps({"id": f"g_second_{i}", "createdAt": last_created_at - 10000 - i * 1000, "pgn": valid_pgn})
                for i in range(160)
            ]
            body = ("\n".join(second_lines) + "\n").encode()
            return httpx.Response(200, headers={"content-type": "application/x-ndjson"}, stream=httpx.ByteStream(body))

    transport = httpx.MockTransport(handler)
    original_client_init = httpx.AsyncClient.__init__

    def patched_init(self, **kwargs):
        kwargs["transport"] = transport
        original_client_init(self, **kwargs)

    try:
        with patch.object(httpx.AsyncClient, "__init__", patched_init), \
             patch("asyncio.sleep", AsyncMock()):
            await _run_ingest(username)

        assert len(captured_requests) >= 2
        req2_params = captured_requests[1].url.params
        assert int(req2_params["until"]) == last_created_at + config_module.OVERLAP_MS
        remaining = config_module.PER_BUCKET_FETCH_MAX - n_first
        assert int(req2_params["max"]) == remaining + config_module.RESUME_SLACK
        assert _jobs[username]["status"] == "completed"
    finally:
        config_module.DATA_ROOT = orig_data_root


