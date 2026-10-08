import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.downloader import stream_games
from app import config
import json
from datetime import datetime, timezone

@pytest.mark.asyncio
async def test_downloader_404_raises():
    """Test that HTTP 404 raises ValueError with user not found message."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, request=request)

    transport = httpx.MockTransport(handler)
    original_client_init = httpx.AsyncClient.__init__

    def patched_init(self, **kwargs):
        kwargs["transport"] = transport
        original_client_init(self, **kwargs)

    with patch.object(httpx.AsyncClient, "__init__", patched_init):
        with pytest.raises(ValueError, match="User not found"):
            async for _ in stream_games("nonexistent_user_xyz"):
                pass


@pytest.mark.asyncio
async def test_downloader_403_raises():
    """Test that HTTP 403 raises ValueError with access forbidden message."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, request=request)

    transport = httpx.MockTransport(handler)
    original_client_init = httpx.AsyncClient.__init__

    def patched_init(self, **kwargs):
        kwargs["transport"] = transport
        original_client_init(self, **kwargs)

    with patch.object(httpx.AsyncClient, "__init__", patched_init):
        with pytest.raises(ValueError, match="Access forbidden"):
            async for _ in stream_games("forbidden_user"):
                pass


@pytest.mark.asyncio
async def test_downloader_429_retries_and_fails():
    """Test that HTTP 429 retries 5 times before raising HTTPStatusError."""
    req_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal req_count
        req_count += 1
        return httpx.Response(429, request=request)

    transport = httpx.MockTransport(handler)
    original_client_init = httpx.AsyncClient.__init__

    def patched_init(self, **kwargs):
        kwargs["transport"] = transport
        original_client_init(self, **kwargs)

    sleep_mock = AsyncMock()

    with patch.object(httpx.AsyncClient, "__init__", patched_init):
        with pytest.raises(httpx.HTTPStatusError):
            async for _ in stream_games("rate_limited_user", sleep_fn=sleep_mock):
                pass

    assert req_count == 5
    assert sleep_mock.call_count == 4


@pytest.mark.asyncio
async def test_downloader_request_contains_clocks_true():
    """T1 clocks: assert via captured_request.url.params['clocks'] == 'true'."""
    captured_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/x-ndjson"},
            stream=httpx.ByteStream(b""),
        )

    transport = httpx.MockTransport(handler)
    original_client_init = httpx.AsyncClient.__init__

    def patched_init(self, **kwargs):
        kwargs["transport"] = transport
        original_client_init(self, **kwargs)

    with patch.object(httpx.AsyncClient, "__init__", patched_init):
        async for _ in stream_games("testuser", max_games=10):
            pass

    assert len(captured_requests) > 0
    assert captured_requests[0].url.params["clocks"] == "true"


@pytest.mark.asyncio
async def test_downloader_resume_on_retry():
    """T2 resume: MockTransport honours until (inclusive) and max.
    Assert exactly 50 unique ids yielded, none repeated;
    2nd request params until == ts of game 19 and max == 31; retries_used == 1.
    Parse request.url.params; no substring checks."""
    games_50 = []
    base_ts = 1700000000000
    for i in range(50):
        ts = base_ts - i * 1000
        games_50.append({"id": f"game_{i:02d}", "createdAt": ts, "pgn": "1. e4 e5"})

    attempt = 0
    captured_requests = []

    class PartialStream(httpx.AsyncByteStream):
        def __init__(self, lines):
            self._lines = lines

        async def __aiter__(self):
            for l in self._lines:
                yield (l + "\n").encode()
            raise httpx.ReadError("Simulated network drop mid-stream")

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempt
        attempt += 1
        captured_requests.append(request)
        params = request.url.params
        until = int(params["until"]) if "until" in params else None
        max_param = int(params["max"]) if "max" in params else 50

        # honour until (inclusive) and max
        filtered = [g for g in games_50 if until is None or g["createdAt"] <= until]
        capped = filtered[:max_param]

        if attempt == 1:
            return httpx.Response(
                200,
                headers={"content-type": "application/x-ndjson"},
                stream=PartialStream([json.dumps(g) for g in capped[:20]]),
            )
        else:
            body = ("\n".join([json.dumps(g) for g in capped]) + "\n").encode()
            return httpx.Response(
                200,
                headers={"content-type": "application/x-ndjson"},
                stream=httpx.ByteStream(body),
            )

    transport = httpx.MockTransport(handler)
    original_client_init = httpx.AsyncClient.__init__

    def patched_init(self, **kwargs):
        kwargs["transport"] = transport
        original_client_init(self, **kwargs)

    yielded_lines = []
    with patch.object(httpx.AsyncClient, "__init__", patched_init), \
         patch("asyncio.sleep", AsyncMock()):
        stream = stream_games("testuser", max_games=50, windowed=False)
        async for line in stream:
            yielded_lines.append(line)

    yielded_ids = [json.loads(l)["id"] for l in yielded_lines]
    assert len(yielded_ids) == 50
    assert len(set(yielded_ids)) == 50
    assert len(captured_requests) == 2

    req2_params = captured_requests[1].url.params
    expected_until = str(games_50[19]["createdAt"])
    assert req2_params["until"] == expected_until
    assert req2_params["max"] == "31"
    assert getattr(stream, "retries_used", None) == 1


@pytest.mark.asyncio
async def test_downloader_fail_before_any_line():
    """T3 fail-before-any-line: stream errors before yielding anything ->
    retry request params are identical to the first request (no until, same max)."""
    attempt = 0
    captured_requests = []

    class FailImmediateStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            if False:
                yield b""
            raise httpx.ReadError("Immediate connection drop")

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempt
        attempt += 1
        captured_requests.append(request)
        if attempt == 1:
            return httpx.Response(
                200,
                headers={"content-type": "application/x-ndjson"},
                stream=FailImmediateStream(),
            )
        else:
            return httpx.Response(
                200,
                headers={"content-type": "application/x-ndjson"},
                stream=httpx.ByteStream(b'{"id": "g1", "createdAt": 1700000000000}\n'),
            )

    transport = httpx.MockTransport(handler)
    original_client_init = httpx.AsyncClient.__init__

    def patched_init(self, **kwargs):
        kwargs["transport"] = transport
        original_client_init(self, **kwargs)

    with patch.object(httpx.AsyncClient, "__init__", patched_init), \
         patch("asyncio.sleep", AsyncMock()):
        stream = stream_games("testuser", max_games=50, windowed=False)
        async for _ in stream:
            pass

    assert len(captured_requests) == 2
    req1_params = captured_requests[0].url.params
    req2_params = captured_requests[1].url.params
    assert "until" not in req2_params
    assert req2_params["max"] == req1_params["max"] == "50"
    assert getattr(stream, "retries_used", None) == 1


@pytest.mark.asyncio
async def test_downloader_warning_log_on_retry(caplog):
    """T4 WARNING log: use caplog; assert one WARNING per retry containing
    attempt number, exception class name, lines yielded, resume timestamp."""
    import logging
    attempt = 0

    class PartialStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"id": "g1", "createdAt": 1700000000000}\n'
            raise httpx.ReadError("TCP reset")

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            return httpx.Response(200, headers={"content-type": "application/x-ndjson"}, stream=PartialStream())
        else:
            return httpx.Response(200, headers={"content-type": "application/x-ndjson"}, stream=httpx.ByteStream(b'{"id": "g2"}\n'))

    transport = httpx.MockTransport(handler)
    original_client_init = httpx.AsyncClient.__init__

    def patched_init(self, **kwargs):
        kwargs["transport"] = transport
        original_client_init(self, **kwargs)

    with caplog.at_level(logging.WARNING), \
         patch.object(httpx.AsyncClient, "__init__", patched_init), \
         patch("asyncio.sleep", AsyncMock()):
        stream = stream_games("testuser", max_games=10)
        async for _ in stream:
            pass

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_records) == 1
    log_text = warning_records[0].message
    assert "1" in log_text
    assert "ReadError" in log_text
    assert "1" in log_text
    assert "1700000000000" in log_text


@pytest.mark.asyncio
async def test_downloader_missing_created_at_fallback():
    """T5 missing createdAt: lines have no createdAt, ReadError after 20 ->
    mock honours max; result is still 50 unique ids, no silent replay."""
    lines_50 = [json.dumps({"id": f"game_{i:02d}", "pgn": "1. e4 e5"}) for i in range(50)]
    attempt = 0
    captured_requests = []

    class PartialStream(httpx.AsyncByteStream):
        def __init__(self, lines):
            self._lines = lines

        async def __aiter__(self):
            for l in self._lines:
                yield (l + "\n").encode()
            raise httpx.ReadError("Drop after 20")

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempt
        attempt += 1
        captured_requests.append(request)
        max_param = int(request.url.params.get("max", "50"))
        served_lines = lines_50[:max_param]
        if attempt == 1:
            return httpx.Response(200, headers={"content-type": "application/x-ndjson"}, stream=PartialStream(served_lines[:20]))
        else:
            body = ("\n".join(served_lines) + "\n").encode()
            return httpx.Response(200, headers={"content-type": "application/x-ndjson"}, stream=httpx.ByteStream(body))

    transport = httpx.MockTransport(handler)
    original_client_init = httpx.AsyncClient.__init__

    def patched_init(self, **kwargs):
        kwargs["transport"] = transport
        original_client_init(self, **kwargs)

    yielded = []
    with patch.object(httpx.AsyncClient, "__init__", patched_init), \
         patch("asyncio.sleep", AsyncMock()):
        stream = stream_games("testuser", max_games=50, windowed=False)
        async for line in stream:
            yielded.append(line)

    yielded_ids = [json.loads(l)["id"] for l in yielded]
    assert len(yielded_ids) == 50
    assert len(set(yielded_ids)) == 50
    assert len(captured_requests) == 2
    assert captured_requests[1].url.params["max"] == "50"


@pytest.mark.asyncio
async def test_downloader_request_contains_since():
    """T8 since: inject clock_fn with fixed datetime; HARD-CODE expected millisecond
    value as literal number and assert exact equality."""
    captured_requests = []
    fixed_now = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/x-ndjson"},
            stream=httpx.ByteStream(b""),
        )

    transport = httpx.MockTransport(handler)
    original_client_init = httpx.AsyncClient.__init__

    def patched_init(self, **kwargs):
        kwargs["transport"] = transport
        original_client_init(self, **kwargs)

    with patch.object(httpx.AsyncClient, "__init__", patched_init):
        stream = stream_games("testuser", max_games=10, clock_fn=lambda: fixed_now)
        async for _ in stream:
            pass

    assert len(captured_requests) == 3
    # Literal 1654171200000 = 2025-06-01 12:00:00 UTC minus 1095 days in milliseconds (bucket 18_36m)
    assert captured_requests[2].url.params["since"] == "1654171200000"
    # Literal 1733227200000 = 2025-06-01 12:00:00 UTC minus 180 days in milliseconds (bucket last_6m)
    assert captured_requests[0].url.params["since"] == "1733227200000"


@pytest.mark.asyncio
async def test_downloader_until_ms_and_resume():
    """T10 window until_ms resume: when until_ms is set, retry resumes from
    tighter until bound min(until_ms, last_created_at) and reduced max."""
    initial_until_ms = 1700000000000
    games_50 = []
    for i in range(50):
        ts = 1699999000000 - i * 1000  # all < initial_until_ms
        games_50.append({"id": f"win_g_{i:02d}", "createdAt": ts, "pgn": "1. e4 e5"})

    attempt = 0
    captured_requests = []

    class PartialStream(httpx.AsyncByteStream):
        def __init__(self, lines):
            self._lines = lines

        async def __aiter__(self):
            for l in self._lines:
                yield (l + "\n").encode()
            raise httpx.ReadError("Drop mid window")

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempt
        attempt += 1
        captured_requests.append(request)
        params = request.url.params
        until = int(params["until"]) if "until" in params else None
        max_param = int(params["max"]) if "max" in params else 50
        filtered = [g for g in games_50 if until is None or g["createdAt"] <= until]
        capped = filtered[:max_param]
        if attempt == 1:
            return httpx.Response(200, headers={"content-type": "application/x-ndjson"}, stream=PartialStream([json.dumps(g) for g in capped[:20]]))
        else:
            body = ("\n".join([json.dumps(g) for g in capped]) + "\n").encode()
            return httpx.Response(200, headers={"content-type": "application/x-ndjson"}, stream=httpx.ByteStream(body))

    transport = httpx.MockTransport(handler)
    original_client_init = httpx.AsyncClient.__init__

    def patched_init(self, **kwargs):
        kwargs["transport"] = transport
        original_client_init(self, **kwargs)

    with patch.object(httpx.AsyncClient, "__init__", patched_init), \
         patch("asyncio.sleep", AsyncMock()):
        stream = stream_games("testuser", max_games=50, until_ms=initial_until_ms)
        yielded = [l async for l in stream]

    assert len(captured_requests) == 2
    req1_params = captured_requests[0].url.params
    req2_params = captured_requests[1].url.params
    assert req1_params["until"] == str(initial_until_ms)
    expected_resume_until = games_50[19]["createdAt"]
    assert req2_params["until"] == str(expected_resume_until)
    assert int(req2_params["max"]) <= 31
    assert len(yielded) == 50


@pytest.mark.asyncio
async def test_downloader_429_waits_at_least_rate_limit_wait_seconds():
    """T11 429 wait: HTTP 429 waits at least RATE_LIMIT_WAIT_SECONDS (60s),
    while ReadError uses standard exponential backoff (< 60s)."""
    # 1. ReadError wait must be < 60.0 (uses standard exponential backoff)
    re_attempt = 0
    re_sleep_durations = []

    async def record_re_sleep(dur):
        re_sleep_durations.append(dur)

    def re_handler(request: httpx.Request) -> httpx.Response:
        nonlocal re_attempt
        re_attempt += 1
        if re_attempt == 1:
            raise httpx.ReadError("Transport glitch")
        else:
            return httpx.Response(200, headers={"content-type": "application/x-ndjson"}, stream=httpx.ByteStream(b'{"id": "g1"}\n'))

    re_transport = httpx.MockTransport(re_handler)
    original_client_init = httpx.AsyncClient.__init__

    def patched_re_init(self, **kwargs):
        kwargs["transport"] = re_transport
        original_client_init(self, **kwargs)

    with patch.object(httpx.AsyncClient, "__init__", patched_re_init):
        stream = stream_games("testuser", max_games=10, sleep_fn=record_re_sleep)
        yielded_re = [l async for l in stream]

    assert len(re_sleep_durations) == 1
    assert re_sleep_durations[0] < 60.0
    assert len(yielded_re) == 1

    # 2. HTTP 429 wait must be >= 60.0 (RATE_LIMIT_WAIT_SECONDS)
    attempt = 0
    sleep_durations = []

    async def record_sleep(dur):
        sleep_durations.append(dur)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            return httpx.Response(429, request=request)
        else:
            return httpx.Response(200, headers={"content-type": "application/x-ndjson"}, stream=httpx.ByteStream(b'{"id": "g1"}\n'))

    transport = httpx.MockTransport(handler)

    def patched_init(self, **kwargs):
        kwargs["transport"] = transport
        original_client_init(self, **kwargs)

    with patch.object(httpx.AsyncClient, "__init__", patched_init):
        stream = stream_games("testuser", max_games=10, sleep_fn=record_sleep)
        yielded = [l async for l in stream]

    assert len(sleep_durations) == 1
    assert sleep_durations[0] >= 60.0
    assert len(yielded) == 1


@pytest.mark.asyncio
async def test_downloader_non_monotonic_stream():
    """T12 non-monotonic stream: stream has an inversion (100, 90, 95),
    ReadError after line 2 (90) -> safe resume delivers the 95 game too."""
    games = [
        {"id": "g_100", "createdAt": 1700000100000, "pgn": "1. e4 e5"},
        {"id": "g_90", "createdAt": 1700000090000, "pgn": "1. e4 e5"},
        {"id": "g_95", "createdAt": 1700000095000, "pgn": "1. e4 e5"},
        {"id": "g_80", "createdAt": 1700000080000, "pgn": "1. e4 e5"},
        {"id": "g_70", "createdAt": 1700000070000, "pgn": "1. e4 e5"},
    ]
    attempt = 0

    class PartialStream(httpx.AsyncByteStream):
        def __init__(self, items):
            self._items = items
        async def __aiter__(self):
            for it in self._items:
                yield (json.dumps(it) + "\n").encode()
            raise httpx.ReadError("Drop after g_90")

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempt
        attempt += 1
        params = request.url.params
        until = int(params["until"]) if "until" in params else None
        max_param = int(params.get("max", 5))
        served = [g for g in games if until is None or g["createdAt"] <= until][:max_param]
        if attempt == 1:
            return httpx.Response(200, headers={"content-type": "application/x-ndjson"}, stream=PartialStream(served[:2]))
        else:
            body = ("\n".join([json.dumps(g) for g in served]) + "\n").encode()
            return httpx.Response(200, headers={"content-type": "application/x-ndjson"}, stream=httpx.ByteStream(body))

    transport = httpx.MockTransport(handler)
    original_client_init = httpx.AsyncClient.__init__

    def patched_init(self, **kwargs):
        kwargs["transport"] = transport
        original_client_init(self, **kwargs)

    with patch.object(httpx.AsyncClient, "__init__", patched_init), \
         patch("asyncio.sleep", AsyncMock()):
        stream = stream_games("testuser", max_games=5, overlap_ms=config.OVERLAP_MS)
        yielded = [l async for l in stream]

    yielded_ids = [json.loads(l)["id"] for l in yielded]
    assert "g_95" in yielded_ids
    assert len(yielded_ids) == 5


def test_downloader_perftype_deterministic():
    """T_A1: the perfType string is 'blitz,bullet,rapid' on repeated calls."""
    from app.services.downloader import _perf_type_param
    for _ in range(10):
        assert _perf_type_param() == "blitz,bullet,rapid"


@pytest.mark.asyncio
async def test_downloader_skipped_duplicates_on_resume():
    """T_A3: 50 games, ReadError after 20, overlap re-sends 3 already-yielded games ->
    skipped_duplicates == 3, yielded ids unique, parser duplicate == 0.
    """
    from app.services.ingest_stats import IngestStats
    from app.services.parser import parse_stream_into_buckets
    from datetime import datetime, timezone, timedelta

    target = "ann"
    opponent = "joanna"
    rel_date = (datetime.now(timezone.utc) - timedelta(days=10)).strftime("%Y.%m.%d")
    valid_pgn = (
        f'[Event "Test"]\n[White "{target}"]\n[Black "{opponent}"]\n'
        f'[UTCDate "{rel_date}"]\n[UTCTime "12:00:00"]\n[TimeControl "180+2"]\n\n'
        f'1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 1-0'
    )

    all_games = [
        {"id": f"game_{i:02d}", "createdAt": 1700000000000 - i * 1000, "pgn": valid_pgn}
        for i in range(50)
    ]

    attempt = 0

    class PartialStream(httpx.AsyncByteStream):
        def __init__(self, items):
            self._items = items

        async def __aiter__(self):
            for item in self._items:
                yield (json.dumps(item) + "\n").encode()
            raise httpx.ReadError("Drop after 20 games")

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            # First request: yields games 0..19 (20 games) then ReadError
            return httpx.Response(200, headers={"content-type": "application/x-ndjson"}, stream=PartialStream(all_games[:20]))
        else:
            # Second request: overlap re-sends 3 already-yielded games (17, 18, 19) + remaining (20..49)
            resend_games = all_games[17:20] + all_games[20:]
            body = ("\n".join([json.dumps(g) for g in resend_games]) + "\n").encode()
            return httpx.Response(200, headers={"content-type": "application/x-ndjson"}, stream=httpx.ByteStream(body))

    transport = httpx.MockTransport(handler)
    original_client_init = httpx.AsyncClient.__init__

    def patched_init(self, **kwargs):
        kwargs["transport"] = transport
        original_client_init(self, **kwargs)

    stats = IngestStats()
    with patch.object(httpx.AsyncClient, "__init__", patched_init), \
         patch("asyncio.sleep", AsyncMock()):
        stream = stream_games(target, max_games=50, stats=stats)
        buckets, total_valid, drop_stats = await parse_stream_into_buckets(
            stream, target, stats=stats
        )

    assert stats.skipped_duplicates == 3
    assert total_valid == 50
    assert drop_stats["duplicate"] == 0


@pytest.mark.asyncio
async def test_downloader_request_log():
    """T_A5: IngestStats.request_log records {bucket, status_code, lines, since, until, max}
    appended per HTTP request by the downloader via MockTransport.
    """
    from app.services.ingest_stats import IngestStats

    attempt = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            return httpx.Response(429, request=request)
        else:
            lines = [json.dumps({"id": f"g_{i}", "createdAt": 1700000000000}) for i in range(2)]
            body = ("\n".join(lines) + "\n").encode()
            return httpx.Response(200, headers={"content-type": "application/x-ndjson"}, stream=httpx.ByteStream(body))

    transport = httpx.MockTransport(handler)
    original_client_init = httpx.AsyncClient.__init__

    def patched_init(self, **kwargs):
        kwargs["transport"] = transport
        original_client_init(self, **kwargs)

    stats = IngestStats()
    with patch.object(httpx.AsyncClient, "__init__", patched_init), \
         patch("asyncio.sleep", AsyncMock()):
        stream = stream_games("testuser", max_games=2, stats=stats, windowed=False)
        yielded = [l async for l in stream]

    assert len(yielded) == 2
    assert len(stats.request_log) == 2
    entry0 = stats.request_log[0]
    assert entry0["bucket"] == "all"
    assert entry0["status_code"] == 429
    assert entry0["lines"] == 0
    assert entry0["max"] == 2
    assert entry0["cap_hit"] is False

    entry1 = stats.request_log[1]
    assert entry1["bucket"] == "all"
    assert entry1["status_code"] == 200
    assert entry1["lines"] == 2
    assert entry1["max"] == 2
    assert entry1["cap_hit"] is True


@pytest.mark.asyncio
async def test_phase_b_three_bucket_requests_expected_windows_and_per_bucket_max():
    """Phase B: exactly 3 requests with the expected since/until values,
    non-overlapping except at shared boundary timestamps, and per-bucket max == 1500.
    """
    from app.services.ingest_stats import IngestStats

    captured_requests = []
    fixed_now = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/x-ndjson"},
            stream=httpx.ByteStream(b""),
        )

    transport = httpx.MockTransport(handler)
    original_client_init = httpx.AsyncClient.__init__

    def patched_init(self, **kwargs):
        kwargs["transport"] = transport
        original_client_init(self, **kwargs)

    stats = IngestStats()
    with patch.object(httpx.AsyncClient, "__init__", patched_init):
        stream = stream_games("testuser", now=fixed_now, stats=stats)
        async for _ in stream:
            pass

    assert len(captured_requests) == 3
    # Bucket 0: last_6m -> [now-180d, now]
    req0 = captured_requests[0]
    assert req0.url.params["since"] == "1733227200000"
    assert req0.url.params["until"] == "1748779200000"
    assert req0.url.params["max"] == "1500"

    # Bucket 1: 6_18m -> [now-540d, now-180d]
    req1 = captured_requests[1]
    assert req1.url.params["since"] == "1702123200000"
    assert req1.url.params["until"] == "1733227200000"
    assert req1.url.params["max"] == "1500"

    # Bucket 2: 18_36m -> [now-1095d, now-540d]
    req2 = captured_requests[2]
    assert req2.url.params["since"] == "1654171200000"
    assert req2.url.params["until"] == "1702123200000"
    assert req2.url.params["max"] == "1500"

    # Non-overlapping except shared boundaries
    assert int(req0.url.params["since"]) == int(req1.url.params["until"]) == 1733227200000
    assert int(req1.url.params["since"]) == int(req2.url.params["until"]) == 1702123200000
    assert int(req0.url.params["since"]) < int(req0.url.params["until"])
    assert int(req1.url.params["since"]) < int(req1.url.params["until"])
    assert int(req2.url.params["since"]) < int(req2.url.params["until"])

    # IngestStats verification
    assert stats.bucket_requests == {"last_6m": 1, "6_18m": 1, "18_36m": 1}
    assert [entry["bucket"] for entry in stats.request_log] == ["last_6m", "6_18m", "18_36m"]


@pytest.mark.asyncio
async def test_phase_b_boundary_game_counted_once_and_raw_line_accounting():
    """Phase B: boundary game present in two windows -> counted once.
    Raw-line accounting holds:
      lines received from network == lines yielded + skipped_duplicates
      and valid + sum(drop_stats) == lines yielded to the parser.
    """
    from app.services.ingest_stats import IngestStats
    from app.services.parser import parse_stream_into_buckets

    fixed_now = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    target = "the_king_crusher"

    def make_game_line(gid: str, ts_ms: int, date_str: str) -> str:
        pgn = (
            f'[Event "Rated Blitz game"]\n[White "{target}"]\n[Black "Opponent"]\n'
            f'[UTCDate "{date_str}"]\n[UTCTime "12:00:00"]\n[TimeControl "180+2"]\n\n'
            f'1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 1-0'
        )
        return json.dumps({"id": gid, "createdAt": ts_ms, "pgn": pgn})

    # Shared boundary timestamp: 1733227200000 (2024-12-03)
    boundary_game = make_game_line("boundary_game_123", 1733227200000, "2024.12.03")
    recent_game = make_game_line("recent_game_456", 1740000000000, "2025.02.19")
    mid_game = make_game_line("mid_game_789", 1715000000000, "2024.05.08")
    old_game = make_game_line("old_game_999", 1680000000000, "2023.03.29")

    attempt = 0
    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            # last_6m request: returns recent_game + boundary_game
            lines = [recent_game, boundary_game]
        elif attempt == 2:
            # 6_18m request: returns boundary_game again (boundary duplicate) + mid_game
            lines = [boundary_game, mid_game]
        else:
            # 18_36m request: returns old_game
            lines = [old_game]
        body = ("\n".join(lines) + "\n").encode()
        return httpx.Response(200, headers={"content-type": "application/x-ndjson"}, stream=httpx.ByteStream(body))

    transport = httpx.MockTransport(handler)
    original_client_init = httpx.AsyncClient.__init__

    def patched_init(self, **kwargs):
        kwargs["transport"] = transport
        original_client_init(self, **kwargs)

    stats = IngestStats()
    with patch.object(httpx.AsyncClient, "__init__", patched_init):
        stream = stream_games(target, now=fixed_now, stats=stats)
        buckets, total_valid, drop_stats = await parse_stream_into_buckets(
            stream, target, now=fixed_now, stats=stats
        )

    # Boundary game appears twice across requests, so skipped_duplicates == 1
    assert stats.skipped_duplicates == 1
    # 5 total lines from network (2 + 2 + 1)
    assert stats.raw_lines == 5

    # 4 distinct games yielded to parser
    lines_yielded_to_parser = total_valid + sum(drop_stats.values())
    assert lines_yielded_to_parser == 4
    assert total_valid == 4
    assert drop_stats["duplicate"] == 0

    # Invariants from B3
    assert stats.raw_lines == lines_yielded_to_parser + stats.skipped_duplicates
    assert total_valid + sum(drop_stats.values()) == lines_yielded_to_parser


@pytest.mark.asyncio
async def test_phase_b_readerror_mid_bucket_resumes_within_bucket_window():
    """Phase B: ReadError mid-bucket resumes within that bucket's window only,
    tightening until to min(b_until, last_created_at + overlap_ms) and setting
    max to remaining + RESUME_SLACK.
    """
    from app.services.ingest_stats import IngestStats

    fixed_now = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    captured_requests = []
    attempt = 0

    class PartialStream(httpx.AsyncByteStream):
        def __init__(self, lines):
            self._lines = lines

        async def __aiter__(self):
            for l in self._lines:
                yield (l + "\n").encode()
            raise httpx.ReadError("Mid-bucket network drop in 6_18m")

    # Games for bucket 6_18m
    g1 = json.dumps({"id": "g1", "createdAt": 1720000000000})
    g2 = json.dumps({"id": "g2", "createdAt": 1715000000000})
    g3 = json.dumps({"id": "g3", "createdAt": 1710000000000})

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempt
        attempt += 1
        captured_requests.append(request)
        if attempt == 1:
            # last_6m: succeeds immediately with empty stream
            return httpx.Response(200, headers={"content-type": "application/x-ndjson"}, stream=httpx.ByteStream(b""))
        elif attempt == 2:
            # 6_18m first try: yields g1 and g2, then drops
            return httpx.Response(200, headers={"content-type": "application/x-ndjson"}, stream=PartialStream([g1, g2]))
        elif attempt == 3:
            # 6_18m retry: yields g3
            return httpx.Response(200, headers={"content-type": "application/x-ndjson"}, stream=httpx.ByteStream((g3 + "\n").encode()))
        else:
            # 18_36m: succeeds immediately with empty stream
            return httpx.Response(200, headers={"content-type": "application/x-ndjson"}, stream=httpx.ByteStream(b""))

    transport = httpx.MockTransport(handler)
    original_client_init = httpx.AsyncClient.__init__

    def patched_init(self, **kwargs):
        kwargs["transport"] = transport
        original_client_init(self, **kwargs)

    stats = IngestStats()
    with patch.object(httpx.AsyncClient, "__init__", patched_init), \
         patch("asyncio.sleep", AsyncMock()):
        stream = stream_games("testuser", now=fixed_now, overlap_ms=config.OVERLAP_MS, stats=stats)
        yielded = [l async for l in stream]

    assert len(yielded) == 3
    assert len(captured_requests) == 4

    # Request 0: last_6m initial
    assert captured_requests[0].url.params["since"] == "1733227200000"
    # Request 1: 6_18m initial
    assert captured_requests[1].url.params["since"] == "1702123200000"
    assert captured_requests[1].url.params["until"] == "1733227200000"
    assert captured_requests[1].url.params["max"] == "1500"

    # Request 2: 6_18m RETRY
    retry_req = captured_requests[2]
    # Window since must still be bucket lower bound
    assert retry_req.url.params["since"] == "1702123200000"
    # resume_until = last_created_at (1715000000000) + overlap_ms (120000) = 1715000120000
    expected_resume_until = 1715000000000 + config.OVERLAP_MS
    assert retry_req.url.params["until"] == str(expected_resume_until)
    # Must NOT exceed bucket upper bound (1733227200000)
    assert int(retry_req.url.params["until"]) <= 1733227200000
    # max = remaining (1500 - 2 = 1498) + RESUME_SLACK (100) = 1598
    assert retry_req.url.params["max"] == "1598"

    # Request 3: 18_36m initial
    assert captured_requests[3].url.params["since"] == "1654171200000"

    assert stats.retries_used == 1
    assert stats.bucket_requests["6_18m"] == 2


def test_phase_b_sampler_fixture_output_byte_identical():
    """Phase B: existing sampler fixture output is byte-identical to before."""
    import hashlib
    from pathlib import Path
    from app.services.sampler import sample_games

    fixture_path = Path(__file__).parent / "fixtures" / "sample_games.jsonl"
    lines = [json.loads(l) for l in fixture_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    buckets = {
        "last_6m": [lines[0]],
        "6_18m": [lines[1]],
        "18_36m": lines[2:],
    }
    result = sample_games(buckets, seed=42)
    serialized = json.dumps(result, sort_keys=True).encode("utf-8")
    result_hash = hashlib.sha256(serialized).hexdigest()

    expected_hash = "833fa6280285e9ff6f9cae3c15027d64f7e97a3ef434ca8bffc1344e8249d140"
    assert result_hash == expected_hash




