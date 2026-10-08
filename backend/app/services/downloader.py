"""
Downloader service for streaming games from the Lichess API.

Streams NDJSON game records from https://lichess.org/api/games/user/{username}.
Each line is a JSON object with a "pgn" field containing the PGN text.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
import httpx
from typing import AsyncIterator, Callable, Optional, Any
from .. import config

logger = logging.getLogger(__name__)

LICHESS_EXPORT_URL = "https://lichess.org/api/games/user/{username}"

# Build the perfType parameter from INCLUDED_CATEGORIES.
# Lichess API names:  ultrabullet, bullet, blitz, rapid, classical
_LICHESS_PERF_MAP = {
    "ultrabullet": "ultraBullet",
    "bullet": "bullet",
    "blitz": "blitz",
    "rapid": "rapid",
    "classical": "classical",
}


def _perf_type_param() -> str:
    """Build the comma-separated perfType value from config.INCLUDED_CATEGORIES."""
    parts = [_LICHESS_PERF_MAP[c] for c in sorted(config.INCLUDED_CATEGORIES)
             if c in _LICHESS_PERF_MAP]
    return ",".join(parts)


class GameStream:
    """
    Async iterator over raw NDJSON lines for *username* from Lichess export API.
    Maintains retries_used, requests, and resumes on network failure using until=<last createdAt>.
    """

    def __init__(
        self,
        username: str,
        max_games: int = config.DEFAULT_MAX_GAMES,
        clock_fn: Callable[[], datetime] | None = None,
        now: Optional[datetime] = None,
        since_ms: Optional[int] = None,
        until_ms: Optional[int] = None,
        sleep_fn: Optional[Callable[[float], Any]] = None,
        overlap_ms: int = 0,
        stats: Any = None,
        bucket: str = "all",
    ):
        self.username = username
        self.max_games = max_games
        self.clock_fn = clock_fn
        self.now = now
        self.since_ms = since_ms
        self.until_ms = until_ms
        self.sleep_fn = sleep_fn
        self.overlap_ms = overlap_ms
        self.stats = stats
        self.bucket = bucket
        self.retries_used = 0
        self.requests = 0
        self._generator = None

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        if self._generator is None:
            self._generator = self._stream()
        return await self._generator.__anext__()

    async def _stream(self) -> AsyncIterator[str]:
        url = LICHESS_EXPORT_URL.format(username=self.username.lower())
        headers = {
            "Accept": "application/x-ndjson",
            "User-Agent": "ChessCloneFactory/1.0 (Educational ML Project)",
        }

        if self.now is not None:
            now_dt = self.now
        elif self.clock_fn is not None:
            now_dt = self.clock_fn()
        else:
            now_dt = datetime.now(timezone.utc)

        now_sec = now_dt.timestamp()

        if self.since_ms is not None or self.until_ms is not None:
            since = self.since_ms if self.since_ms is not None else config.default_since_ms(now_dt)
            bucket_defs = [(self.bucket, since, self.until_ms, self.max_games)]
        elif self.max_games < config.PER_BUCKET_FETCH_MAX and self.now is None and self.clock_fn is None:
            since = config.default_since_ms(now_dt)
            bucket_defs = [("all", since, None, self.max_games)]
        else:
            # Phase B: 3 age buckets sequentially
            # last_6m : [now-180d, now]
            # 6_18m   : [now-540d, now-180d]
            # 18_36m  : [now-1095d, now-540d]
            b1_since = int((now_sec - config.BUCKET_MAX_AGE_6M * 86400) * 1000)
            b1_until = int(now_sec * 1000)

            b2_since = int((now_sec - config.BUCKET_MAX_AGE_18M * 86400) * 1000)
            b2_until = int((now_sec - config.BUCKET_MAX_AGE_6M * 86400) * 1000)

            b3_since = int((now_sec - config.MAX_AGE_DAYS * 86400) * 1000)
            b3_until = int((now_sec - config.BUCKET_MAX_AGE_18M * 86400) * 1000)

            bucket_defs = [
                ("last_6m", b1_since, b1_until, config.PER_BUCKET_FETCH_MAX),
                ("6_18m", b2_since, b2_until, config.PER_BUCKET_FETCH_MAX),
                ("18_36m", b3_since, b3_until, config.PER_BUCKET_FETCH_MAX),
            ]

        base_params: dict[str, str | int | bool] = {
            "clocks": "true",
            "evals": "false",
            "opening": "true",
            "pgnInJson": "true",
            "rated": "true",
            "perfType": _perf_type_param(),
        }

        timeout = httpx.Timeout(30.0, read=120.0)
        max_retries = 5

        yielded_ids: set[str] = set()
        yielded_count = 0

        for b_name, b_since, b_until, b_max in bucket_defs:
            if yielded_count >= self.max_games:
                return

            bucket_yielded_count = 0
            last_created_at: int | None = None

            for attempt in range(1, max_retries + 1):
                if yielded_count >= self.max_games:
                    return

                req_params = dict(base_params)
                req_params["since"] = b_since

                if bucket_yielded_count > 0:
                    if last_created_at is not None:
                        resume_until = last_created_at + self.overlap_ms
                        if b_until is not None:
                            req_params["until"] = min(b_until, resume_until)
                        else:
                            req_params["until"] = resume_until

                        remaining = b_max - bucket_yielded_count
                        slack = config.RESUME_SLACK if self.overlap_ms > 0 else 1
                        req_params["max"] = remaining + slack
                    else:
                        if b_until is not None:
                            req_params["until"] = b_until
                        req_params["max"] = b_max
                        logger.warning(
                            "Resuming download retry without createdAt; falling back to id-based skipping for %d already yielded games",
                            len(yielded_ids),
                        )
                else:
                    if b_until is not None:
                        req_params["until"] = b_until
                    req_params["max"] = b_max

                self.requests += 1
                if self.stats is not None:
                    self.stats.requests = self.requests

                lines_in_request = 0
                status_code = None
                try:
                    try:
                        async with httpx.AsyncClient(timeout=timeout) as client:
                            async with client.stream("GET", url, headers=headers, params=req_params) as response:
                                status_code = response.status_code
                                if response.status_code == 404:
                                    raise ValueError(f"User not found: '{self.username}'")
                                if response.status_code == 403:
                                    raise ValueError(f"Access forbidden: '{self.username}'")
                                response.raise_for_status()

                                async for line in response.aiter_lines():
                                    line = line.strip()
                                    if line:
                                        lines_in_request += 1
                                        if self.stats is not None:
                                            self.stats.raw_lines += 1
                                        gid = None
                                        try:
                                            obj = json.loads(line)
                                            gid = obj.get("id")
                                            ca = obj.get("createdAt")
                                            if ca is not None:
                                                last_created_at = ca
                                        except Exception:
                                            pass

                                        if gid and gid in yielded_ids:
                                            if self.stats is not None:
                                                self.stats.skipped_duplicates += 1
                                            continue

                                        if gid:
                                            yielded_ids.add(gid)

                                        yielded_count += 1
                                        bucket_yielded_count += 1
                                        yield line

                                        if yielded_count >= self.max_games:
                                            return
                                break
                    finally:
                        if self.stats is not None:
                            self.stats.request_log.append({
                                "bucket": b_name,
                                "status_code": status_code,
                                "lines": lines_in_request,
                                "since": req_params.get("since"),
                                "until": req_params.get("until"),
                                "max": req_params.get("max"),
                            })
                            self.stats.bucket_requests[b_name] = self.stats.bucket_requests.get(b_name, 0) + 1
                except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                    if isinstance(exc, httpx.HTTPStatusError):
                        if exc.response.status_code == 404:
                            raise ValueError(f"User not found: '{self.username}'") from exc
                        if exc.response.status_code == 403:
                            raise ValueError(f"Access forbidden: '{self.username}'") from exc
                        if exc.response.status_code != 429 and exc.response.status_code < 500:
                            raise exc
                    if attempt == max_retries:
                        raise exc

                    self.retries_used += 1
                    if self.stats is not None:
                        self.stats.retries_used = self.retries_used

                    logger.warning(
                        "Lichess download retry attempt %d/%d due to %s: %d lines yielded so far, resuming from until=%s",
                        attempt,
                        max_retries,
                        exc.__class__.__name__,
                        len(yielded_ids),
                        last_created_at,
                    )

                    # Rate limit handling: HTTP 429 waits >= RATE_LIMIT_WAIT_SECONDS (60s)
                    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
                        wait_sec = max(config.RATE_LIMIT_WAIT_SECONDS, 1.0 * (2 ** (attempt - 1)))
                    else:
                        wait_sec = 1.0 * (2 ** (attempt - 1))

                    if self.sleep_fn:
                        res = self.sleep_fn(wait_sec)
                        if asyncio.iscoroutine(res):
                            await res
                    else:
                        await asyncio.sleep(wait_sec)


def stream_games(
    username: str,
    max_games: int = config.DEFAULT_MAX_GAMES,
    clock_fn: Callable[[], datetime] | None = None,
    now: Optional[datetime] = None,
    since_ms: Optional[int] = None,
    until_ms: Optional[int] = None,
    sleep_fn: Optional[Callable[[float], Any]] = None,
    overlap_ms: int = 0,
    stats: Any = None,
    bucket: str = "all",
) -> GameStream:
    """
    Stream raw NDJSON lines for *username* from the Lichess export API.

    Parameters
    ----------
    username : str
        The Lichess username (case-insensitive).
    max_games: int
        Upper bound on the number of games to download (Lichess default = all).
    clock_fn : Callable[[], datetime] | None
        Optional clock provider for deterministic 'since' parameter calculation.
    now : Optional[datetime]
        Optional fixed datetime for bucket boundary calculations.
    since_ms : Optional[int]
        Optional since timestamp in milliseconds to override automatic calculation.
    until_ms : Optional[int]
        Optional until timestamp in milliseconds for resuming.
    sleep_fn : Optional[Callable[[float], Any]]
        Optional injectable sleep function.
    overlap_ms : int
        Optional overlap window in ms on resume for non-monotonic streams.
    stats : Any
        Optional IngestStats object to record requests and retries.
    bucket : str
        Bucket identifier for request log ('all' or specific bucket name).

    Returns
    -------
    GameStream
        AsyncIterator yielding non-empty NDJSON lines, with retries_used exposed.
    """
    return GameStream(
        username=username,
        max_games=max_games,
        clock_fn=clock_fn,
        now=now,
        since_ms=since_ms,
        until_ms=until_ms,
        sleep_fn=sleep_fn,
        overlap_ms=overlap_ms,
        stats=stats,
        bucket=bucket,
    )
