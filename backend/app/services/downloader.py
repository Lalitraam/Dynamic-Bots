"""
Downloader service for streaming games from the Lichess API.

Streams NDJSON game records from https://lichess.org/api/games/user/{username}.
Each line is a JSON object with a "pgn" field containing the PGN text.
"""

import asyncio
import httpx
from typing import AsyncIterator
from .. import config

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


async def stream_games(username: str, max_games: int = config.DEFAULT_MAX_GAMES) -> AsyncIterator[str]:
    """
    Stream raw NDJSON lines for *username* from the Lichess export API.

    Parameters
    ----------
    username : str
        The Lichess username (case-insensitive).
    max_games : int
        Upper bound on the number of games to download (Lichess default = all).

    Yields
    ------
    str
        Non-empty NDJSON lines, each representing one game.
    """
    url = LICHESS_EXPORT_URL.format(username=username.lower())
    headers = {
        "Accept": "application/x-ndjson",
        "User-Agent": "ChessCloneFactory/1.0 (Educational ML Project)",
    }
    params: dict[str, str | int | bool] = {
        "max": max_games,
        "evals": "false",
        "opening": "true",
        "pgnInJson": "true",
        "rated": "true",
        "perfType": _perf_type_param(),
    }

    timeout = httpx.Timeout(30.0, read=120.0)
    max_retries = 5

    for attempt in range(1, max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("GET", url, headers=headers, params=params) as response:
                    if response.status_code == 404:
                        raise ValueError(f"User not found: '{username}'")
                    if response.status_code == 403:
                        raise ValueError(f"Access forbidden: '{username}'")
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        line = line.strip()
                        if line:
                            yield line
                    return
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            if isinstance(exc, ValueError):
                raise
            if isinstance(exc, httpx.HTTPStatusError):
                if exc.response.status_code == 404:
                    raise ValueError(f"User not found: '{username}'") from exc
                if exc.response.status_code == 403:
                    raise ValueError(f"Access forbidden: '{username}'") from exc
                if exc.response.status_code != 429 and exc.response.status_code < 500:
                    raise exc
            if attempt == max_retries:
                raise exc
            await asyncio.sleep(1.0 * (2 ** (attempt - 1)))
