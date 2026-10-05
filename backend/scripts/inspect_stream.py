#!/usr/bin/env python3
"""
Inspect the Lichess stream for a given username.
Fetches 3 games and prints:
- Top-level JSON keys
- First 300 characters of the PGN field
- Whether "[%clk" appears in the PGN
"""
import sys
import json
import httpx
import asyncio

LICHESS_EXPORT_URL = "https://lichess.org/api/games/user/{username}"

async def inspect_stream(username: str, max_games: int = 3):
    url = LICHESS_EXPORT_URL.format(username=username.lower())
    headers = {
        "Accept": "application/x-ndjson",
        "User-Agent": "ChessCloneFactory/1.0 (Educational ML Project)"
    }
    params = {
        "max": max_games,
        "evals": "false",
        "opening": "true",
        "pgnInJson": "true",
        "rated": "true"
    }

    async with httpx.AsyncClient() as client:
        try:
            async with client.stream("GET", url, headers=headers, params=params) as response:
                response.raise_for_status()
                game_count = 0
                async for line in response.aiter_lines():
                    if line.strip():
                        game_count += 1
                        try:
                            data = json.loads(line)
                            print(f"--- Game {game_count} ---")
                            print("Top-level JSON keys:", list(data.keys()))
                            pgn = data.get("pgn", "")
                            print("First 300 chars of PGN:", pgn[:300])
                            has_clk = "[%clk" in pgn
                            print("Contains [%clk]:", has_clk)
                            print()
                        except json.JSONDecodeError:
                            print(f"--- Game {game_count} (non-JSON line) ---")
                            print("Line:", line[:200])
                            print()
        except httpx.HTTPStatusError as e:
            print(f"HTTP error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_stream.py <username>")
        sys.exit(1)
    username = sys.argv[1]
    asyncio.run(inspect_stream(username))