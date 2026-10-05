"""
Parser service: consumes an NDJSON stream and produces bucketed game records.

Each NDJSON line is a Lichess game object with a "pgn" field.  We extract
PGN headers using python-chess, apply all required filters, and bucket valid games by age.

Buckets
-------
last_6m  : games played within the last 6 months (<= 180 days)
6_18m    : games played between 6 and 18 months ago (<= 540 days)
18_36m   : games played between 18 and 36 months ago (<= 1095 days)

Returns
-------
(buckets, total_valid, drop_stats)
  buckets      : dict[str, list[dict]]
  total_valid  : int  – games that actually landed in a bucket
  drop_stats   : dict[str, int] – count per drop reason
"""

import io
import json
from datetime import datetime, timezone
from typing import AsyncIterable, Callable

import chess.pgn

from .. import config
from .classifier import classify_time_control, parse_time_control, meets_min_time_control

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Termination values that disqualify a game.
_BAD_TERMINATIONS = {"Abandoned", "Unterminated", "Rules infraction"}


# ---------------------------------------------------------------------------
# Main parsing function
# ---------------------------------------------------------------------------

async def parse_stream_into_buckets(
    stream: AsyncIterable[str],
    username: str,
    on_progress: Callable[[int], None] | None = None,
) -> tuple[dict[str, list[dict]], int, dict[str, int]]:
    """
    Consume *stream* (async iterable of NDJSON lines) and return
    (buckets, total_valid, drop_stats).

    Checks are applied in order so each game is counted under exactly one
    drop reason.
    """
    target = username.lower()
    now = datetime.now(timezone.utc)
    seen_ids: set[str] = set()
    line_count = 0

    buckets: dict[str, list[dict]] = {
        "last_6m": [],
        "6_18m": [],
        "18_36m": [],
    }
    drop_stats: dict[str, int] = {
        "no_pgn": 0,
        "unparseable_pgn": 0,
        "duplicate": 0,
        "variant": 0,
        "custom_start": 0,
        "bad_termination": 0,
        "unfinished": 0,
        "too_short": 0,
        "bad_date": 0,
        "player_not_found": 0,
        "bad_time_control": 0,
        "below_min_time_control": 0,
        "excluded_category": 0,
        "too_old": 0,
    }

    async for raw_line in stream:
        line_count += 1
        if on_progress:
            on_progress(line_count)

        raw_line = raw_line.strip()
        if not raw_line:
            continue

        # --- Parse JSON ---
        try:
            obj = json.loads(raw_line)
        except json.JSONDecodeError:
            drop_stats["no_pgn"] += 1
            continue

        # --- Require a non-empty PGN field ---
        pgn = obj.get("pgn", "")
        if not pgn or not pgn.strip():
            drop_stats["no_pgn"] += 1
            continue

        # --- Parse PGN using python-chess ---
        try:
            game = chess.pgn.read_game(io.StringIO(pgn))
        except Exception:
            game = None

        if game is None or game.errors:
            drop_stats["unparseable_pgn"] += 1
            continue

        headers = game.headers

        # --- game_id from JSON "id" or Site URL last path segment ---
        game_id = obj.get("id", "")
        if not game_id:
            site = headers.get("Site", "")
            game_id = site.rstrip("/").split("/")[-1] if site else ""

        # --- Duplicate filter ---
        if game_id and game_id in seen_ids:
            drop_stats["duplicate"] += 1
            continue

        # --- Variant filter ---
        variant = headers.get("Variant", "")
        if variant and variant != "Standard":
            drop_stats["variant"] += 1
            continue

        # --- Custom start position filter ---
        if "FEN" in headers or "SetUp" in headers:
            drop_stats["custom_start"] += 1
            continue

        # --- Termination filter ---
        termination = headers.get("Termination", "")
        if termination in _BAD_TERMINATIONS:
            drop_stats["bad_termination"] += 1
            continue

        # --- Unfinished result ---
        result = headers.get("Result", "")
        if result == "*":
            drop_stats["unfinished"] += 1
            continue

        # --- Ply count with python-chess ---
        n_plies = sum(1 for _ in game.mainline_moves())
        if n_plies < config.MIN_PLIES:
            drop_stats["too_short"] += 1
            continue

        # --- Date parsing ---
        utc_date = headers.get("UTCDate", "") or headers.get("Date", "")
        utc_time = headers.get("UTCTime", "")
        game_dt: datetime | None = None
        if utc_date:
            date_str = utc_date.replace(".", "-")
            try:
                if utc_time:
                    game_dt = datetime.strptime(
                        f"{date_str} {utc_time}", "%Y-%m-%d %H:%M:%S"
                    ).replace(tzinfo=timezone.utc)
                else:
                    game_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(
                        tzinfo=timezone.utc
                    )
            except ValueError:
                pass

        if game_dt is None:
            drop_stats["bad_date"] += 1
            continue

        # --- Player identification (exact match after lowercasing) ---
        white = headers.get("White", "").lower()
        black = headers.get("Black", "").lower()
        white_elo_str = headers.get("WhiteElo", "") or ""
        black_elo_str = headers.get("BlackElo", "") or ""

        if white == target:
            player_color = "white"
            try:
                rating = int(white_elo_str)
            except ValueError:
                rating = config.DEFAULT_RATING
            try:
                opponent_rating = int(black_elo_str)
            except ValueError:
                opponent_rating = config.DEFAULT_RATING
        elif black == target:
            player_color = "black"
            try:
                rating = int(black_elo_str)
            except ValueError:
                rating = config.DEFAULT_RATING
            try:
                opponent_rating = int(white_elo_str)
            except ValueError:
                opponent_rating = config.DEFAULT_RATING
        else:
            drop_stats["player_not_found"] += 1
            continue

        # --- Time control parsing ---
        tc_raw = headers.get("TimeControl", "")
        if not tc_raw:
            drop_stats["bad_time_control"] += 1
            continue

        parsed_tc = parse_time_control(tc_raw)
        if parsed_tc is None:
            drop_stats["bad_time_control"] += 1
            continue

        # --- Minimum base time ---
        if not meets_min_time_control(tc_raw):
            drop_stats["below_min_time_control"] += 1
            continue

        # --- Category classification ---
        category = classify_time_control(tc_raw)
        if category not in config.INCLUDED_CATEGORIES:
            drop_stats["excluded_category"] += 1
            continue

        # --- Age filter (too old) ---
        age_days = (now - game_dt).days
        if age_days > config.MAX_AGE_DAYS:
            drop_stats["too_old"] += 1
            continue

        # --- is_fast flag (bullet category) ---
        is_fast = category == "bullet"

        # --- Assemble game record ---
        game_record = {
            "game_id": game_id,
            "date": game_dt,
            "category": category,
            "is_fast": is_fast,
            "player_color": player_color,
            "rating": rating,
            "opponent_rating": opponent_rating,
            "time_control": tc_raw,
            "termination": termination,
            "pgn": pgn,
        }

        # --- Mark game_id seen & Bucket by age ---
        if game_id:
            seen_ids.add(game_id)

        if age_days <= config.BUCKET_MAX_AGE_6M:
            buckets["last_6m"].append(game_record)
        elif age_days <= config.BUCKET_MAX_AGE_18M:
            buckets["6_18m"].append(game_record)
        else:
            buckets["18_36m"].append(game_record)

    total_valid = sum(len(v) for v in buckets.values())
    return buckets, total_valid, drop_stats
