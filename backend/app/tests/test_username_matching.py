import pytest
from datetime import datetime, timedelta, timezone
from app.services.parser import parse_stream_into_buckets
import io
import json

_RELATIVE_UTC_DATE = (datetime.now(timezone.utc) - timedelta(days=10)).strftime("%Y.%m.%d")

def make_pgn_line(username, opponent, is_white, tc="180+2", utc_date=None, utc_time="12:00:00", game_id="game123", pgn_body=None):
    """Create a JSON line representing a Lichess game with 12 plies by default."""
    if utc_date is None:
        utc_date = _RELATIVE_UTC_DATE
    white = username if is_white else opponent
    black = opponent if is_white else username
    if pgn_body is None:
        pgn_body = f'[White "{white}"]\n[Black "{black}"]\n[UTCDate "{utc_date}"]\n[UTCTime "{utc_time}"]\n[TimeControl "{tc}"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 {{[%clk 1:00:00]}} 1-0'
    return json.dumps({
        "id": game_id,
        "pgn": pgn_body
    })

@pytest.mark.asyncio
async def test_exact_match_username_white():
    """Test that username 'ann' is correctly identified as White when playing 'joanna'."""
    target = "ann"
    opponent = "joanna"
    line = make_pgn_line(target, opponent, is_white=True)

    async def mock_stream():
        yield line

    buckets, total_valid, drop_stats = await parse_stream_into_buckets(
        mock_stream(), target
    )
    assert drop_stats["player_not_found"] == 0
    assert total_valid == 1

@pytest.mark.asyncio
async def test_exact_match_username_black():
    """Test that username 'ann' is correctly identified as Black when playing 'joanna'."""
    target = "ann"
    opponent = "joanna"
    line = make_pgn_line(target, opponent, is_white=False)

    async def mock_stream():
        yield line

    buckets, total_valid, drop_stats = await parse_stream_into_buckets(
        mock_stream(), target
    )
    assert drop_stats["player_not_found"] == 0
    assert total_valid == 1

@pytest.mark.asyncio
async def test_substring_of_opponent_name_does_not_cause_false_match():
    """Test that substring matching (the old buggy behavior) is now fixed.
    User 'ann' should NOT match opponent 'joannette' (where 'ann' is a substring but not equal)."""
    target = "ann"
    opponent = "joannette"  # 'ann' is a substring of 'joannette' but not equal
    line = make_pgn_line(target, opponent, is_white=False)  # ann is Black, joannette is White

    async def mock_stream():
        yield line

    buckets, total_valid, drop_stats = await parse_stream_into_buckets(
        mock_stream(), target
    )
    # Should NOT have player_not_found drop
    assert drop_stats["player_not_found"] == 0
    # Should have processed the game (went into one of the buckets)
    total_bucketed = len(buckets["last_6m"]) + len(buckets["6_18m"]) + len(buckets["18_36m"])
    assert total_bucketed == 1, f"Expected 1 game in buckets, got {total_bucketed}"

@pytest.mark.asyncio
async def test_missing_rating_defaults_to_1500():
    """Test that missing or garbage WhiteElo/BlackElo defaults to 1500."""
    target = "ann"
    opponent = "joanna"
    # PGN line without WhiteElo and BlackElo headers
    line = make_pgn_line(target, opponent, is_white=True)

    async def mock_stream():
        yield line

    buckets, total_valid, drop_stats = await parse_stream_into_buckets(
        mock_stream(), target
    )
    assert total_valid == 1
    game = buckets["18_36m"][0] if buckets["18_36m"] else buckets["last_6m"][0]
    assert game["rating"] == 1500
    assert game["opponent_rating"] == 1500

@pytest.mark.asyncio
async def test_duplicate_game_protection():
    """Test that feeding the same game_id twice counts it once and increments drop_stats['duplicate']."""
    target = "ann"
    opponent = "joanna"
    line1 = make_pgn_line(target, opponent, is_white=True, game_id="same_id_1")
    line2 = make_pgn_line(target, opponent, is_white=True, game_id="same_id_1")

    async def mock_stream():
        yield line1
        yield line2

    buckets, total_valid, drop_stats = await parse_stream_into_buckets(
        mock_stream(), target
    )
    assert total_valid == 1
    assert drop_stats["duplicate"] == 1

@pytest.mark.asyncio
async def test_unparseable_pgn():
    """Test that invalid/unparseable PGN text is caught and recorded under unparseable_pgn."""
    target = "ann"
    # Illegal move 6. e4 (pawn already on e4) causes python-chess to record a ValueError in game.errors
    bad_line = json.dumps({"id": "bad1", "pgn": f'[White "ann"]\n[Black "joanna"]\n[UTCDate "{_RELATIVE_UTC_DATE}"]\n[UTCTime "12:00:00"]\n[TimeControl "180+2"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. e4 1-0'})

    async def mock_stream():
        yield bad_line

    buckets, total_valid, drop_stats = await parse_stream_into_buckets(
        mock_stream(), target
    )
    assert total_valid == 0
    assert drop_stats["unparseable_pgn"] == 1

@pytest.mark.asyncio
async def test_min_plies_boundary_exact():
    """Test a game with exactly MIN_PLIES (10 plies) is accepted."""
    target = "ann"
    opponent = "joanna"
    # Exactly 10 plies: 5 full moves
    pgn_10_plies = f'[White "{target}"]\n[Black "{opponent}"]\n[UTCDate "{_RELATIVE_UTC_DATE}"]\n[UTCTime "12:00:00"]\n[TimeControl "180+2"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0'
    line = json.dumps({"id": "plies_10", "pgn": pgn_10_plies})

    async def mock_stream():
        yield line

    buckets, total_valid, drop_stats = await parse_stream_into_buckets(
        mock_stream(), target
    )
    assert drop_stats["too_short"] == 0
    assert total_valid == 1

@pytest.mark.asyncio
async def test_min_plies_boundary_below():
    """Test a game with MIN_PLIES - 1 (9 plies) is dropped under too_short."""
    target = "ann"
    opponent = "joanna"
    # 9 plies: 4 full moves + 1 White move
    pgn_9_plies = f'[White "{target}"]\n[Black "{opponent}"]\n[UTCDate "{_RELATIVE_UTC_DATE}"]\n[UTCTime "12:00:00"]\n[TimeControl "180+2"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O 1-0'
    line = json.dumps({"id": "plies_9", "pgn": pgn_9_plies})

    async def mock_stream():
        yield line

    buckets, total_valid, drop_stats = await parse_stream_into_buckets(
        mock_stream(), target
    )
    assert drop_stats["too_short"] == 1
    assert total_valid == 0


@pytest.mark.asyncio
async def test_parser_dedup_ordering():
    """T6 parser ordering:
    Replay stream (first 20 lines then all 50, 5 of first 20 too_short) ->
    too_short == 5 AND duplicate == 20.
    """
    target = "ann"
    opponent = "joanna"

    lines_50 = []
    for i in range(50):
        gid = f"GAME{i:04d}"
        if i < 5:
            pgn = f'[Event "Test"]\n[White "{target}"]\n[Black "{opponent}"]\n[UTCDate "{_RELATIVE_UTC_DATE}"]\n[UTCTime "12:00:00"]\n[TimeControl "180+2"]\n\n1. e4 e5 2. Nf3 1-0'
        else:
            pgn = f'[Event "Test"]\n[White "{target}"]\n[Black "{opponent}"]\n[UTCDate "{_RELATIVE_UTC_DATE}"]\n[UTCTime "12:00:00"]\n[TimeControl "180+2"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 1-0'
        lines_50.append(json.dumps({"id": gid, "pgn": pgn}))

    stream_70 = lines_50[:20] + lines_50

    async def mock_stream():
        for l in stream_70:
            yield l

    buckets, total_valid, drop_stats = await parse_stream_into_buckets(
        mock_stream(), target
    )

    assert drop_stats["too_short"] == 5
    assert drop_stats["duplicate"] == 20


@pytest.mark.asyncio
async def test_parser_invariant_multiple_failure_reasons():
    """T7 parser invariant:
    Stream with replays and at least 5 different failure reasons:
    variant, bad_date, player_not_found, bad_time_control, too_short,
    unparseable_pgn, no_pgn.
    Assert total_valid + sum(drop_stats.values()) == raw line count.
    """
    target = "ann"
    opponent = "joanna"

    mixed_lines = [
        # 1. Valid game
        json.dumps({
            "id": "valid_1",
            "pgn": f'[Event "Test"]\n[White "{target}"]\n[Black "{opponent}"]\n[UTCDate "{_RELATIVE_UTC_DATE}"]\n[UTCTime "12:00:00"]\n[TimeControl "180+2"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0'
        }),
        # 2. variant
        json.dumps({
            "id": "variant_1",
            "pgn": f'[Event "Test"]\n[Variant "Crazyhouse"]\n[White "{target}"]\n[Black "{opponent}"]\n[UTCDate "{_RELATIVE_UTC_DATE}"]\n[UTCTime "12:00:00"]\n[TimeControl "180+2"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0'
        }),
        # 3. bad_date
        json.dumps({
            "id": "baddate_1",
            "pgn": f'[Event "Test"]\n[White "{target}"]\n[Black "{opponent}"]\n[UTCDate "????.??.??"]\n[TimeControl "180+2"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0'
        }),
        # 4. player_not_found
        json.dumps({
            "id": "notfound_1",
            "pgn": f'[Event "Test"]\n[White "other1"]\n[Black "other2"]\n[UTCDate "{_RELATIVE_UTC_DATE}"]\n[UTCTime "12:00:00"]\n[TimeControl "180+2"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0'
        }),
        # 5. bad_time_control
        json.dumps({
            "id": "badtc_1",
            "pgn": f'[Event "Test"]\n[White "{target}"]\n[Black "{opponent}"]\n[UTCDate "{_RELATIVE_UTC_DATE}"]\n[UTCTime "12:00:00"]\n[TimeControl "-"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0'
        }),
        # 6. too_short
        json.dumps({
            "id": "short_1",
            "pgn": f'[Event "Test"]\n[White "{target}"]\n[Black "{opponent}"]\n[UTCDate "{_RELATIVE_UTC_DATE}"]\n[UTCTime "12:00:00"]\n[TimeControl "180+2"]\n\n1. e4 1-0'
        }),
        # 7. unparseable_pgn
        json.dumps({
            "id": "unparseable_1",
            "pgn": f'[Event "Test"]\n[White "{target}"]\n[Black "{opponent}"]\n[UTCDate "{_RELATIVE_UTC_DATE}"]\n[UTCTime "12:00:00"]\n[TimeControl "180+2"]\n\n1. e4 e4 1-0'
        }),
        # 8. no_pgn
        json.dumps({
            "id": "nopgn_1",
            "pgn": ""
        }),
    ]

    replayed_stream = mixed_lines + mixed_lines

    async def mock_stream():
        for l in replayed_stream:
            yield l

    buckets, total_valid, drop_stats = await parse_stream_into_buckets(
        mock_stream(), target
    )

    assert total_valid == 1
    assert drop_stats["variant"] == 1
    assert drop_stats["bad_date"] == 1
    assert drop_stats["player_not_found"] == 1
    assert drop_stats["bad_time_control"] == 1
    assert drop_stats["too_short"] == 1
    assert drop_stats["unparseable_pgn"] == 1
    assert drop_stats["no_pgn"] == 1
    assert drop_stats["duplicate"] == len(mixed_lines)
    assert total_valid + sum(drop_stats.values()) == len(replayed_stream)


@pytest.mark.asyncio
async def test_parser_injectable_now():
    """T13 injectable now: asserts the same game lands in different buckets
    under two injected `now` values, regardless of system wall clock."""
    game_line = make_pgn_line("target_p", "opp", is_white=True, utc_date="2024.01.01")

    # Under now_1 (2024-03-01): game is 60 days old -> lands in 'last_6m'
    now_1 = datetime(2024, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
    async def stream_1():
        yield game_line

    buckets_1, total_valid_1, drop_stats_1 = await parse_stream_into_buckets(
        stream_1(), "target_p", now=now_1
    )
    assert total_valid_1 == 1
    assert len(buckets_1["last_6m"]) == 1
    assert len(buckets_1["6_18m"]) == 0

    # Under now_2 (2025-01-01): same game is ~366 days old -> lands in '6_18m'
    now_2 = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    async def stream_2():
        yield game_line

    buckets_2, total_valid_2, drop_stats_2 = await parse_stream_into_buckets(
        stream_2(), "target_p", now=now_2
    )
    assert total_valid_2 == 1
    assert len(buckets_2["last_6m"]) == 0
    assert len(buckets_2["6_18m"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])