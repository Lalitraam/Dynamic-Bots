import pytest
from datetime import datetime, timedelta, timezone
from app.services.parser import parse_stream_into_buckets
import io
import json

def make_pgn_line(username, opponent, is_white, tc="180+2", utc_date="2024.01.01", utc_time="12:00:00", game_id="game123", pgn_body=None):
    """Create a JSON line representing a Lichess game with 12 plies by default."""
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
async def test_substring_match_should_fail():
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
    bad_line = json.dumps({"id": "bad1", "pgn": '[White "ann"]\n[Black "joanna"]\n[UTCDate "2024.01.01"]\n[UTCTime "12:00:00"]\n[TimeControl "180+2"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. e4 1-0'})

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
    pgn_10_plies = f'[White "{target}"]\n[Black "{opponent}"]\n[UTCDate "2024.01.01"]\n[UTCTime "12:00:00"]\n[TimeControl "180+2"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0'
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
    pgn_9_plies = f'[White "{target}"]\n[Black "{opponent}"]\n[UTCDate "2024.01.01"]\n[UTCTime "12:00:00"]\n[TimeControl "180+2"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O 1-0'
    line = json.dumps({"id": "plies_9", "pgn": pgn_9_plies})

    async def mock_stream():
        yield line

    buckets, total_valid, drop_stats = await parse_stream_into_buckets(
        mock_stream(), target
    )
    assert drop_stats["too_short"] == 1
    assert total_valid == 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])