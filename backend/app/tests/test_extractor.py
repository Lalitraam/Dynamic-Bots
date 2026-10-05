import json
from pathlib import Path
from unittest.mock import patch
import pytest

from app.services.extractor import extract_samples
from app import config

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_games.jsonl"

def load_fixture_games():
    games = []
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            game = json.loads(line)
            # storage.load_games normally converts date to datetime, but let's mock the dict
            # extractor.py just passes "date" forward, so string is fine for testing.
            games.append(game)
    return games

@pytest.fixture
def mock_storage():
    games = load_fixture_games()
    with patch("app.services.extractor.storage.load_games", return_value=games):
        yield games

def test_extract_samples_basic(mock_storage):
    # 'TargetPlayer' is White in game1, Black in game2, White in game3, Black in game_noclock
    # game1: 5 moves (White plays 5 times)
    # game2: 6 moves (Black plays 6 times)
    # game3: 47 moves (White plays 47 times, last is mate)
    # game_noclock: 4 moves (Black plays 3 times)
    # Total White samples for TargetPlayer: 5 + 47 = 52
    # Total Black samples for TargetPlayer: 6 + 3 = 9
    # Wait, game_noclock: 1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. d3 Nf6 1/2-1/2
    # Black plays e5, Nc6, Bc5, Nf6 (4 moves).
    
    samples, drop_stats = extract_samples("targetplayer")
    
    assert drop_stats["unparseable_pgn"] == 0
    assert drop_stats["illegal_move"] == 0
    assert drop_stats["no_target_moves"] == 0

    assert len(samples) == 5 + 6 + 47 + 4

    # Check clock logic on a game with clocks
    # Game 1, target is White. Base time = 180
    game1_samples = [s for s in samples if s["game_id"] == "game1"]
    assert len(game1_samples) == 5
    # First move: clock should be base time (180.0)
    assert game1_samples[0]["clock_before_move"] == 180.0
    # Second move: clock should be the time remaining AFTER White's first move
    # White 1. e4 { [%clk 0:03:00] }, so after move 1, clock was 180.
    assert game1_samples[1]["clock_before_move"] == 180.0

    # Game with no clocks
    noclock_samples = [s for s in samples if s["game_id"] == "game_noclock"]
    assert len(noclock_samples) == 4
    for s in noclock_samples:
        assert s["clock_before_move"] is None

def test_extractor_reuses_classifier():
    # Make sure we don't re-implement parsing, we should call classifier.parse_time_control
    with patch("app.services.extractor.classifier.parse_time_control") as mock_parse:
        mock_parse.return_value = (60.0, 0.0)
        
        games = load_fixture_games()[:1] # just game 1
        with patch("app.services.extractor.storage.load_games", return_value=games):
            extract_samples("targetplayer")
            
        mock_parse.assert_called_with("180+2")


def test_mid_game_missing_clock_carries_forward():
    """
    Decision: when a game HAS %clk annotations overall but one specific move's
    %clk tag is absent, the last-known clock value is carried forward and
    clock_available is treated as True for that move (see extractor.py docstring).

    PGN layout (White = TargetPlayer, base_time = 180s, inc = 0):
      ply 0 (W move 1): e4   [%clk 0:03:00]   last_clock[W] <- 180 after push
      ply 1 (B move 1): e5   [%clk 0:03:00]
      ply 2 (W move 2): Nf3  [%clk 0:02:58]   clock_before = 180; last_clock[W] <- 178
      ply 3 (B move 2): Nc6  [%clk 0:02:58]
      ply 4 (W move 3): Bc4  (NO %clk)        clock_before = 178 (carry-forward);
                                               last_clock[W] unchanged
      ply 5 (B move 3): Bc5  [%clk 0:02:55]
      ply 6 (W move 4): d3   [%clk 0:02:55]   clock_before = 178 (still carry-forward)
      ply 7 (B move 4): Nf6  [%clk 0:02:55]
    """
    pgn = (
        "[Event \"Test\"]\n"
        "[White \"TargetPlayer\"]\n"
        "[Black \"Opponent\"]\n"
        "[Result \"*\"]\n"
        "[TimeControl \"180+0\"]\n"
        "\n"
        "1. e4 { [%clk 0:03:00] } 1... e5 { [%clk 0:03:00] } "
        "2. Nf3 { [%clk 0:02:58] } 2... Nc6 { [%clk 0:02:58] } "
        "3. Bc4 3... Bc5 { [%clk 0:02:55] } "
        "4. d3 { [%clk 0:02:55] } 4... Nf6 { [%clk 0:02:55] } *"
    )

    game_record = {
        "game_id": "test_missing_clk",
        "date": "2024-06-01T00:00:00+00:00",
        "category": "blitz",
        "is_fast": True,
        "player_color": "white",
        "rating": 1500,
        "opponent_rating": 1500,
        "time_control": "180+0",
        "termination": "Normal",
        "pgn": pgn,
    }

    with patch("app.services.extractor.storage.load_games", return_value=[game_record]):
        samples, drop_stats = extract_samples("targetplayer")

    assert drop_stats["unparseable_pgn"] == 0
    assert drop_stats["illegal_move"] == 0

    # White plays 4 moves -> 4 samples
    assert len(samples) == 4, f"Expected 4 White samples, got {len(samples)}"

    # Sort by ply to make assertions order-stable
    samples_sorted = sorted(samples, key=lambda s: s["ply"])

    w1 = samples_sorted[0]  # ply 0: e4
    w2 = samples_sorted[1]  # ply 2: Nf3
    w3 = samples_sorted[2]  # ply 4: Bc4 -- missing %clk
    w4 = samples_sorted[3]  # ply 6: d3

    # Move 1: seeded with base time (180s)
    assert w1["clock_before_move"] == 180.0, f"W move 1 clock: {w1['clock_before_move']}"

    # Move 2: used last_clock[W] = 180 (still the base-time seed; e4 annotation of 180
    # is stored AFTER the move, so clock_before for Nf3 is still 180)
    assert w2["clock_before_move"] == 180.0, f"W move 2 clock: {w2['clock_before_move']}"

    # Move 3 (missing %clk): must carry forward value stored after Nf3 annotation (178s)
    assert w3["clock_before_move"] is not None, (
        "W move 3 clock_before_move must not be None -- carry-forward from clocked game"
    )
    assert w3["clock_before_move"] == 178.0, (
        f"W move 3 should carry forward 178.0 from Nf3 annotation, got {w3['clock_before_move']}"
    )

    # Move 4: last_clock[W] was NOT updated by the missing-clock move, still 178
    assert w4["clock_before_move"] == 178.0, (
        f"W move 4 should still see 178.0 (carry-forward persists), got {w4['clock_before_move']}"
    )

    # All White samples must have non-None clocks (game has clocks -> clock_available=True path)
    for s in samples_sorted:
        assert s["clock_before_move"] is not None, (
            f"Clocked game: all moves must have non-None clock_before_move, ply={s['ply']}"
        )

