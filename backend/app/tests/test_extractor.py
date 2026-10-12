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


def test_extractor_standard_300_plus_0_clock_sequence():
    """Phase C: 300+0 game, white %clk after each white move = 0:04:55, 0:04:50, 0:04:45
    -> white's 3 values are 300, 295, 290.
    Assert the 2nd value is NOT 290 (the clock after the predicted move must never be used).
    """
    pgn = (
        '[Event "Test"]\n'
        '[White "TargetPlayer"]\n'
        '[Black "Opponent"]\n'
        '[Result "*"]\n'
        '[TimeControl "300+0"]\n\n'
        '1. e4 { [%clk 0:04:55] } 1... e5 { [%clk 0:04:58] } '
        '2. Nf3 { [%clk 0:04:50] } 2... Nc6 { [%clk 0:04:56] } '
        '3. Bb5 { [%clk 0:04:45] } 3... a6 { [%clk 0:04:54] } *'
    )
    game_record = {
        "game_id": "g_std_300",
        "date": "2024-06-01T00:00:00+00:00",
        "category": "blitz",
        "is_fast": False,
        "player_color": "white",
        "rating": 1500,
        "opponent_rating": 1500,
        "time_control": "300+0",
        "termination": "Normal",
        "pgn": pgn,
    }
    with patch("app.services.extractor.storage.load_games", return_value=[game_record]):
        samples, _ = extract_samples("targetplayer")

    s = sorted(samples, key=lambda x: x["ply"])
    assert len(s) == 3
    assert s[0]["clock_before_move"] == 300.0
    assert s[1]["clock_before_move"] == 295.0
    assert s[2]["clock_before_move"] == 290.0
    # The 2nd value must NOT be 290 (clock after predicted move)
    assert s[1]["clock_before_move"] != 290.0
    assert s[0]["target_berserk"] is False


def test_extractor_white_berserk_clock_sequence():
    """Phase C: same PGN with [WhiteBerserk "true"] and white clocks 0:02:25, 0:02:20, 0:02:15
    -> first value 150, then 145, 140.
    """
    pgn = (
        '[Event "Test"]\n'
        '[White "TargetPlayer"]\n'
        '[Black "Opponent"]\n'
        '[Result "*"]\n'
        '[TimeControl "300+0"]\n'
        '[WhiteBerserk "true"]\n\n'
        '1. e4 { [%clk 0:02:25] } 1... e5 { [%clk 0:04:58] } '
        '2. Nf3 { [%clk 0:02:20] } 2... Nc6 { [%clk 0:04:56] } '
        '3. Bb5 { [%clk 0:02:15] } 3... a6 { [%clk 0:04:54] } *'
    )
    game_record = {
        "game_id": "g_wb_300",
        "date": "2024-06-01T00:00:00+00:00",
        "category": "blitz",
        "is_fast": False,
        "player_color": "white",
        "rating": 1500,
        "opponent_rating": 1500,
        "time_control": "300+0",
        "termination": "Normal",
        "pgn": pgn,
    }
    with patch("app.services.extractor.storage.load_games", return_value=[game_record]):
        samples, _ = extract_samples("targetplayer")

    s = sorted(samples, key=lambda x: x["ply"])
    assert len(s) == 3
    assert s[0]["clock_before_move"] == 150.0
    assert s[1]["clock_before_move"] == 145.0
    assert s[2]["clock_before_move"] == 140.0
    assert s[0]["target_berserk"] is True


def test_extractor_black_berserk_only_target_white():
    """Phase C: Black-berserk-only game, target = White -> target's first value stays 300."""
    pgn = (
        '[Event "Test"]\n'
        '[White "TargetPlayer"]\n'
        '[Black "Opponent"]\n'
        '[Result "*"]\n'
        '[TimeControl "300+0"]\n'
        '[BlackBerserk "true"]\n\n'
        '1. e4 { [%clk 0:04:55] } 1... e5 { [%clk 0:02:25] } *'
    )
    game_record = {
        "game_id": "g_bb_target_white",
        "date": "2024-06-01T00:00:00+00:00",
        "category": "blitz",
        "is_fast": False,
        "player_color": "white",
        "rating": 1500,
        "opponent_rating": 1500,
        "time_control": "300+0",
        "termination": "Normal",
        "pgn": pgn,
    }
    with patch("app.services.extractor.storage.load_games", return_value=[game_record]):
        samples, _ = extract_samples("targetplayer")

    s = sorted(samples, key=lambda x: x["ply"])
    assert len(s) == 1
    assert s[0]["clock_before_move"] == 300.0
    assert s[0]["target_berserk"] is False


def test_extractor_target_black_with_black_berserk():
    """Phase C: target = Black with BlackBerserk -> first value 150."""
    pgn = (
        '[Event "Test"]\n'
        '[White "Opponent"]\n'
        '[Black "TargetPlayer"]\n'
        '[Result "*"]\n'
        '[TimeControl "300+0"]\n'
        '[BlackBerserk "true"]\n\n'
        '1. e4 { [%clk 0:04:58] } 1... e5 { [%clk 0:02:25] } *'
    )
    game_record = {
        "game_id": "g_bb_target_black",
        "date": "2024-06-01T00:00:00+00:00",
        "category": "blitz",
        "is_fast": False,
        "player_color": "black",
        "rating": 1500,
        "opponent_rating": 1500,
        "time_control": "300+0",
        "termination": "Normal",
        "pgn": pgn,
    }
    with patch("app.services.extractor.storage.load_games", return_value=[game_record]):
        samples, _ = extract_samples("targetplayer")

    s = sorted(samples, key=lambda x: x["ply"])
    assert len(s) == 1
    assert s[0]["clock_before_move"] == 150.0
    assert s[0]["target_berserk"] is True


def test_extractor_no_clock_game_gives_none():
    """Phase C: no-%clk game -> all None, clock_available False."""
    pgn = (
        '[Event "Test"]\n'
        '[White "TargetPlayer"]\n'
        '[Black "Opponent"]\n'
        '[Result "*"]\n'
        '[TimeControl "300+0"]\n\n'
        '1. e4 1... e5 2. Nf3 2... Nc6 *'
    )
    game_record = {
        "game_id": "g_noclock",
        "date": "2024-06-01T00:00:00+00:00",
        "category": "blitz",
        "is_fast": False,
        "player_color": "white",
        "rating": 1500,
        "opponent_rating": 1500,
        "time_control": "300+0",
        "termination": "Normal",
        "pgn": pgn,
    }
    with patch("app.services.extractor.storage.load_games", return_value=[game_record]):
        samples, _ = extract_samples("targetplayer")

    s = sorted(samples, key=lambda x: x["ply"])
    assert len(s) == 2
    for row in s:
        assert row["clock_before_move"] is None
        assert row["target_berserk"] is False


def test_extract_samples_illegal_move_value_error_discards_game(mock_storage, monkeypatch):
    """Defect D13: ValueError during move push discards that game's rows and increments drop_stats."""
    import sys
    import chess
    orig_push = chess.Board.push

    push_count = 0
    def faulty_push(board_self, move):
        nonlocal push_count
        caller = sys._getframe(1).f_code.co_name
        if caller == "extract_samples":
            push_count += 1
            if push_count == 6:  # ply 5 of the first game (game1)
                assert len(board_self.move_stack) == 5
                raise ValueError("Simulated illegal move at ply 5")
        return orig_push(board_self, move)

    monkeypatch.setattr(chess.Board, "push", faulty_push)

    samples, drop_stats = extract_samples("targetplayer")

    game1_samples = [s for s in samples if s["game_id"] == "game1"]
    assert len(game1_samples) == 0
    assert drop_stats["illegal_move"] == 1
    assert len(samples) == 57
    assert len([s for s in samples if s["game_id"] == "game2"]) == 6
    assert len([s for s in samples if s["game_id"] == "game3_ep_promo"]) == 47
    assert len([s for s in samples if s["game_id"] == "game_noclock"]) == 4


def test_extract_samples_illegal_move_assertion_error_discards_game(mock_storage, monkeypatch):
    """Defect D13: AssertionError during move push discards that game's rows and increments drop_stats."""
    import sys
    import chess
    orig_push = chess.Board.push

    push_count = 0
    def faulty_push(board_self, move):
        nonlocal push_count
        caller = sys._getframe(1).f_code.co_name
        if caller == "extract_samples":
            push_count += 1
            if push_count == 6:  # ply 5 of the first game (game1)
                assert len(board_self.move_stack) == 5
                raise AssertionError("Simulated assertion error at ply 5")
        return orig_push(board_self, move)

    monkeypatch.setattr(chess.Board, "push", faulty_push)

    samples, drop_stats = extract_samples("targetplayer")

    game1_samples = [s for s in samples if s["game_id"] == "game1"]
    assert len(game1_samples) == 0
    assert drop_stats["illegal_move"] == 1
    assert len(samples) == 57
    assert len([s for s in samples if s["game_id"] == "game2"]) == 6
    assert len([s for s in samples if s["game_id"] == "game3_ep_promo"]) == 47
    assert len([s for s in samples if s["game_id"] == "game_noclock"]) == 4


def test_extract_samples_runtime_error_propagates(mock_storage, monkeypatch):
    """Defect D13: Unexpected exceptions (e.g. RuntimeError) propagate out of extract_samples."""
    import sys
    import chess
    orig_push = chess.Board.push

    def faulty_push(board_self, move):
        caller = sys._getframe(1).f_code.co_name
        if caller == "extract_samples":
            raise RuntimeError("Simulated unexpected runtime error")
        return orig_push(board_self, move)

    monkeypatch.setattr(chess.Board, "push", faulty_push)

    with pytest.raises(RuntimeError, match="Simulated unexpected runtime error"):
        extract_samples("targetplayer")


def test_extractor_phase_b_columns_300_plus_0():
    """
    Phase B.1 test for 300+0 game, target player is White.
    Expected values hand-computed:
    - Base seconds: 300, Increment seconds: 0
    - Move 1 (White, ply 0, e4):
      * prev_move_1 = None (no prior opponent move)
      * prev_move_2 = None (no prior target move)
      * num_legal_moves = 20 (standard opening position)
      * clock_before_move = 300.0 (base time)
      * clock_after_move = 300.0 (%clk 0:05:00)
      * time_spent_s = None (first move of target player)
      * increment_s = 0, base_s = 300
      * game_result_for_player = 1.0 (Result 1-0, target is White -> 1.0)
    - Move 2 (White, ply 2, Nf3):
      * prev_move_1 = 'e7e5' (Black's move 1)
      * prev_move_2 = 'e2e4' (White's move 1)
      * num_legal_moves = 29
      * clock_before_move = 300.0 (carried from White's move 1)
      * clock_after_move = 295.0 (%clk 0:04:55)
      * time_spent_s = 300.0 - 295.0 + 0 = 5.0
      * increment_s = 0, base_s = 300
      * game_result_for_player = 1.0
    """
    from datetime import datetime, timezone
    pgn = (
        '[Event "Test 300+0"]\n'
        '[White "TargetPlayer"]\n'
        '[Black "Opponent"]\n'
        '[Result "1-0"]\n'
        '[TimeControl "300+0"]\n'
        '\n'
        '1. e4 { [%clk 0:05:00] } 1... e5 { [%clk 0:05:00] } '
        '2. Nf3 { [%clk 0:04:55] } 2... Nc6 { [%clk 0:04:58] } 1-0\n'
    )
    game_record = [{
        "game_id": "game_300_0",
        "pgn": pgn,
        "player_color": "white",
        "time_control": "300+0",
        "date": datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc),
        "category": "blitz",
        "is_fast": False,
        "rating": 1800,
        "opponent_rating": 1750,
        "termination": "Normal",
    }]
    with patch("app.services.extractor.storage.load_games", return_value=game_record):
        samples, drop_stats = extract_samples("targetplayer")

    assert drop_stats["unparseable_pgn"] == 0
    assert len(samples) == 2

    # Move 1
    m1 = samples[0]
    assert m1["ply"] == 0
    assert m1["uci"] == "e2e4"
    assert m1["prev_move_1"] is None
    assert m1["prev_move_2"] is None
    assert m1["num_legal_moves"] == 20
    assert m1["clock_before_move"] == 300.0
    assert m1["clock_after_move"] == 300.0
    assert m1["time_spent_s"] is None
    assert m1["increment_s"] == 0
    assert m1["base_s"] == 300
    assert m1["created_at_ms"] == int(datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)
    assert m1["game_result_for_player"] == 1.0

    # Move 2
    m2 = samples[1]
    assert m2["ply"] == 2
    assert m2["uci"] == "g1f3"
    assert m2["prev_move_1"] == "e7e5"
    assert m2["prev_move_2"] == "e2e4"
    assert m2["clock_before_move"] == 300.0
    assert m2["clock_after_move"] == 295.0
    assert m2["time_spent_s"] == 5.0
    assert m2["increment_s"] == 0
    assert m2["base_s"] == 300
    assert m2["game_result_for_player"] == 1.0


def test_extractor_phase_b_columns_180_plus_2():
    """
    Phase B.1 test for 180+2 game, target player is White.
    Expected values hand-computed:
    - Base seconds: 180, Increment seconds: 2
    - Move 1 (White, ply 0, e4):
      * clock_before_move = 180.0
      * clock_after_move = 180.0 (%clk 0:03:00)
      * time_spent_s = None (first move)
      * increment_s = 2, base_s = 180
      * game_result_for_player = 0.0 (Result 0-1, target White -> 0.0)
    - Move 2 (White, ply 2, Nf3):
      * clock_before_move = 180.0
      * clock_after_move = 177.0 (%clk 0:02:57)
      * time_spent_s = 180.0 - 177.0 + 2 = 5.0 (clamped >= 0)
      * increment_s = 2, base_s = 180
      * game_result_for_player = 0.0
    """
    from datetime import datetime, timezone
    pgn = (
        '[Event "Test 180+2"]\n'
        '[White "TargetPlayer"]\n'
        '[Black "Opponent"]\n'
        '[Result "0-1"]\n'
        '[TimeControl "180+2"]\n'
        '\n'
        '1. e4 { [%clk 0:03:00] } 1... e5 { [%clk 0:03:00] } '
        '2. Nf3 { [%clk 0:02:57] } 2... Nc6 { [%clk 0:02:59] } 0-1\n'
    )
    game_record = [{
        "game_id": "game_180_2",
        "pgn": pgn,
        "player_color": "white",
        "time_control": "180+2",
        "date": datetime(2025, 6, 2, 12, 0, tzinfo=timezone.utc),
        "category": "blitz",
        "is_fast": False,
        "rating": 1800,
        "opponent_rating": 1850,
        "termination": "Normal",
    }]
    with patch("app.services.extractor.storage.load_games", return_value=game_record):
        samples, drop_stats = extract_samples("targetplayer")

    assert len(samples) == 2
    m1, m2 = samples[0], samples[1]
    assert m1["time_spent_s"] is None
    assert m1["increment_s"] == 2
    assert m1["base_s"] == 180
    assert m1["game_result_for_player"] == 0.0

    assert m2["clock_before_move"] == 180.0
    assert m2["clock_after_move"] == 177.0
    assert m2["time_spent_s"] == 5.0  # 180.0 - 177.0 + 2 = 5.0
    assert m2["increment_s"] == 2
    assert m2["base_s"] == 180
    assert m2["game_result_for_player"] == 0.0


def test_extractor_phase_b_white_berserk_180_plus_2():
    """
    Test 3 (a): White berserk in 180+2.
    Moves: "1. e4 {[%clk 0:01:30]} e5 {[%clk 0:03:00]} 2. Nf3 {[%clk 0:01:28]} Nc6 {[%clk 0:03:00]} *"
    Hand-computed expected values:
    - Base = 180s, Berserk base = 180 / 2 = 90.0s.
    - Berserk removes increment, so effective_increment = 0 (even though increment_s = 2 in header).
    - Move 1 (e4): first move -> time_spent_s = None. clock_after = 90.0.
    - Move 2 (Nf3): clock_before = 90.0, clock_after = 88.0.
      time_spent_s = max(0, 90.0 - 88.0 + 0) = 2.0.
    Target white rows time_spent_s: [None, 2.0].
    """
    from datetime import datetime, timezone
    pgn = (
        '[Event "Test Berserk 180+2"]\n'
        '[White "TargetPlayer"]\n'
        '[Black "Opponent"]\n'
        '[Result "*"]\n'
        '[TimeControl "180+2"]\n'
        '[WhiteBerserk "true"]\n'
        '\n'
        '1. e4 { [%clk 0:01:30] } 1... e5 { [%clk 0:03:00] } '
        '2. Nf3 { [%clk 0:01:28] } 2... Nc6 { [%clk 0:03:00] } *\n'
    )
    game_record = [{
        "game_id": "game_berserk_180_2",
        "pgn": pgn,
        "player_color": "white",
        "time_control": "180+2",
        "date": datetime(2025, 6, 3, 12, 0, tzinfo=timezone.utc),
        "category": "blitz",
        "is_fast": False,
        "rating": 1800,
        "opponent_rating": 1800,
        "termination": "Normal",
    }]
    with patch("app.services.extractor.storage.load_games", return_value=game_record):
        samples, drop_stats = extract_samples("targetplayer")

    assert len(samples) == 2
    m1, m2 = samples[0], samples[1]
    assert m1["target_berserk"] is True
    assert m1["increment_s"] == 2
    assert m1["clock_before_move"] == 90.0
    assert m1["clock_after_move"] == 90.0
    assert m1["time_spent_s"] is None

    assert m2["target_berserk"] is True
    assert m2["increment_s"] == 2
    assert m2["clock_before_move"] == 90.0
    assert m2["clock_after_move"] == 88.0
    # Expected: 90 - 88 + 0 = 2.0 (effective_increment = 0)
    assert m2["time_spent_s"] == 2.0


def test_extractor_phase_b_non_berserk_180_plus_2():
    """
    Test 3 (b): Same moves, NOT berserk, TC 180+2.
    Moves: "1. e4 {[%clk 0:03:00]} e5 {[%clk 0:03:00]} 2. Nf3 {[%clk 0:03:01]} Nc6 {[%clk 0:03:00]} *"
    Hand-computed expected values:
    - Base = 180s, increment = 2s.
    - Move 1 (e4): first move -> time_spent_s = None. clock_after = 180.0.
    - Move 2 (Nf3): clock_before = 180.0, clock_after = 181.0.
      time_spent_s = max(0, 180.0 - 181.0 + 2) = 1.0.
    Target white rows time_spent_s: [None, 1.0].
    """
    from datetime import datetime, timezone
    pgn = (
        '[Event "Test Non-Berserk 180+2"]\n'
        '[White "TargetPlayer"]\n'
        '[Black "Opponent"]\n'
        '[Result "*"]\n'
        '[TimeControl "180+2"]\n'
        '\n'
        '1. e4 { [%clk 0:03:00] } 1... e5 { [%clk 0:03:00] } '
        '2. Nf3 { [%clk 0:03:01] } 2... Nc6 { [%clk 0:03:00] } *\n'
    )
    game_record = [{
        "game_id": "game_non_berserk_180_2",
        "pgn": pgn,
        "player_color": "white",
        "time_control": "180+2",
        "date": datetime(2025, 6, 3, 12, 0, tzinfo=timezone.utc),
        "category": "blitz",
        "is_fast": False,
        "rating": 1800,
        "opponent_rating": 1800,
        "termination": "Normal",
    }]
    with patch("app.services.extractor.storage.load_games", return_value=game_record):
        samples, drop_stats = extract_samples("targetplayer")

    assert len(samples) == 2
    m1, m2 = samples[0], samples[1]
    assert m1["target_berserk"] is False
    assert m1["clock_before_move"] == 180.0
    assert m1["clock_after_move"] == 180.0
    assert m1["time_spent_s"] is None

    assert m2["target_berserk"] is False
    assert m2["clock_before_move"] == 180.0
    assert m2["clock_after_move"] == 181.0
    assert m2["time_spent_s"] == 1.0


def test_extractor_phase_b_missing_clock_stale_before_180_plus_0():
    """
    Test 3 (c): Missing %clk, white 180+0.
    PGN: "1. e4 {[%clk 0:03:00]} e5 {[%clk 0:03:00]} 2. Nf3 Nc6 3. Bc4 {[%clk 0:02:50]} Bc5 {[%clk 0:02:55]} 4. d3 {[%clk 0:02:48]} Nf6 {[%clk 0:02:50]} *"
    (Nf3 and Nc6 have no %clk).
    Expected white time_spent_s per row:
    - e4: None (first move)
    - Nf3: None (clock_after missing)
    - Bc4: None (stale before, because previous move Nf3 had no %clk)
    - d3: 2.0 (170 - 168 = 2.0; previous move Bc4 had %clk 0:02:50 = 170s, current move d3 has %clk 0:02:48 = 168s).
    Also assert clock_before_move values are unchanged from existing carry-forward behaviour:
    - e4: 180.0
    - Nf3: 180.0
    - Bc4: 180.0
    - d3: 170.0
    """
    from datetime import datetime, timezone
    pgn = (
        '[Event "Test Missing Clock Stale"]\n'
        '[White "TargetPlayer"]\n'
        '[Black "Opponent"]\n'
        '[Result "*"]\n'
        '[TimeControl "180+0"]\n'
        '\n'
        '1. e4 { [%clk 0:03:00] } 1... e5 { [%clk 0:03:00] } '
        '2. Nf3 2... Nc6 '
        '3. Bc4 { [%clk 0:02:50] } 3... Bc5 { [%clk 0:02:55] } '
        '4. d3 { [%clk 0:02:48] } 4... Nf6 { [%clk 0:02:50] } *\n'
    )
    game_record = [{
        "game_id": "game_missing_stale_180_0",
        "pgn": pgn,
        "player_color": "white",
        "time_control": "180+0",
        "date": datetime(2025, 6, 4, 12, 0, tzinfo=timezone.utc),
        "category": "blitz",
        "is_fast": False,
        "rating": 1800,
        "opponent_rating": 1800,
        "termination": "Normal",
    }]
    with patch("app.services.extractor.storage.load_games", return_value=game_record):
        samples, drop_stats = extract_samples("targetplayer")

    assert len(samples) == 4
    m1, m2, m3, m4 = samples

    # clock_before_move carry-forward assertions
    assert m1["clock_before_move"] == 180.0
    assert m2["clock_before_move"] == 180.0
    assert m3["clock_before_move"] == 180.0
    assert m4["clock_before_move"] == 170.0

    # time_spent_s assertions
    assert m1["time_spent_s"] is None
    assert m2["time_spent_s"] is None
    assert m3["time_spent_s"] is None
    assert m4["time_spent_s"] == 2.0


def test_extractor_phase_b_prev_move_sequence_and_legal_moves():
    """
    Phase B.1 test:
    Verify prev_move_1, prev_move_2, and num_legal_moves on '1. e4 e5 2. Nf3 Nc6'.
    Hand-computed expected values:
    - Target = White:
      * 1st White move (ply 0, e4): prev_move_1 = None, prev_move_2 = None, num_legal_moves = 20
      * 2nd White move (ply 2, Nf3): prev_move_1 = 'e7e5', prev_move_2 = 'e2e4'
    - Target = Black:
      * 1st Black move (ply 1, e5): prev_move_1 = 'e2e4', prev_move_2 = None, num_legal_moves = 20
      * 2nd Black move (ply 3, Nc6): prev_move_1 = 'g1f3', prev_move_2 = 'e7e5'
    """
    from datetime import datetime, timezone
    pgn = (
        '[Event "Sequence Test"]\n'
        '[White "WhitePlayer"]\n'
        '[Black "BlackPlayer"]\n'
        '[Result "1/2-1/2"]\n'
        '[TimeControl "300+0"]\n'
        '\n'
        '1. e4 { [%clk 0:05:00] } 1... e5 { [%clk 0:05:00] } '
        '2. Nf3 { [%clk 0:04:55] } 2... Nc6 { [%clk 0:04:58] } 1/2-1/2\n'
    )

    # 1. White target
    game_white = [{
        "game_id": "game_seq_w",
        "pgn": pgn,
        "player_color": "white",
        "time_control": "300+0",
        "date": datetime(2025, 6, 5, 12, 0, tzinfo=timezone.utc),
        "category": "blitz",
        "is_fast": False,
        "rating": 1800,
        "opponent_rating": 1800,
        "termination": "Normal",
    }]
    with patch("app.services.extractor.storage.load_games", return_value=game_white):
        samples_w, _ = extract_samples("whiteplayer")

    assert len(samples_w) == 2
    assert samples_w[0]["ply"] == 0
    assert samples_w[0]["prev_move_1"] is None
    assert samples_w[0]["prev_move_2"] is None
    assert samples_w[0]["num_legal_moves"] == 20

    assert samples_w[1]["ply"] == 2
    assert samples_w[1]["prev_move_1"] == "e7e5"
    assert samples_w[1]["prev_move_2"] == "e2e4"

    # 2. Black target
    game_black = [{
        "game_id": "game_seq_b",
        "pgn": pgn,
        "player_color": "black",
        "time_control": "300+0",
        "date": datetime(2025, 6, 5, 12, 0, tzinfo=timezone.utc),
        "category": "blitz",
        "is_fast": False,
        "rating": 1800,
        "opponent_rating": 1800,
        "termination": "Normal",
    }]
    with patch("app.services.extractor.storage.load_games", return_value=game_black):
        samples_b, _ = extract_samples("blackplayer")

    assert len(samples_b) == 2
    assert samples_b[0]["ply"] == 1
    assert samples_b[0]["prev_move_1"] == "e2e4"
    assert samples_b[0]["prev_move_2"] is None
    assert samples_b[0]["num_legal_moves"] == 20

    assert samples_b[1]["ply"] == 3
    assert samples_b[1]["prev_move_1"] == "g1f3"
    assert samples_b[1]["prev_move_2"] == "e7e5"


def test_extractor_phase_b_game_result_for_player_all():
    """
    Phase B.1 test:
    Hand-computed mapping:
    - Result 1-0: White target -> 1.0; Black target -> 0.0
    - Result 0-1: White target -> 0.0; Black target -> 1.0
    - Result 1/2-1/2: White target -> 0.5; Black target -> 0.5
    """
    from datetime import datetime, timezone

    for result, exp_w, exp_b in [("1-0", 1.0, 0.0), ("0-1", 0.0, 1.0), ("1/2-1/2", 0.5, 0.5)]:
        pgn = f'[Event "T"]\n[White "W"]\n[Black "B"]\n[Result "{result}"]\n[TimeControl "300+0"]\n\n1. e4 e5 {result}\n'
        gw = [{
            "game_id": f"gw_{result}", "pgn": pgn, "player_color": "white",
            "time_control": "300+0", "date": datetime(2025, 6, 6, tzinfo=timezone.utc),
            "category": "blitz", "is_fast": False, "rating": 1500, "opponent_rating": 1500, "termination": "Normal"
        }]
        gb = [{
            "game_id": f"gb_{result}", "pgn": pgn, "player_color": "black",
            "time_control": "300+0", "date": datetime(2025, 6, 6, tzinfo=timezone.utc),
            "category": "blitz", "is_fast": False, "rating": 1500, "opponent_rating": 1500, "termination": "Normal"
        }]
        with patch("app.services.extractor.storage.load_games", return_value=gw):
            sw, _ = extract_samples("w")
            assert sw[0]["game_result_for_player"] == exp_w

        with patch("app.services.extractor.storage.load_games", return_value=gb):
            sb, _ = extract_samples("b")
            assert sb[0]["game_result_for_player"] == exp_b


