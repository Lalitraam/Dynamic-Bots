"""Tests for Authoritative GameClock (Phase F)."""

import pytest
from app.game.clock import GameClock


def test_clock_initialization():
    """Test clock initial state with given base and increment."""
    clock = GameClock(base_s=180, inc_s=2)
    assert clock.base_ms == 180000
    assert clock.inc_ms == 2000
    assert clock.remaining_ms == {"white": 180000, "black": 180000}
    assert clock.running_side == "white"
    assert clock.last_tick is None
    assert clock.started == {"white": False, "black": False}


def test_started_dict_prevents_deduction_on_move_1():
    """Test Lichess rule: started dict prevents time deduction on move 1."""
    t0 = 1000.0
    clock = GameClock(base_s=180, inc_s=2)
    clock.start(t0)

    # White takes 15 seconds on move 1
    t1 = t0 + 15.0
    ok = clock.on_move("white", t1)
    assert ok is True
    # Started flag is set, increment added, no deduction
    assert clock.started["white"] is True
    assert clock.started["black"] is False
    assert clock.remaining_ms["white"] == 182000
    assert clock.running_side == "black"
    assert clock.last_tick == t1

    # Black takes 10 seconds on move 1
    t2 = t1 + 10.0
    ok = clock.on_move("black", t2)
    assert ok is True
    assert clock.started["black"] is True
    assert clock.remaining_ms["black"] == 182000
    assert clock.running_side == "white"
    assert clock.last_tick == t2


def test_increment_and_deduction_on_subsequent_moves():
    """Test normal elapsed time deduction and increment addition after move 1."""
    t0 = 1000.0
    clock = GameClock(base_s=180, inc_s=2)
    clock.start(t0)

    # Move 1
    clock.on_move("white", t0 + 5.0)
    clock.on_move("black", t0 + 10.0)

    # Move 2 for White: takes 8 seconds (from t0 + 10.0 to t0 + 18.0)
    t_white_move2 = t0 + 18.0
    ok = clock.on_move("white", t_white_move2)
    assert ok is True
    # 182000 - 8000 + 2000 = 176000
    assert clock.remaining_ms["white"] == 176000
    assert clock.running_side == "black"

    # Mid-turn get_remaining check: Black thinking for 4 seconds
    t_check = t_white_move2 + 4.0
    rem = clock.get_remaining(t_check)
    assert rem["white"] == 176000
    # Black: 182000 - 4000 = 178000
    assert rem["black"] == 178000


def test_flagging_when_time_expires():
    """Test flagging returns False and clamps remaining time to 0 when time expires."""
    t0 = 100.0
    clock = GameClock(base_s=5, inc_s=1)
    clock.start(t0)

    # Move 1: no deduction
    clock.on_move("white", t0 + 1.0)  # remaining: 5000 + 1000 = 6000
    clock.on_move("black", t0 + 2.0)  # remaining: 5000 + 1000 = 6000

    # White takes 8 seconds (from t0 + 2.0 to t0 + 10.0), having only 6000ms
    t_flag = t0 + 10.0
    ok = clock.on_move("white", t_flag)
    assert ok is False
    assert clock.remaining_ms["white"] == 0


def test_get_remaining_clamps_to_zero():
    """Test get_remaining does not return negative time."""
    t0 = 100.0
    clock = GameClock(base_s=5, inc_s=0)
    clock.start(t0)

    clock.on_move("white", t0 + 1.0)
    clock.on_move("black", t0 + 2.0)

    # 10 seconds pass while White is thinking, with 5 seconds remaining
    rem = clock.get_remaining(t0 + 12.0)
    assert rem["white"] == 0
