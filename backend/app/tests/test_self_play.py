"""Tests for self-play harness (Phase SP)."""

from io import StringIO
import json
from pathlib import Path
import random
from typing import Literal
import chess
import chess.pgn
import pytest

from app import config
from app.analysis.self_play import (
    GameRecord,
    MoveRecord,
    play_match,
    load_pooled_think_time,
)
from app.engine.provider import MoveContext, MoveDecision, MoveProvider


class StubProvider(MoveProvider):
    """A trivial MoveProvider for self-play testing.

    Defaults to picking the first legal move, with think_ms=100 and source='policy'.
    Plays Fool's Mate when the position matches to ensure checkmate termination
    is exercised.
    """

    def __init__(
        self,
        think_ms: int = 100,
        moves: list[str] | None = None,
        cycling_moves: list[str] | None = None,
    ):
        self.think_ms = think_ms
        self.scripted_moves = list(moves) if moves is not None else None
        self.cycling_moves = list(cycling_moves) if cycling_moves is not None else None
        self._cycle_idx = 0

    def decide(self, ctx: MoveContext, rng: random.Random) -> MoveDecision:
        legal_moves = list(ctx.board.legal_moves)
        if not legal_moves:
            raise ValueError(f"No legal moves in position {ctx.board.fen()}")

        chosen_move = None

        if self.scripted_moves:
            cand_uci = self.scripted_moves.pop(0)
            cand = chess.Move.from_uci(cand_uci)
            if cand in ctx.board.legal_moves:
                chosen_move = cand

        if chosen_move is None and self.cycling_moves:
            cand_uci = self.cycling_moves[self._cycle_idx % len(self.cycling_moves)]
            self._cycle_idx += 1
            cand = chess.Move.from_uci(cand_uci)
            if cand in ctx.board.legal_moves:
                chosen_move = cand

        if chosen_move is None:
            # Check for immediate checkmate move
            for m in legal_moves:
                ctx.board.push(m)
                mate = ctx.board.is_checkmate()
                ctx.board.pop()
                if mate:
                    chosen_move = m
                    break

        if chosen_move is None:
            # Standard miniature (Fool's Mate) if available
            fools = ["f2f3", "e7e5", "g2g4", "d8h4"]
            if ctx.ply < len(fools):
                cand = chess.Move.from_uci(fools[ctx.ply])
                if cand in ctx.board.legal_moves:
                    chosen_move = cand

        if chosen_move is None:
            chosen_move = legal_moves[0]

        return MoveDecision(
            move=chosen_move,
            source="policy",
            think_ms=self.think_ms,
            top5=[(chosen_move.uci(), 1.0)],
            p_err=None,
            model_version="stub",
        )


def test_0_pooled_thinktime_loads():
    pooled_path = config.BASE_THINKTIME_DIR / "pooled_thinktime.json"
    if not pooled_path.exists():
        pytest.skip("SP-0 not yet built")

    with open(pooled_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "metadata" in data
    assert "buckets" in data
    buckets = data["buckets"]
    assert len(buckets) > 0, "Buckets dict is empty"

    valid_speeds = {"bullet", "blitz", "rapid", "classical"}
    valid_phases = {"opening", "middlegame", "endgame"}
    valid_ratios = {"<0.1", "0.1-0.3", ">0.3"}

    for key, samples in buckets.items():
        assert len(samples) <= config.BASE_THINKTIME_MAX_PER_BUCKET
        parts = key.split("|")
        assert len(parts) == 5, f"Invalid key format: {key}"
        rating_band_s, speed, phase, clock_ratio, is_forced_s = parts
        int(rating_band_s)  # Must be integer rating band
        assert speed in valid_speeds
        assert phase in valid_phases
        assert clock_ratio in valid_ratios
        assert is_forced_s in {"True", "False"}


def test_1_play_match_finishes_20_games():
    bot = StubProvider(think_ms=100)
    opp = StubProvider(think_ms=100)

    results = play_match(
        bot=bot,
        opponent=opp,
        n_games=20,
        base_s=180,
        inc_s=0,
        seed=0,
        bot_colors="alternate",
    )

    assert len(results) == 20
    allowed_terminations = {
        "checkmate",
        "stalemate",
        "insufficient_material",
        "fivefold",
        "75move",
        "flag",
        "cap",
    }

    has_checkmate = False
    for game in results:
        assert isinstance(game.pgn, str) and len(game.pgn) > 0
        assert len(game.moves) > 0
        assert game.termination in allowed_terminations
        if game.termination == "checkmate":
            has_checkmate = True
        assert game.termination != "cap", "at least one game goes to cap that shouldn't"

        if game.termination == "flag":
            last_move = game.moves[-1]
            assert last_move.clock_before_ms - last_move.think_ms <= 0

    assert has_checkmate, "Expected at least one checkmate game in 20 games"


def test_2_legal_pgns():
    bot = StubProvider(think_ms=100)
    opp = StubProvider(think_ms=100)

    results = play_match(
        bot=bot,
        opponent=opp,
        n_games=5,
        base_s=180,
        inc_s=2,
        seed=123,
    )

    for game in results:
        parsed = chess.pgn.read_game(StringIO(game.pgn))
        assert parsed is not None
        board = chess.Board()
        for move in parsed.mainline_moves():
            assert move in board.legal_moves
            board.push(move)


def test_3_clock_non_negative():
    # Test clock deductions and increments
    # With base_s=2, think_ms=800, without increment a player would flag on move 3.
    # With inc_s=1 (+1000ms), clock increases (+200ms net per move), so game does not flag.
    bot = StubProvider(think_ms=800, cycling_moves=["g1f3", "f3g1"])
    opp = StubProvider(think_ms=800, cycling_moves=["g8f6", "f6g8"])

    results = play_match(
        bot=bot,
        opponent=opp,
        n_games=2,
        base_s=2,
        inc_s=1,
        seed=7,
    )

    for game in results:
        assert game.termination != "flag", "at least one game terminates on flag when it shouldn't have"

        white_clock = 2 * 1000
        black_clock = 2 * 1000
        turn = chess.WHITE

        for m in game.moves:
            clock_before = white_clock if turn == chess.WHITE else black_clock
            assert m.clock_before_ms == clock_before
            new_clock = clock_before - m.think_ms
            if new_clock <= 0:
                assert game.termination == "flag"
                break
            new_clock += 1 * 1000  # inc_s = 1
            if turn == chess.WHITE:
                white_clock = new_clock
            else:
                black_clock = new_clock
            turn = not turn

        if game.termination != "flag":
            assert white_clock > 0
            assert black_clock > 0


def test_4_deterministic_with_seed():
    bot = StubProvider(think_ms=100)
    opp = StubProvider(think_ms=100)

    run_a = play_match(bot, opp, n_games=5, base_s=180, inc_s=2, seed=42)
    run_b = play_match(bot, opp, n_games=5, base_s=180, inc_s=2, seed=42)

    assert len(run_a) == len(run_b)
    for g_a, g_b in zip(run_a, run_b):
        assert g_a.result == g_b.result
        assert g_a.termination == g_b.termination
        assert g_a.bot_color == g_b.bot_color
        assert [m.uci for m in g_a.moves] == [m.uci for m in g_b.moves]


def test_5_bot_color_alternation():
    bot = StubProvider(think_ms=100)
    opp = StubProvider(think_ms=100)

    results = play_match(
        bot, opp, n_games=4, base_s=180, inc_s=0, seed=0, bot_colors="alternate"
    )

    assert results[0].bot_color == "white"
    assert results[1].bot_color == "black"
    assert results[2].bot_color == "white"
    assert results[3].bot_color == "black"


def test_6_cap_termination(monkeypatch):
    monkeypatch.setattr(config, "SP_MAX_PLIES", 20)

    # Irreversible pawn moves that will not repeat or checkmate before 20 plies
    w_pawns = ["a2a3", "b2b3", "c2c3", "d2d3", "e2e3", "g2g3", "h2h3", "a3a4", "b3b4", "c3c4"]
    b_pawns = ["a7a6", "b7b6", "c7c6", "d7d6", "e7e6", "g7g6", "h7h6", "a6a5", "b6b5", "c6c5"]

    bot = StubProvider(think_ms=1, moves=w_pawns)
    opp = StubProvider(think_ms=1, moves=b_pawns)

    results = play_match(
        bot, opp, n_games=1, base_s=180, inc_s=0, seed=0
    )

    assert results[0].termination == "cap"
    assert results[0].result == "1/2-1/2"
    assert len(results[0].moves) == 20


def test_7_fivefold_repetition_termination():
    bot = StubProvider(think_ms=100, cycling_moves=["g1f3", "f3g1"])
    opp = StubProvider(think_ms=100, cycling_moves=["g8f6", "f6g8"])

    results = play_match(
        bot, opp, n_games=1, base_s=180, inc_s=0, seed=0
    )

    assert results[0].termination in {"fivefold", "75move"}
    assert results[0].termination == "fivefold"
    assert results[0].result == "1/2-1/2"


def test_8_pgn_has_headers_and_legal_moves():
    bot = StubProvider(think_ms=100)
    opp = StubProvider(think_ms=100)

    results = play_match(
        bot, opp, n_games=1, base_s=180, inc_s=2, seed=99
    )

    first_game = results[0]
    parsed = chess.pgn.read_game(StringIO(first_game.pgn))
    assert parsed is not None

    for header in ["Event", "White", "Black", "Result"]:
        assert header in parsed.headers
        assert len(parsed.headers[header]) > 0

    board = chess.Board()
    for move in parsed.mainline_moves():
        assert move in board.legal_moves
        board.push(move)
