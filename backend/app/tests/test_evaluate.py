"""
Unit tests for evaluation module and baselines.

All expected values are HAND-COMPUTED in docstrings as required.
"""

from __future__ import annotations

import math
import random
from typing import Any

import chess
import numpy as np
import pytest
import torch

from app.ml.baselines import (
    BOOK_EVAL_EPS,
    BookThenModel,
    CurrentModel,
    PositionFrequency,
    UniformLegal,
)
from app.ml.evaluate import (
    Predictor,
    bootstrap_by_game,
    bootstrap_distribution_by_game,
    evaluate,
    paired_bootstrap_by_game,
    row_to_ctx,
    score_position,
)
from app.ml.model import ChessPolicyNet


def test_1_fake_predictor_hand_computed():
    """
    Test 1: Fake predictor on the start position (20 legal moves).

    Hand-computed expected values:
    Position A (true = e2e4):
      Probs: e2e4 = 0.5, d2d4 = 0.3, g1f3 = 0.1, other 17 share 0.1 (0.1/17 each)
      Strictly better moves (g): 0
      Tied moves (e): 1 (e2e4 itself)
      top1 = clip((1 - 0) / 1, 0, 1) = 1.0
      top3 = clip((3 - 0) / 1, 0, 1) = 1.0
      nll = -ln(0.5) = ln(2) = 0.6931471805599453
      rank = 0 + (1 + 1)/2 = 1.0

    Position B (true = g1f3):
      Probs: e2e4 = 0.5, d2d4 = 0.3, g1f3 = 0.1, other 17 share 0.1 (0.1/17 each)
      Strictly better moves (g): 2 (e2e4, d2d4)
      Tied moves (e): 1 (g1f3 itself)
      top1 = clip((1 - 2) / 1, 0, 1) = 0.0
      top3 = clip((3 - 2) / 1, 0, 1) = 1.0
      nll = -ln(0.1) = ln(10) = 2.302585092994046
      rank = 2 + (1 + 1)/2 = 3.0

    Position C (true = a2a3):
      Probs: e2e4 = 0.5, d2d4 = 0.3, g1f3 = 0.1, c2c4 = 0.05, a2a3 = 0.01,
             other 15 share 0.04 (0.04/15 each)
      Strictly better moves (g): 4 (e2e4, d2d4, g1f3, c2c4)
      Tied moves (e): 1 (a2a3 itself)
      top1 = clip((1 - 4) / 1, 0, 1) = 0.0
      top3 = clip((3 - 4) / 1, 0, 1) = 0.0
      nll = -ln(0.01) = ln(100) = 4.605170185988092
      rank = 4 + (1 + 1)/2 = 5.0

    Dataset Means (3 positions across 3 different games):
      top1 mean = (1 + 0 + 0) / 3 = 1/3 = 0.333333333...
      top3 mean = (1 + 1 + 0) / 3 = 2/3 = 0.666666666...
      nll mean  = (ln(2) + ln(10) + ln(100)) / 3 = 7.60090245954 / 3 = 2.53363415318...
      rank mean = (1 + 3 + 5) / 3 = 9 / 3 = 3.0
    """
    board = chess.Board(chess.STARTING_FEN)
    legal_moves = [m.uci() for m in board.legal_moves]
    assert len(legal_moves) == 20

    # Dist 1 for A and B
    dist_ab: dict[str, float] = {
        "e2e4": 0.5,
        "d2d4": 0.3,
        "g1f3": 0.1,
    }
    rem_moves_ab = [m for m in legal_moves if m not in dist_ab]
    for m in rem_moves_ab:
        dist_ab[m] = 0.1 / len(rem_moves_ab)

    # Dist 2 for C
    dist_c: dict[str, float] = {
        "e2e4": 0.5,
        "d2d4": 0.3,
        "g1f3": 0.1,
        "c2c4": 0.05,
        "a2a3": 0.01,
    }
    rem_moves_c = [m for m in legal_moves if m not in dist_c]
    for m in rem_moves_c:
        dist_c[m] = 0.04 / len(rem_moves_c)

    logp_ab = {m: math.log(p) for m, p in dist_ab.items()}
    logp_c = {m: math.log(p) for m, p in dist_c.items()}

    # Verify per-position score_position
    score_a = score_position(logp_ab, board, "e2e4")
    assert score_a["top1"] == 1.0
    assert score_a["top3"] == 1.0
    assert score_a["nll"] == pytest.approx(math.log(2), rel=1e-6)
    assert score_a["rank"] == 1.0

    score_b = score_position(logp_ab, board, "g1f3")
    assert score_b["top1"] == 0.0
    assert score_b["top3"] == 1.0
    assert score_b["nll"] == pytest.approx(math.log(10), rel=1e-6)
    assert score_b["rank"] == 3.0

    score_c = score_position(logp_c, board, "a2a3")
    assert score_c["top1"] == 0.0
    assert score_c["top3"] == 0.0
    assert score_c["nll"] == pytest.approx(math.log(100), rel=1e-6)
    assert score_c["rank"] == 5.0

    class FakePredictor:
        def legal_log_probs(self, b: chess.Board, ctx: dict) -> dict[str, float]:
            if ctx.get("game_id") == "game_c":
                return logp_c
            return logp_ab

    rows = [
        {
            "game_id": "game_a",
            "ply": 1,
            "fen": chess.STARTING_FEN,
            "uci": "e2e4",
            "time_control": "180+2",
            "clock_before_move": 180.0,
            "player_rating": 1500,
            "opponent_rating": 1500,
            "halfmove_clock": 0,
            "target_berserk": False,
        },
        {
            "game_id": "game_b",
            "ply": 1,
            "fen": chess.STARTING_FEN,
            "uci": "g1f3",
            "time_control": "180+2",
            "clock_before_move": 180.0,
            "player_rating": 1500,
            "opponent_rating": 1500,
            "halfmove_clock": 0,
            "target_berserk": False,
        },
        {
            "game_id": "game_c",
            "ply": 1,
            "fen": chess.STARTING_FEN,
            "uci": "a2a3",
            "time_control": "180+2",
            "clock_before_move": 180.0,
            "player_rating": 1500,
            "opponent_rating": 1500,
            "halfmove_clock": 0,
            "target_berserk": False,
        },
    ]

    res = evaluate(FakePredictor(), rows)
    assert res["top1"] == pytest.approx(1.0 / 3.0, rel=1e-6)
    assert res["top3"] == pytest.approx(2.0 / 3.0, rel=1e-6)
    assert res["nll"] == pytest.approx((math.log(2) + math.log(10) + math.log(100)) / 3.0, rel=1e-6)
    assert res["rank"] == pytest.approx(3.0, rel=1e-6)


def test_2_uniform_legal_tie_rule():
    """
    Test 2: UniformLegal on start position, true e2e4.

    Hand-computed expected values:
      Total legal moves: 20
      Each move prob: 1/20 = 0.05
      logp for every move: ln(1/20) = -ln(20) = -2.99573227355...
      Strictly better moves than true move (g): 0 (all moves have identical logp)
      Tied moves (e): 20
      top1 credit = clip((1 - 0) / 20, 0, 1) = 1/20 = 0.05
      top3 credit = clip((3 - 0) / 20, 0, 1) = 3/20 = 0.15
      nll = -ln(1/20) = ln(20) = 2.99573227355...
      rank = 0 + (20 + 1) / 2 = 21 / 2 = 10.5
    """
    predictor = UniformLegal()
    board = chess.Board(chess.STARTING_FEN)
    ctx = {"game_id": "g1", "ply": 1}
    log_probs = predictor.legal_log_probs(board, ctx)

    scores = score_position(log_probs, board, "e2e4")
    assert scores["top1"] == pytest.approx(0.05, rel=1e-6)
    assert scores["top3"] == pytest.approx(0.15, rel=1e-6)
    assert scores["nll"] == pytest.approx(math.log(20), rel=1e-6)
    assert scores["rank"] == pytest.approx(10.5, rel=1e-6)


def test_3_position_frequency():
    """
    Test 3: PositionFrequency Laplace (+1) smoothing.

    Hand-computed expected values:
      Position seen 3 times in training:
        e2e4 x 2
        d2d4 x 1
      Position has 20 legal moves.
      Total count of legal moves in train = 2 + 1 = 3.
      Laplace smoothing denominator = total + n_legal = 3 + 20 = 23.
      p(e2e4) = (count + 1) / (total + n_legal) = (2 + 1) / 23 = 3/23
      p(d2d4) = (count + 1) / (total + n_legal) = (1 + 1) / 23 = 2/23
      Other 18 legal moves = (0 + 1) / 23 = 1/23 each
      Unseen position with 20 legal moves:
      uniform probability = 1/20 = 0.05
    """
    train_rows = [
        {"fen": chess.STARTING_FEN, "uci": "e2e4"},
        {"fen": chess.STARTING_FEN, "uci": "e2e4"},
        {"fen": chess.STARTING_FEN, "uci": "d2d4"},
    ]
    pf = PositionFrequency(train_rows)
    board = chess.Board(chess.STARTING_FEN)
    log_probs = pf.legal_log_probs(board, {})

    probs = {u: math.exp(lp) for u, lp in log_probs.items()}
    assert probs["e2e4"] == pytest.approx(3.0 / 23.0, rel=1e-6)
    assert probs["d2d4"] == pytest.approx(2.0 / 23.0, rel=1e-6)
    assert probs["g1f3"] == pytest.approx(1.0 / 23.0, rel=1e-6)
    assert sum(probs.values()) == pytest.approx(1.0, rel=1e-6)

    # Unseen key defaults to uniform
    unseen_board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    unseen_lps = pf.legal_log_probs(unseen_board, {})
    unseen_probs = {u: math.exp(lp) for u, lp in unseen_lps.items()}
    n_legal = len(list(unseen_board.legal_moves))
    for p in unseen_probs.values():
        assert p == pytest.approx(1.0 / n_legal, rel=1e-6)


def test_4_identical_predictors_paired_delta():
    """
    Test 4: Identical predictors paired bootstrap delta.

    Hand-computed expected values:
      When predictor A == predictor B:
      For all positions i: diff_i = A_i - B_i = 0.0
      delta point estimate = mean(diff) = 0.0
      In every bootstrap replication: delta_b = 0.0
      lo95 = 0.0, hi95 = 0.0
      delta == 0.0 and CI [0.0, 0.0] contains 0.
    """
    records = []
    for g in range(30):
        for p in range(5):
            records.append({
                "game_id": f"game_{g}",
                "ply": p,
                "top1": 0.5,
                "nll": 2.0,
            })

    delta, lo95, hi95 = paired_bootstrap_by_game(records, records, "top1", n=2000, seed=0)
    assert delta == 0.0
    assert lo95 == 0.0
    assert hi95 == 0.0
    assert lo95 <= 0.0 <= hi95


def test_5_game_level_resampling():
    """
    Test 5: Game-level resampling.

    Hand-computed expected values:
      1 game of 100 positions with metric = 1
      19 games of 1 position with metric = 0
      Total games G = 20.
      Drawing 20 games with replacement:
      P(big game is not drawn in 20 draws) = (19/20)^20 = (0.95)^20 ~ 0.3584859...
      When big game is not drawn, all sampled positions have metric = 0, so mean is exactly 0.
      When big game is drawn at least once, at least 100 positions have metric = 1, so mean > 0.
      Therefore, fraction of bootstrap statistics exactly 0 must be close to ~0.358,
      bounded within [0.30, 0.42].
      Position-level resampling (119 positions total, 100 with 1, 19 with 0) would have:
      P(all 0) = (19/119)^119 ~ 1e-95 ~ 0.
    """
    per_position = []
    # Game 0: 100 positions with metric = 1
    for p in range(100):
        per_position.append({"game_id": "game_0", "ply": p, "metric": 1.0})
    # Games 1..19: 1 position each with metric = 0
    for g in range(1, 20):
        per_position.append({"game_id": f"game_{g}", "ply": 0, "metric": 0.0})

    dist = bootstrap_distribution_by_game(per_position, "metric", n=2000, seed=0)
    frac_zero = float(np.mean(dist == 0.0))
    assert 0.30 <= frac_zero <= 0.42, f"frac_zero {frac_zero} outside [0.30, 0.42]"


def test_6_insufficient_evidence_and_illegal_move_raise():
    """
    Test 6:
    - Less than 20 games returns status insufficient_evidence for CI fields
      while point estimates remain present.
    - True move not legal raises ValueError.
    """
    # 6a: 5 games (< 20 games)
    rows = []
    for g in range(5):
        rows.append({
            "game_id": f"game_{g}",
            "ply": 1,
            "fen": chess.STARTING_FEN,
            "uci": "e2e4",
            "time_control": "180+2",
            "clock_before_move": 170.0,
            "player_rating": 1500,
            "opponent_rating": 1500,
            "halfmove_clock": 0,
            "target_berserk": False,
        })

    res = evaluate(UniformLegal(), rows)
    assert res["status"] == "insufficient_evidence"
    assert res["top1_ci"] == {"status": "insufficient_evidence"}
    assert res["top3_ci"] == {"status": "insufficient_evidence"}
    assert res["nll_ci"] == {"status": "insufficient_evidence"}
    assert res["rank_ci"] == {"status": "insufficient_evidence"}
    # Point estimates present as numeric floats
    assert isinstance(res["top1"], float)
    assert res["top1"] == pytest.approx(0.05, rel=1e-6)
    assert isinstance(res["nll"], float)

    # 6b: Illegal true move -> ValueError
    bad_rows = [
        {
            "game_id": "game_bad",
            "ply": 1,
            "fen": chess.STARTING_FEN,
            "uci": "e2e5",  # Illegal move in starting position
            "time_control": "180+2",
            "clock_before_move": 170.0,
            "player_rating": 1500,
            "opponent_rating": 1500,
            "halfmove_clock": 0,
            "target_berserk": False,
        }
    ]
    with pytest.raises(ValueError, match="not among legal moves"):
        evaluate(UniformLegal(), bad_rows)


def test_7_current_model_probabilities_and_properties():
    """
    Test 7:
    CurrentModel with a tiny untrained ChessPolicyNet:
    - Over 50 random positions (both colours) + promotion FEN,
      probabilities sum to 1 within 1e-4 and are 0 for illegal moves.
    - A single-legal-move position gives probability exactly 1 (logp 0).
    """
    torch.manual_seed(0)
    model = ChessPolicyNet(channels=16, num_res_blocks=1, hidden_dim=32, dropout=0.0)
    cm = CurrentModel(model)

    ctx = {
        "player_rating": 1500,
        "opponent_rating": 1500,
        "base_seconds": 300,
        "increment_seconds": 0,
        "ply": 10,
        "clock_before": 150.0,
        "clock_available": True,
        "halfmove_clock": 0,
    }

    # 1. Promotion FEN: White pawn promotes on e8
    promo_fen = "7k/4P3/8/8/8/8/8/4K3 w - - 0 1"
    promo_board = chess.Board(promo_fen)
    promo_lps = cm.legal_log_probs(promo_board, ctx)
    total_p = sum(math.exp(lp) for lp in promo_lps.values())
    assert abs(total_p - 1.0) < 1e-4
    for u in promo_lps:
        assert chess.Move.from_uci(u) in promo_board.legal_moves

    # 2. 50 random positions (both colours)
    rng = random.Random(42)
    positions: list[chess.Board] = []
    while len(positions) < 50:
        b = chess.Board()
        steps = rng.randint(4, 30)
        for _ in range(steps):
            if b.is_game_over() or not list(b.legal_moves):
                break
            b.push(rng.choice(list(b.legal_moves)))
        if not b.is_game_over() and list(b.legal_moves):
            positions.append(b)

    has_white = any(b.turn == chess.WHITE for b in positions)
    has_black = any(b.turn == chess.BLACK for b in positions)
    assert has_white and has_black

    for b in positions:
        lps = cm.legal_log_probs(b, ctx)
        tot = sum(math.exp(lp) for lp in lps.values())
        assert abs(tot - 1.0) < 1e-4
        for u in lps:
            assert chess.Move.from_uci(u) in b.legal_moves

    # 3. Single-legal-move position: Black to move, only move is a8b8
    single_fen = "k7/p7/2B5/8/8/8/8/K7 b - - 0 1"
    single_board = chess.Board(single_fen)
    assert len(list(single_board.legal_moves)) == 1
    single_lps = cm.legal_log_probs(single_board, ctx)
    assert len(single_lps) == 1
    only_move = list(single_lps.keys())[0]
    assert only_move == "a8b8"
    assert math.exp(single_lps[only_move]) == pytest.approx(1.0, rel=1e-6)
    assert single_lps[only_move] == pytest.approx(0.0, abs=1e-6)


def test_8_paired_bootstrap_known_deltas():
    """
    Test 8: Paired difference bootstrap with known deltas.

    Hand-computed expected values:
    (i) 30 games x 5 positions:
        Predictor A top1 = 0.75 everywhere, Predictor B top1 = 0.5 everywhere.
        For every position: diff = 0.75 - 0.5 = 0.25.
        Point estimate delta = 0.25.
        In every bootstrap sample of games: resampled mean diff = 0.25.
        Therefore: delta == 0.25, lo95 == 0.25, hi95 == 0.25.

    (ii) Predictors A and B with mismatched (game_id, ply) keys:
        Raises ValueError due to key alignment mismatch.

    (iii) 30 games, 1 position each:
        25 games with a=1.0, b=0.0 (diff = +1.0)
        5 games with a=0.0, b=1.0 (diff = -1.0)
        Sum of diffs = 25 * (+1.0) + 5 * (-1.0) = 20.0
        Mean delta = 20 / 30 = 2/3 = 0.666667.
        Since not all games have identical values, bootstrap distribution has variance:
        lo95 > 0.0, hi95 < 1.0, and lo95 < delta < hi95.
    """
    # (i) 30 games x 5 positions, a=0.75, b=0.5
    a_i = []
    b_i = []
    for g in range(30):
        for p in range(5):
            a_i.append({"game_id": f"game_{g}", "ply": p, "top1": 0.75})
            b_i.append({"game_id": f"game_{g}", "ply": p, "top1": 0.50})

    delta_i, lo_i, hi_i = paired_bootstrap_by_game(a_i, b_i, "top1", n=2000, seed=0)
    assert delta_i == 0.25
    assert lo_i == 0.25
    assert hi_i == 0.25

    # (ii) Mismatched keys -> ValueError
    a_bad = [{"game_id": "game_1", "ply": 0, "top1": 1.0}]
    b_bad = [{"game_id": "game_1", "ply": 1, "top1": 1.0}]
    with pytest.raises(ValueError):
        paired_bootstrap_by_game(a_bad, b_bad, "top1", n=2000, seed=0)

    # (iii) 30 games, 1 pos each: 25 games +1, 5 games -1
    a_iii = []
    b_iii = []
    for g in range(30):
        if g < 25:
            a_iii.append({"game_id": f"game_{g}", "ply": 0, "top1": 1.0})
            b_iii.append({"game_id": f"game_{g}", "ply": 0, "top1": 0.0})
        else:
            a_iii.append({"game_id": f"game_{g}", "ply": 0, "top1": 0.0})
            b_iii.append({"game_id": f"game_{g}", "ply": 0, "top1": 1.0})

    delta_iii, lo_iii, hi_iii = paired_bootstrap_by_game(a_iii, b_iii, "top1", n=2000, seed=0)
    assert delta_iii == pytest.approx(20.0 / 30.0, rel=1e-5)
    assert lo_iii > 0.0
    assert hi_iii < 1.0
    assert lo_iii < delta_iii < hi_iii


def test_9_book_then_model():
    """
    Test 9: BookThenModel interpolation and fallback.

    Hand-computed expected values:
    Start position (20 legal moves).
    Fake book query returns: {"e2e4": 0.6, "d2d4": 0.4}
    BOOK_EVAL_EPS = 0.001 (1e-3).
    1 - BOOK_EVAL_EPS = 0.999.
    Uniform prob over 20 legal moves = 1 / 20 = 0.05.
    Smoothing addition = 0.001 * (1 / 20) = 0.00005.

    p(e2e4) = 0.999 * 0.6 + 0.001 / 20 = 0.5994 + 0.00005 = 0.59945
    p(d2d4) = 0.999 * 0.4 + 0.00005 = 0.3996 + 0.00005 = 0.39965
    Each of the other 18 legal moves: 0.00005
    Sum = 0.59945 + 0.39965 + 18 * 0.00005 = 1.0

    Fallback check:
    When fake book query returns None, BookThenModel output must equal
    CurrentModel output exactly.
    """
    torch.manual_seed(0)
    model = ChessPolicyNet(channels=16, num_res_blocks=1, hidden_dim=32, dropout=0.0)
    cm = CurrentModel(model)

    class FakeBook:
        def __init__(self, ret: dict[str, float] | None):
            self.ret = ret

        def query(self, board: chess.Board) -> dict[str, float] | None:
            return self.ret

    book = FakeBook({"e2e4": 0.6, "d2d4": 0.4})
    btm = BookThenModel(cm, book)

    board = chess.Board(chess.STARTING_FEN)
    ctx = {
        "player_rating": 1500,
        "opponent_rating": 1500,
        "base_seconds": 300,
        "increment_seconds": 0,
        "ply": 0,
        "clock_before": None,
        "clock_available": False,
        "halfmove_clock": 0,
    }

    log_probs = btm.legal_log_probs(board, ctx)
    assert len(log_probs) == 20

    probs = {m: math.exp(lp) for m, lp in log_probs.items()}
    assert probs["e2e4"] == pytest.approx(0.59945, rel=1e-6)
    assert probs["d2d4"] == pytest.approx(0.39965, rel=1e-6)
    for m in board.legal_moves:
        u = m.uci()
        if u not in ("e2e4", "d2d4"):
            assert probs[u] == pytest.approx(0.00005, rel=1e-6)

    assert sum(probs.values()) == pytest.approx(1.0, rel=1e-6)

    # When book returns None, output matches CurrentModel exactly
    book_none = FakeBook(None)
    btm_none = BookThenModel(cm, book_none)
    lps_none = btm_none.legal_log_probs(board, ctx)
    lps_cm = cm.legal_log_probs(board, ctx)
    assert lps_none == lps_cm


def test_10_splits_and_ece():
    """
    Test 10: Subgroup splits and ECE calculation.

    Hand-computed expected values:
    Start position (20 legal moves).
    Predictor: p(e2e4) = 0.95, other 19 moves share 0.05 / 19 = 1 / 380 each.
    Four rows across four distinct games:
      P1: ply 10, true e2e4, base 180, clock 90, berserk False
          -> ply <= 20, clock_ratio = 90/180 = 0.5 (>0.3), berserk False
          -> top1 credit = 1.0, nll = -ln(0.95) = 0.051293
      P2: ply 30, true d2d4, base 180, clock 36, berserk False
          -> 21 <= ply <= 60, clock_ratio = 36/180 = 0.2 (0.1-0.3), berserk False
          -> top1 credit = 0.0, nll = -ln(1/380) = ln(380) = 5.940171
      P3: ply 70, true e2e4, base 180, clock 9, berserk False
          -> ply > 60, clock_ratio = 9/180 = 0.05 (<0.1), berserk False
          -> top1 credit = 1.0, nll = -ln(0.95) = 0.051293
      P4: ply 20, true g1f3, base 180, clock NaN (None), berserk True
          -> ply <= 20, clock_ratio = unknown, berserk True
          -> top1 credit = 0.0, nll = -ln(1/380) = ln(380) = 5.940171

    by_phase breakdown:
      ply<=20 (P1, P4): n = 2, top1 = (1 + 0)/2 = 0.5,
                        nll = (0.051293 + 5.940171)/2 = 2.995732 = ln(20)
      21-60 (P2): n = 1, top1 = 0.0, nll = ln(380) = 5.940171
      >60 (P3): n = 1, top1 = 1.0, nll = -ln(0.95) = 0.051293

    by_clock_ratio breakdown:
      >0.3: P1 (n = 1)
      0.1-0.3: P2 (n = 1)
      <0.1: P3 (n = 1)
      unknown: P4 (n = 1)

    by_berserk breakdown:
      true: P4 (n = 1, top1 = 0.0)
      false: P1, P2, P3 (n = 3, top1 = (1 + 0 + 1)/3 = 2/3 = 0.666667)

    ECE breakdown:
      All 4 positions have max prob = 0.95, so all fall in bin [0.9, 1.0].
      bin confidence = 0.95.
      bin accuracy = (1 + 0 + 1 + 0) / 4 = 0.5.
      ECE = |0.5 - 0.95| = 0.45.
    """
    board = chess.Board(chess.STARTING_FEN)
    legal_moves = [m.uci() for m in board.legal_moves]
    probs = {"e2e4": 0.95}
    for m in legal_moves:
        if m != "e2e4":
            probs[m] = 0.05 / 19.0
    log_probs = {m: math.log(p) for m, p in probs.items()}

    class FakePredictor:
        def legal_log_probs(self, b: chess.Board, ctx: dict) -> dict[str, float]:
            return log_probs

    rows = [
        {
            "game_id": "g1",
            "ply": 10,
            "fen": chess.STARTING_FEN,
            "uci": "e2e4",
            "time_control": "180",
            "clock_before_move": 90.0,
            "player_rating": 1500,
            "opponent_rating": 1500,
            "halfmove_clock": 0,
            "target_berserk": False,
        },
        {
            "game_id": "g2",
            "ply": 30,
            "fen": chess.STARTING_FEN,
            "uci": "d2d4",
            "time_control": "180",
            "clock_before_move": 36.0,
            "player_rating": 1500,
            "opponent_rating": 1500,
            "halfmove_clock": 0,
            "target_berserk": False,
        },
        {
            "game_id": "g3",
            "ply": 70,
            "fen": chess.STARTING_FEN,
            "uci": "e2e4",
            "time_control": "180",
            "clock_before_move": 9.0,
            "player_rating": 1500,
            "opponent_rating": 1500,
            "halfmove_clock": 0,
            "target_berserk": False,
        },
        {
            "game_id": "g4",
            "ply": 20,
            "fen": chess.STARTING_FEN,
            "uci": "g1f3",
            "time_control": "180",
            "clock_before_move": float("nan"),
            "player_rating": 1500,
            "opponent_rating": 1500,
            "halfmove_clock": 0,
            "target_berserk": True,
        },
    ]

    res = evaluate(FakePredictor(), rows)

    # by_phase checks
    bp = res["by_phase"]
    assert bp["ply<=20"]["n_positions"] == 2
    assert bp["ply<=20"]["top1"] == pytest.approx(0.5, rel=1e-5)
    assert bp["ply<=20"]["nll"] == pytest.approx(math.log(20), rel=1e-5)

    assert bp["21-60"]["n_positions"] == 1
    assert bp["21-60"]["top1"] == pytest.approx(0.0, abs=1e-6)
    assert bp["21-60"]["nll"] == pytest.approx(math.log(380), rel=1e-5)

    assert bp[">60"]["n_positions"] == 1
    assert bp[">60"]["top1"] == pytest.approx(1.0, rel=1e-5)
    assert bp[">60"]["nll"] == pytest.approx(-math.log(0.95), rel=1e-5)

    # by_clock_ratio checks
    bcr = res["by_clock_ratio"]
    assert bcr[">0.3"]["n_positions"] == 1
    assert bcr["0.1-0.3"]["n_positions"] == 1
    assert bcr["<0.1"]["n_positions"] == 1
    assert bcr["unknown"]["n_positions"] == 1

    # by_berserk checks
    bb = res["by_berserk"]
    assert bb["true"]["n_positions"] == 1
    assert bb["true"]["top1"] == pytest.approx(0.0, abs=1e-6)
    assert bb["false"]["n_positions"] == 3
    assert bb["false"]["top1"] == pytest.approx(2.0 / 3.0, rel=1e-5)

    # ECE check
    assert res["ece"] == pytest.approx(0.45, rel=1e-5)
