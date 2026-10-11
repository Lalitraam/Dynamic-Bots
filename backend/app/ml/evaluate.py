"""
Evaluation module for chess imitation learning models and baselines.

Provides:
- Predictor Protocol
- row_to_ctx helper (matches dataset.py metadata extraction)
- Per-position evaluation metrics (top-1, top-3, NLL, rank) with exact tie handling
- Game-level block bootstrap confidence intervals and paired difference test
- Expected Calibration Error (ECE)
- Subgroup splits by game phase, clock ratio, and berserk flag

CRITICAL: This module must NOT import PyTorch.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Protocol, Sequence

import chess
import numpy as np

from ..services import classifier


class Predictor(Protocol):
    """Protocol for chess move probability predictors."""

    def legal_log_probs(self, board: chess.Board, ctx: dict) -> dict[str, float]:
        """UCI -> log prob over LEGAL moves only; exp() sums to 1 (promotions as e7e8q etc.)."""
        ...


def row_to_ctx(row: Any) -> dict:
    """
    Build context dictionary from a sample row exactly like ml/dataset.py builds meta.

    Extracts:
    - player_rating
    - opponent_rating
    - base_seconds (via classifier.parse_time_control)
    - increment_seconds (via classifier.parse_time_control)
    - ply
    - clock_before (None if NaN)
    - clock_available
    - halfmove_clock
    """
    tc = row.get("time_control") if isinstance(row, dict) else row["time_control"]
    parsed_tc = classifier.parse_time_control(tc) if tc else None
    base_seconds = parsed_tc[0] if parsed_tc else 0
    inc_seconds = parsed_tc[1] if parsed_tc else 0

    cb = row.get("clock_before_move") if isinstance(row, dict) else row["clock_before_move"]
    if cb is None:
        clock_before = None
        clock_available = False
    else:
        try:
            val = float(cb)
            if math.isnan(val):
                clock_before = None
                clock_available = False
            else:
                clock_before = val
                clock_available = True
        except (ValueError, TypeError):
            clock_before = None
            clock_available = False

    def _get(key: str, default: Any = None) -> Any:
        if isinstance(row, dict):
            return row.get(key, default)
        try:
            val = row[key]
            return default if val is None else val
        except (KeyError, IndexError):
            return default

    player_rating = _get("player_rating", 1500)
    opponent_rating = _get("opponent_rating", 1500)
    ply = int(_get("ply", 0))
    halfmove_clock = int(_get("halfmove_clock", 0))

    return {
        "player_rating": int(player_rating) if player_rating is not None else 1500,
        "opponent_rating": int(opponent_rating) if opponent_rating is not None else 1500,
        "base_seconds": int(base_seconds),
        "increment_seconds": int(inc_seconds),
        "ply": ply,
        "clock_before": clock_before,
        "clock_available": clock_available,
        "halfmove_clock": halfmove_clock,
        "game_id": _get("game_id"),
    }


def score_position(
    log_probs: dict[str, float],
    board: chess.Board,
    true_uci: str,
) -> dict[str, float]:
    """
    Score a single position under the mandatory tie rule.

    TIE RULE:
    - treat |logp diff| <= 1e-9 as equal
    - g = #moves strictly better than the true move
    - e = #moves tied with it (incl. itself)
    - top-k credit = clip((k - g) / e, 0, 1)
    - rank = g + (e + 1) / 2
    - True move not among legal moves -> raise ValueError
    - Validate sum(exp(logp)) within 1e-4 of 1, else raise ValueError
    """
    try:
        true_move = chess.Move.from_uci(true_uci)
    except Exception as exc:
        raise ValueError(
            f"True move '{true_uci}' cannot be parsed as UCI in position {board.fen()}"
        ) from exc

    if true_move not in board.legal_moves:
        raise ValueError(
            f"True move '{true_uci}' is not among legal moves in position {board.fen()}"
        )

    if true_uci not in log_probs:
        raise ValueError(
            f"True move '{true_uci}' not provided in predictor log_probs for {board.fen()}"
        )

    # Validate sum(exp(logp)) within 1e-4 of 1
    total_prob = sum(math.exp(lp) for lp in log_probs.values())
    if abs(total_prob - 1.0) > 1e-4:
        raise ValueError(
            f"Sum of exp(logp) is {total_prob:.6f}, which is not within 1e-4 of 1.0 "
            f"in position {board.fen()}"
        )

    target_lp = log_probs[true_uci]
    g = 0
    e = 0
    max_prob = 0.0

    for lp in log_probs.values():
        p = math.exp(lp)
        if p > max_prob:
            max_prob = p

        diff = lp - target_lp
        if diff > 1e-9:
            g += 1
        elif abs(diff) <= 1e-9:
            e += 1

    top1 = max(0.0, min(1.0, (1.0 - g) / e))
    top3 = max(0.0, min(1.0, (3.0 - g) / e))
    nll = -target_lp
    rank = g + (e + 1.0) / 2.0

    return {
        "top1": float(top1),
        "top3": float(top3),
        "nll": float(nll),
        "rank": float(rank),
        "max_prob": float(max_prob),
    }


def _extract_metric(pos: dict, metric: str | Callable[[dict], float]) -> float:
    if isinstance(metric, str):
        return float(pos[metric])
    return float(metric(pos))


def bootstrap_distribution_by_game(
    per_position: Sequence[dict],
    metric: str | Callable[[dict], float],
    n: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """
    Resample GAMES (draw G game ids with replacement, pool all their positions)
    and compute the mean metric distribution.
    """
    if not per_position:
        return np.zeros(n, dtype=np.float64)

    game_groups: dict[Any, list[float]] = {}
    for pos in per_position:
        gid = pos["game_id"]
        val = _extract_metric(pos, metric)
        if gid not in game_groups:
            game_groups[gid] = []
        game_groups[gid].append(val)

    gids = list(game_groups.keys())
    g_count = len(gids)
    if g_count == 0:
        return np.zeros(n, dtype=np.float64)

    game_sums = np.array([sum(game_groups[g]) for g in gids], dtype=np.float64)
    game_counts = np.array([len(game_groups[g]) for g in gids], dtype=np.int64)

    rng = np.random.default_rng(seed)
    sampled_indices = rng.choice(g_count, size=(n, g_count), replace=True)

    rep_sums = game_sums[sampled_indices].sum(axis=1)
    rep_counts = game_counts[sampled_indices].sum(axis=1)

    return rep_sums / rep_counts


def bootstrap_by_game(
    per_position: Sequence[dict],
    metric: str | Callable[[dict], float],
    n: int = 2000,
    seed: int = 0,
) -> tuple[float, float | dict[str, str], float | dict[str, str]]:
    """
    Compute game-level bootstrap confidence interval: (mean, lo95, hi95).
    mean = point estimate on full data.
    CI = 2.5 and 97.5 percentiles.
    If < 20 distinct games, returns (mean, {"status": "insufficient_evidence"}, {"status": "insufficient_evidence"}).
    """
    if not per_position:
        return 0.0, 0.0, 0.0

    all_vals = [_extract_metric(p, metric) for p in per_position]
    point_estimate = float(np.mean(all_vals))

    distinct_games = {p["game_id"] for p in per_position}
    if len(distinct_games) < 20:
        return point_estimate, {"status": "insufficient_evidence"}, {"status": "insufficient_evidence"}

    dist = bootstrap_distribution_by_game(per_position, metric, n=n, seed=seed)
    lo95 = float(np.percentile(dist, 2.5))
    hi95 = float(np.percentile(dist, 97.5))

    return point_estimate, lo95, hi95


def paired_bootstrap_by_game(
    a: Sequence[dict],
    b: Sequence[dict],
    metric: str | Callable[[dict], float],
    n: int = 2000,
    seed: int = 0,
) -> tuple[float, float | dict[str, str], float | dict[str, str]]:
    """
    Paired difference bootstrap by game: (delta, lo95, hi95).
    delta = a - b.
    a, b aligned by (game_id, ply); raise ValueError if keys differ.
    """
    if len(a) != len(b):
        raise ValueError(f"Lengths differ in paired_bootstrap_by_game: {len(a)} vs {len(b)}")

    def _make_key(pos: dict) -> tuple[Any, int]:
        return (pos["game_id"], int(pos["ply"]))

    a_dict = {_make_key(p): _extract_metric(p, metric) for p in a}
    b_dict = {_make_key(p): _extract_metric(p, metric) for p in b}

    if a_dict.keys() != b_dict.keys():
        raise ValueError("Keys (game_id, ply) differ between predictors a and b")

    diff_records: list[dict] = []
    for key, val_a in a_dict.items():
        val_b = b_dict[key]
        diff_records.append({
            "game_id": key[0],
            "ply": key[1],
            "diff": val_a - val_b,
        })

    delta_vals = [r["diff"] for r in diff_records]
    delta = float(np.mean(delta_vals))

    distinct_games = {r["game_id"] for r in diff_records}
    if len(distinct_games) < 20:
        return delta, {"status": "insufficient_evidence"}, {"status": "insufficient_evidence"}

    dist = bootstrap_distribution_by_game(diff_records, "diff", n=n, seed=seed)
    lo95 = float(np.percentile(dist, 2.5))
    hi95 = float(np.percentile(dist, 97.5))

    return delta, lo95, hi95


def compute_ece(per_position: Sequence[dict], n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE) over 10 bins of top-1 probability.
    confidence = max prob
    correct = top-1 credit
    """
    if not per_position:
        return 0.0

    confs = np.array([p["max_prob"] for p in per_position], dtype=np.float64)
    accs = np.array([p["top1"] for p in per_position], dtype=np.float64)
    total_n = len(per_position)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        low = bin_boundaries[i]
        high = bin_boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = (confs >= low) & (confs <= high)
        else:
            in_bin = (confs >= low) & (confs < high)

        n_in_bin = int(np.sum(in_bin))
        if n_in_bin > 0:
            bin_acc = float(np.mean(accs[in_bin]))
            bin_conf = float(np.mean(confs[in_bin]))
            ece += (n_in_bin / total_n) * abs(bin_acc - bin_conf)

    return float(ece)


def _compute_group_metrics(records: Sequence[dict]) -> dict[str, Any]:
    if not records:
        return {
            "n_positions": 0,
            "top1": 0.0,
            "top3": 0.0,
            "nll": 0.0,
            "rank": 0.0,
        }
    return {
        "n_positions": len(records),
        "top1": float(np.mean([p["top1"] for p in records])),
        "top3": float(np.mean([p["top3"] for p in records])),
        "nll": float(np.mean([p["nll"] for p in records])),
        "rank": float(np.mean([p["rank"] for p in records])),
    }


def evaluate(
    predictor: Predictor,
    rows: Iterable[dict],
    phase_bins: tuple[int, int] = (20, 60),
) -> dict[str, Any]:
    """
    Evaluate predictor on an iterable of sample rows.

    Aggregates:
    - Overall means: top1, top3, nll, rank
    - Bootstrap CIs for top1, top3, nll, rank
    - Split by phase (ply <= phase_bins[0], phase_bins[0] < ply <= phase_bins[1], ply > phase_bins[1])
    - Split by clock ratio clock_before_move / base_seconds (<0.1, 0.1-0.3, >0.3, unknown)
    - Split by target_berserk (true, false)
    - ECE over 10 bins of top-1 probability
    """
    per_position: list[dict[str, Any]] = []

    for row in rows:
        fen = row["fen"]
        board = chess.Board(fen)
        true_uci = row["uci"]
        ctx = row_to_ctx(row)

        log_probs = predictor.legal_log_probs(board, ctx)
        scores = score_position(log_probs, board, true_uci)

        berserk_val = row.get("target_berserk", False) if isinstance(row, dict) else row["target_berserk"]
        if berserk_val is None or (isinstance(berserk_val, float) and math.isnan(berserk_val)):
            is_berserk = False
        else:
            is_berserk = bool(berserk_val)

        rec = {
            "game_id": row["game_id"],
            "ply": int(row["ply"]),
            "top1": scores["top1"],
            "top3": scores["top3"],
            "nll": scores["nll"],
            "rank": scores["rank"],
            "max_prob": scores["max_prob"],
            "clock_before_move": ctx["clock_before"],
            "base_seconds": ctx["base_seconds"],
            "target_berserk": is_berserk,
        }
        per_position.append(rec)

    n_positions = len(per_position)
    distinct_games = {p["game_id"] for p in per_position}
    n_games = len(distinct_games)

    if n_positions == 0:
        return {
            "n_games": 0,
            "n_positions": 0,
            "status": "insufficient_evidence",
            "ci": {"status": "insufficient_evidence"},
            "top1": 0.0,
            "top1_ci": {"status": "insufficient_evidence"},
            "top3": 0.0,
            "top3_ci": {"status": "insufficient_evidence"},
            "nll": 0.0,
            "nll_ci": {"status": "insufficient_evidence"},
            "rank": 0.0,
            "rank_ci": {"status": "insufficient_evidence"},
            "ece": 0.0,
            "by_phase": {},
            "by_clock_ratio": {},
            "by_berserk": {},
            "per_position": [],
        }

    mean_top1 = float(np.mean([p["top1"] for p in per_position]))
    mean_top3 = float(np.mean([p["top3"] for p in per_position]))
    mean_nll = float(np.mean([p["nll"] for p in per_position]))
    mean_rank = float(np.mean([p["rank"] for p in per_position]))

    if n_games < 20:
        top1_ci: tuple[float, float] | dict[str, str] = {"status": "insufficient_evidence"}
        top3_ci: tuple[float, float] | dict[str, str] = {"status": "insufficient_evidence"}
        nll_ci: tuple[float, float] | dict[str, str] = {"status": "insufficient_evidence"}
        rank_ci: tuple[float, float] | dict[str, str] = {"status": "insufficient_evidence"}
        status_str = "insufficient_evidence"
    else:
        _, lo1, hi1 = bootstrap_by_game(per_position, "top1")
        _, lo3, hi3 = bootstrap_by_game(per_position, "top3")
        _, lon, hin = bootstrap_by_game(per_position, "nll")
        _, lor, hir = bootstrap_by_game(per_position, "rank")
        top1_ci = (float(lo1), float(hi1))
        top3_ci = (float(lo3), float(hi3))
        nll_ci = (float(lon), float(hin))
        rank_ci = (float(lor), float(hir))
        status_str = "ok"

    # By phase split
    p_open = [p for p in per_position if p["ply"] <= phase_bins[0]]
    p_mid = [p for p in per_position if phase_bins[0] < p["ply"] <= phase_bins[1]]
    p_end = [p for p in per_position if p["ply"] > phase_bins[1]]

    by_phase = {
        f"ply<={phase_bins[0]}": _compute_group_metrics(p_open),
        f"{phase_bins[0]+1}-{phase_bins[1]}": _compute_group_metrics(p_mid),
        f">{phase_bins[1]}": _compute_group_metrics(p_end),
    }

    # By clock ratio split
    def _clock_cat(p: dict) -> str:
        bs = p["base_seconds"]
        cb = p["clock_before_move"]
        if bs <= 0 or cb is None:
            return "unknown"
        ratio = cb / bs
        if ratio < 0.1:
            return "<0.1"
        if ratio <= 0.3:
            return "0.1-0.3"
        return ">0.3"

    clock_groups: dict[str, list[dict]] = {
        "<0.1": [],
        "0.1-0.3": [],
        ">0.3": [],
        "unknown": [],
    }
    for p in per_position:
        clock_groups[_clock_cat(p)].append(p)

    by_clock_ratio = {k: _compute_group_metrics(v) for k, v in clock_groups.items()}

    # By berserk split
    berserk_groups: dict[str, list[dict]] = {
        "true": [p for p in per_position if p["target_berserk"]],
        "false": [p for p in per_position if not p["target_berserk"]],
    }
    by_berserk = {k: _compute_group_metrics(v) for k, v in berserk_groups.items()}

    # ECE
    ece_val = compute_ece(per_position, n_bins=10)

    return {
        "n_games": n_games,
        "n_positions": n_positions,
        "status": status_str,
        "ci": {"status": "insufficient_evidence"} if n_games < 20 else {"status": "ok"},
        "top1": mean_top1,
        "top1_ci": top1_ci,
        "top3": mean_top3,
        "top3_ci": top3_ci,
        "nll": mean_nll,
        "nll_ci": nll_ci,
        "rank": mean_rank,
        "rank_ci": rank_ci,
        "ece": ece_val,
        "by_phase": by_phase,
        "by_clock_ratio": by_clock_ratio,
        "by_berserk": by_berserk,
        "per_position": per_position,
    }


# TODO: implement evaluate_cached (needs Phase B cache)
