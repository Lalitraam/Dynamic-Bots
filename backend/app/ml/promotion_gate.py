"""
Promotion gate module for Phase D chess imitation learning.
Compares fine-tuned models against base models using game-level paired bootstrap
and dual-metric significance testing.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence
import numpy as np

from .. import config


def _extract_metric(pos: dict, metric: str | Callable[[dict], float]) -> float:
    if isinstance(metric, str):
        return float(pos[metric])
    return float(metric(pos))


def paired_game_bootstrap(
    per_pos_base: Sequence[dict],
    per_pos_ft: Sequence[dict],
    metric: str | Callable[[dict], float],
    n: int = getattr(config, "PROMOTION_GATE_BOOTSTRAP_N", 2000),
    seed: int = getattr(config, "PROMOTION_GATE_SEED", 0),
) -> tuple[float, float, float]:
    """
    Return (delta_mean, lo95, hi95). Resample GAMES; match by (game_id, ply).
    delta = ft - base.
    Raise ValueError if keys differ or lengths differ.
    """
    if len(per_pos_base) != len(per_pos_ft):
        raise ValueError(
            f"Lengths differ in paired_game_bootstrap: {len(per_pos_base)} vs {len(per_pos_ft)}"
        )

    def _make_key(pos: dict) -> tuple[Any, int]:
        return (pos["game_id"], int(pos["ply"]))

    base_dict = {_make_key(p): _extract_metric(p, metric) for p in per_pos_base}
    ft_dict = {_make_key(p): _extract_metric(p, metric) for p in per_pos_ft}

    if base_dict.keys() != ft_dict.keys():
        raise ValueError("Keys (game_id, ply) differ between per_pos_base and per_pos_ft")

    game_groups: dict[Any, list[float]] = {}
    diffs_all: list[float] = []

    for key, base_val in base_dict.items():
        gid = key[0]
        ft_val = ft_dict[key]
        diff = ft_val - base_val
        diffs_all.append(diff)
        if gid not in game_groups:
            game_groups[gid] = []
        game_groups[gid].append(diff)

    delta_mean = float(np.mean(diffs_all)) if diffs_all else 0.0

    gids = list(game_groups.keys())
    g_count = len(gids)
    if g_count == 0:
        return 0.0, 0.0, 0.0

    game_sums = np.array([sum(game_groups[g]) for g in gids], dtype=np.float64)
    game_counts = np.array([len(game_groups[g]) for g in gids], dtype=np.int64)

    rng = np.random.default_rng(seed)
    sampled_indices = rng.choice(g_count, size=(n, g_count), replace=True)

    rep_sums = game_sums[sampled_indices].sum(axis=1)
    rep_counts = game_counts[sampled_indices].sum(axis=1)
    dist = rep_sums / np.maximum(rep_counts, 1)

    lo95 = float(np.percentile(dist, 2.5))
    hi95 = float(np.percentile(dist, 97.5))

    return delta_mean, lo95, hi95


def _extract_nll(obj: Any) -> float:
    if isinstance(obj, (int, float)):
        return float(obj)
    if isinstance(obj, dict):
        if "nll" in obj:
            return float(obj["nll"])
        if "val_nll" in obj:
            return float(obj["val_nll"])
    if isinstance(obj, Sequence) and len(obj) > 0:
        return float(np.mean([p["nll"] for p in obj]))
    raise ValueError(f"Cannot extract NLL from object of type {type(obj)}")


def should_promote(
    test_split_rows_base: Sequence[dict],
    test_split_rows_ft: Sequence[dict],
    n_test_games: int,
    val_split_half1_base: Any = None,
    val_split_half1_ft: Any = None,
    val_split_half2_base: Any = None,
    val_split_half2_ft: Any = None,
) -> dict[str, Any]:
    """
    Rules from plan section 9 D.3:
      test_games >= 20:
        promote_ft iff CI_nll.hi < 0 AND CI_top1.lo > 0
      test_games < 20 with val halves:
        promote_ft iff FT NLL < base NLL on BOTH halves
      else: decision = "insufficient"
    Returns dict with decision, reason, delta_nll, ci_nll, delta_top1, ci_top1.
    """
    min_games = getattr(config, "PROMOTION_GATE_MIN_GAMES", 20)

    delta_nll: float | None = None
    ci_nll: tuple[float, float] | None = None
    delta_top1: float | None = None
    ci_top1: tuple[float, float] | None = None

    if test_split_rows_base and test_split_rows_ft:
        try:
            dn, lon, hin = paired_game_bootstrap(test_split_rows_base, test_split_rows_ft, "nll")
            dt, lot, hit = paired_game_bootstrap(test_split_rows_base, test_split_rows_ft, "top1")
            delta_nll = dn
            ci_nll = (lon, hin)
            delta_top1 = dt
            ci_top1 = (lot, hit)
        except Exception:
            pass

    if n_test_games >= min_games:
        if delta_nll is None or ci_nll is None or delta_top1 is None or ci_top1 is None:
            dn, lon, hin = paired_game_bootstrap(test_split_rows_base, test_split_rows_ft, "nll")
            dt, lot, hit = paired_game_bootstrap(test_split_rows_base, test_split_rows_ft, "top1")
            delta_nll = dn
            ci_nll = (lon, hin)
            delta_top1 = dt
            ci_top1 = (lot, hit)

        if ci_nll[1] < 0:
            decision = "promote_ft"
            reason = (
                f"FT significantly better: NLL 95% CI = "
                f"[{ci_nll[0]:.4f}, {ci_nll[1]:.4f}] (upper < 0)"
            )
        else:
            decision = "promote_base"
            reason = (
                f"FT not significantly better: NLL 95% CI = "
                f"[{ci_nll[0]:.4f}, {ci_nll[1]:.4f}]"
            )
    else:
        has_val_halves = all(
            v is not None
            for v in [val_split_half1_base, val_split_half1_ft, val_split_half2_base, val_split_half2_ft]
        )
        if has_val_halves:
            h1_base = _extract_nll(val_split_half1_base)
            h1_ft = _extract_nll(val_split_half1_ft)
            h2_base = _extract_nll(val_split_half2_base)
            h2_ft = _extract_nll(val_split_half2_ft)

            if h1_ft < h1_base and h2_ft < h2_base:
                decision = "promote_ft"
                reason = (
                    f"test_games ({n_test_games}) < {min_games}; "
                    f"FT NLL lower on both val halves (H1: {h1_ft:.4f} < {h1_base:.4f}, H2: {h2_ft:.4f} < {h2_base:.4f})"
                )
            else:
                decision = "promote_base"
                reason = (
                    f"test_games ({n_test_games}) < {min_games}; "
                    f"FT NLL not lower on both val halves (H1: {h1_ft:.4f} vs {h1_base:.4f}, H2: {h2_ft:.4f} vs {h2_base:.4f})"
                )
        else:
            decision = "insufficient"
            reason = f"test_games ({n_test_games}) < {min_games} and validation halves not provided"

    return {
        "decision": decision,
        "reason": reason,
        "delta_nll": delta_nll,
        "ci_nll": ci_nll,
        "delta_top1": delta_top1,
        "ci_top1": ci_top1,
    }
