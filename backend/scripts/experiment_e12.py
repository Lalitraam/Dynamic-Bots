"""E12 Experiment: Pattern-Conditioned vs Temperature-Only Sampling (Phase P)."""

from __future__ import annotations
import argparse
import math
from pathlib import Path
import random
import sys

# Ensure backend root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from app import config
from app.analysis.error_model import ErrorModel
from app.analysis.features import compute_features
from app.engine.model_cache import ModelCache
from app.engine.provider import MoveContext
from app.engine.provider_v3 import V3Provider, _build_board_planes, _build_meta
from app.engine.think_time import ThinkTimeModel


def run_e12(username: str = "the_king_crusher", max_rows: int | None = 1000, seed: int = 0):
    data_root = Path(config.DATA_ROOT)
    player_dir = data_root / "players" / username

    samples_path = player_dir / "samples_v3.parquet"
    if not samples_path.exists():
        samples_path = player_dir / "samples.parquet"
    if not samples_path.exists():
        raise FileNotFoundError(f"Samples file not found at {samples_path}")

    print(f"Loading test split from {samples_path}...")
    df_samples = pq.read_table(samples_path).to_pandas()
    if "split" in df_samples.columns:
        df_test = df_samples[df_samples["split"] == "test"].copy()
    else:
        df_test = df_samples.iloc[-5000:].copy()

    # If analysis_test.parquet exists, merge is_error
    analysis_test_path = player_dir / "analysis_test.parquet"
    if analysis_test_path.exists():
        try:
            df_an = pq.read_table(analysis_test_path, columns=["game_id", "ply", "is_error"]).to_pandas()
            df_test = df_test.merge(df_an, on=["game_id", "ply"], how="left")
            print(f"Merged is_error column from {analysis_test_path.name}")
        except Exception as e:
            print(f"Note: Could not merge analysis_test.parquet: {e}")

    if max_rows is not None and len(df_test) > max_rows:
        df_test = df_test.iloc[:max_rows].copy()

    n_rows = len(df_test)
    print(f"Running E12 evaluation on {n_rows} test positions...")

    cache = ModelCache(capacity=4, data_root=data_root)
    base_model = cache.get("base", 1)
    base_model.eval()
    tt_model = ThinkTimeModel.from_parquet(samples_path)

    error_model_path = player_dir / "models" / "v1" / "error_model.joblib"
    if not error_model_path.exists():
        raise FileNotFoundError(f"Error model not found at {error_model_path}. Train it first.")
    error_model = ErrorModel.load(error_model_path)
    print(f"Successfully loaded error model for branch (b) from: {error_model_path}")

    provider_a = V3Provider(
        username=username,
        model_version=1,
        model_cache=cache,
        think_time_model=tt_model,
        temperature=1.0,
        data_root=data_root,
        error_model=None,
    )
    provider_a.error_model = None  # Guarantee temperature-only fallback for baseline

    provider_b = V3Provider(
        username=username,
        model_version=1,
        model_cache=cache,
        think_time_model=tt_model,
        temperature=1.0,
        data_root=data_root,
        error_model=error_model,
    )

    results_a = {"matches": 0, "nlls": [], "errors_triggered": 0, "actual_errors": 0, "p_errs": []}
    results_b = {"matches": 0, "nlls": [], "errors_triggered": 0, "actual_errors": 0, "p_errs": []}

    # Estimate human error rate from analysis_train or fallback
    human_error_rate = 0.130
    analysis_train_path = player_dir / "analysis_train.parquet"
    if analysis_train_path.exists():
        df_tr_an = pq.read_table(analysis_train_path, columns=["is_error"]).to_pandas()
        human_error_rate = float(df_tr_an["is_error"].mean())

    for idx, (_, row) in enumerate(df_test.iterrows()):
        fen = str(row["fen"])
        human_uci = str(row["uci"])
        board = chess.Board(fen)

        clock_before = row.get("clock_before_move")
        clock_sec = float(clock_before) if pd.notna(clock_before) else None
        clock_ms = int(float(clock_before) * 1000.0) if pd.notna(clock_before) else 120000
        base_s = int(row.get("base_s", 180)) if pd.notna(row.get("base_s")) else 180
        inc_s = int(row.get("increment_s", 2)) if pd.notna(row.get("increment_s")) else 2
        ply = int(row.get("ply", 0)) if pd.notna(row.get("ply")) else 0

        prev_moves = []
        if pd.notna(row.get("prev_move_2")):
            prev_moves.append(str(row["prev_move_2"]))
        if pd.notna(row.get("prev_move_1")):
            prev_moves.append(str(row["prev_move_1"]))

        ctx = MoveContext(
            board=board,
            prev_moves=prev_moves,
            bot_rating=int(row.get("player_rating", config.DEFAULT_RATING)),
            opp_rating=int(row.get("opponent_rating", config.DEFAULT_RATING)),
            base_seconds=base_s,
            increment_seconds=inc_s,
            bot_clock_ms=clock_ms,
            opp_clock_ms=clock_ms,
            ply=ply,
        )

        # Human move ground truth error status
        is_human_err = bool(row["is_error"]) if ("is_error" in row and pd.notna(row["is_error"])) else False

        # (a) Temperature-only baseline
        rng_a = random.Random(seed + idx)
        dec_a = provider_a.decide(ctx, rng_a)
        matched_a = (dec_a.move.uci() == human_uci)
        if matched_a:
            results_a["matches"] += 1
            if is_human_err:
                results_a["actual_errors"] += 1
        prob_a = dict(dec_a.top5).get(human_uci, 1e-4)
        results_a["nlls"].append(-math.log(max(1e-4, prob_a)))
        p_err_a = float(getattr(config, "ERROR_MODEL_FALLBACK_RATE", 0.08))
        results_a["p_errs"].append(p_err_a)
        u_a = random.Random(seed + idx * 7).random()
        if u_a < p_err_a:
            results_a["errors_triggered"] += 1

        # (b) Pattern-conditioned: Compute 12 features using base model policy features
        bt = torch.from_numpy(_build_board_planes(ctx)).unsqueeze(0).float()
        mt = torch.from_numpy(_build_meta(ctx, ctx.bot_rating)).unsqueeze(0).float()
        with torch.no_grad():
            b_out = base_model(bt, mt)
            logits = b_out["policy_logits"][0].cpu().numpy()

        is_black = (board.turn == chess.BLACK)
        legal_lps = {}
        for m in board.legal_moves:
            f = (m.from_square ^ 56) if is_black else m.from_square
            t = (m.to_square ^ 56) if is_black else m.to_square
            legal_lps[m.uci()] = float(logits[f * 64 + t])
        max_l = max(legal_lps.values())
        sum_exp = sum(math.exp(v - max_l) for v in legal_lps.values())
        log_sum = max_l + math.log(max(1e-12, sum_exp))
        lps = {u: (v - log_sum) for u, v in legal_lps.items()}

        feats_b = compute_features(board, clock_before=clock_sec, base_s=base_s, ply=ply, policy_log_probs=lps)
        p_err_b = float(error_model.predict(feats_b)[0])
        results_b["p_errs"].append(p_err_b)

        u_b = random.Random(seed + idx * 7).random()
        if u_b < p_err_b:
            results_b["errors_triggered"] += 1

        rng_b = random.Random(seed + idx)
        dec_b = provider_b.decide(ctx, rng_b)
        matched_b = (dec_b.move.uci() == human_uci)
        if matched_b:
            results_b["matches"] += 1
            if is_human_err:
                results_b["actual_errors"] += 1
        prob_b = dict(dec_b.top5).get(human_uci, 1e-4)
        results_b["nlls"].append(-math.log(max(1e-4, prob_b)))

        if (idx + 1) % 500 == 0 or (idx + 1) == n_rows:
            print(f"Evaluated {idx + 1}/{n_rows} positions...")

    top1_a = results_a["matches"] / n_rows
    top1_b = results_b["matches"] / n_rows
    mean_nll_a = float(np.mean(results_a["nlls"]))
    mean_nll_b = float(np.mean(results_b["nlls"]))
    avg_p_err_a = float(np.mean(results_a["p_errs"]))
    avg_p_err_b = float(np.mean(results_b["p_errs"]))
    err_rate_a = results_a["errors_triggered"] / n_rows
    err_rate_b = results_b["errors_triggered"] / n_rows
    actual_err_rate_a = results_a["actual_errors"] / n_rows
    actual_err_rate_b = results_b["actual_errors"] / n_rows

    top1_drop = (top1_a - top1_b) * 100.0
    err_diff_a = abs(err_rate_a - human_error_rate)
    err_diff_b = abs(err_rate_b - human_error_rate)
    closer_to_human = (err_diff_b <= err_diff_a)
    adopted = closer_to_human and (top1_drop < 1.0)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("\n" + "=" * 74)
    print("E12 EXPERIMENT: Pattern-Conditioned vs Temperature-Only Sampling")
    print("=" * 74)
    print(f"Average predicted p_err:")
    print(f"  Branch (a) Fallback:           {avg_p_err_a * 100:.1f}% ({avg_p_err_a:.4f})")
    print(f"  Branch (b) Pattern ErrorModel: {avg_p_err_b * 100:.1f}% ({avg_p_err_b:.4f})")
    print("-" * 74)
    print(f"{'Metric':<26} {'(a) Temperature-Only':<24} {'(b) Pattern-Conditioned':<24}")
    print("-" * 74)
    print(f"{'Top-1 Accuracy':<26} {top1_a * 100:.1f}%{'':<19} {top1_b * 100:.1f}%")
    print(f"{'Mean NLL':<26} {mean_nll_a:.4f}{'':<18} {mean_nll_b:.4f}")
    print(f"{'Average p_err':<26} {avg_p_err_a * 100:.1f}%{'':<19} {avg_p_err_b * 100:.1f}%")
    print(f"{'Simulated Error Rate':<26} {err_rate_a * 100:.1f}%{'':<19} {err_rate_b * 100:.1f}%")
    print(f"{'Actual Error Rate (Proxy)':<26} {actual_err_rate_a * 100:.2f}%{'':<18} {actual_err_rate_b * 100:.2f}%")
    print(f"{'Human Error Rate':<26} {human_error_rate * 100:.1f}%{'':<19} {human_error_rate * 100:.1f}%")
    print("-" * 74)
    if adopted:
        status_str = f"Adopted (b) - Error rate closer to human ({err_rate_b*100:.1f}% vs {human_error_rate*100:.1f}%), top-1 drop {top1_drop:.2f} pt (< 1.0 pt)"
    else:
        status_str = f"Rejected (b) - Error rate diff {err_diff_b*100:.1f}% vs baseline {err_diff_a*100:.1f}%, top-1 drop {top1_drop:.2f} pt"
    print(f"Decision: {status_str}")
    print("=" * 74 + "\n")

    return {
        "top1_a": top1_a,
        "top1_b": top1_b,
        "mean_nll_a": mean_nll_a,
        "mean_nll_b": mean_nll_b,
        "avg_p_err_a": avg_p_err_a,
        "avg_p_err_b": avg_p_err_b,
        "err_rate_a": err_rate_a,
        "err_rate_b": err_rate_b,
        "human_error_rate": human_error_rate,
        "adopted": adopted,
    }


def main():
    parser = argparse.ArgumentParser(description="Run E12 Experiment: Pattern vs Temperature-Only.")
    parser.add_argument("--username", type=str, default="the_king_crusher")
    parser.add_argument("--max-rows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_e12(username=args.username, max_rows=args.max_rows, seed=args.seed)


if __name__ == "__main__":
    main()
