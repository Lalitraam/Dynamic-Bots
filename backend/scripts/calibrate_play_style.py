"""Paired Replay Calibration: Grid Search over t_base, top_p, and error_multiplier."""

from __future__ import annotations
import argparse
import math
from pathlib import Path
import random
import sys

# Ensure backend root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import chess
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from app import config
from app.analysis.error_model import ErrorModel
from app.analysis.features import compute_features
from app.analysis.stockfish import StockfishAnalyser, compute_cp_loss
from app.engine.model_cache import ModelCache
from app.engine.provider import MoveContext
from app.engine.provider_v3 import _build_board_planes, _build_meta, _apply_top_p, _sample_from


def run_calibration(
    username: str = "the_king_crusher",
    max_rows: int | None = None,
    seed: int = 42,
):
    data_root = Path(config.DATA_ROOT)
    player_dir = data_root / "players" / username

    samples_path = player_dir / "samples_v3.parquet"
    if not samples_path.exists():
        samples_path = player_dir / "samples.parquet"
    if not samples_path.exists():
        raise FileNotFoundError(f"Samples file not found at {samples_path}")

    print(f"Loading test split from {samples_path}...", flush=True)
    df_samples = pq.read_table(samples_path).to_pandas()
    if "split" in df_samples.columns:
        df_test = df_samples[df_samples["split"] == "test"].copy()
    else:
        df_test = df_samples.iloc[-5004:].copy()

    # Load player median rating
    median_rating = config.DEFAULT_RATING
    if "player_rating" in df_test.columns and not df_test["player_rating"].empty:
        median_rating = int(df_test["player_rating"].median())
    print(f"Using player median rating: {median_rating}", flush=True)

    # Merge cp_loss from analysis_test.parquet if available
    analysis_test_path = player_dir / "analysis_test.parquet"
    if analysis_test_path.exists():
        try:
            df_an = pq.read_table(analysis_test_path, columns=["game_id", "ply", "cp_loss"]).to_pandas()
            df_test = df_test.merge(df_an, on=["game_id", "ply"], how="left")
            print(f"Merged cp_loss from {analysis_test_path.name} ({len(df_an)} rows)", flush=True)
        except Exception as e:
            print(f"Note: Could not merge analysis_test.parquet: {e}", flush=True)

    if max_rows is not None and len(df_test) > max_rows:
        df_test = df_test.iloc[:max_rows].copy()

    n_rows = len(df_test)
    print(f"Running Paired Replay Calibration on {n_rows} test positions...\n", flush=True)

    # Load fine-tuned model
    cache = ModelCache(capacity=4, data_root=data_root)
    model = cache.get(username, 1)
    model.eval()
    print(f"Loaded fine-tuned model for {username}:v1", flush=True)

    # Load error model
    error_model_path = player_dir / "models" / "v1" / "error_model.joblib"
    if not error_model_path.exists():
        raise FileNotFoundError(f"Error model not found at {error_model_path}. Train it first.")
    error_model = ErrorModel.load(error_model_path)
    print(f"Loaded error model from {error_model_path}", flush=True)

    # Initialize StockfishAnalyser
    if not Path(config.STOCKFISH_PATH).exists():
        raise FileNotFoundError(f"Stockfish not found at {config.STOCKFISH_PATH}")
    analyser = StockfishAnalyser(depth=config.STOCKFISH_DEPTH)
    print(f"Initialized StockfishAnalyser (depth={config.STOCKFISH_DEPTH})\n", flush=True)

    # Define grid
    t_base_list = [0.7, 0.8, 0.9, 1.0]
    top_p_list = [0.85, 0.90, 0.95]
    error_mult_list = [1.0, 1.1, 1.3]

    configs = []
    for tb in t_base_list:
        for tp in top_p_list:
            for em in error_mult_list:
                configs.append({"t_base": tb, "top_p": tp, "error_mult": em})
    n_configs = len(configs)
    print(f"Testing grid of {n_configs} configurations...", flush=True)

    # Cache for Stockfish evaluations: (fen, move_uci) -> cp_loss
    sf_cache: dict[tuple[str, str], float] = {}

    # Store choices for each configuration: list of move_uci per config
    cfg_moves: list[list[str]] = [[] for _ in range(n_configs)]

    positions_data = []

    print("Step 1/2: Evaluating positions and sampling moves across configurations...", flush=True)
    for idx, (_, row) in enumerate(df_test.iterrows()):
        fen = str(row["fen"])
        human_uci = str(row["uci"])
        human_cp_loss = row.get("cp_loss")
        if pd.notna(human_cp_loss):
            sf_cache[(fen, human_uci)] = float(human_cp_loss)

        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            continue

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
            bot_rating=median_rating,
            opp_rating=int(row.get("opponent_rating", config.DEFAULT_RATING)),
            base_seconds=base_s,
            increment_seconds=inc_s,
            bot_clock_ms=clock_ms,
            opp_clock_ms=clock_ms,
            ply=ply,
        )

        # Forward pass on fine-tuned model
        bt = torch.from_numpy(_build_board_planes(ctx)).unsqueeze(0).float()
        mt = torch.from_numpy(_build_meta(ctx, ctx.bot_rating)).unsqueeze(0).float()
        with torch.no_grad():
            out = model(bt, mt)
            raw_logits = out["policy_logits"][0].cpu().numpy()

        is_black = (board.turn == chess.BLACK)
        legal_lps = {}
        for m in legal_moves:
            f = (m.from_square ^ 56) if is_black else m.from_square
            t = (m.to_square ^ 56) if is_black else m.to_square
            legal_lps[m.uci()] = float(raw_logits[f * 64 + t])

        max_l = max(legal_lps.values())
        sum_exp = sum(math.exp(v - max_l) for v in legal_lps.values())
        log_sum = max_l + math.log(max(1e-12, sum_exp))
        norm_lps = {u: (v - log_sum) for u, v in legal_lps.items()}

        feats = compute_features(board, clock_before=clock_sec, base_s=base_s, ply=ply, policy_log_probs=norm_lps)
        p_err = float(error_model.predict(feats)[0])
        u_err = random.Random(seed + idx * 13).random()
        is_error_triggered = (u_err < p_err)

        legal_arr = np.array([legal_lps[m.uci()] for m in legal_moves], dtype=np.float64)

        positions_data.append({
            "fen": fen,
            "board": board,
            "human_uci": human_uci,
            "human_cp_loss": float(human_cp_loss) if pd.notna(human_cp_loss) else None,
        })

        # Sample for each configuration
        for c_idx, cfg in enumerate(configs):
            tb = cfg["t_base"]
            tp = cfg["top_p"]
            em = cfg["error_mult"]

            eff_temp = max(1e-4, tb * em) if is_error_triggered else max(1e-4, tb)
            scaled_logits = legal_arr / eff_temp
            l_max = scaled_logits.max()
            probs = np.exp(scaled_logits - l_max)
            s_sum = probs.sum()
            probs = probs / s_sum if s_sum > 0 else (np.ones_like(probs) / len(probs))

            filtered = _apply_top_p(probs, tp)
            f_sum = filtered.sum()
            final_p = (filtered / f_sum) if f_sum > 0 else probs

            rng_sample = random.Random(seed + idx)
            chosen_idx = _sample_from(final_p, rng_sample)
            chosen_move = legal_moves[chosen_idx]
            cfg_moves[c_idx].append(chosen_move.uci())

        if (idx + 1) % 500 == 0 or (idx + 1) == n_rows:
            print(f"Evaluated & sampled {idx + 1}/{n_rows} positions...", flush=True)

    # Step 2: Compute Stockfish evaluations with caching
    print("\nStep 2/2: Computing Stockfish centipawn losses with caching...", flush=True)
    # Collect all needed (fen, move_uci) pairs
    needed_pairs: list[tuple[int, str]] = []
    seen = set()
    for c_idx in range(n_configs):
        for pos_idx, move_uci in enumerate(cfg_moves[c_idx]):
            fen = positions_data[pos_idx]["fen"]
            pair = (fen, move_uci)
            if pair not in sf_cache and pair not in seen:
                seen.add(pair)
                needed_pairs.append((pos_idx, move_uci))

    print(f"Total unique (position, move) pairs requiring Stockfish analysis: {len(needed_pairs)}", flush=True)
    print(f"Pre-cached from analysis_test: {len(sf_cache)} entries", flush=True)

    try:
        eval_count = 0
        total_evals = len(needed_pairs)
        for pos_idx, move_uci in needed_pairs:
            fen = positions_data[pos_idx]["fen"]
            if (fen, move_uci) in sf_cache:
                continue
            board = positions_data[pos_idx]["board"]
            move = chess.Move.from_uci(move_uci)
            loss, _ = compute_cp_loss(board, move, analyser)
            sf_cache[(fen, move_uci)] = loss
            eval_count += 1
            if eval_count % 500 == 0 or eval_count == total_evals:
                print(f"Stockfish evaluated {eval_count}/{total_evals} moves (cache size: {len(sf_cache)})...", flush=True)
    finally:
        analyser.close()
        print("Stockfish analysis complete.", flush=True)

    # Compute metrics for each configuration
    human_acpl = 53.2
    human_err_rate = 0.130
    human_blunder_rate = 0.0382

    results = []
    for c_idx, cfg in enumerate(configs):
        tb = cfg["t_base"]
        tp = cfg["top_p"]
        em = cfg["error_mult"]

        moves = cfg_moves[c_idx]
        matches = 0
        cp_losses = []
        errors = 0
        blunders = 0

        for pos_idx, move_uci in enumerate(moves):
            human_uci = positions_data[pos_idx]["human_uci"]
            if move_uci == human_uci:
                matches += 1
            fen = positions_data[pos_idx]["fen"]
            loss = sf_cache.get((fen, move_uci), 0.0)
            cp_losses.append(loss)
            if loss >= 100.0:
                errors += 1
            if loss >= 300.0:
                blunders += 1

        top1_acc = (matches / n_rows) * 100.0
        acpl = float(np.mean(cp_losses)) if cp_losses else 0.0
        err_rate = (errors / n_rows) * 100.0
        blunder_rate = (blunders / n_rows) * 100.0

        # Normalized distance score to human baselines
        d_acpl = abs(acpl - human_acpl) / human_acpl
        d_err = abs((err_rate / 100.0) - human_err_rate) / human_err_rate
        d_blunder = abs((blunder_rate / 100.0) - human_blunder_rate) / human_blunder_rate
        dist_score = math.sqrt(d_acpl**2 + d_err**2 + d_blunder**2)

        results.append({
            "t_base": tb,
            "top_p": tp,
            "error_mult": em,
            "top1_acc": top1_acc,
            "acpl": acpl,
            "err_rate": err_rate,
            "blunder_rate": blunder_rate,
            "dist_score": dist_score,
        })

    # Sort results by dist_score ascending (closest to human first)
    sorted_by_score = sorted(results, key=lambda x: x["dist_score"])
    best = sorted_by_score[0]

    # Print Table
    print("\n" + "=" * 104)
    print("PAIRED REPLAY CALIBRATION: Grid Search over (t_base, top_p, error_mult)")
    print("=" * 104)
    print(f"Human Real Baseline: ACPL: {human_acpl:.1f} cp | >=100 cp: {human_err_rate*100:.1f}% | >=300 cp: {human_blunder_rate*100:.1f}%")
    print("-" * 104)
    header = f"{'t_base':<8} {'top_p':<8} {'err_mult':<10} {'Top-1 Acc':<12} {'ACPL (cp)':<12} {'>=100 cp Rate':<16} {'>=300 cp Rate':<16} {'Dist Score':<12}"
    print(header)
    print("-" * 104)

    for r in results:
        acc_str = f"{r['top1_acc']:.1f}%"
        acpl_str = f"{r['acpl']:.1f}"
        err_str = f"{r['err_rate']:.1f}%"
        blunder_str = f"{r['blunder_rate']:.1f}%"
        is_best_mark = " *" if r == best else ""
        print(
            f"{r['t_base']:<8.2f} {r['top_p']:<8.2f} {r['error_mult']:<10.1f} "
            f"{acc_str:<12} {acpl_str:<12} {err_str:<16} {blunder_str:<16} "
            f"{r['dist_score']:<10.4f}{is_best_mark}"
        )

    print("-" * 104)
    print("OPTIMAL CONFIGURATION (Closest to Human Baselines):")
    print(f"  t_base:           {best['t_base']:.2f}")
    print(f"  top_p:            {best['top_p']:.2f}")
    print(f"  error_multiplier: {best['error_mult']:.1f}")
    print(f"  Top-1 Accuracy:   {best['top1_acc']:.1f}%")
    print(f"  ACPL:             {best['acpl']:.1f} cp (Human: {human_acpl:.1f} cp, diff: {abs(best['acpl']-human_acpl):.1f} cp)")
    print(f"  >=100 cp Rate:    {best['err_rate']:.1f}% (Human: {human_err_rate*100:.1f}%, diff: {abs(best['err_rate']-human_err_rate*100):.1f}%)")
    print(f"  >=300 cp Rate:    {best['blunder_rate']:.1f}% (Human: {human_blunder_rate*100:.1f}%, diff: {abs(best['blunder_rate']-human_blunder_rate*100):.1f}%)")
    print(f"  Distance Score:   {best['dist_score']:.4f}")
    print("=" * 104 + "\n", flush=True)

    return best


def main():
    parser = argparse.ArgumentParser(description="Paired Replay Calibration across t_base, top_p, and error_multiplier.")
    parser.add_argument("--username", type=str, default="the_king_crusher")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_calibration(username=args.username, max_rows=args.max_rows, seed=args.seed)


if __name__ == "__main__":
    main()
