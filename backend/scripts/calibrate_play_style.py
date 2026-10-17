"""Paired Replay Calibration: Extended Grid Search over (t_base, top_p) with Multi-Seed Evaluation and Diversity Metrics."""

from __future__ import annotations
import argparse
import math
from pathlib import Path
import random
import sys
import time

# Ensure backend root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import chess
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from app import config
from app.analysis.error_model import ErrorModel
from app.analysis.features import compute_features
from app.analysis.stockfish import StockfishAnalyser, compute_cp_loss
from app.db.players import get_player_median_rating
from app.engine.model_cache import ModelCache
from app.engine.provider import MoveContext
from app.engine.provider_v3 import _build_board_planes, _build_meta, _apply_top_p, _sample_from


HUMAN_ACPL = 55.62
HUMAN_100_RATE = 13.49
HUMAN_300_RATE = 3.86


def compute_game_level_bootstrap_ci(
    game_ids: np.ndarray,
    cp_losses: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """Compute 95% bootstrap confidence intervals resampled at the game level."""
    unique_games = np.unique(game_ids)
    n_games = len(unique_games)

    # Precompute per-game aggregates
    game_sums = {}
    game_100s = {}
    game_300s = {}
    game_counts = {}

    for g in unique_games:
        mask = (game_ids == g)
        losses_g = cp_losses[mask]
        game_sums[g] = float(np.sum(losses_g))
        game_100s[g] = float(np.sum(losses_g >= 100.0))
        game_300s[g] = float(np.sum(losses_g >= 300.0))
        game_counts[g] = float(len(losses_g))

    arr_sums = np.array([game_sums[g] for g in unique_games], dtype=np.float64)
    arr_100s = np.array([game_100s[g] for g in unique_games], dtype=np.float64)
    arr_300s = np.array([game_300s[g] for g in unique_games], dtype=np.float64)
    arr_counts = np.array([game_counts[g] for g in unique_games], dtype=np.float64)

    rng = np.random.RandomState(seed)
    boot_idx = rng.choice(n_games, size=(n_bootstrap, n_games), replace=True)

    boot_sum = arr_sums[boot_idx].sum(axis=1)
    boot_count = arr_counts[boot_idx].sum(axis=1)
    boot_100 = arr_100s[boot_idx].sum(axis=1)
    boot_300 = arr_300s[boot_idx].sum(axis=1)

    boot_acpl = boot_sum / np.maximum(boot_count, 1e-9)
    boot_100_rate = (boot_100 / np.maximum(boot_count, 1e-9)) * 100.0
    boot_300_rate = (boot_300 / np.maximum(boot_count, 1e-9)) * 100.0

    return {
        "acpl_ci": (float(np.percentile(boot_acpl, 2.5)), float(np.percentile(boot_acpl, 97.5))),
        "rate100_ci": (float(np.percentile(boot_100_rate, 2.5)), float(np.percentile(boot_100_rate, 97.5))),
        "rate300_ci": (float(np.percentile(boot_300_rate, 2.5)), float(np.percentile(boot_300_rate, 97.5))),
    }


def run_calibration(
    username: str = "the_king_crusher",
    n_seeds: int = 5,
    max_rows: int | None = None,
    with_cap: bool = False,
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

    # Load player median rating from unified db.players
    median_rating = get_player_median_rating(username)
    print(f"Using player median rating: {median_rating}", flush=True)

    # Merge cp_loss from analysis_test.parquet
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
    print(f"Running Paired Replay Calibration on {n_rows} test positions with {n_seeds} seeds...\n", flush=True)

    # Compute p99 cp loss for cap if requested
    player_p99 = 300.0
    if "cp_loss" in df_test.columns and not df_test["cp_loss"].dropna().empty:
        player_p99 = float(np.percentile(df_test["cp_loss"].dropna(), 99))
    cap_threshold = min(player_p99, 300.0)
    if with_cap:
        print(f"Error capping enabled: max cp_loss threshold = {cap_threshold:.1f} (p99={player_p99:.1f})", flush=True)

    # Load fine-tuned model
    cache = ModelCache(capacity=4, data_root=data_root)
    model = cache.get(username, 1)
    model.eval()
    print(f"Loaded fine-tuned model for {username}:v1", flush=True)

    # Load error model
    error_model_path = player_dir / "models" / "v1" / "error_model.joblib"
    error_model = None
    if error_model_path.exists():
        error_model = ErrorModel.load(error_model_path)
        print(f"Loaded error model from {error_model_path}", flush=True)

    # Versioned Stockfish Cache: (fen, move_uci, depth, version)
    stockfish_depth = 12
    stockfish_version = "sf_16.1"
    cache_file = player_dir / ".sf_calibration_cache.joblib"
    sf_cache: dict[tuple[str, str, int, str], float] = {}

    if cache_file.exists():
        try:
            raw_cache = joblib.load(cache_file)
            for k, v in raw_cache.items():
                if len(k) == 2:
                    sf_cache[(k[0], k[1], stockfish_depth, stockfish_version)] = float(v)
                elif len(k) == 4:
                    sf_cache[k] = float(v)
            print(f"Loaded {len(sf_cache)} versioned cached Stockfish evaluations from {cache_file.name}", flush=True)
        except Exception:
            sf_cache = {}

    # Pre-cache any human evaluations from df_test
    if "cp_loss" in df_test.columns:
        for _, row in df_test.iterrows():
            f = str(row["fen"])
            u = str(row["uci"])
            l = row.get("cp_loss")
            if pd.notna(l):
                sf_cache[(f, u, stockfish_depth, stockfish_version)] = float(l)

    # Initialize Stockfish Analyser
    if not Path(config.STOCKFISH_PATH).exists():
        raise FileNotFoundError(f"Stockfish not found at {config.STOCKFISH_PATH}")
    analyser = StockfishAnalyser(depth=config.STOCKFISH_DEPTH)
    print(f"Initialized StockfishAnalyser (depth={config.STOCKFISH_DEPTH})\n", flush=True)

    # Define the 29 configurations:
    # 7 t_base x 4 top_p = 28 + 1 greedy = 29
    t_base_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    top_p_list = [0.5, 0.7, 0.85, 0.95]

    configs = []
    # 1. Greedy config
    configs.append({
        "name": "greedy",
        "t_base": "greedy",
        "top_p": "greedy",
        "error_mult": 1.0,
        "is_greedy": True,
    })
    # 2. Sampling configs
    for tb in t_base_list:
        for tp in top_p_list:
            configs.append({
                "name": f"t={tb:.2f}_p={tp:.2f}",
                "t_base": tb,
                "top_p": tp,
                "error_mult": 1.0,
                "is_greedy": False,
            })

    n_configs = len(configs)
    print(f"Total configurations: {n_configs} (1 greedy + 28 sampling)", flush=True)

    # Precompute neural net policy logits & features for all test positions
    print("\nStep 1/3: Computing model policy logits for all positions...", flush=True)
    positions_data = []

    for idx, (_, row) in enumerate(df_test.iterrows()):
        fen = str(row["fen"])
        human_uci = str(row["uci"])
        game_id = str(row.get("game_id", f"game_{idx}"))
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

        prev_moves_list = []
        p1 = row.get("prev_move_1")
        p2 = row.get("prev_move_2")
        if pd.notna(p1):
            prev_moves_list.append(str(p1))
        if pd.notna(p2):
            prev_moves_list.append(str(p2))

        ctx = MoveContext(
            board=board,
            prev_moves=prev_moves_list,
            bot_rating=median_rating,
            opp_rating=config.DEFAULT_RATING,
            base_seconds=base_s,
            increment_seconds=inc_s,
            bot_clock_ms=clock_ms,
            opp_clock_ms=clock_ms,
            ply=ply,
        )
        planes = _build_board_planes(ctx)
        meta = _build_meta(ctx, median_rating)

        board_t = torch.from_numpy(planes).unsqueeze(0).float()
        meta_t = torch.from_numpy(meta).unsqueeze(0).float()
        with torch.no_grad():
            out = model(board_t, meta_t)
            raw_logits = out["policy_logits"][0].cpu().numpy()

        is_black = (board.turn == chess.BLACK)
        legal_lps = {}
        for m in legal_moves:
            f = (m.from_square ^ 56) if is_black else m.from_square
            t = (m.to_square ^ 56) if is_black else m.to_square
            legal_lps[m.uci()] = float(raw_logits[f * 64 + t])

        legal_arr = np.array([legal_lps[m.uci()] for m in legal_moves], dtype=np.float64)

        positions_data.append({
            "fen": fen,
            "game_id": game_id,
            "board": board,
            "human_uci": human_uci,
            "legal_moves": legal_moves,
            "legal_arr": legal_arr,
        })

        if (idx + 1) % 1000 == 0 or (idx + 1) == n_rows:
            print(f"  Processed {idx + 1}/{n_rows} positions...", flush=True)

    # Step 2: Sample moves for each config and each seed
    print(f"\nStep 2/3: Sampling moves across {n_configs} configs and {n_seeds} seeds...", flush=True)
    # sampled_moves[config_idx][seed_idx] -> list of move_uci
    sampled_moves: list[list[list[str]]] = [[[] for _ in range(n_seeds)] for _ in range(n_configs)]
    needed_eval_pairs: set[tuple[str, str]] = set()

    for pos_idx, p_data in enumerate(positions_data):
        fen = p_data["fen"]
        legal_moves = p_data["legal_moves"]
        legal_arr = p_data["legal_arr"]
        human_uci = p_data["human_uci"]

        # 1. Greedy choice (deterministic)
        greedy_move = legal_moves[int(np.argmax(legal_arr))].uci()
        for s in range(n_seeds):
            sampled_moves[0][s].append(greedy_move)
        cache_key_g = (fen, greedy_move, stockfish_depth, stockfish_version)
        if cache_key_g not in sf_cache:
            needed_eval_pairs.add((fen, greedy_move))

        # 2. Sampling configs
        for c_idx in range(1, n_configs):
            cfg = configs[c_idx]
            tb = cfg["t_base"]
            tp = cfg["top_p"]

            # Temperature scaling
            eff_temp = max(1e-4, tb)
            scaled = legal_arr / eff_temp
            l_max = scaled.max()
            probs = np.exp(scaled - l_max)
            s_sum = probs.sum()
            probs = probs / s_sum if s_sum > 0 else (np.ones_like(probs) / len(probs))

            filtered = _apply_top_p(probs, tp)
            f_sum = filtered.sum()
            final_p = (filtered / f_sum) if f_sum > 0 else probs

            for s in range(n_seeds):
                rng = random.Random(s * 100000 + pos_idx)
                chosen_idx = _sample_from(final_p, rng)
                m_uci = legal_moves[chosen_idx].uci()
                sampled_moves[c_idx][s].append(m_uci)
                cache_key_m = (fen, m_uci, stockfish_depth, stockfish_version)
                if cache_key_m not in sf_cache:
                    needed_eval_pairs.add((fen, m_uci))

        if (pos_idx + 1) % 1000 == 0 or (pos_idx + 1) == len(positions_data):
            print(f"  Sampled {pos_idx + 1}/{len(positions_data)} positions...", flush=True)

    # Step 3: Stockfish evaluation of unique moves
    print(f"\nStep 3/3: Evaluating unique moves with Stockfish depth {stockfish_depth} ({stockfish_version})...", flush=True)
    print(f"Pre-cached move pairs: {len(sf_cache)}")
    print(f"Unique move pairs needing Stockfish: {len(needed_eval_pairs)}", flush=True)

    fen_to_board = {p["fen"]: p["board"] for p in positions_data}
    eval_count = 0
    total_evals = len(needed_eval_pairs)

    try:
        for fen, m_uci in needed_eval_pairs:
            cache_key = (fen, m_uci, stockfish_depth, stockfish_version)
            if cache_key in sf_cache:
                continue
            board = fen_to_board[fen]
            m = chess.Move.from_uci(m_uci)
            loss, _ = compute_cp_loss(board, m, analyser)
            sf_cache[cache_key] = loss
            eval_count += 1
            if eval_count % 500 == 0 or eval_count == total_evals:
                print(f"  Stockfish evaluated {eval_count}/{total_evals} (cache size: {len(sf_cache)})...", flush=True)
    finally:
        analyser.close()
        # Save cache to disk
        try:
            joblib.dump(sf_cache, cache_file)
            print(f"Saved {len(sf_cache)} Stockfish evaluations to {cache_file.name}", flush=True)
        except Exception:
            pass

    # Step 4: Calculate metrics, seed mean ± std, diversity, and Maximum Relative Error score
    results = []
    game_ids = np.array([p["game_id"] for p in positions_data])
    n_positions = len(positions_data)

    for c_idx, cfg in enumerate(configs):
        seed_top1 = []
        seed_acpl = []
        seed_100 = []
        seed_300 = []
        seed_losses_list = []

        for s in range(n_seeds):
            moves_s = sampled_moves[c_idx][s]
            matches = 0
            losses_s = []

            for pos_idx, m_uci in enumerate(moves_s):
                h_uci = positions_data[pos_idx]["human_uci"]
                fen = positions_data[pos_idx]["fen"]
                cache_key = (fen, m_uci, stockfish_depth, stockfish_version)
                loss = sf_cache.get(cache_key, 0.0)

                # Optional error capping test
                if with_cap and loss > cap_threshold:
                    loss = cap_threshold

                if m_uci == h_uci:
                    matches += 1
                losses_s.append(loss)

            arr_l = np.array(losses_s, dtype=np.float64)
            seed_losses_list.append(arr_l)
            seed_top1.append((matches / len(moves_s)) * 100.0)
            seed_acpl.append(float(np.mean(arr_l)))
            seed_100.append(float(np.mean(arr_l >= 100.0) * 100.0))
            seed_300.append(float(np.mean(arr_l >= 300.0) * 100.0))

        mean_top1 = float(np.mean(seed_top1))
        std_top1 = float(np.std(seed_top1))
        mean_acpl = float(np.mean(seed_acpl))
        std_acpl = float(np.std(seed_acpl))
        mean_100 = float(np.mean(seed_100))
        std_100 = float(np.std(seed_100))
        mean_300 = float(np.mean(seed_300))
        std_300 = float(np.std(seed_300))

        # Diversity metrics across the n_seeds for each position
        unique_counts = []
        is_stochastic = []
        for pos_idx in range(n_positions):
            pos_moves = [sampled_moves[c_idx][s][pos_idx] for s in range(n_seeds)]
            u_cnt = len(set(pos_moves))
            unique_counts.append(u_cnt)
            is_stochastic.append(1.0 if u_cnt >= 2 else 0.0)

        mean_unique_moves = float(np.mean(unique_counts))
        stochastic_fraction = float(np.mean(is_stochastic) * 100.0)

        # Composite Distance Score: Maximum Relative Error
        err_acpl = abs(mean_acpl - HUMAN_ACPL) / HUMAN_ACPL
        err_100 = abs(mean_100 - HUMAN_100_RATE) / HUMAN_100_RATE
        err_300 = abs(mean_300 - HUMAN_300_RATE) / HUMAN_300_RATE
        max_rel_error = max(err_acpl, err_100, err_300)

        # Average losses across seeds for bootstrap analysis
        avg_losses = np.mean(seed_losses_list, axis=0)

        results.append({
            "c_idx": c_idx,
            "name": cfg["name"],
            "t_base": cfg["t_base"],
            "top_p": cfg["top_p"],
            "is_greedy": cfg["is_greedy"],
            "top1_mean": mean_top1,
            "top1_std": std_top1,
            "acpl_mean": mean_acpl,
            "acpl_std": std_acpl,
            "err_100_mean": mean_100,
            "err_100_std": std_100,
            "err_300_mean": mean_300,
            "err_300_std": std_300,
            "mean_unique_moves": mean_unique_moves,
            "stochastic_fraction": stochastic_fraction,
            "err_acpl": err_acpl,
            "err_100": err_100,
            "err_300": err_300,
            "max_rel_error": max_rel_error,
            "avg_losses": avg_losses,
        })

    # Rank configs by Maximum Relative Error (ascending)
    ranked = sorted(results, key=lambda x: x["max_rel_error"])
    top3 = ranked[:3]

    # Compute bootstrap CIs for top 3 configs
    print("\nComputing game-level bootstrap CIs (2,000 iterations) for top 3 configurations...", flush=True)
    for r in top3:
        ci = compute_game_level_bootstrap_ci(game_ids, r["avg_losses"], n_bootstrap=2000, seed=0)
        r["bootstrap_ci"] = ci

    # Print Table
    print("\n" + "=" * 138)
    print("PAIRED REPLAY CALIBRATION: Extended Grid Search across 29 Configurations (5 Seeds)")
    print("=" * 138)
    print(f"HUMAN BASELINE:   ACPL: {HUMAN_ACPL:.2f} cp  |  >=100 cp: {HUMAN_100_RATE:.2f}%  |  >=300 cp: {HUMAN_300_RATE:.2f}%")
    print("-" * 138)
    header = (
        f"{'Config':<16} {'t_base':<8} {'top_p':<8} "
        f"{'Top-1 Acc (%)':<16} {'ACPL (cp)':<18} {'>=100 cp (%)':<18} {'>=300 cp (%)':<18} "
        f"{'Uniq/Pos':<10} {'Stoch (%)':<12} {'MaxRelErr':<10}"
    )
    print(header)
    print("-" * 138)

    for r in ranked:
        cfg_name = r["name"]
        tb_str = f"{r['t_base']:.2f}" if isinstance(r["t_base"], (int, float)) else str(r["t_base"])
        tp_str = f"{r['top_p']:.2f}" if isinstance(r["top_p"], (int, float)) else str(r["top_p"])

        acc_str = f"{r['top1_mean']:.1f} ± {r['top1_std']:.2f}"
        acpl_str = f"{r['acpl_mean']:.1f} ± {r['acpl_std']:.2f}"
        e100_str = f"{r['err_100_mean']:.1f} ± {r['err_100_std']:.2f}"
        e300_str = f"{r['err_300_mean']:.1f} ± {r['err_300_std']:.2f}"
        uniq_str = f"{r['mean_unique_moves']:.2f}"
        stoch_str = f"{r['stochastic_fraction']:.1f}%"
        is_best_mark = " *" if r == ranked[0] else ""

        print(
            f"{cfg_name:<16} {tb_str:<8} {tp_str:<8} "
            f"{acc_str:<16} {acpl_str:<18} {e100_str:<18} {e300_str:<18} "
            f"{uniq_str:<10} {stoch_str:<12} {r['max_rel_error']:<10.4f}{is_best_mark}"
        )

    print("-" * 138)
    print("TOP-3 CONFIGURATIONS & 95% GAME-LEVEL BOOTSTRAP CONFIDENCE INTERVALS (2,000 iterations):")
    print("-" * 138)

    for rank_idx, r in enumerate(top3, 1):
        tb = f"{r['t_base']:.2f}" if isinstance(r["t_base"], (int, float)) else str(r["t_base"])
        tp = f"{r['top_p']:.2f}" if isinstance(r["top_p"], (int, float)) else str(r["top_p"])
        ci = r["bootstrap_ci"]
        print(f"Rank {rank_idx}: Config {r['name']} (t_base={tb}, top_p={tp}) | Max Rel Error: {r['max_rel_error']:.4f}")
        print(f"  • Top-1 Acc:      {r['top1_mean']:.2f}% (std: {r['top1_std']:.2f}%)")
        print(f"  • ACPL:           {r['acpl_mean']:.2f} cp (Human: {HUMAN_ACPL:.2f} cp, 95% CI: [{ci['acpl_ci'][0]:.2f}, {ci['acpl_ci'][1]:.2f}])")
        print(f"  • >=100 cp Rate:  {r['err_100_mean']:.2f}% (Human: {HUMAN_100_RATE:.2f}%, 95% CI: [{ci['rate100_ci'][0]:.2f}%, {ci['rate100_ci'][1]:.2f}%])")
        print(f"  • >=300 cp Rate:  {r['err_300_mean']:.2f}% (Human: {HUMAN_300_RATE:.2f}%, 95% CI: [{ci['rate300_ci'][0]:.2f}%, {ci['rate300_ci'][1]:.2f}%])")
        print(f"  • Diversity:      {r['mean_unique_moves']:.2f} unique moves/pos, {r['stochastic_fraction']:.1f}% stochastic positions")
        print()

    best = ranked[0]
    print("=" * 138)
    print(f"OPTIMAL CONFIGURATION: {best['name']}")
    tb_val = best['t_base'] if isinstance(best['t_base'], (int, float)) else best['t_base']
    tp_val = best['top_p'] if isinstance(best['top_p'], (int, float)) else best['top_p']
    print(f"  t_base:           {tb_val}")
    print(f"  top_p:            {tp_val}")
    print(f"  error_mult:       1.0")
    print(f"  Max Rel Error:    {best['max_rel_error']:.4f}")
    print(f"  Diversity:        {best['mean_unique_moves']:.2f} unique moves/pos, {best['stochastic_fraction']:.1f}% stochastic positions")
    print("=" * 138 + "\n", flush=True)

    return ranked


def main():
    parser = argparse.ArgumentParser(description="Extended Grid Multi-Seed Calibration.")
    parser.add_argument("--username", type=str, default="the_king_crusher")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds (default: 5)")
    parser.add_argument("--max-rows", type=int, default=None, help="Max test rows (for quick testing)")
    parser.add_argument("--with-cap", action="store_true", help="Enable error cap test")

    args = parser.parse_args()
    run_calibration(
        username=args.username,
        n_seeds=args.seeds,
        max_rows=args.max_rows,
        with_cap=args.with_cap,
    )


if __name__ == "__main__":
    main()
