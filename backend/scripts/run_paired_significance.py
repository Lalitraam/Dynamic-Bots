"""Paired significance evaluation of 4 sampling configs vs greedy across 5 seeds on the 5,004 test positions."""

from __future__ import annotations
import json
from pathlib import Path
import random
import sys
import chess
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

# Ensure backend root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config
from app.db.players import get_player_median_rating
from app.engine.model_cache import ModelCache
from app.engine.provider import MoveContext
from app.engine.provider_v3 import _apply_top_p, _build_board_planes, _build_meta, _sample_from


def main():
    date_str = "2026-10-17"
    print(f"=== Paired Significance Evaluation ({date_str}) ===")

    # 1. Load human baseline
    baseline_path = Path(__file__).resolve().parents[1] / "data_baselines" / "human_baseline_the_king_crusher.json"
    with open(baseline_path, "r", encoding="utf-8") as f:
        human_b = json.load(f)
    h_acpl = human_b["metrics"]["acpl"]["value"]
    h_100 = human_b["metrics"]["rate_100_cp"]["value"]
    h_300 = human_b["metrics"]["rate_300_cp"]["value"]
    print(f"Human baseline: ACPL={h_acpl:.4f}, >=100={h_100*100:.2f}%, >=300={h_300*100:.2f}%")

    # 2. Load Stockfish cache and verify keys
    data_root = Path(config.DATA_ROOT)
    player_dir = data_root / "players" / "the_king_crusher"
    cache_file = player_dir / ".sf_calibration_cache.joblib"
    sf_cache = joblib.load(cache_file)
    stockfish_depth = 12
    stockfish_version = "sf_16.1"

    invalid_keys = [
        k for k in sf_cache
        if not (isinstance(k, tuple) and len(k) == 4 and k[2] == stockfish_depth and k[3] == stockfish_version)
    ]
    print(f"Stockfish cache: {len(sf_cache)} entries. Invalid/dropped: {len(invalid_keys)}")

    # 3. Load samples and model
    samples_path = player_dir / "samples_v3.parquet"
    df_samples = pq.read_table(samples_path).to_pandas()
    df_test = df_samples[df_samples["split"] == "test"].copy().reset_index(drop=True)
    n_positions = len(df_test)
    print(f"Loaded {n_positions} test positions.")

    median_rating = get_player_median_rating("the_king_crusher")
    model_cache = ModelCache(capacity=4, data_root=data_root)
    model = model_cache.get("the_king_crusher", 1)
    model.eval()

    # Precompute neural network policy logits for each position
    print("Computing model policy logits for all positions...")
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
            "legal_moves": legal_moves,
            "legal_arr": legal_arr,
        })

    print(f"Precomputed logits for {len(positions_data)} positions.")

    configs = [
        {"name": "greedy", "t_base": None, "top_p": None, "is_greedy": True},
        {"name": "t=0.30_p=0.70", "t_base": 0.30, "top_p": 0.70, "is_greedy": False},
        {"name": "t=0.50_p=0.70", "t_base": 0.50, "top_p": 0.70, "is_greedy": False},
        {"name": "t=0.70_p=0.70", "t_base": 0.70, "top_p": 0.70, "is_greedy": False},
    ]

    n_seeds = 5
    results = {}

    for cfg in configs:
        name = cfg["name"]
        is_greedy = cfg["is_greedy"]
        tb = cfg["t_base"]
        tp = cfg["top_p"]

        moves_mat = []
        losses_mat = []

        for pos_idx, p in enumerate(positions_data):
            fen = p["fen"]
            legal_moves = p["legal_moves"]
            legal_arr = p["legal_arr"]

            if is_greedy:
                best_idx = int(np.argmax(legal_arr))
                m_uci = legal_moves[best_idx].uci()
                loss = sf_cache.get((fen, m_uci, stockfish_depth, stockfish_version), 0.0)
                moves_mat.append([m_uci] * n_seeds)
                losses_mat.append([loss] * n_seeds)
            else:
                eff_temp = max(1e-4, tb)
                scaled = legal_arr / eff_temp
                l_max = scaled.max()
                probs = np.exp(scaled - l_max)
                s_sum = probs.sum()
                probs = probs / s_sum if s_sum > 0 else (np.ones_like(probs) / len(probs))
                filtered = _apply_top_p(probs, tp)
                f_sum = filtered.sum()
                final_p = (filtered / f_sum) if f_sum > 0 else probs

                p_moves = []
                p_losses = []
                for s in range(n_seeds):
                    rng = random.Random(s * 100000 + pos_idx)
                    chosen_idx = _sample_from(final_p, rng)
                    m_uci = legal_moves[chosen_idx].uci()
                    loss = sf_cache.get((fen, m_uci, stockfish_depth, stockfish_version), 0.0)
                    p_moves.append(m_uci)
                    p_losses.append(loss)
                moves_mat.append(p_moves)
                losses_mat.append(p_losses)

        losses_arr = np.array(losses_mat, dtype=np.float64)
        mean_pos_losses = losses_arr.mean(axis=1)
        mean_pos_100 = (losses_arr >= 100.0).mean(axis=1)
        mean_pos_300 = (losses_arr >= 300.0).mean(axis=1)

        uniq_per_pos = [len(set(moves_mat[i])) for i in range(n_positions)]
        mean_uniq = float(np.mean(uniq_per_pos))
        stoch_frac = float(np.mean([u > 1 for u in uniq_per_pos])) * 100.0

        results[name] = {
            "losses_arr": losses_arr,
            "mean_pos_losses": mean_pos_losses,
            "mean_pos_100": mean_pos_100,
            "mean_pos_300": mean_pos_300,
            "acpl": float(np.mean(mean_pos_losses)),
            "rate_100": float(np.mean(mean_pos_100)),
            "rate_300": float(np.mean(mean_pos_300)),
            "mean_uniq": mean_uniq,
            "stoch_frac": stoch_frac,
        }

    # Bootstrap CIs and paired deltas vs greedy
    game_ids = np.array([p["game_id"] for p in positions_data])
    game_groups = df_test.groupby("game_id")
    game_list = list(game_groups.groups.keys())
    n_games = len(game_list)
    g_indices = [game_groups.get_group(g).index.to_numpy() for g in game_list]
    g_counts = np.array([len(idx) for idx in g_indices])

    rng = np.random.default_rng(42)
    boot_draws = [rng.choice(n_games, size=n_games, replace=True) for _ in range(2000)]

    greedy_pos_losses = results["greedy"]["mean_pos_losses"]
    greedy_pos_300 = results["greedy"]["mean_pos_300"]

    summary_rows = []

    for cfg in configs:
        name = cfg["name"]
        res = results[name]
        pos_losses = res["mean_pos_losses"]
        pos_300 = res["mean_pos_300"]

        delta_loss = pos_losses - greedy_pos_losses
        delta_300 = pos_300 - greedy_pos_300

        g_acpl_sums = np.array([pos_losses[idx].sum() for idx in g_indices])
        g_300_sums = np.array([pos_300[idx].sum() for idx in g_indices])
        g_d_acpl_sums = np.array([delta_loss[idx].sum() for idx in g_indices])
        g_d_300_sums = np.array([delta_300[idx].sum() for idx in g_indices])

        boot_acpl = np.empty(2000)
        boot_300 = np.empty(2000)
        boot_d_acpl = np.empty(2000)
        boot_d_300 = np.empty(2000)

        for b, draw in enumerate(boot_draws):
            tot_c = g_counts[draw].sum()
            boot_acpl[b] = g_acpl_sums[draw].sum() / tot_c
            boot_300[b] = g_300_sums[draw].sum() / tot_c
            boot_d_acpl[b] = g_d_acpl_sums[draw].sum() / tot_c
            boot_d_300[b] = g_d_300_sums[draw].sum() / tot_c

        acpl_ci = [np.percentile(boot_acpl, 2.5), np.percentile(boot_acpl, 97.5)]
        r300_ci = [np.percentile(boot_300, 2.5), np.percentile(boot_300, 97.5)]
        d_acpl_ci = [np.percentile(boot_d_acpl, 2.5), np.percentile(boot_d_acpl, 97.5)]
        d_300_ci = [np.percentile(boot_d_300, 2.5), np.percentile(boot_d_300, 97.5)]

        d_acpl_val = float(np.mean(delta_loss))
        d_300_val = float(np.mean(delta_300)) * 100.0

        summary_rows.append({
            "config": name,
            "acpl": res["acpl"],
            "acpl_ci": acpl_ci,
            "r100": res["rate_100"] * 100.0,
            "r300": res["rate_300"] * 100.0,
            "r300_ci": [r300_ci[0] * 100.0, r300_ci[1] * 100.0],
            "mean_uniq": res["mean_uniq"],
            "stoch_frac": res["stoch_frac"],
            "delta_acpl": d_acpl_val,
            "delta_acpl_ci": d_acpl_ci,
            "delta_r300": d_300_val,
            "delta_r300_ci": [d_300_ci[0] * 100.0, d_300_ci[1] * 100.0],
            "ratio_human_acpl": res["acpl"] / h_acpl,
            "ratio_human_300": (res["rate_300"] * 100.0) / (h_300 * 100.0),
        })

    # Print markdown table
    print("\n| config | ACPL (95% CI) | >=300 cp (%) (95% CI) | % positions stochastic | Uniq/Pos | delta ACPL vs greedy (95% CI) | delta >=300 cp vs greedy (95% CI) | Ratio vs Human ACPL |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in summary_rows:
        cfg_name = r["config"]
        acpl_s = f"{r['acpl']:.2f} [{r['acpl_ci'][0]:.2f}, {r['acpl_ci'][1]:.2f}]"
        r300_s = f"{r['r300']:.2f}% [{r['r300_ci'][0]:.2f}%, {r['r300_ci'][1]:.2f}%]"
        stoch_s = f"{r['stoch_frac']:.2f}%"
        uniq_s = f"{r['mean_uniq']:.2f}"
        if cfg_name == "greedy":
            d_acpl_s = "0.00 (ref)"
            d_300_s = "0.00% (ref)"
        else:
            d_acpl_s = f"{r['delta_acpl']:+.2f} [{r['delta_acpl_ci'][0]:+.2f}, {r['delta_acpl_ci'][1]:+.2f}]"
            d_300_s = f"{r['delta_r300']:+.2f}% [{r['delta_r300_ci'][0]:+.2f}%, {r['delta_r300_ci'][1]:+.2f}%]"
        ratio_s = f"{r['ratio_human_acpl']:.2f}x"
        print(f"| {cfg_name} | {acpl_s} | {r300_s} | {stoch_s} | {uniq_s} | {d_acpl_s} | {d_300_s} | {ratio_s} |")

    # Also save json output for documentation
    out_json = Path(__file__).resolve().parents[1] / "data_baselines" / "paired_significance_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "command": "python scripts/run_paired_significance.py",
            "date": date_str,
            "stockfish_depth": stockfish_depth,
            "stockfish_version": stockfish_version,
            "human_baseline": human_b,
            "configs": summary_rows,
        }, f, indent=2)
    print(f"\nSaved paired significance results to {out_json.name}")


if __name__ == "__main__":
    main()
