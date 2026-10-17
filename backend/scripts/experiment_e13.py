"""E13 Experiment: Strength Calibration via Self-Play and ACPL Evaluation (Phase P)."""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

# Ensure backend root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from app import config
from app.analysis.error_model import ErrorModel
from app.analysis.self_play import load_pooled_think_time, play_match
from app.analysis.stockfish import StockfishAnalyser, compute_cp_loss
from app.engine.model_cache import ModelCache
from app.engine.provider_v3 import V3Provider
from app.engine.think_time import ThinkTimeModel


def run_e13(
    username: str = "the_king_crusher",
    n_games: int = 20,
    base_s: int = 180,
    inc_s: int = 2,
    seed: int = 0,
):
    data_root = Path(config.DATA_ROOT)
    player_dir = data_root / "players" / username

    samples_path = player_dir / "samples_v3.parquet"
    if not samples_path.exists():
        samples_path = player_dir / "samples.parquet"
    if not samples_path.exists():
        raise FileNotFoundError(f"Samples file not found at {samples_path}")

    # Estimate human ACPL, error rate and blunder rate from analyzed data
    human_acpl = 53.2
    human_error_rate = 0.130
    human_blunder_rate = 0.0382
    analysis_train_path = player_dir / "analysis_train.parquet"
    if analysis_train_path.exists():
        df_tr = pq.read_table(analysis_train_path).to_pandas()
        if "cp_loss" in df_tr.columns:
            human_acpl = float(df_tr["cp_loss"].mean())
            blunder_threshold = getattr(config, "ERROR_MODEL_BLUNDER_CP", 300)
            human_blunder_rate = float((df_tr["cp_loss"] >= blunder_threshold).mean())
            human_error_rate = float((df_tr["cp_loss"] >= getattr(config, "ERROR_MODEL_MIN_CP_LOSS", 100)).mean())

    print(f"Human baseline estimates from {analysis_train_path.name if analysis_train_path.exists() else 'prior'}:")
    print(f"  Real ACPL: {human_acpl:.1f}")
    print(f"  Real Error Rate (>= 100 cp): {human_error_rate * 100:.1f}%")
    print(f"  Real Blunder Rate (>= 300 cp): {human_blunder_rate * 100:.1f}%\n")

    # Load think time models
    print("Loading bot and opponent think time models...")
    bot_tt = ThinkTimeModel.from_parquet(samples_path)
    pooled_path = config.BASE_THINKTIME_DIR / "pooled_thinktime.json"
    opp_tt = load_pooled_think_time(pooled_path, rating_band=1800)

    # Load error model
    error_model_path = player_dir / "models" / "v1" / "error_model.joblib"
    if not error_model_path.exists():
        raise FileNotFoundError(f"Error model not found at {error_model_path}. Train it first.")
    error_model = ErrorModel.load(error_model_path)
    print(f"Successfully loaded error model for {username}:v1 from {error_model_path}")

    # Compute player median rating from samples
    median_rating = config.DEFAULT_RATING
    df_samples = pq.read_table(samples_path, columns=["player_rating"]).to_pandas()
    if "player_rating" in df_samples.columns and not df_samples["player_rating"].empty:
        median_rating = int(df_samples["player_rating"].median())
    print(f"Using player median rating: {median_rating}")

    # Instantiate Stockfish analyser for runtime blunder capping and evaluation
    stockfish_analyser = (
        StockfishAnalyser(depth=config.STOCKFISH_DEPTH)
        if Path(config.STOCKFISH_PATH).exists()
        else None
    )

    cache = ModelCache(capacity=8, data_root=data_root)
    bot_provider = V3Provider(
        username=username,
        model_version=1,
        model_cache=cache,
        think_time_model=bot_tt,
        temperature=1.0,
        data_root=data_root,
        error_model=error_model,
        stockfish_analyser=stockfish_analyser,
    )
    bot_provider.bot_rating_from_model = median_rating
    assert bot_provider.error_model is not None, "Error model not active on bot provider!"

    opp_provider = V3Provider(
        username="base",
        model_version=1,
        model_cache=cache,
        think_time_model=opp_tt,
        temperature=1.0,
        data_root=data_root,
        error_model=None,
    )

    try:
        print(f"Playing {n_games}-game match: {username}:v1 (with error model) vs base...")
        records = play_match(
            bot=bot_provider,
            opponent=opp_provider,
            n_games=n_games,
            base_s=base_s,
            inc_s=inc_s,
            seed=seed,
            bot_colors="alternate",
            bot_rating=median_rating,
        )

        bot_wins = 0
        opp_wins = 0
        draws = 0
        for r in records:
            if r.result == "1/2-1/2":
                draws += 1
            elif (r.result == "1-0" and r.bot_color == "white") or (r.result == "0-1" and r.bot_color == "black"):
                bot_wins += 1
            else:
                opp_wins += 1

        win_rate = (bot_wins / n_games) * 100.0 if n_games > 0 else 0.0
        print(f"Match result: {bot_wins} wins, {opp_wins} losses, {draws} draws (win rate: {win_rate:.1f}%)\n")

        # Evaluate ACPL on bot moves with Stockfish
        print("Evaluating bot ACPL on self-play games with Stockfish (depth 12)...")
        eval_analyser = stockfish_analyser if stockfish_analyser is not None else StockfishAnalyser(depth=config.STOCKFISH_DEPTH)

        bot_cp_losses: list[float] = []
        bot_errors = 0
        bot_blunders = 0

        for g_idx, game in enumerate(records):
            board = chess.Board()
            bot_is_white = (game.bot_color == "white")

            for m_rec in game.moves:
                move = chess.Move.from_uci(m_rec.uci)
                is_bot_move = (m_rec.by == "bot")

                if is_bot_move:
                    loss, _ = compute_cp_loss(board, move, eval_analyser)
                    bot_cp_losses.append(loss)
                    if loss >= config.ERROR_MODEL_MIN_CP_LOSS:
                        bot_errors += 1
                    if loss >= config.ERROR_MODEL_BLUNDER_CP:
                        bot_blunders += 1

                board.push(move)

            if (g_idx + 1) % 5 == 0 or (g_idx + 1) == n_games:
                print(f"Evaluated moves from {g_idx + 1}/{n_games} games...")

    finally:
        if stockfish_analyser is not None:
            stockfish_analyser.close()

    bot_acpl = float(np.mean(bot_cp_losses)) if bot_cp_losses else 0.0
    bot_error_rate = float(bot_errors / len(bot_cp_losses)) if bot_cp_losses else 0.0
    bot_blunder_rate = float(bot_blunders / len(bot_cp_losses)) if bot_cp_losses else 0.0

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("\n" + "=" * 74)
    print("E13 EXPERIMENT: Strength Calibration & Behavioral Fidelity")
    print("=" * 74)
    print(f"{'Metric':<30} {'Human Real':<22} {'Bot Simulated':<22}")
    print("-" * 74)
    print(f"{'ACPL (Centipawn Loss)':<30} {human_acpl:.1f}{'':<18} {bot_acpl:.1f}")
    print(f"{'Error Rate (>= 100 cp)':<30} {human_error_rate * 100:.1f}%{'':<17} {bot_error_rate * 100:.1f}%")
    print(f"{'Blunder Rate (>= 300 cp)':<30} {human_blunder_rate * 100:.1f}%{'':<17} {bot_blunder_rate * 100:.1f}%")
    print(f"{'Win Rate vs Base':<30} {'N/A':<22} {win_rate:.1f}%")
    print("-" * 74)
    delta_acpl = abs(bot_acpl - human_acpl)
    delta_blunder = abs(bot_blunder_rate - human_blunder_rate)
    is_calibrated = (delta_acpl < 30.0) and (delta_blunder < 0.05)
    takeaway = (
        f"Well-calibrated to human fidelity (ACPL delta {delta_acpl:.1f} cp, blunder delta {delta_blunder*100:.1f}%)"
        if is_calibrated
        else f"Calibration gap remains (ACPL delta {delta_acpl:.1f} cp, blunder delta {delta_blunder*100:.1f}%)"
    )
    print(f"ACPL Difference: {delta_acpl:.1f} cp (human: {human_acpl:.1f}, bot: {bot_acpl:.1f})")
    print(f"Takeaway: {takeaway}")
    print("=" * 74 + "\n")


    return {
        "human_acpl": human_acpl,
        "bot_acpl": bot_acpl,
        "human_blunder_rate": human_blunder_rate,
        "bot_blunder_rate": bot_blunder_rate,
        "win_rate": win_rate,
    }


def main():
    parser = argparse.ArgumentParser(description="Run E13 Experiment: Strength Calibration.")
    parser.add_argument("--username", type=str, default="the_king_crusher")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_e13(username=args.username, n_games=args.games, seed=args.seed)


if __name__ == "__main__":
    main()
