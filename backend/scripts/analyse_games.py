"""Analyze player game positions with Stockfish and compute complexity features (Phase P)."""

from __future__ import annotations
import argparse
import math
from pathlib import Path
import sys

# Ensure backend root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from app import config
from app.analysis.features import FEATURE_NAMES, compute_features
from app.analysis.stockfish import StockfishAnalyser, compute_cp_loss
from app.engine.model_cache import ModelCache
from app.ml.encoding_v3 import encode_board_v3, encode_meta_v3


def main():
    parser = argparse.ArgumentParser(description="Analyze player game positions for error modeling.")
    parser.add_argument("username", type=str, nargs="?", default="the_king_crusher", help="Player username")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-rows", type=int, default=None, help="Maximum number of rows to analyze")
    parser.add_argument("--stockfish-path", type=str, default=config.STOCKFISH_PATH)
    parser.add_argument("--depth", type=int, default=config.STOCKFISH_DEPTH)
    parser.add_argument("--out", type=str, default=None, help="Output parquet path")

    args = parser.parse_args()

    data_root = Path(config.DATA_ROOT)
    player_dir = data_root / "players" / args.username

    samples_path = player_dir / "samples_v3.parquet"
    if not samples_path.exists():
        samples_path = player_dir / "samples.parquet"
    if not samples_path.exists():
        raise FileNotFoundError(f"Samples file not found for {args.username} at {samples_path}")

    print(f"Loading samples from {samples_path} for split '{args.split}'...")
    df_samples = pq.read_table(samples_path).to_pandas()
    if "split" in df_samples.columns:
        df_split = df_samples[df_samples["split"] == args.split].copy()
    else:
        df_split = df_samples.copy()

    if args.max_rows is not None:
        df_split = df_split.iloc[: args.max_rows].copy()

    print(f"Total rows to analyze: {len(df_split)}")
    if len(df_split) == 0:
        print("No rows to analyze.")
        return

    # Load base model for policy log probs
    print("Loading base model for policy log-probabilities...")
    cache = ModelCache(capacity=2, data_root=data_root)
    base_model = cache.get("base", 1)
    base_model.eval()

    # Initialize Stockfish
    print(f"Initializing Stockfish from {args.stockfish_path} at depth {args.depth}...")
    analyser = StockfishAnalyser(path=args.stockfish_path, depth=args.depth)

    results: list[dict] = []

    try:
        for idx, (_, row) in enumerate(df_split.iterrows()):
            fen = str(row["fen"])
            uci_str = str(row["uci"])
            board = chess.Board(fen)
            played_move = chess.Move.from_uci(uci_str)

            # Compute policy log probs from base model
            prev_1 = row.get("prev_move_1")
            prev_2 = row.get("prev_move_2")
            if pd.isna(prev_1):
                prev_1 = None
            if pd.isna(prev_2):
                prev_2 = None

            planes = encode_board_v3(board, prev_move_1=prev_1, prev_move_2=prev_2, history=True)
            clock_before = row.get("clock_before_move")
            clock_sec = float(clock_before) if pd.notna(clock_before) else None
            base_s = int(row.get("base_s", 180)) if pd.notna(row.get("base_s")) else 180
            inc_s = int(row.get("increment_s", 0)) if pd.notna(row.get("increment_s")) else 0
            ply = int(row.get("ply", 0)) if pd.notna(row.get("ply")) else 0

            meta = encode_meta_v3(
                player_rating=int(row.get("player_rating", config.DEFAULT_RATING)),
                opponent_rating=int(row.get("opponent_rating", config.DEFAULT_RATING)),
                base_seconds=base_s,
                increment_seconds=inc_s,
                ply=ply,
                clock_before=clock_sec,
                clock_available=clock_sec is not None,
                halfmove_clock=int(row.get("halfmove_clock", 0)),
                num_legal_moves=board.legal_moves.count(),
                berserk=False,
            )

            board_t = torch.from_numpy(planes).unsqueeze(0).float()
            meta_t = torch.from_numpy(meta).unsqueeze(0).float()

            with torch.no_grad():
                out = base_model(board_t, meta_t)
                logits = out["policy_logits"][0].cpu().numpy()

            is_black = (board.turn == chess.BLACK)
            legal_lps: dict[str, float] = {}
            for m in board.legal_moves:
                f = (m.from_square ^ 56) if is_black else m.from_square
                t = (m.to_square ^ 56) if is_black else m.to_square
                legal_lps[m.uci()] = float(logits[f * 64 + t])

            if legal_lps:
                max_l = max(legal_lps.values())
                sum_exp = sum(math.exp(v - max_l) for v in legal_lps.values())
                log_sum = max_l + math.log(max(1e-12, sum_exp))
                policy_log_probs = {u: (v - log_sum) for u, v in legal_lps.items()}
            else:
                policy_log_probs = None

            # Compute features
            feats = compute_features(
                board=board,
                clock_before=clock_sec,
                base_s=base_s,
                ply=ply,
                policy_log_probs=policy_log_probs,
            )

            # Compute cp loss and is_error with Stockfish
            cp_loss, is_error = compute_cp_loss(board, played_move, analyser)

            record = {
                "game_id": row.get("game_id", ""),
                "ply": ply,
                "fen": fen,
                "uci": uci_str,
                "cp_loss": round(cp_loss, 2),
                "is_error": int(is_error),
                **feats,
            }
            results.append(record)

            print_interval = max(25, len(df_split) // 20)
            if (idx + 1) % print_interval == 0 or (idx + 1) == len(df_split):
                print(f"Analyzed {idx + 1}/{len(df_split)} positions...")

    finally:
        analyser.close()

    out_df = pd.DataFrame(results)

    out_path = Path(args.out) if args.out is not None else (player_dir / f"analysis_{args.split}.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"Analysis saved to {out_path} ({len(out_df)} rows).")


if __name__ == "__main__":
    main()
