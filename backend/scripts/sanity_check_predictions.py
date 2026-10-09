"""
Sanity check predictions for The_King_Crusher model.
Performs:
1. 30-ply self-play game starting from initial board, printing UCI and SAN moves.
2. 50 independent first-move calls from initial position, tabulating frequency.
"""
import sys
from pathlib import Path
from collections import Counter
import random
import chess

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.ml import export
from app.services.opening_book import OpeningBook
from app.services import storage
from app import config


def main():
    username = sys.argv[1] if len(sys.argv) > 1 else "the_king_crusher"
    p_dir = storage.player_dir(username, create=False)

    print("=" * 60)
    if not export.is_model_current(username):
        print("!" * 70)
        print(f"WARNING: Model checkpoint for '{username}' is NOT current with samples.parquet!")
        print("!" * 70)

    print(f"Loading trained model for '{username}'...")
    model = export.load_player_model(username, device="cpu")
    print("Model loaded successfully.")

    # Read meta_defaults if present
    info_path = p_dir / "training_info.json"
    meta_inputs = None
    if info_path.exists():
        try:
            import json
            with open(info_path, "r", encoding="utf-8") as f:
                t_info = json.load(f)
            if "meta_defaults" in t_info:
                meta_inputs = export.meta_inputs_from_defaults(t_info["meta_defaults"])
        except Exception:
            meta_inputs = None

    print(f"Loading OpeningBook for '{username}'...")
    book = OpeningBook(username)
    print("OpeningBook loaded successfully.")
    print("=" * 60)

    # 1. 30-ply playthrough
    print("\n--- Part 1: 30-Ply Game Simulation ---")
    board = chess.Board()
    ply = 0
    max_plies = 30
    move_log = []

    rng = random.Random(42)

    while ply < max_plies and not board.is_game_over():
        ply += 1
        move = export.predict_move(
            model=model,
            board=board,
            meta_inputs=meta_inputs,
            device="cpu",
            temperature=1.0,
            opening_book=book,
            rng=rng,
        )
        san = board.san(move)
        uci = move.uci()
        turn_str = "White" if board.turn == chess.WHITE else "Black"
        move_number = (ply + 1) // 2
        prefix = f"{move_number}." if board.turn == chess.WHITE else f"{move_number}..."
        print(f"Ply {ply:2d} ({turn_str:5s}): {prefix:5s} {san:7s} (UCI: {uci})")
        move_log.append((ply, turn_str, san, uci))
        board.push(move)

    print("\nSimulation outcome:")
    print(f"Final FEN: {board.fen()}")
    print(f"Is game over: {board.is_game_over()}")
    if board.is_game_over():
        print(f"Outcome: {board.outcome()}")

    # 2. 50 independent first-move samples
    print("\n--- Part 2: 50 Independent First Move Predictions (from startpos) ---")
    first_moves = Counter()
    for i in range(50):
        fresh_board = chess.Board()
        sample_rng = random.Random(1000 + i)
        move = export.predict_move(
            model=model,
            board=fresh_board,
            meta_inputs=meta_inputs,
            device="cpu",
            temperature=1.0,
            opening_book=book,
            rng=sample_rng,
        )
        first_moves[move.uci()] += 1

    print("\nFirst Move Frequency Table (50 trials):")
    print(f"{'Move (UCI)':<12} | {'SAN':<8} | {'Count':<6} | {'Percentage':<10}")
    print("-" * 45)
    for uci_move, count in first_moves.most_common():
        san_move = chess.Board().san(chess.Move.from_uci(uci_move))
        pct = (count / 50) * 100
        print(f"{uci_move:<12} | {san_move:<8} | {count:<6} | {pct:>5.1f}%")

    print("=" * 60)


if __name__ == "__main__":
    main()
