"""
Hand-check script for visual verification of tensor encoding.
"""
import sys
from pathlib import Path

# Add backend directory to sys.path so we can import app
sys.path.append(str(Path(__file__).resolve().parent.parent))

import chess
from app.ml.encoding import encode_board, encode_move, decode_move

def print_plane(plane, title="Plane"):
    print(f"--- {title} ---")
    # plane is 8x8. print rows from top to bottom (7 down to 0) to match standard chess board view
    for r in range(7, -1, -1):
        row_str = " ".join(str(plane[r, c]) for c in range(8))
        print(row_str)
    print()

def hand_check(fen: str, move_uci: str):
    board = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)
    
    print("==================================================")
    print(f"FEN:  {fen}")
    print(f"Move: {move_uci}")
    print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
    print("Real board:")
    print(board)
    print()

    tensor = encode_board(board)
    labels = encode_move(board, move)

    print("Mover's view (working board used for encoding):")
    work = board.mirror() if board.turn == chess.BLACK else board
    print(work)
    print()

    print("--- Encoded Move Labels ---")
    print(f"from_sq: {labels['from_sq']}  to_sq: {labels['to_sq']}")
    print(f"promo: {labels['promo']}  piece_type: {labels['piece_type']}")
    print(f"capture: {labels['is_capture']}  check: {labels['is_check']}  castle: {labels['is_castle']}")
    
    decoded = decode_move(board, labels["from_sq"], labels["to_sq"], labels["promo"])
    print(f"Decoded move back to real frame: {decoded.uci()}")
    assert decoded == move, "Mismatch!"
    
    # Print some interesting planes
    print_plane(tensor[5], "Plane 5 (Mover's King)")
    print_plane(tensor[11], "Plane 11 (Opponent's King)")
    print_plane(tensor[12], "Plane 12 (Original side white?)")

if __name__ == "__main__":
    print("Running hand check...")
    
    # E.g. Black to move, en passant available
    hand_check("rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 3", "e4d3")
    
    # White castling
    hand_check("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5", "e1g1")
