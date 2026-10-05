"""
Opening book service: builds and queries a probabilistic opening book from training games.
"""
from collections import Counter
import json
import random
import chess
from pathlib import Path
from typing import Any

from .. import config
from . import storage

def _fen_key(board: chess.Board) -> str:
    """Return the first 4 fields of the FEN (ignores halfmove and fullmove clocks)."""
    return " ".join(board.fen().split()[:4])

def build_opening_book(samples: list[dict[str, Any]], username: str) -> None:
    """
    Build the opening book from TRAIN split samples and save to JSON.
    
    Parameters
    ----------
    samples : list of dict
        The fully processed sample rows (with 'split' assigned).
    username : str
        The player's username (to save the file).
    """
    book: dict[str, Counter] = {}
    
    for row in samples:
        if row.get("split") != "train":
            continue
            
        fen = row["fen"]
        uci = row["uci"]
        
        board = chess.Board(fen)
        key = _fen_key(board)
        
        if key not in book:
            book[key] = Counter()
        book[key][uci] += 1
        
    # Convert counters to dicts for JSON serialization
    serializable_book = {k: dict(v) for k, v in book.items()}
    
    out_path = storage.player_dir(username) / "opening_book.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable_book, f, indent=2)

class OpeningBook:
    """Loads and queries an opening book."""
    
    def __init__(self, username: str):
        self.username = username
        self.book = {}
        
        path = storage.player_dir(username) / "opening_book.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                self.book = json.load(f)
                
    def query(self, board: chess.Board) -> dict[str, float] | None:
        """
        Returns a dict mapping UCI move -> probability.
        Returns None if position not in book, or seen fewer times than MIN_BOOK_VISITS.
        """
        key = _fen_key(board)
        if key not in self.book:
            return None
            
        moves = self.book[key]
        total_visits = sum(moves.values())
        
        if total_visits < config.MIN_BOOK_VISITS:
            return None
            
        return {uci: count / total_visits for uci, count in moves.items()}
        
    def sample(self, board: chess.Board, rng: random.Random | None = None) -> chess.Move | None:
        """
        Sample a move probabilistically from the book.
        Returns None if the position is not in the book (or below visit threshold).
        """
        probs = self.query(board)
        if probs is None:
            return None
            
        if rng is None:
            rng = random.Random()
            
        ucis = list(probs.keys())
        weights = list(probs.values())
        
        chosen_uci = rng.choices(ucis, weights=weights, k=1)[0]
        return chess.Move.from_uci(chosen_uci)
