import pytest
import os
from pathlib import Path
from unittest.mock import patch
import chess

from app.services.dataset_builder import build_dataset
from app.services.opening_book import OpeningBook
from app import config

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_games.jsonl"

def load_fixture_games():
    import json
    from datetime import datetime
    games = []
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            game = json.loads(line)
            game["date"] = datetime.fromisoformat(game["date"])
            games.append(game)
    return games

def test_opening_book(tmp_path):
    import app.services.storage as storage_module
    import app.config as config_module
    
    orig_data_root = config_module.DATA_ROOT
    config_module.DATA_ROOT = tmp_path
    
    games = load_fixture_games()
    
    # We use a mocked min book visits of 1 for testing, since we have few games
    orig_min_visits = config_module.MIN_BOOK_VISITS
    config_module.MIN_BOOK_VISITS = 1
    
    with patch("app.services.extractor.storage.load_games", return_value=games):
        # The fixture has 4 games sorted chronologically: game1, game2, game3_ep_promo, game_noclock.
        # assign_splits() for n_games=4:
        #   n_train_raw = round(4 * 0.80) = 3, but safeguard forces n_val=max(1,round(4*0.10))=1,
        #   n_test=max(1, 4-3-1)=1, so n_train_final = 4-1-1 = 2.
        # Result: game1→train, game2→train, game3_ep_promo→val, game_noclock→test.
        # Both train games have TargetPlayer moves, so the opening book will be populated.
        # With MIN_BOOK_VISITS=1 (set above), a single occurrence per position is enough.
        info = build_dataset("testuser")
        
    book = OpeningBook("testuser")
    
    # Starting position
    board = chess.Board()
    probs = book.query(board)
    assert probs is not None
    
    # At least one move played from starting pos
    assert sum(probs.values()) == 1.0
    
    # Sample a move
    move = book.sample(board)
    assert move is not None
    assert move.uci() in probs
    
    # Check threshold logic
    config_module.MIN_BOOK_VISITS = 100
    assert book.query(board) is None
    assert book.sample(board) is None
    
    config_module.MIN_BOOK_VISITS = orig_min_visits
    config_module.DATA_ROOT = orig_data_root
