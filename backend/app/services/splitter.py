"""
Splitter service: assigns train/val/test splits at the game level chronologically.
"""
from typing import Any

from .. import config

def assign_splits(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Assigns each row in `samples` to a split (train, val, or test).
    Splits are assigned at the game level, chronologically:
    oldest 80% train, next 10% val, newest 10% test.
    
    Parameters
    ----------
    samples : list of dict
        Sample rows produced by extractor.
        
    Returns
    -------
    list of dict
        The same sample rows, with a "split" string key added.

    With fewer than three unique games, all games are assigned to train
    because there are not enough games to populate all three splits.
    """
    if not samples:
        return []

    # 1. Gather all unique games and their dates
    # Assuming samples have 'game_id' and 'date'
    games_info = {}
    for row in samples:
        g_id = row["game_id"]
        if g_id not in games_info:
            games_info[g_id] = row["date"]
            
    # 2. Sort unique games chronologically. Tie-break by game_id
    sorted_games = sorted(games_info.items(), key=lambda x: (x[1], x[0]))
    
    # 3. Determine split indices
    n_games = len(sorted_games)
    if n_games < 3:
        n_train = n_games
        n_val = 0
    else:
        n_train = int(round(n_games * config.TRAIN_SPLIT))
        n_val = max(1, int(round(n_games * config.VAL_SPLIT)))
        n_test = max(1, n_games - n_train - n_val)
        n_train = n_games - n_val - n_test
    
    game_to_split = {}
    for i, (g_id, _) in enumerate(sorted_games):
        if i < n_train:
            game_to_split[g_id] = "train"
        elif i < n_train + n_val:
            game_to_split[g_id] = "val"
        else:
            game_to_split[g_id] = "test"
            
    # 4. Assign splits to rows
    for row in samples:
        row["split"] = game_to_split[row["game_id"]]
        
    return samples
