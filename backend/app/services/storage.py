"""
Storage service for persisting sampled games to disk.
"""
import json
from pathlib import Path
from .. import config

def player_dir(username: str) -> Path:
    """
    Returns the directory for a given player's data.
    Creates the directory if it doesn't exist.
    """
    username = username.lower()
    dir_path = Path(config.DATA_ROOT) / "players" / username
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_games(username: str, games: list):
    """
    Saves a list of games to JSONL format.
    Each game is written as a separate JSON line.
    """
    username = username.lower()
    dir_path = player_dir(username)
    file_path = dir_path / "games.jsonl"

    with open(file_path, "w", encoding="utf-8") as f:
        for game in games:
            # Convert datetime to ISO string for JSON serialization
            game_copy = game.copy()
            if "date" in game_copy and hasattr(game_copy["date"], "isoformat"):
                game_copy["date"] = game_copy["date"].isoformat()
            f.write(json.dumps(game_copy) + "\n")

def load_games(username: str) -> list:
    """
    Loads games from JSONL format.
    Returns a list of game dictionaries.
    """
    username = username.lower()
    dir_path = player_dir(username)
    file_path = dir_path / "games.jsonl"

    if not file_path.exists():
        return []

    games = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                game = json.loads(line)
                # Convert ISO date string back to datetime if needed
                if "date" in game and isinstance(game["date"], str):
                    from datetime import datetime
                    game["date"] = datetime.fromisoformat(game["date"])
                games.append(game)
            except json.JSONDecodeError:
                # Skip invalid lines
                continue
    return games