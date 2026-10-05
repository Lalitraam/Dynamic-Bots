import json
from datetime import datetime, timezone
from app.services.storage import player_dir, save_games, load_games

def test_storage_round_trip(tmp_path):
    # Temporarily override the data root for testing
    import app.services.storage as storage_module
    import app.config as config_module

    original_data_root = config_module.DATA_ROOT
    config_module.DATA_ROOT = tmp_path

    try:
        username = "testuser"
        now = datetime.now(timezone.utc)

        # Create sample games
        games = [
            {
                "game_id": "game1",
                "date": now,
                "category": "blitz",
                "is_fast": True,
                "player_color": "white",
                "rating": 1500,
                "opponent_rating": 1600,
                "time_control": "180+2",
                "termination": "normal",
                "pgn": "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0"
            },
            {
                "game_id": "game2",
                "date": now.replace(day=now.day-1),
                "category": "rapid",
                "is_fast": False,
                "player_color": "black",
                "rating": 1400,
                "opponent_rating": 1550,
                "time_control": "600+0",
                "termination": "normal",
                "pgn": "1. d4 d5 2. c4 e6 3. Nc3 Nf6 1/2-1/2"
            }
        ]

        # Save games
        save_games(username, games)

        # Check that the file exists
        dir_path = player_dir(username)
        file_path = dir_path / "games.jsonl"
        assert file_path.exists(), "games.jsonl file was not created"

        # Load games back
        loaded_games = load_games(username)

        # Assertions
        assert len(loaded_games) == 2, f"Expected 2 games, got {len(loaded_games)}"

        # Check first game
        g1 = loaded_games[0]
        assert g1["game_id"] == "game1"
        assert g1["category"] == "blitz"
        assert g1["is_fast"] == True
        assert g1["player_color"] == "white"
        assert g1["rating"] == 1500
        assert g1["opponent_rating"] == 1600
        assert g1["time_control"] == "180+2"
        assert g1["termination"] == "normal"
        assert g1["pgn"] == "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0"
        # Date should be a datetime object
        assert isinstance(g1["date"], datetime)
        assert g1["date"] == now

        # Check second game
        g2 = loaded_games[1]
        assert g2["game_id"] == "game2"
        assert g2["category"] == "rapid"
        assert g2["is_fast"] == False
        assert g2["player_color"] == "black"
        assert g2["rating"] == 1400
        assert g2["opponent_rating"] == 1550
        assert g2["time_control"] == "600+0"
        assert g2["termination"] == "normal"
        assert g2["pgn"] == "1. d4 d5 2. c4 e6 3. Nc3 Nf6 1/2-1/2"
        assert isinstance(g2["date"], datetime)
        assert g2["date"] == now.replace(day=now.day-1)

    finally:
        # Restore original data root
        config_module.DATA_ROOT = original_data_root

def test_load_games_empty():
    import app.services.storage as storage_module
    import app.config as config_module

    original_data_root = config_module.DATA_ROOT
    # Use a temporary directory that doesn't have the file
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        config_module.DATA_ROOT = tmpdir
        try:
            username = "nonexistent"
            games = load_games(username)
            assert games == [], f"Expected empty list, got {games}"
        finally:
            config_module.DATA_ROOT = original_data_root

if __name__ == "__main__":
    # For manual testing
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        test_storage_round_trip(tmpdir)
        test_load_games_empty()
    print("All storage tests passed!")