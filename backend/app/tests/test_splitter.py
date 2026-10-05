import pytest
from app.services.splitter import assign_splits

def test_assign_splits():
    # Create 100 sample rows, representing 10 games (10 rows each)
    samples = []
    for g in range(10):
        for ply in range(10):
            samples.append({
                "game_id": f"game_{g}",
                "date": f"2024-01-{g+1:02d}", # increasing dates
                "ply": ply
            })
            
    samples_out = assign_splits(samples)
    assert len(samples_out) == 100
    
    # 10 games total. 80% train = 8. 10% val = 1. 10% test = 1.
    train_games = set(s["game_id"] for s in samples_out if s["split"] == "train")
    val_games = set(s["game_id"] for s in samples_out if s["split"] == "val")
    test_games = set(s["game_id"] for s in samples_out if s["split"] == "test")
    
    assert len(train_games) == 8
    assert len(val_games) == 1
    assert len(test_games) == 1
    
    # No game in more than one split
    assert len(train_games.intersection(val_games)) == 0
    assert len(train_games.intersection(test_games)) == 0
    assert len(val_games.intersection(test_games)) == 0

    # Oldest 80% should be game_0 to game_7
    for g in range(8):
        assert f"game_{g}" in train_games
    
    # Next is game_8 in val
    assert "game_8" in val_games
    
    # Newest is game_9 in test
    assert "game_9" in test_games


def test_small_game_count_gets_nonzero_val_and_test():
    for n_games in (3, 4, 5):
        samples = [
            {"game_id": f"game_{game}", "date": f"2024-01-{game + 1:02d}"}
            for game in range(n_games)
        ]

        samples_out = assign_splits(samples)
        train_games = {sample["game_id"] for sample in samples_out if sample["split"] == "train"}
        val_games = {sample["game_id"] for sample in samples_out if sample["split"] == "val"}
        test_games = {sample["game_id"] for sample in samples_out if sample["split"] == "test"}

        assert len(val_games) >= 1
        assert len(test_games) >= 1
        assert len(train_games) >= len(val_games)
        assert len(train_games) >= len(test_games)
        if n_games > 3:
            assert len(train_games) > len(val_games)
            assert len(train_games) > len(test_games)


def test_fewer_than_three_games_stay_in_train():
    for n_games in (1, 2):
        samples = [
            {"game_id": f"game_{game}", "date": f"2024-01-{game + 1:02d}"}
            for game in range(n_games)
        ]

        samples_out = assign_splits(samples)

        assert all(sample["split"] == "train" for sample in samples_out)
