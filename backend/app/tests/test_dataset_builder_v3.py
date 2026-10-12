"""
Tests for dataset_builder_v3 (Phase B.1 and B.2).
"""
import pytest
import json
import hashlib
from pathlib import Path
from unittest.mock import patch
from datetime import datetime, timezone, timedelta
import pyarrow.parquet as pq

from app.services.dataset_builder_v3 import build_dataset_v3
from app import config

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_games.jsonl"


def load_fixture_games():
    games = []
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            game = json.loads(line)
            game["date"] = datetime.fromisoformat(game["date"])
            games.append(game)
    return games


def test_v3_recency_weights_hand_computed(tmp_path):
    """
    Phase B.2 test for recency weights with hand-computed values.
    Setup:
      Two games in the train split:
      - Game New: date = 2026-01-01T00:00:00 UTC (newest game in dataset)
        age_days = 0.0 -> raw_weight = 0.5 ** (0 / 365) = 1.0
      - Game Old: date = 2025-01-01T00:00:00 UTC (exactly 365 days before newest)
        age_days = 365.0 -> raw_weight = 0.5 ** (365 / 365) = 0.5
      Train raw mean = (1.0 + 0.5) / 2 = 0.75.
      Normalized train weights:
      - Game New: 1.0 / 0.75 = 4/3 ~ 1.3333334
      - Game Old: 0.5 / 0.75 = 2/3 ~ 0.6666667
      Hand-computed ratio = (4/3) / (2/3) = 2.0 exactly.
      Train mean is normalized to 1.0 +/- 1e-6.
      Weights strictly decrease with age: weight(New) > weight(Old).
      Val and test weights are exactly 1.0.
    """
    import app.config as config_module
    from app.services.storage import player_dir

    orig_data_root = config_module.DATA_ROOT
    config_module.DATA_ROOT = tmp_path
    username = "recency_user"

    dt_new = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    dt_old = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
    dt_val = datetime(2025, 6, 1, 0, 0, tzinfo=timezone.utc)
    dt_test = datetime(2025, 8, 1, 0, 0, tzinfo=timezone.utc)

    pgn = '[Event "T"]\n[White "recency_user"]\n[Black "Opp"]\n[Result "1-0"]\n[TimeControl "300+0"]\n\n1. e4 e5 2. Nf3 Nc6 1-0\n'

    mock_games = [
        {"game_id": "g_new", "pgn": pgn, "player_color": "white", "time_control": "300+0", "date": dt_new,
         "category": "blitz", "is_fast": False, "rating": 1500, "opponent_rating": 1500, "termination": "Normal"},
        {"game_id": "g_old", "pgn": pgn, "player_color": "white", "time_control": "300+0", "date": dt_old,
         "category": "blitz", "is_fast": False, "rating": 1500, "opponent_rating": 1500, "termination": "Normal"},
        {"game_id": "g_val", "pgn": pgn, "player_color": "white", "time_control": "300+0", "date": dt_val,
         "category": "blitz", "is_fast": False, "rating": 1500, "opponent_rating": 1500, "termination": "Normal"},
        {"game_id": "g_test", "pgn": pgn, "player_color": "white", "time_control": "300+0", "date": dt_test,
         "category": "blitz", "is_fast": False, "rating": 1500, "opponent_rating": 1500, "termination": "Normal"},
    ]

    try:
        # Mock extractor and splitter to directly control split assignment
        def mock_extract(u):
            # 2 samples per game
            samples = []
            for g in mock_games:
                for ply in [0, 2]:
                    samples.append({
                        "game_id": g["game_id"], "ply": ply, "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                        "uci": "e2e4" if ply == 0 else "g1f3", "date": g["date"], "category": "blitz", "is_fast": False,
                        "time_control": "300+0", "player_rating": 1500, "opponent_rating": 1500, "color": "white",
                        "termination": "Normal", "clock_before_move": 300.0, "halfmove_clock": 0, "sample_weight": 1.0,
                        "target_berserk": False, "prev_move_1": None, "prev_move_2": None, "clock_after_move": 300.0,
                        "time_spent_s": None, "increment_s": 0, "base_s": 300,
                        "created_at_ms": int(g["date"].timestamp() * 1000), "game_result_for_player": 1.0, "num_legal_moves": 20,
                    })
            return samples, {"unparseable_pgn": 0, "illegal_move": 0, "no_target_moves": 0}

        def mock_split(samples):
            for s in samples:
                if s["game_id"] in ("g_new", "g_old"):
                    s["split"] = "train"
                elif s["game_id"] == "g_val":
                    s["split"] = "val"
                else:
                    s["split"] = "test"
            return samples

        with patch("app.services.dataset_builder_v3.extractor.extract_samples", side_effect=mock_extract), \
             patch("app.services.dataset_builder_v3.splitter.assign_splits", side_effect=mock_split):
            info = build_dataset_v3(username)

        parquet_path = player_dir(username) / "samples_v3.parquet"
        assert parquet_path.exists()
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        train_df = df[df["split"] == "train"]
        w_new = train_df[train_df["game_id"] == "g_new"]["sample_weight"].iloc[0]
        w_old = train_df[train_df["game_id"] == "g_old"]["sample_weight"].iloc[0]

        # Hand-computed ratio between games 365 days apart is exactly 2.0
        ratio = float(w_new / w_old)
        assert pytest.approx(ratio, rel=1e-5) == 2.0

        # Weights strictly decrease with age
        assert w_new > w_old

        # Train mean is normalized to 1.0 +/- 1e-6
        train_mean = float(train_df["sample_weight"].mean())
        assert abs(train_mean - 1.0) < 1e-6

        # Val and test weights are exactly 1.0
        val_df = df[df["split"] == "val"]
        test_df = df[df["split"] == "test"]
        assert all(w == 1.0 for w in val_df["sample_weight"])
        assert all(w == 1.0 for w in test_df["sample_weight"])
    finally:
        config_module.DATA_ROOT = orig_data_root


def test_v3_idempotency_and_two_builds_identical_bytes(tmp_path):
    """
    Two consecutive calls to build_dataset_v3 on fixture games write identical parquet bytes.
    """
    import app.config as config_module
    from app.services.storage import player_dir

    orig_data_root = config_module.DATA_ROOT
    config_module.DATA_ROOT = tmp_path
    username = "idempotent_v3_user"
    games = load_fixture_games()

    try:
        with patch("app.services.extractor.storage.load_games", return_value=games):
            build_dataset_v3(username)
        parquet_path = player_dir(username) / "samples_v3.parquet"
        bytes_1 = parquet_path.read_bytes()
        hash_1 = hashlib.sha256(bytes_1).hexdigest()

        with patch("app.services.extractor.storage.load_games", return_value=games):
            build_dataset_v3(username)
        bytes_2 = parquet_path.read_bytes()
        hash_2 = hashlib.sha256(bytes_2).hexdigest()

        assert hash_1 == hash_2
        assert bytes_1 == bytes_2
    finally:
        config_module.DATA_ROOT = orig_data_root


def test_v3_schema_artifacts_and_no_v2_leakage(tmp_path):
    """
    Check that build_dataset_v3 creates samples_v3.parquet and info_v3.json,
    does NOT write samples.parquet, info.json, opening_book.json, or dataset_report.md.
    Verify explicit pyarrow schema including all 9 new columns and target_berserk.
    """
    import app.config as config_module
    from app.services.storage import player_dir

    orig_data_root = config_module.DATA_ROOT
    config_module.DATA_ROOT = tmp_path
    username = "artifacts_v3_user"
    games = load_fixture_games()

    try:
        with patch("app.services.extractor.storage.load_games", return_value=games):
            info = build_dataset_v3(username)

        pdir = player_dir(username)
        assert (pdir / "samples_v3.parquet").exists()
        assert (pdir / "info_v3.json").exists()

        # Must NOT write v2 files or reports
        assert not (pdir / "samples.parquet").exists()
        assert not (pdir / "info.json").exists()
        assert not (pdir / "opening_book.json").exists()
        assert not (pdir / "dataset_report.md").exists()

        assert info["schema_version"] == "v3"

        table = pq.read_table(pdir / "samples_v3.parquet")
        schema = table.schema

        expected_columns = [
            "game_id", "ply", "fen", "uci", "date", "category", "is_fast",
            "time_control", "player_rating", "opponent_rating", "color",
            "termination", "clock_before_move", "halfmove_clock", "sample_weight",
            "split", "schema_version", "target_berserk",
            # Phase B.1 new columns
            "prev_move_1", "prev_move_2", "clock_after_move", "time_spent_s",
            "increment_s", "base_s", "created_at_ms", "game_result_for_player",
            "num_legal_moves",
        ]
        assert schema.names == expected_columns
    finally:
        config_module.DATA_ROOT = orig_data_root


def test_v3_tier_boundaries():
    """
    Test tier boundaries: <50 rejected, 50-149 base_only, 150-799 standard, >=800 full.
    Boundaries tested: 49, 50, 149, 150, 799, 800.
    """
    assert config.model_tier(49) == "rejected"
    assert config.model_tier(50) == "base_only"
    assert config.model_tier(149) == "base_only"
    assert config.model_tier(150) == "standard"
    assert config.model_tier(799) == "standard"
    assert config.model_tier(800) == "full"
