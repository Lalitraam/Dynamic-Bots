import pytest

from app import config
from app.services import player_paths
from app.services.player_paths import InvalidUsernameError


@pytest.mark.parametrize("name", ["ab", "The_King_Crusher", "a-b_c9", "X" * 30])
def test_valid_usernames_are_lowercased(name):
    assert player_paths.normalize_username(name) == name.lower()


@pytest.mark.parametrize(
    "name",
    ["", "a", "X" * 31, "../etc", "a/b", "a\\b", "a b", "bad.name", "name!", "..", "ab\n"],
)
def test_invalid_usernames_rejected(name):
    with pytest.raises(InvalidUsernameError):
        player_paths.normalize_username(name)


def test_non_string_rejected():
    with pytest.raises(InvalidUsernameError):
        player_paths.normalize_username(None)


def test_player_path_is_read_only(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path)
    path = player_paths.player_path("Some_Player")
    assert path == tmp_path / "players" / "some_player"
    # Nothing may be created on disk by building a path.
    assert not (tmp_path / "players").exists()


def test_artifact_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path)
    base = tmp_path / "players" / "abc"
    assert player_paths.model_path("ABC") == base / "model.pt"
    assert player_paths.info_path("ABC") == base / "info.json"
    assert player_paths.training_info_path("ABC") == base / "training_info.json"
    assert player_paths.opening_book_path("ABC") == base / "opening_book.json"
