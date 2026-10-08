"""
Safe, read-only path helpers for per-player artifacts (Milestone 4).

storage.player_dir() creates the directory as a side effect and does not
validate the username. That is fine for ingestion, but unsafe for serving
endpoints where the username comes straight from an untrusted URL:

  * every typo or probe would create an empty folder under data/players/
  * a crafted name could point outside the data folder

These helpers validate first and NEVER create anything on disk. They read
config.DATA_ROOT at call time so tests can override it with monkeypatch.
"""
import re
from pathlib import Path

from .. import config

_USERNAME_RE = re.compile(config.USERNAME_PATTERN)


class InvalidUsernameError(ValueError):
    """Raised when a username does not look like a valid Lichess username."""


def normalize_username(username: str) -> str:
    """Validate *username* and return its lowercase form (the storage key)."""
    if not isinstance(username, str) or not _USERNAME_RE.fullmatch(username):
        raise InvalidUsernameError(
            "Invalid username: use 2-30 letters, digits, '_' or '-'."
        )
    return username.lower()


def players_root() -> Path:
    """Root folder holding every player's directory (not created here)."""
    return Path(config.DATA_ROOT) / "players"


def player_path(username: str) -> Path:
    """
    Path of a player's directory. Validates the username; does not create it.
    """
    key = normalize_username(username)
    path = players_root() / key
    # Defence in depth: the regex already forbids separators and dots.
    if path.resolve().parent != players_root().resolve():
        raise InvalidUsernameError("Invalid username.")
    return path


def info_path(username: str) -> Path:
    return player_path(username) / "info.json"


def model_path(username: str) -> Path:
    return player_path(username) / "model.pt"


def training_info_path(username: str) -> Path:
    return player_path(username) / "training_info.json"


def opening_book_path(username: str) -> Path:
    return player_path(username) / "opening_book.json"
