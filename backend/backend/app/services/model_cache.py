"""
In-memory LRU cache of loaded player bots (Milestone 4, Phase 2).

Why
---
Loading model.pt (and the opening book JSON) from disk on every move request
is wasteful. This cache keeps the most recently used players' bots in memory
and drops the least recently used one when it is full, so memory stays bounded
as more players get trained.

Design
------
* Keyed by lowercase username. Capacity: config.MODEL_CACHE_MAX_SIZE (LRU).
* An entry is a LoadedBot: model + opening book + a file "version"
  (mtime_ns, size) of model.pt.
* Retraining: every get() compares the current model.pt version with the
  cached one. If the file changed, the bot is reloaded, so NEW games always get
  the newest model. A LoadedBot is immutable and sessions keep their own
  reference to it, so a game already in progress keeps playing with the model
  it started with, even after eviction, invalidation or retraining.
* Thread safety: FastAPI runs sync endpoints in a thread pool, so
  - one lock protects the cache dict (held only for quick dict operations), and
  - one lock PER USERNAME serialises loading, so ten simultaneous requests for
    the same player trigger a single disk load instead of ten.
  Loading never happens while the dict lock is held, so loading one player
  never blocks cache hits for other players.
* torch is imported lazily (inside the default loader), so importing this
  module is cheap and does not require torch.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .. import config
from . import player_paths

logger = logging.getLogger(__name__)

FileVersion = tuple[int, int]  # (mtime_ns, size_in_bytes) of model.pt


class ModelNotFoundError(LookupError):
    """No usable model.pt exists for this player (missing or empty)."""


class ModelLoadError(RuntimeError):
    """model.pt exists but could not be loaded (corrupt / incompatible)."""


@dataclass(frozen=True)
class LoadedBot:
    """Everything needed to play as one player. Immutable."""
    username: str
    model: Any
    opening_book: Any
    version: FileVersion
    loaded_at: float


Loader = Callable[[str, FileVersion], LoadedBot]


def load_bot_from_disk(username: str, version: FileVersion) -> LoadedBot:
    """
    Default loader: read model.pt + opening_book.json for *username*.

    Models are loaded on CPU: the network is small and serving evaluates one
    position at a time. The architecture comes from config.py (the same
    defaults train_player_model() uses); a model trained with overrides will
    not match and is reported as a ModelLoadError.
    """
    try:
        from ..ml import export  # lazy: keeps torch out of module import
        from .opening_book import OpeningBook

        model = export.load_model(
            player_paths.model_path(username),
            device="cpu",
            channels=config.MODEL_CHANNELS,
            num_res_blocks=config.MODEL_RES_BLOCKS,
            hidden_dim=config.MODEL_HIDDEN_DIM,
            dropout=config.MODEL_DROPOUT,
        )
        opening_book = OpeningBook(username)
    except Exception as exc:  # noqa: BLE001 - any failure means "not loadable"
        # Details (file paths, torch errors) go to the server log, not to API clients.
        logger.exception("Failed to load model for '%s'", username)
        raise ModelLoadError(
            "The model file could not be loaded (it may be corrupt or built "
            "with a different architecture)."
        ) from exc

    return LoadedBot(
        username=username,
        model=model,
        opening_book=opening_book,
        version=version,
        loaded_at=time.time(),
    )


class ModelCache:
    """Thread-safe LRU cache of LoadedBot objects keyed by username."""

    def __init__(
        self,
        max_size: Optional[int] = None,
        loader: Optional[Loader] = None,
    ) -> None:
        self._max_size = max_size              # None -> read config at use time
        self._loader: Loader = loader or load_bot_from_disk
        self._entries: "OrderedDict[str, LoadedBot]" = OrderedDict()
        self._lock = threading.Lock()          # guards _entries, _key_locks, counters
        self._key_locks: dict[str, threading.Lock] = {}
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------------ utils
    @property
    def max_size(self) -> int:
        size = self._max_size if self._max_size is not None else config.MODEL_CACHE_MAX_SIZE
        return max(1, int(size))

    @staticmethod
    def _file_version(key: str) -> FileVersion:
        """Current (mtime_ns, size) of model.pt; raises ModelNotFoundError."""
        path = player_paths.model_path(key)
        try:
            st = path.stat()
        except OSError:
            raise ModelNotFoundError(f"No model found for '{key}'.") from None
        if not path.is_file() or st.st_size == 0:
            raise ModelNotFoundError(f"No usable model file for '{key}'.")
        return (st.st_mtime_ns, st.st_size)

    def _key_lock(self, key: str) -> threading.Lock:
        # Only usernames with an existing model.pt reach here (validated and
        # stat-checked first), so this dict is bounded by the number of
        # trained players and each entry is tiny.
        with self._lock:
            return self._key_locks.setdefault(key, threading.Lock())

    def _lookup(self, key: str, version: FileVersion) -> Optional[LoadedBot]:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.version == version:
                self._entries.move_to_end(key)
                self.hits += 1
                return entry
        return None

    # -------------------------------------------------------------- public API
    def get(self, username: str) -> LoadedBot:
        """
        Return the LoadedBot for *username*, loading it if needed.

        Raises
        ------
        player_paths.InvalidUsernameError  malformed username
        ModelNotFoundError                 no usable model.pt on disk
        ModelLoadError                     model.pt could not be loaded
        """
        key = player_paths.normalize_username(username)
        version = self._file_version(key)

        cached = self._lookup(key, version)
        if cached is not None:
            return cached

        with self._key_lock(key):
            # Another thread may have finished loading while we waited.
            cached = self._lookup(key, version)
            if cached is not None:
                return cached

            with self._lock:
                self.misses += 1
            try:
                bot = self._loader(key, version)
            except ModelLoadError:
                self._drop(key)   # a stale older model must not outlive a failed reload
                raise
            except Exception as exc:  # noqa: BLE001
                self._drop(key)
                logger.exception("Unexpected error while loading model for '%s'", key)
                raise ModelLoadError("The model could not be loaded.") from exc

            with self._lock:
                self._entries[key] = bot
                self._entries.move_to_end(key)
                while len(self._entries) > self.max_size:
                    self._entries.popitem(last=False)   # evict least recently used
            return bot

    def _drop(self, key: str) -> bool:
        with self._lock:
            return self._entries.pop(key, None) is not None

    def invalidate(self, username: str) -> bool:
        """Forget one player's cached bot. Returns True if something was removed."""
        return self._drop(player_paths.normalize_username(username))

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def keys(self) -> list[str]:
        """Cached usernames, least recently used first."""
        with self._lock:
            return list(self._entries.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "size": len(self._entries),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "keys": list(self._entries.keys()),
            }


# Process-wide cache used by the routers. Tests replace this attribute with
# their own ModelCache(loader=fake) via monkeypatch.
default_cache = ModelCache()
