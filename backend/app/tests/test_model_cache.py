"""
Tests for the LRU model cache (Milestone 4, Phase 2).

Most tests use a fake loader, so they need no real model and no torch.
The last two tests use the real loader and are skipped if torch is missing.
"""
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import chess
import pytest

from app import config
from app.services import player_paths
from app.services.model_cache import (
    LoadedBot,
    ModelCache,
    ModelLoadError,
    ModelNotFoundError,
)
from app.services.player_paths import InvalidUsernameError


class FakeLoader:
    """Records calls; can be told to fail for some usernames or to be slow."""

    def __init__(self, delay: float = 0.0):
        self.calls: list[str] = []
        self.fail: set[str] = set()
        self.fail_with_plain_error: set[str] = set()
        self.delay = delay
        self._lock = threading.Lock()

    def __call__(self, username, version):
        with self._lock:
            self.calls.append(username)
        if self.delay:
            time.sleep(self.delay)
        if username in self.fail:
            raise ModelLoadError("boom")
        if username in self.fail_with_plain_error:
            raise ValueError("plain error")
        return LoadedBot(
            username=username,
            model=object(),
            opening_book=object(),
            version=version,
            loaded_at=time.time(),
        )


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path)
    return tmp_path


def make_model(root, name, content=b"weights"):
    d = root / "players" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.pt").write_bytes(content)
    return d / "model.pt"


# ---------------------------------------------------------------------------
# Hits, misses, LRU eviction
# ---------------------------------------------------------------------------

def test_miss_then_hit_loads_once(data_root):
    make_model(data_root, "alice")
    loader = FakeLoader()
    cache = ModelCache(max_size=3, loader=loader)

    first = cache.get("Alice")
    second = cache.get("alice")

    assert first is second
    assert loader.calls == ["alice"]
    stats = cache.stats()
    assert stats["hits"] == 1 and stats["misses"] == 1


def test_lru_eviction_drops_least_recently_used(data_root):
    for name in ["aa", "bb", "cc"]:
        make_model(data_root, name)
    cache = ModelCache(max_size=2, loader=FakeLoader())

    cache.get("aa")
    cache.get("bb")
    cache.get("aa")        # aa is now the most recently used
    cache.get("cc")        # evicts bb

    assert cache.keys() == ["aa", "cc"]
    assert len(cache) == 2


def test_evicted_bot_reference_stays_usable(data_root):
    for name in ["aa", "bb"]:
        make_model(data_root, name)
    cache = ModelCache(max_size=1, loader=FakeLoader())

    held = cache.get("aa")   # e.g. a game session holding its bot
    cache.get("bb")          # evicts aa from the cache

    assert cache.keys() == ["bb"]
    assert held.username == "aa"   # the held object is untouched


def test_default_max_size_comes_from_config(data_root, monkeypatch):
    monkeypatch.setattr(config, "MODEL_CACHE_MAX_SIZE", 1)
    for name in ["aa", "bb"]:
        make_model(data_root, name)
    cache = ModelCache(loader=FakeLoader())
    cache.get("aa")
    cache.get("bb")
    assert cache.keys() == ["bb"]


# ---------------------------------------------------------------------------
# Retraining
# ---------------------------------------------------------------------------

def test_retrained_model_is_reloaded_for_new_requests(data_root):
    path = make_model(data_root, "alice", b"weights-v1")
    loader = FakeLoader()
    cache = ModelCache(max_size=3, loader=loader)

    old = cache.get("alice")
    path.write_bytes(b"weights-v2-retrained-and-longer")   # file changed
    new = cache.get("alice")

    assert new is not old
    assert loader.calls == ["alice", "alice"]
    assert old.version != new.version
    assert len(cache) == 1                                  # replaced, not duplicated


def test_failed_reload_after_retrain_drops_stale_entry(data_root):
    path = make_model(data_root, "alice", b"weights-v1")
    loader = FakeLoader()
    cache = ModelCache(max_size=3, loader=loader)
    cache.get("alice")

    path.write_bytes(b"corrupt-new-file-with-different-size")
    loader.fail.add("alice")
    with pytest.raises(ModelLoadError):
        cache.get("alice")

    assert len(cache) == 0   # the old model must not linger after a failed reload


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def test_missing_model_raises_not_found_and_creates_nothing(data_root):
    cache = ModelCache(loader=FakeLoader())
    with pytest.raises(ModelNotFoundError):
        cache.get("ghost")
    assert not (data_root / "players").exists()


def test_empty_model_file_raises_not_found(data_root):
    make_model(data_root, "alice", b"")
    cache = ModelCache(loader=FakeLoader())
    with pytest.raises(ModelNotFoundError):
        cache.get("alice")


def test_invalid_username_raises(data_root):
    cache = ModelCache(loader=FakeLoader())
    with pytest.raises(InvalidUsernameError):
        cache.get("../etc")


def test_loader_failure_is_not_cached(data_root):
    make_model(data_root, "alice")
    loader = FakeLoader()
    loader.fail.add("alice")
    cache = ModelCache(loader=loader)

    with pytest.raises(ModelLoadError):
        cache.get("alice")
    assert len(cache) == 0

    loader.fail.clear()               # fixed: now it loads
    assert cache.get("alice").username == "alice"


def test_unexpected_loader_exception_is_wrapped(data_root):
    make_model(data_root, "alice")
    loader = FakeLoader()
    loader.fail_with_plain_error.add("alice")
    cache = ModelCache(loader=loader)
    with pytest.raises(ModelLoadError):
        cache.get("alice")


def test_loader_error_text_is_logged_not_exposed(data_root, caplog):
    make_model(data_root, "alice")
    loader = FakeLoader()
    loader.fail_with_plain_error.add("alice")        # raises ValueError("plain error")
    cache = ModelCache(loader=loader)
    with caplog.at_level(logging.ERROR, logger="app.services.model_cache"):
        with pytest.raises(ModelLoadError) as excinfo:
            cache.get("alice")
    assert "plain error" not in str(excinfo.value)    # nothing internal reaches clients
    assert "plain error" in caplog.text               # but the server log has the details


# ---------------------------------------------------------------------------
# invalidate / clear
# ---------------------------------------------------------------------------

def test_invalidate_and_clear(data_root):
    for name in ["aa", "bb"]:
        make_model(data_root, name)
    loader = FakeLoader()
    cache = ModelCache(loader=loader)
    cache.get("aa")
    cache.get("bb")

    assert cache.invalidate("AA") is True
    assert cache.invalidate("aa") is False
    assert cache.keys() == ["bb"]

    cache.clear()
    assert len(cache) == 0


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

def test_concurrent_requests_for_same_player_load_once(data_root):
    make_model(data_root, "alice")
    loader = FakeLoader(delay=0.2)
    cache = ModelCache(max_size=3, loader=loader)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: cache.get("alice"), range(8)))

    assert loader.calls == ["alice"]                 # one disk load, not eight
    assert all(r is results[0] for r in results)


def test_concurrent_mixed_players_never_exceed_max_size(data_root):
    names = ["aa", "bb", "cc", "dd", "ee"]
    for name in names:
        make_model(data_root, name)
    cache = ModelCache(max_size=3, loader=FakeLoader(delay=0.01))

    with ThreadPoolExecutor(max_workers=10) as pool:
        list(pool.map(lambda i: cache.get(names[i % len(names)]), range(60)))

    assert len(cache) <= 3


# ---------------------------------------------------------------------------
# Real loader (needs torch; skipped otherwise)
# ---------------------------------------------------------------------------

def test_real_loader_wraps_corrupt_file(data_root):
    pytest.importorskip("torch")
    make_model(data_root, "alice", b"this is not a torch checkpoint")
    cache = ModelCache()    # default loader
    with pytest.raises(ModelLoadError):
        cache.get("alice")


def test_real_loader_roundtrip_and_play(data_root):
    pytest.importorskip("torch")
    from app.ml import export
    from app.ml.model import ChessPolicyNet

    model = ChessPolicyNet(
        channels=config.MODEL_CHANNELS,
        num_res_blocks=config.MODEL_RES_BLOCKS,
        hidden_dim=config.MODEL_HIDDEN_DIM,
        dropout=config.MODEL_DROPOUT,
    )
    export.save_model(model, player_paths.model_path("trainedbot"))

    cache = ModelCache()    # default loader
    bot = cache.get("trainedbot")

    assert bot.model.training is False                 # eval mode
    assert bot.opening_book.book == {}                 # no opening_book.json written
    move = export.predict_move(
        bot.model, chess.Board(), opening_book=bot.opening_book, temperature=1.0
    )
    assert move in chess.Board().legal_moves
