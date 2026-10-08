"""Phase 2 demo: watch the model cache load, hit, and evict."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config
from app.ml import export
from app.ml.model import ChessPolicyNet
from app.services import player_paths
from app.services.model_cache import ModelCache

# Create 3 fake players with untrained models (for demonstration only).
for name in ["demo_bot", "demo_bot2", "demo_bot3"]:
    model = ChessPolicyNet(
        channels=config.MODEL_CHANNELS,
        num_res_blocks=config.MODEL_RES_BLOCKS,
        hidden_dim=config.MODEL_HIDDEN_DIM,
        dropout=config.MODEL_DROPOUT,
    )
    export.save_model(model, player_paths.model_path(name))

cache = ModelCache(max_size=2)  # small size so eviction is easy to see


def timed_get(name):
    start = time.perf_counter()
    cache.get(name)
    ms = (time.perf_counter() - start) * 1000
    print(f"get({name}): {ms:7.1f} ms   cached now: {cache.keys()}")


timed_get("demo_bot")    # slow: loaded from disk
timed_get("demo_bot")    # fast: served from memory
timed_get("demo_bot2")   # loaded, cache now holds 2
timed_get("demo_bot3")   # loaded, evicts demo_bot (least recently used)
timed_get("demo_bot")    # slow again: it had been evicted
print(cache.stats())