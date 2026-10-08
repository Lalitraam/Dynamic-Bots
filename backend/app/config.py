"""
Configuration constants for the Chess Bot Clone Factory.
"""

from pathlib import Path

# Data root directory (relative to backend)
DATA_ROOT = Path(__file__).resolve().parents[2] / "data"

# Default maximum games to download from Lichess API
DEFAULT_MAX_GAMES = 4000

# Minimum base time in seconds for a time control to be considered (60 seconds)
MIN_BASE_SECONDS = 60

# Minimum plies in a game to be considered valid
MIN_PLIES = 10

# Default rating for players when missing or unparseable
DEFAULT_RATING = 1500

# Age boundaries for game bucketing in days
BUCKET_MAX_AGE_6M = 180    # 6 months
BUCKET_MAX_AGE_18M = 540   # 18 months
MAX_AGE_DAYS = 1095        # 3 years (36 months)

# Included categories (time control categories) for training
INCLUDED_CATEGORIES = {"bullet", "blitz", "rapid"}

# Quotas for each bucket (last_6m, 6_18m, 18_36m) in the sampling stage
BUCKET_QUOTAS = {
    "last_6m": 500,
    "6_18m": 400,
    "18_36m": 300
}

# Maximum total games to sample after redistribution
MAX_TOTAL_SAMPLED = 1200

# Minimum number of valid games required to proceed (after filtering and bucketing)
MIN_GAMES_REQUIRED = 150

# Schema version for saved games
SCHEMA_VERSION = "v1"



def model_tier(n_games: int) -> str:
    """
    Determine the model tier based on the number of games sampled.
    Raises ValueError if n_games < MIN_GAMES_REQUIRED.
    """
    if n_games < MIN_GAMES_REQUIRED:
        return "rejected"
    if n_games < 400:
        return "small"
    if n_games < 800:
        return "standard"
    return "full"


# ---------------------------------------------------------------------------
# Milestone 2: Tensor Generation & Feature Engineering
# ---------------------------------------------------------------------------

# Board tensor dimensions
BOARD_PLANES = 18  # 18×8×8 board representation

# Move promotion classes: 0=none/queen, 1=knight, 2=bishop, 3=rook
PROMO_CLASSES = 4

# Metadata vector
META_DIM = 8                           # length of metadata feature vector
META_RATING_CENTER = DEFAULT_RATING    # 1500 — center for rating normalization
META_RATING_SCALE = 400                # rating normalization divisor
META_BASE_TIME_MIN = MIN_BASE_SECONDS  # 60s — lower bound for base-time normalization
META_BASE_TIME_MAX = 1499              # just under classical — upper bound (raw base_seconds, NOT estimated)
META_INCREMENT_CLIP = 30               # max increment seconds for normalization
META_PLY_CLIP = 200                    # max ply for normalization
META_CLOCK_CLIP = 3.0                  # max clock ratio (remaining / base)
META_HALFMOVE_CLIP = 100              # max halfmove clock for normalization
META_CLOCK_DEFAULT = 0.5               # default clock ratio when clock unavailable

# Train/val/test split ratios (by game, chronologically)
TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10

# Opening book
MIN_BOOK_VISITS = 2  # minimum times a position must be seen to appear in book


# ---------------------------------------------------------------------------
# Milestone 3: Model Architecture & Training
# ---------------------------------------------------------------------------

TRAIN_BATCH_SIZE = 64
TRAIN_LR = 1e-3
TRAIN_WEIGHT_DECAY = 1e-4
TRAIN_MAX_EPOCHS = 100
TRAIN_EARLY_STOP_PATIENCE = 8
TRAIN_LR_PATIENCE = 3
TRAIN_LR_FACTOR = 0.5
LABEL_SMOOTHING = 0.05
PROMO_LOSS_WEIGHT = 0.2
AUX_LOSS_WEIGHT = 0.1
MODEL_CHANNELS = 64
MODEL_RES_BLOCKS = 5
MODEL_HIDDEN_DIM = 256
MODEL_DROPOUT = 0.3
TRAIN_SEED = 42
INFERENCE_TEMPERATURE_DEFAULT = 1.0
OPENING_BOOK_PLY_CUTOFF = 10   # used only by evaluate.py's "opening divergence" metric


# ---------------------------------------------------------------------------
# Milestone 4: Serving & Inference API
# ---------------------------------------------------------------------------

# Lichess usernames: 2-30 characters, letters / digits / underscore / hyphen.
# Serving endpoints validate against this BEFORE touching the filesystem.
USERNAME_PATTERN = r"^[A-Za-z0-9_-]{2,30}$"

# Model cache (Phase 2): max number of loaded models kept in memory (LRU).
MODEL_CACHE_MAX_SIZE = 8

# Game sessions (Phase 3): idle expiry and hard cap on live sessions.
SESSION_IDLE_TTL_SECONDS = 30 * 60
SESSION_MAX_COUNT = 200

# Temperature bounds accepted from API callers (Phase 4).
# The default stays INFERENCE_TEMPERATURE_DEFAULT (Milestone 3).
TEMPERATURE_MIN = 0.1
TEMPERATURE_MAX = 2.0



# Rating metadata bounds accepted for the human / bot in a session (Phase 3).
RATING_MIN = 100
RATING_MAX = 3500