"""
Configuration constants for the Chess Bot Clone Factory.
"""

import os
from pathlib import Path

# Data root directory (resolves to the repo-root data/ folder, the parent of backend/)
DATA_ROOT = Path(__file__).resolve().parents[2] / "data"

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
FETCH_SINCE_DAYS = MAX_AGE_DAYS
PER_BUCKET_FETCH_MAX = 1500  # max per bucket request

# Default maximum games to download from Lichess API (3 * per-bucket cap)
DEFAULT_MAX_GAMES = 3 * PER_BUCKET_FETCH_MAX


def default_since_ms(now=None) -> int:
    from datetime import datetime, timezone
    if now is None:
        now = datetime.now(timezone.utc)
    return int((now.timestamp() - FETCH_SINCE_DAYS * 86400) * 1000)

# Rate limiting and resume constants
RATE_LIMIT_WAIT_SECONDS = 60.0
OVERLAP_MS = 120000        # 2 minutes overlap window on resume
RESUME_SLACK = 100         # Additional games requested on resume

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
MIN_GAMES_REQUIRED = 50

# Schema versions for saved games and datasets
SCHEMA_VERSION = "v2"
SCHEMA_VERSION_V3 = "v3"

# Recency weighting half-life in days (Phase B.2)
RECENCY_HALF_LIFE_DAYS = 365


def model_tier(n_games: int) -> str:
    """
    Determine the model tier based on the number of games sampled.
    Returns "rejected" if n_games < MIN_GAMES_REQUIRED (50).
    50-149: "base_only"
    150-799: "standard"
    >=800: "full"
    """
    if n_games < MIN_GAMES_REQUIRED:
        return "rejected"
    if n_games < 150:
        return "base_only"
    if n_games < 800:
        return "standard"
    return "full"


# ---------------------------------------------------------------------------
# Milestone 2: Tensor Generation & Feature Engineering
# ---------------------------------------------------------------------------

# Board tensor dimensions
BOARD_PLANES = 18  # 18×8×8 board representation
HISTORY_PLANES = True

# Move promotion classes: 0=none/queen, 1=knight, 2=bishop, 3=rook
PROMO_CLASSES = 4

# Metadata vector
META_DIM = 8                           # length of metadata feature vector
META_DIM_V3 = 10                       # length of metadata feature vector (v3)
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
OPENING_BOOK_PLY_CUTOFF = 10   # currently unused / reserved

# Opening book v2 constants (B.5)
BOOK_PLY_CUTOFF = 16
BOOK_MIN_VISITS = 5
BOOK_ALPHA_K = 10

# Phase D — fine-tuning constants
FT_MAX_MINUTES = 20
FT_MAX_EPOCHS = 8
FT_PATIENCE = 2
FT_BATCH_SIZE = 256
FT_LR_FULL = 3e-4
FT_LR_FREEZE = 5e-4
FT_LR_FILM_ONLY = 1e-3
FT_FREEZE_UNTIL_BLOCK = 3
FT_MIN_VAL_GAMES = 10
TEMPERATURE_GRID = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
PROMOTION_GATE_MIN_GAMES = 20
PROMOTION_GATE_BOOTSTRAP_N = 2000
PROMOTION_GATE_SEED = 0

# Phase E — inference v2
PROVIDER = "v3"                        # "legacy" or "v3"
MODEL_CACHE_SIZE = 8
TOP_P = 0.95
THINK_MIN_SAMPLES_PER_BUCKET = 20
THINK_MAX_SAMPLES_PER_BUCKET = 500
THINK_FLOOR_MS = 150
THINK_MARGIN_MS = 300
GUARD_ENABLED = False
GUARD_CP = 300
STOCKFISH_PATH = None
STOCKFISH_DEPTH = 12
TELEMETRY_ENABLED = True

# Phase SP — self-play
SP_MAX_PLIES = 300
SP_DEFAULT_GAMES = 20
SP_THINK_FLOOR_MS = 150
SP_THINK_MARGIN_MS = 300
BASE_THINKTIME_DIR = DATA_ROOT / "base" / "thinktime"
BASE_THINKTIME_SEED = 0
BASE_THINKTIME_MAX_PER_BUCKET = 500

# Phase P — analysis and error model
STOCKFISH_PATH = r"E:\dynamic_bots\tools\stockfish\stockfish.exe"
STOCKFISH_DEPTH = 12
STOCKFISH_THREADS = 1
STOCKFISH_HASH_MB = 64
ERROR_MODEL_MIN_CP_LOSS = 100
ERROR_MODEL_BLUNDER_CP = 300
ERROR_MODEL_CP_CLIP = 1000
ERROR_MODEL_POOLED_WEIGHT = 200
ERROR_MODEL_FALLBACK_RATE = 0.08

# Phase G — Database & Auth
DB_PATH = Path(os.environ.get("DB_PATH", str(DATA_ROOT / "app.db")))
INVITE_CODE_REQUIRED = True
MAX_USER_BOTS = 10
SESSION_EXPIRE_DAYS = 30
SECRET_KEY = os.environ.get("SECRET_KEY", "phase_g_secret_key_development_only")