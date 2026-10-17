-- Initial database schema for Chess Bot Clone Factory (Phase G)

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS invite_codes (
    code TEXT PRIMARY KEY,
    created_by INTEGER REFERENCES users(id),
    used_by INTEGER REFERENCES users(id),
    used_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS players (
    username TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'queued',
    active_version INTEGER DEFAULT 1,
    last_trained_at TIMESTAMP,
    games_used INTEGER DEFAULT 0,
    tier TEXT DEFAULT 'free',
    blocked INTEGER DEFAULT 0,
    error TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_bots (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    username TEXT NOT NULL REFERENCES players(username) ON DELETE CASCADE,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    archived INTEGER DEFAULT 0,
    PRIMARY KEY (user_id, username)
);

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    username TEXT NOT NULL,
    requested_by INTEGER REFERENCES users(id),
    status TEXT NOT NULL DEFAULT 'queued',
    step TEXT,
    progress REAL DEFAULT 0.0,
    error TEXT,
    payload_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    finished_at TIMESTAMP,
    heartbeat_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS games (
    id TEXT PRIMARY KEY,
    owner_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    player_username TEXT NOT NULL REFERENCES players(username),
    model_version TEXT NOT NULL,
    human_color TEXT NOT NULL,
    base_s INTEGER NOT NULL,
    inc_s INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    result TEXT,
    termination TEXT,
    secret_hash TEXT NOT NULL,
    pgn TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS game_moves (
    game_id TEXT NOT NULL REFERENCES games(id) ON DELETE CASCADE,
    ply INTEGER NOT NULL,
    uci TEXT NOT NULL,
    san TEXT,
    by TEXT NOT NULL,
    clock_after_ms INTEGER,
    elapsed_ms INTEGER,
    server_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (game_id, ply)
);

CREATE TABLE IF NOT EXISTS move_telemetry (
    game_id TEXT NOT NULL REFERENCES games(id) ON DELETE CASCADE,
    ply INTEGER NOT NULL,
    source TEXT NOT NULL,
    top1_prob REAL,
    entropy REAL,
    p_err REAL,
    guarded INTEGER DEFAULT 0,
    infer_ms REAL,
    think_ms REAL,
    model_version TEXT,
    PRIMARY KEY (game_id, ply)
);
