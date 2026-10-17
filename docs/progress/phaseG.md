# Phase G — Database, Authentication, and Game Engine APIs

**Date:** 2026-10-18  
**Branch:** `phase-b`  
**Status:** Completed  

---

## 1. Database Foundation (Task DB.1)
- **Schema Initializer (`app/db/schema/001_init.sql`):**
  - `users`: id, email, password_hash, role, created_at.
  - `sessions`: token_hash (PK), user_id, created_at, expires_at.
  - `invite_codes`: code (PK), created_by, used_by, used_at.
  - `players`: username (PK), status, active_version, last_trained_at, games_used, tier, blocked, error, updated_at.
  - `user_bots`: user_id, username, added_at, archived, PK(user_id, username).
  - `jobs`: id, kind, username, requested_by, status, step, progress, error, payload_json, created_at, started_at, finished_at, heartbeat_at.
  - `games`: id (PK), owner_id, player_username, model_version, human_color, base_s, inc_s, status, result, termination, secret_hash, pgn, created_at, ended_at.
  - `game_moves`: game_id, ply, uci, san, by, clock_after_ms, elapsed_ms, server_ts, PK(game_id, ply).
  - `move_telemetry`: game_id, ply, source, top1_prob, entropy, p_err, guarded, infer_ms, think_ms, model_version, PK(game_id, ply).
- **Migration Runner (`app/db/migrate.py`):**
  - Manages SQLite database connection (`config.DB_PATH`).
  - Sets `PRAGMA journal_mode=WAL` and `PRAGMA foreign_keys=ON`.
  - Tracks applied schema version numbers in `schema_migrations`.
  - Ensures clean idempotent execution on startup.

---

## 2. Authentication & Users (Task G.1)
- **Password & Token Security (`app/auth.py`):**
  - Argon2id password hashing and verification via `argon2-cffi`.
  - 32-byte URL-safe session token generator with SHA-256 token hashing for database storage.
  - FastAPI dependency `get_current_user` enforcing bearer authentication with 401 Unauthorized handling.
- **Auth Endpoints (`app/routers/auth.py`):**
  - `POST /api/auth/register`: Validates invite code, enforces single-use, creates user account.
  - `POST /api/auth/login`: Authenticates credentials, creates 30-day session, returns bearer access token.
  - `GET /api/auth/me`: Authenticated user profile lookup.
- **Admin Management (`scripts/create_admin.py`):**
  - CLI script to seed an initial admin account directly into the database.

---

## 3. Bots API (Task G.2)
- **Bots Router (`app/routers/bots.py`):**
  - `GET /api/bots`: Lists user bots joined with `players` metadata.
  - `POST /api/bots`:
    - Validates target player username on Lichess API.
    - Enforces 10-bot active limit per user account.
    - Creates `players` entry, links `user_bots`, and queues `jobs` pipeline job.
  - `POST /api/bots/{username}/archive`: Archives user bot to release an active slot.

---

## 4. Games API & V3Provider Runtime (Task G.3)
- **Games Router (`app/routers/games.py`):**
  - `POST /api/games`: Initializes chess match, generates game secret, initiates opening move if human is Black.
  - `POST /api/games/{game_id}/move`:
    - Validates game secret against `secret_hash`.
    - Reconstructs board state, validates human legal move, updates move history.
    - Evaluates bot counter-move using `V3Provider` with model caching, think time model, and error probability modeling.
    - Records move telemetry (`top1_prob`, `entropy`, `p_err`, `infer_ms`, `think_ms`).
    - Detects checkmate/draw/resignation and finalizes game status.

---

## 5. Verification & Tests (Task G.4)
- **Unit & Integration Tests:**
  - `app/tests/test_db.py`: Migration execution and idempotency verification.
  - `app/tests/test_auth.py`: Password hashing, invite code registration, login tokens, bad token rejection.
  - `app/tests/test_bots_api.py`: Bot registration, 10-bot limit, archive slot freeing, end-to-end game creation and moves.
- **Results:**
  - 10 passed in `test_db.py`, `test_auth.py`, `test_bots_api.py`.
  - Full backend test suite passing.
