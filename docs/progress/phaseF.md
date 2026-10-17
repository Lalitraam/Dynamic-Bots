# Phase F: Authoritative Clocks, In-Memory Game State, and WebSockets

**Date:** 2026-10-17  
**Branch:** `phase-b`

---

## 1. Executive Summary

Phase F introduces real-time multiplayer capabilities, authoritative server-side time management, and WebSocket infrastructure for the Chess Bot Clone Factory:
1. **Authoritative Game Clock (`app/game/clock.py`)**: Pure injectable `GameClock` handling precise increment calculations, time deduction, and the Lichess move 1 rule (`started` dict preventing time forfeiture/deduction on move 1).
2. **Game State & Manager (`app/game/state.py`, `app/game/manager.py`)**: In-memory single-process manager with per-game async locks (`asyncio.Lock`), database persistence (`games`, `game_moves`, `move_telemetry`), and seamless integration with `V3Provider` for bot counter-moves.
3. **WebSockets (`app/routers/ws.py`)**: Real-time bidirectional WebSocket endpoint (`/ws/{game_id}` and `/api/ws/{game_id}`) featuring secure secret handshake (`{"t": "hello", "secret": ...}`), broadcast snapshots, move dispatching with duplicate ply deduplication, and resignations.
4. **REST Compatibility (`app/routers/games.py`)**: Maintained backward-compatible game creation and move delegation through `GameManager`.

---

## 2. Architecture & Components

### 2.1 GameClock (`backend/app/game/clock.py`)
- **State Representation**:
  - `base_ms`: Base time in milliseconds.
  - `inc_ms`: Increment in milliseconds.
  - `remaining_ms`: Dictionary storing milliseconds remaining for white and black.
  - `running_side`: Active turn side ("white" or "black").
  - `last_tick`: Floating point timestamp of last move/start.
  - `started`: `{"white": False, "black": False}` tracking whether each player has played their move 1.
- **Rules Enforced**:
  - **Lichess Move 1 Rule**: When `started[side]` is `False`, elapsed time is not deducted from the player's clock; increment is added, and `started[side]` is marked `True`.
  - **Subsequent Moves**: Elapsed time since `last_tick` is deducted. If `remaining_ms[side] <= 0`, the clock flags and returns `False`. Increment is added upon successful move.
  - **Clamping**: `get_remaining(now)` dynamically calculates time left for the active side and clamps at 0.

### 2.2 GameState & GameManager (`backend/app/game/state.py`, `backend/app/game/manager.py`)
- **Concurreny Control**: Each active match has an isolated `asyncio.Lock` preventing race conditions during simultaneous move submissions.
- **V3Provider Integration**: Upon human move, the server evaluates game termination. If active, `V3Provider` generates a bot move matching the cloned player's style, records think time, adds telemetry to `move_telemetry`, and updates clocks authoritatively.
- **Idempotency**: Clients passing `client_ply` (or `n`) will have duplicate network retransmissions safely ignored without mutating board state or double-clocking.

### 2.3 WebSockets (`backend/app/routers/ws.py`)
- **Handshake Protocol**:
  1. Client connects to `/ws/{game_id}`.
  2. Server accepts and waits for `{"t": "hello", "secret": "<game_secret>"}`.
  3. Secret is validated against SHA-256 hash stored in DB.
  4. Server replies with full snapshot: `{"t": "snapshot", "fen": ..., "moves": [...], "clocks": {...}}`.
- **Game Event Loop**:
  - `{"t": "move", "uci": "...", "n": ...}`: Server applies human move and bot response, broadcasting updated snapshot to all connected clients for that game.
  - `{"t": "resign"}`: Server resigns game for human, records time/resignation termination, and broadcasts final snapshot.

---

## 3. Verification & Test Results

### 3.1 Unit & Integration Suite (`app/tests/test_clock.py`, `app/tests/test_game_manager.py`)
- `test_clock_initialization`: Verified default clock parameters and started state.
- `test_started_dict_prevents_deduction_on_move_1`: Verified Lichess move 1 time exemption.
- `test_increment_and_deduction_on_subsequent_moves`: Verified accurate clock countdown and increment additions.
- `test_flagging_when_time_expires`: Verified flagging detection and zero-clamping.
- `test_get_remaining_clamps_to_zero`: Verified non-negative clock snapshotting.
- `test_create_game_and_legal_move`: Verified end-to-end game creation, human move application, and bot counter-move generation.
- `test_idempotent_duplicate_moves`: Verified network duplicate retry suppression.
- `test_human_timeout_flagging`: Verified time forfeit detection and game completion.
- `test_websocket_flow`: Verified full WebSocket handshake, live move push, snapshot broadcast, and resignation.

All 9 new tests passed cleanly. Full backend test suite passing 100%.
