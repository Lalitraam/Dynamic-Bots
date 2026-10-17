# Doppelgambit API Documentation

This document describes the interface contract between the Doppelgambit backend and frontend clients (React/Vite).

---

## 1. Authentication (REST)

All authenticated endpoints expect the HTTP header:
```http
Authorization: Bearer <access_token>
```
The frontend should store the `access_token` in `localStorage` upon successful login.

### Register User
- **Method / Path:** `POST /api/auth/register`
- **Auth Required:** No
- **Request Body:**
  ```json
  {
    "email": "player@example.com",
    "password": "SecurePassword123!",
    "invite_code": "BETA-A1B2C3D4"
  }
  ```
- **Response (`201 Created`):**
  ```json
  {
    "id": 1,
    "email": "player@example.com",
    "role": "user",
    "created_at": "2026-10-17T12:00:00Z"
  }
  ```
  *(Note: Returns 400 Bad Request if the invite code is invalid or already claimed, or if the email is registered).*

### Login
- **Method / Path:** `POST /api/auth/login`
- **Auth Required:** No
- **Request Body:**
  ```json
  {
    "email": "player@example.com",
    "password": "SecurePassword123!"
  }
  ```
- **Response (`200 OK`):**
  ```json
  {
    "access_token": "3a8f1b2c...",
    "token_type": "bearer"
  }
  ```
  *(Store `access_token` in `localStorage`).*

### Get Current User Profile
- **Method / Path:** `GET /api/auth/me`
- **Auth Required:** Yes (`Bearer <token>`)
- **Response (`200 OK`):**
  ```json
  {
    "id": 1,
    "email": "player@example.com",
    "role": "user",
    "created_at": "2026-10-17T12:00:00Z"
  }
  ```

---

## 2. Bots Management (REST)

### List User Bots
- **Method / Path:** `GET /api/bots`
- **Auth Required:** Yes (`Bearer <token>`)
- **Response (`200 OK`):**
  ```json
  [
    {
      "username": "the_king_crusher",
      "status": "ready",
      "active_version": 1,
      "added_at": "2026-10-17T12:00:00Z",
      "archived": false,
      "last_trained_at": "2026-10-17T12:30:00Z",
      "games_used": 1500,
      "tier": "free",
      "error": null
    }
  ]
  ```

### Add / Clone a New Bot
- **Method / Path:** `POST /api/bots`
- **Auth Required:** Yes (`Bearer <token>`)
- **Request Body:**
  ```json
  {
    "username": "the_king_crusher"
  }
  ```
- **Response (`200 OK`):**
  ```json
  {
    "username": "the_king_crusher",
    "status": "queued",
    "active_version": 1,
    "added_at": "2026-10-17T12:00:00Z",
    "archived": false,
    "last_trained_at": null,
    "games_used": 0,
    "tier": "free",
    "error": null
  }
  ```

### Archive a Bot
- **Method / Path:** `POST /api/bots/{username}/archive`
- **Auth Required:** Yes (`Bearer <token>`)
- **Response (`200 OK`):**
  ```json
  {
    "archived": true
  }
  ```

---

## 3. Bot Analytics Dashboard (REST)

### Get Bot Dashboard Telemetry
- **Method / Path:** `GET /api/bots/{username}/dashboard`
- **Auth Required:** Yes (`Bearer <token>`)
- **Response (`200 OK`):**
  ```json
  {
    "username": "the_king_crusher",
    "rating": 2180,
    "model_card": {
      "version": "v1",
      "top1": 0.4724,
      "nll": 1.6561,
      "personalisation_status": "Personalised",
      "personalized": true,
      "decision": "promote_ft"
    },
    "openings": {
      "white": [
        { "move": "e2e3", "count": 228, "probability": 0.4625 },
        { "move": "c2c3", "count": 210, "probability": 0.4260 },
        { "move": "g1f3", "count": 30,  "probability": 0.0609 },
        { "move": "e2e4", "count": 12,  "probability": 0.0243 },
        { "move": "d2d4", "count": 9,   "probability": 0.0183 }
      ],
      "black": [
        { "move": "c7c6", "count": 307, "probability": 0.6574 },
        { "move": "d7d5", "count": 99,  "probability": 0.2120 },
        { "move": "g8f6", "count": 25,  "probability": 0.0535 },
        { "move": "c7c5", "count": 24,  "probability": 0.0514 },
        { "move": "e7e6", "count": 7,   "probability": 0.0150 }
      ]
    },
    "time_management": {
      "opening": 0.9135,
      "middlegame": 2.9019,
      "endgame": 2.0013
    },
    "errors": [
      { "bucket": "<0.1", "clock_ratio": "<0.1", "error_rate": 0.1556, "count": 302 },
      { "bucket": "0.1-0.3", "clock_ratio": "0.1-0.3", "error_rate": 0.2121, "count": 646 },
      { "bucket": ">0.3", "clock_ratio": ">0.3", "error_rate": 0.1150, "count": 4052 }
    ]
  }
  ```

---

## 4. Games & Real-Time Gameplay (REST & WebSocket)

### Create a Game
- **Method / Path:** `POST /api/games`
- **Auth Required:** Yes (`Bearer <token>`)
- **Request Body:**
  ```json
  {
    "username": "the_king_crusher",
    "human_color": "white",
    "base_s": 180,
    "inc_s": 2
  }
  ```
- **Response (`201 Created`):**
  ```json
  {
    "game_id": "8f3b20e1a49c",
    "secret": "9a12c84e...",
    "human_color": "white",
    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "status": "active"
  }
  ```
  *(Important: The frontend client must store `secret` in memory or session storage to authenticate the WebSocket session).*

### Real-Time Game WebSocket
- **Connection URL:** `ws://<host>/ws/{game_id}` (or `ws://<host>/api/ws/{game_id}`)
- **Protocol:** JSON messages with a type tag `"t"`.

#### Handshake (Client -> Server)
Upon establishing the WebSocket connection, the client MUST send the handshake first:
```json
{
  "t": "hello",
  "secret": "<game_secret>"
}
```

#### Server Snapshot (Server -> Client)
Sent immediately after successful handshake and broadcast whenever a move is made:
```json
{
  "t": "snapshot",
  "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
  "moves": [
    {
      "ply": 1,
      "uci": "e2e4",
      "san": "e4",
      "clock_ms": 179500,
      "time_spent_s": 0.5
    }
  ],
  "clocks": {
    "white": 179500,
    "black": 180000
  },
  "status": "active",
  "result": null,
  "termination": null
}
```

#### Make Move (Client -> Server)
When the human player makes a move:
```json
{
  "t": "move",
  "uci": "e2e4",
  "n": 1
}
```
*(Where `uci` is UCI move format e.g. `e2e4` or `e7e8q`, and `n` is the client move/ply index).*

#### Resign Game (Client -> Server)
```json
{
  "t": "resign"
}
```

#### Game Over Broadcast (Server -> Client)
When checkmate, resignation, timeout, or draw occurs, the snapshot reflects the terminal status:
```json
{
  "t": "snapshot",
  "fen": "...",
  "moves": [...],
  "clocks": {
    "white": 120000,
    "black": 0
  },
  "status": "ended",
  "result": "1-0",
  "termination": "out_of_time"
}
```

#### Heartbeat (Client -> Server)
Optional ping:
```json
{
  "t": "ping"
}
```
Server responds:
```json
{
  "t": "pong"
}
```
