# Phase J & Phase I Progress

## Phase J: Bot Dashboard API

### Endpoint
`GET /api/bots/{username}/dashboard` (Protected via `get_current_user`)

### Response Fields
1. **Model Card (`model_card`)**:
   - Reads `eval.json` from `data/players/{username}/models/v1/`.
   - Returns model version, top1 accuracy, NLL loss, personalization status (`Personalised` if promoted via `promote_ft`, otherwise `Base Model`), and decision.
2. **Opening Repertoire (`openings`)**:
   - Reads `opening_book.json` from `data/players/{username}/`.
   - White first moves: Top 5 moves from starting position weighted by visit frequency.
   - Black replies: Top 5 replies across 1-ply responses from starting position weighted by visit frequency.
3. **Time Management (`time_management`)**:
   - Reads `samples_v3.parquet` from `data/players/{username}/`.
   - Computes mean think time `time_spent_s` grouped by phase (`opening`, `middlegame`, `endgame`).
4. **Error Rates (`errors`)**:
   - Reads `analysis_train.parquet` from `data/players/{username}/`.
   - Computes error rate (`cp_loss >= 100`) grouped by clock ratio buckets (`<0.1`, `0.1-0.3`, `>0.3`).
   - Returns empty list `[]` if analysis data is unavailable.
5. **Rating (`rating`)**:
   - Fetches cached median rating via `app.db.players.get_player_median_rating`.

---

## Phase I: Docker & Production Deployment Scaffold

### Scaffolded Files
- `backend/Dockerfile`: Python 3.13-slim image installing `sqlite3`, `zstd`, `curl`, python dependencies, running uvicorn.
- `backend/worker.py`: Background worker stub with heartbeat loop.
- `frontend/Dockerfile`: Nginx Alpine container serving static bundle.
- `frontend/dist/index.html`: Static placeholder landing page.
- `Caddyfile`: Reverse proxy directing `/api/*` and `/ws/*` to FastAPI backend (`api:8000`) and root `/*` to Nginx frontend (`frontend:80`).
- `docker-compose.yml`: Multi-service compose specification orchestrating `api`, `worker`, `frontend`, and `caddy`.
- `.env.example`: Environment template defining `SECRET_KEY` and `DATA_ROOT`.
