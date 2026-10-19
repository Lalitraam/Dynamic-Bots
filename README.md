# Doppelgambit: Behavioral Chess Cloning Engine ♟️🤖

An end-to-end imitation learning platform that clones the authentic playing style, opening repertoires, tactical quirks, and time management of any real human chess player from their public Lichess history.

Unlike traditional chess engines that search for objectively optimal moves, **Doppelgambit** performs behavioral cloning. It integrates a probabilistic opening book, a deep residual policy network conditioned on match metadata, Stockfish-powered blunder guardrails, and dynamic think-time models behind a modern React web interface.

---

## 🌟 Key Capabilities

### 1. Behavioral Cloning Pipeline
* **Streaming Ingestion**: Live streaming of games via the Lichess ndjson API with rate limiting, deduplication, and exponential backoff.
* **Stratified Recency Sampling**: Buckets games across 3 recency tiers (0–6 mo, 6–18 mo, 18–36 mo) to reflect contemporary style while maintaining baseline depth.
* **Rich Spatial & Contextual Tensors**:
  * 18-plane $8\times 8$ board representation (pieces, attacks, castling rights, en-passant).
  * 8-dimensional metadata conditioning vector (ratings, clock times, increment, ply count).
* **Probabilistic Opening Book**: Extracts signature player lines from training games, sampling natural openings before handing off to neural policy.
* **Residual Policy Network (`ChessPolicyNet`)**:
  * 5 residual blocks with Batch Normalization, Dropout, and metadata MLP feature fusion.
  * Dual-head action decoding (`from_square`, `to_square`, promotion) with 100% legal move masking.

### 2. Tactical Guardrails & Blunder Filtering
* **Stockfish-Powered Safety Layer**: Evaluates candidate policy moves and filters catastrophic tactical blunders without destroying the unique human style.
* **Engine Process Management**: Robust UCI pipe with thread safety locks and auto-recovery.

### 3. Humanized Dynamic Think-Time Modeling
* **Clock-Adaptive Think Model**:
  * **Time Trouble (<5s)**: Rapid moves (50–350ms) to prevent flagging.
  * **Scramble (5–25s)**: Quick tactical responses (300–1200ms).
  * **Comfortable (≥25s)**: Thoughtful calculation baseline (1.5–4.5s) with log-normal variance and blunder avoidance.

### 4. Interactive Web Platform
* **Play Page**: Real-time WebSocket game engine, interactive clocks, move reviews, piece themes, and Lichess-style premove queuing.
* **Curved Light Fall Background**: High-performance animated light curtain with dynamic tunnel curvature and glowing nodes across Play and Home pages.
* **Ghost Fibers Dashboard**: Ethereal purple particle ribbon waves behind the telemetry and analytics suite.
* **Deep Behavioral Telemetry**: Blunder/mistake error rates by time control and phase, opening repertoire frequencies, and playstyle archetype classification.
* **Email & OTP Authentication**: Password sign-in, direct 6-digit OTP code login, SMTP integration with dev console fallback, and multi-bot roster support.

---

## 📂 Repository Structure

```
dynamic_bots/
├── backend/
│   ├── app/
│   │   ├── analysis/             # Stockfish evaluation & blunder guardrails
│   │   ├── db/                   # SQLite schema, migrations, user/session storage
│   │   ├── engine/               # Policy inference providers (V1, V2, V3) & think-time
│   │   ├── game/                 # Live WebSocket match management & clock controls
│   │   ├── ml/                   # Neural architecture, dataset encoders, loss functions
│   │   ├── routers/              # FastAPI routers (auth, bots, play, dashboard, ingest)
│   │   ├── services/             # Lichess downloaders, PGN extractors, SMTP email service
│   │   ├── tests/                # Comprehensive pytest suite (280+ tests)
│   │   ├── config.py             # Hyperparameters & environment configuration
│   │   └── main.py               # FastAPI entrypoint
│   ├── requirements.txt          # Python dependencies
│   └── worker.py                 # Async background training & extraction worker
├── frontend/
│   ├── src/
│   │   ├── components/           # UI components (LightFallBackground, GhostFibers, etc.)
│   │   ├── context/              # AuthContext & session state
│   │   ├── pages/                # Home, Play, Dashboard, Auth
│   │   └── api/                  # Axios HTTP & WebSocket clients
│   ├── package.json              # Vite & React dependencies
│   └── components.json           # Shadcn / React Bits registry configuration
├── .gitignore                    # Git ignore specifications
└── README.md                     # Platform documentation
```

---

## 🛠️ Getting Started

### Prerequisites
* **Python 3.10+** (tested on Python 3.12 and 3.13)
* **Node.js 18+** & npm
* Optional: NVIDIA GPU with CUDA for training acceleration

---

### Backend Setup

1. **Activate virtual environment**:
   ```bash
   # Windows PowerShell
   .\venv\Scripts\Activate.ps1
   # Linux/macOS
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Configure environment (Optional)**:
   Copy `.env.example` to `.env` and set optional SMTP credentials for real email delivery:
   ```ini
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USER=your-email@gmail.com
   SMTP_PASSWORD=your-app-password
   ```
   *(If SMTP is not configured, OTP codes are logged directly to the backend terminal and surfaced in local dev).*

4. **Run the Backend Server**:
   ```bash
   python -m uvicorn app.main:app --port 8000 --reload
   ```

5. **Run the Background Job Worker**:
   ```bash
   python worker.py
   ```

---

### Frontend Setup

1. **Install packages**:
   ```bash
   cd frontend
   npm install
   ```

2. **Start the development server**:
   ```bash
   npm run dev
   ```
   Access the web app at `http://localhost:5173`.

---

## 🧪 Testing & Verification

Run the comprehensive test suite across the backend:
```bash
cd backend
pytest
```

Build the production frontend bundle:
```bash
cd frontend
npm run build
```

---

## 📜 License
Licensed under **GPL-3.0-or-later** (consistent with dependencies including `python-chess` and `Stockfish`).
