# Chess Bot Clone Factory ♟️🤖

An end-to-end machine learning system that clones the unique playing style, opening preferences, and tactical habits of any real human chess player from their public Lichess game history.

Unlike traditional chess engines (e.g. Stockfish) that search for objectively optimal moves, this system performs **behavioral cloning (imitation learning)**. It combines a probabilistic opening book with a deep residual policy network (ResNet + metadata conditioning) and real-time legal move masking to emulate a specific player's personal chess identity.

---

## 🚀 Key Features

* **Live Streaming Ingestion**: Streams player games directly from the Lichess ndjson API with automatic exponential backoff, rate limiting, and deduplication.
* **Stratified Recency Sampling**: Buckets games across 3 age tiers (last 6 months, 6–18 months, 18–36 months) to capture a player's latest repertoire while preserving long-term style.
* **Rich Spatial & Contextual Tensors**:
  * **18-Plane $8\times 8$ Board Representation**: Piece locations, legal attack maps, castling rights, and en-passant availability.
  * **8-Dimensional Metadata Vector**: Player rating, opponent rating, base time, increment, ply count, remaining clock, and halfmove clock.
* **Probabilistic Opening Book**: Dynamically extracts opening graphs from training games, sampling true player openings before transitioning to neural policy.
* **Deep Residual Policy Network (`ChessPolicyNet`)**:
  * 5 Residual blocks with Batch Normalization and Dropout.
  * Metadata embedding MLP with trunk feature concatenation.
  * Dual-head architecture predicting `from_square` (64) and `to_square` (64), plus pawn promotion classes (4) and auxiliary supervisory heads (piece type, capture, check, castle).
* **Guaranteed 100% Move Legality**: Dual-head conditional decoding with dynamic legal move masking prevents impossible or invalid moves at inference time.

---

## 📂 Project Structure

```
dynamic_bots/
├── backend/
│   ├── app/
│   │   ├── ml/                       # Machine Learning core
│   │   │   ├── dataset.py            # PyArrow Parquet DataLoader & Tensor conversions
│   │   │   ├── encoding.py           # 18-plane board & 8-dim metadata vector encoders
│   │   │   ├── export.py             # Model checkpointing & predict_move inference
│   │   │   ├── losses.py             # Multi-task loss (from, to, promo, auxiliary)
│   │   │   ├── model.py              # ChessPolicyNet (ResNet + Metadata MLP)
│   │   │   └── train.py              # Training loop with ReduceLROnPlateau & Early Stopping
│   │   ├── routers/
│   │   │   └── ingest.py             # FastAPI endpoints (POST /api/ingest, GET /api/status)
│   │   ├── services/                 # Core domain services
│   │   │   ├── classifier.py         # Time control classification (bullet, blitz, rapid)
│   │   │   ├── dataset_builder.py    # Chronological splitting & Parquet generation
│   │   │   ├── downloader.py         # Asynchronous Lichess streaming client
│   │   │   ├── extractor.py          # PGN parser to board state transition records
│   │   │   ├── opening_book.py       # Probabilistic opening book builder and query engine
│   │   │   ├── parser.py             # Game validation, ply filtering, and sanitization
│   │   │   ├── report.py             # Dataset distribution markdown report generator
│   │   │   ├── sampler.py            # Time-decayed game sampling & quota redistribution
│   │   │   ├── splitter.py           # Chronological 80/10/10 game split logic
│   │   │   ├── storage.py            # Local JSONL and artifact persistence
│   │   │   └── training_runner.py    # End-to-end model training orchestrator
│   │   ├── tests/                    # 76-test comprehensive unit and integration suite
│   │   ├── config.py                 # Central hyperparameter & pipeline configuration
│   │   └── main.py                   # FastAPI application entrypoint
│   ├── scripts/                      # Operational & validation scripts
│   │   ├── evaluate_test_split.py    # Quantitative evaluation on held-out test split
│   │   ├── hand_check.py             # Manual diagnostic utility for sample inspection
│   │   ├── inspect_stream.py         # Lichess stream debugging tool
│   │   ├── sanity_check_predictions.py# 30-ply simulation & first-move distribution check
│   │   └── train_real_player.py      # Standalone training launcher with live telemetry
│   └── conftest.py                   # Pytest fixtures and shared configuration
├── .gitignore                        # Git exclusion rules
└── README.md                         # Project documentation
```

---

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.10+ (tested on Python 3.12)
* Optional: NVIDIA GPU with CUDA support for accelerated training (falls back automatically to CPU)

### Setup Virtual Environment
```bash
# Clone the repository
git clone <repo-url>
cd dynamic_bots

# Create and activate virtual environment
python -m venv backend/venv

# Windows:
backend\venv\Scripts\activate
# Linux/macOS:
source backend/venv/bin/activate

# Install dependencies
pip install fastapi uvicorn httpx python-chess torch numpy pyarrow matplotlib pytest
```

---

## 🚦 How to Run

### 1. Start the Ingestion API
Run the FastAPI development server from the `backend/` directory:
```bash
cd backend
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Ingest Player Data
Trigger automated streaming download and dataset creation for any Lichess username:
```bash
# Trigger ingestion
curl -X POST http://127.0.0.1:8000/api/ingest/The_King_Crusher

# Check ingestion progress and dataset status
curl http://127.0.0.1:8000/api/status/The_King_Crusher
```

### 3. Train the Player Model
Train the policy neural network using production defaults:
```bash
python scripts/train_real_player.py
```

### 4. Evaluate & Sanity Check
Test move generation against the trained model and opening book:
```bash
# Run 30-ply simulation and 50 first-move distributions
python scripts/sanity_check_predictions.py

# Evaluate quantitative Top-1 and Top-3 accuracy on the held-out test split
python scripts/evaluate_test_split.py
```

### 5. Run Unit & Integration Tests
```bash
pytest backend/app/tests
```

---

## 📊 Real-World Validation: `The_King_Crusher`

The complete pipeline was validated end-to-end on live data from Lichess master streamer **`The_King_Crusher`**:

| Pipeline Stage | Key Metrics |
| :--- | :--- |
| **Ingestion** | **3,770 raw games downloaded** $\rightarrow$ **1,200 sampled games** across 3 time buckets (`model_tier = "full"`, 0 extraction drops). |
| **Dataset Volume** | **49,147 positions** in Parquet (Train: 39,025 / Val: 5,167 / Test: 4,955). |
| **Opening Book** | **34,915 unique position keys** indexed into probabilistic opening graph. |
| **Training (CPU)** | **22 epochs** run (71 min); train loss dropped $8.17 \rightarrow 3.03$, best val loss **$4.9756$** at Epoch 14 (Early Stopping patience 8). |
| **From-Square Acc** | **`48.21%` Top-1** and **`79.94%` Top-3** piece intention accuracy on unseen test set. |
| **Exact Move Acc** | **`23.79%` Top-1** exact move replication on 4,955 held-out positions. |
| **Opening Fidelity** | First-move sampling selected `c3` (44%) and `e3` (44%), matching his signature repertoire. |

---

## 🗺️ Project Milestones & Roadmap

- [x] **Milestone 1**: Ingestion & Sampling Engine (Lichess streaming, recency bucketing, quota redistribution).
- [x] **Milestone 2**: Feature Engineering & Dataset Builder (18-plane tensors, metadata vectors, opening book).
- [x] **Milestone 3**: Policy Architecture & Training (ResNet trunk, multi-task losses, 100% legal move decoding).
- [x] **Milestone 3.5**: Real-World Player Validation (`The_King_Crusher` full pipeline test).
- [ ] **Milestone 4 (Next)**: Serving API & Bot Engine (FastAPI bot endpoints, live game clock handling, Lichess Bot API integration).
- [ ] **Milestone 5**: Evaluation Suite & Human vs. Bot Benchmarking.

---

## 📜 License
This project is open-source and intended for educational and research purposes in behavioral imitation learning.
