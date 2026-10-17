# Kaggle & Google Colab Base Data & Model Training Runbook (Phase C.1 / C.2)

This runbook guides running the base data processing and pretraining pipeline on external GPU cloud environments (Kaggle or Google Colab) to avoid local compute bottleneck and rate limits.

---

## 1. Prerequisites

1. **Account & GPU Configuration**:
   - Create or log into your [Kaggle](https://www.kaggle.com/) or [Google Colab](https://colab.research.google.com/) account.
   - In Kaggle Notebook Settings:
     - **Accelerator**: GPU (select NVIDIA P100 or T4 x2).
     - **Internet**: Enabled (mandatory for streaming Lichess archives).
     - **Persistence**: Files only (or Variables and Files).
   - Check weekly GPU quota: [https://www.kaggle.com/settings](https://www.kaggle.com/settings) (typically 30 hours/week).
   - *Note on GPU Precision*: NVIDIA T4 and P100 GPUs do not support native bfloat16. `train_v2.py` automatically detects this and falls back to `fp16` mixed precision using `torch.amp.GradScaler("cuda")`.

---

## 2. Execution Steps

### Step A: Upload Pipeline Code & Configuration
Upload the following repository files as a private Kaggle Dataset (e.g., named `base-data-tools`):
- `tools/base_data/__init__.py`
- `tools/base_data/stream_pgn.py`
- `tools/base_data/filter.py`
- `tools/base_data/build_shards.py`
- `config/cohorts.yaml`
- `backend/app/ml/model_v2.py`
- `backend/app/ml/losses_v2.py`
- `backend/app/ml/encoding_v3.py`
- `backend/scripts/train_v2.py`

In your Kaggle notebook, set up the environment:
```bash
!pip install -q zstandard python-chess pyarrow
import sys
sys.path.append("/kaggle/working")
sys.path.append("/kaggle/input/base-data-tools")
```

---

### Step B: Stream and Build Shards
Stream 1–2 recent monthly Lichess archives directly without decompressing to disk and produce v3 parquet shards (250k positions each):
```python
from pathlib import Path
from tools.base_data.build_shards import build_shards

# Recent Lichess monthly dumps (e.g. September 2026 and August 2026)
months = [
    "https://database.lichess.org/standard/lichess_db_standard_rated_2026-08.pgn.zst",
    "https://database.lichess.org/standard/lichess_db_standard_rated_2026-09.pgn.zst",
]

out_shards = Path("/kaggle/working/base_shards")
report = build_shards(
    months=months,
    out_dir=out_shards,
    cohort=["the_king_crusher"],  # loaded from config/cohorts.yaml
)
print("Shard construction report:", report)
```

---

### Step C: Snapshot Shards to Persistent Storage
Because Kaggle sessions can be preempted or time out, save your generated shards to a persistent Kaggle Dataset or cloud bucket:
```bash
!kaggle datasets init -p /kaggle/working/base_shards/
# or export tarball
!tar -cvf /kaggle/working/base_shards.tar /kaggle/working/base_shards/
```

---

### Step D: Train Base Policy Model (`train_v2.py`)
Run the training script with 0.5-epoch checkpoints:
```bash
python scripts/train_v2.py \
  --shards-dir /kaggle/working/base_shards/ \
  --out-dir /kaggle/working/base_v1/ \
  --epochs 3 \
  --batch-size 1024 \
  --lr 2e-3 \
  --wd 1e-4 \
  --device cuda
```

---

### Step E: C2-0 Evidence Collection (Benchmark 50 Batches)
For performance verification, execute a 50-batch smoke benchmark and record the output:
```bash
python scripts/train_v2.py \
  --shards-dir /kaggle/working/base_shards/ \
  --out-dir /kaggle/working/base_v1_bench/ \
  --epochs 1 \
  --batch-size 1024 \
  --max-batches 50 \
  --device cuda
```
Paste the stdout log with step times, positions/sec, and GPU memory utilization.

---

### Step F: Download Model Artifacts & Verify Integrity
1. Download `base_v1.pt`, `checkpoint_latest.pt`, `training_info_v2.json`, and `model_card_v2.md` from `/kaggle/working/base_v1/` to `data/base/`.
2. Compute and record the SHA-256 checksum locally:
```powershell
Get-FileHash data/base/base_v1.pt -Algorithm SHA256
```

---

## 3. Quota & Fault-Tolerance Notes
- **Preemption & Timeout**: Kaggle execution limit is 12 hours per session.
- **Mid-Epoch Checkpoints**: `train_v2.py` writes a full state checkpoint every 0.5 epochs to `checkpoint_latest.pt` containing model parameters, optimizer momentum, OneCycleLR step, AMP GradScaler state, and RNG state.
- **Resume Command**: If interrupted, rerun with `--resume` to seamlessly continue without loss of progress:
  ```bash
  python scripts/train_v2.py --shards-dir ... --out-dir ... --resume --device cuda
  ```
