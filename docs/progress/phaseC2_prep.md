# Phase C2 Prep: Readiness for Kaggle Pretraining Run

## Status
UNVERIFIED until owner pastes Kaggle output.

---

## 1. Local Pipeline Readiness & Patches
All required patches from Steps 1–4 have been implemented and verified locally:
- **Patch A (Target in Legal Slice)**: `ParquetV3Dataset.__getitem__` replaces defensive write with an explicit `assert legal_mask[target_move_idx]` assertion.
- **Patch B (Band × Speed Stratification)**: `tools/base_data/build_shards.py` stratifies cell keys by `f"{band_str}|{category}"` with per-cell balancing and reporting.
- **Patch D (Val Metric Labeling)**: Early stopping comment added above validation NLL; `training_info_v2.json` explicitly logs `"val_metric": "policy_nll_eps0_no_promo"` and `"train_metric": "smoothed_policy_ce_0.05_plus_0.2_promo"`.
- **CLI Flags**: `--log-steps N`, `--dump-config PATH`, and `--time-50-batches` added to `backend/scripts/train_v2.py`.
- **Gate Evaluation**: `backend/scripts/gates_c_a_c_b_c_c.py` created for Gates C-a, C-b, and C-c.
- **Test Suite**: Pytest suite passes completely with **185 passed** (183 baseline + 2 new tests).

---

## 2. Exact Commands to Paste into Kaggle

### Step 1: Environment & Dependency Installation
```bash
!pip install -q zstandard python-chess pyarrow pandas pyyaml
import sys
sys.path.insert(0, "/kaggle/input/base-data-tools")
```

### Step 2: Stream Lichess Archives and Build v3 Shards
```python
from pathlib import Path
import yaml
from tools.base_data.build_shards import build_shards

cohort_path = Path("/kaggle/input/base-data-tools/config/cohorts.yaml")
if not cohort_path.exists():
    cohort_path = Path("/kaggle/working/cohorts.yaml")
with open(cohort_path, "r", encoding="utf-8") as f:
    cohort_cfg = yaml.safe_load(f)
cohort = cohort_cfg.get("evaluation_cohort", ["the_king_crusher"])

months = [
    "https://database.lichess.org/standard/lichess_db_standard_rated_2026-08.pgn.zst",
    "https://database.lichess.org/standard/lichess_db_standard_rated_2026-09.pgn.zst",
]

out_shards = Path("/kaggle/working/base_shards")
report = build_shards(
    months=months,
    out_dir=out_shards,
    cohort=cohort,
)

print(f"total_games_streamed: {report['total_games_streamed']}")
print(f"games_passed_filters: {report['games_passed_filters']}")
print(f"split_position_counts: {report['split_position_counts']}")
top_cells = sorted(report["cell_counts"].items(), key=lambda x: x[1], reverse=True)[:5]
print("cell_counts summary (top 5 bands):")
for cell, cnt in top_cells:
    print(f"  {cell}: {cnt}")
```

### Step 3: C2-0 Evidence Run (50 Batches Benchmark)
```bash
%cd /kaggle/working
!python /kaggle/input/base-data-tools/scripts/train_v2.py \
    --shards-dir /kaggle/working/base_shards \
    --out-dir /kaggle/working/base_v1_bench \
    --time-50-batches --device cuda
```
*(Owner copies stdout into evidence).*

### Step 4: C2 Full Training
```bash
!python /kaggle/input/base-data-tools/scripts/train_v2.py \
    --shards-dir /kaggle/working/base_shards \
    --out-dir /kaggle/working/base_v1 \
    --epochs 4 --batch-size 1024 --lr 2e-3 --device cuda \
    --log-steps 50 --dump-config /kaggle/working/base_v1/config.json
```
*(If interrupted, rerun the same command with `--resume`).*

### Step 5: Post-Download Local Verification & Gate Evaluation
```powershell
Get-FileHash .\base_v1.pt -Algorithm SHA256
```
```powershell
python scripts/gates_c_a_c_b_c_c.py \
    --base-checkpoint ..\data\base\base_v1.pt \
    --base-shards-dir ..\data\base\shards \
    --out ..\docs\progress\phaseC2_gates.md
```

---

## Fixes applied (2026-10-12)

- **Blocker 1 (Kaggle layout) -> `prepare_kaggle_dataset.py`**:
  Created `tools/base_data/prepare_kaggle_dataset.py` to assemble the flattened `base-data-tools` hierarchy and package it into `build/kaggle_dataset.tar.gz`. Verified via extraction to temporary directory and standalone import simulation:
  ```text
  <string>:1: DeprecationWarning: Python 3.14 will, by default, filter extracted tar archives and reject files or modify their metadata. Use the filter argument to control this behavior.
    import sys,os,tempfile,shutil,tarfile,pathlib; d=pathlib.Path(tempfile.mkdtemp()); tarfile.open('build/kaggle_dataset.tar.gz').extractall(d); sys.path.insert(0, str(d/'base-data-tools')); import app.ml.model_v2, app.ml.encoding_v3, tools.base_data.build_shards, scripts.train_v2; print('IMPORTS OK')
  IMPORTS OK
  ```

- **Blocker 2 (zstd untested) -> `test_5`/`test_6` in `test_base_data.py`**:
  Added test fixtures `test_5_stream_games_from_zstd_file` and `test_6_stream_games_url_zstd_magic_check`. Patched `tools/base_data/stream_pgn.py` to support seekable streams (e.g. `io.BytesIO`) that lack a native `.peek()` method, ensuring full zstd detection and decompression across file paths, raw IO buffers, and URLs.

- **Blocker 3 (UA contact) -> `_USER_AGENT` constant in `stream_pgn.py`**:
  Added compliant User-Agent identifier with contact placeholder:
  `_USER_AGENT = "EchoChess-base-data/1.0 (+https://github.com/<owner>/dynamic_bots; contact: <owner-email>)"`.

- **Minor (classifier delegation) -> `filter.classify_tc` wrapper**:
  Inspected `app/services/classifier.py` for signature compatibility. The active service defines `def classify_time_control(tc: str) -> str:` accepting a string and returning `"other"` for unincluded categories, whereas `filter.classify_tc(base_s: int, inc_s: int) -> str` operates directly on numeric base/increment integers. Per instruction, signature was reported and `filter.py` was preserved unchanged.

