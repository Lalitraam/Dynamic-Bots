# Kaggle Runbook: Base Data Streaming and Model Training (Phase C1 / C2)

## Section 1: Notebook Setup (GPU on, internet on, quota check)
```bash
# Check GPU accelerator presence
!nvidia-smi

# Check internet connectivity to Lichess archive
!curl -s -I https://database.lichess.org | head -n 5
```

## Section 2: Dataset Preparation & Path Setup

### 2.1 Local Preparation (Run before starting Kaggle session)
Run `prepare_kaggle_dataset.py` locally to build the bundle and archive:
```bash
python tools/base_data/prepare_kaggle_dataset.py --out-dir build/kaggle_dataset --tar
```
Upload the resulting `build/kaggle_dataset.tar.gz` (or the extracted `base-data-tools/` directory) to Kaggle as a private Dataset named **`base-data-tools`**. Attach the dataset to your Kaggle notebook so that it mounts at `/kaggle/input/base-data-tools`.

### 2.2 Directory layout
`prepare_kaggle_dataset.py` creates the following flattened structure in the dataset root:
```text
base-data-tools/
  app/
    __init__.py
    config.py
    ml/
      __init__.py
      dataset_v3.py
      encoding.py
      encoding_v3.py
      losses_v2.py
      model_v2.py
      tensor_cache.py
    services/
      __init__.py
      classifier.py
      storage.py
  config/
    cohorts.yaml
  scripts/
    baselines_v3.py
    gates_c_a_c_b_c_c.py
    train_v2.py
  tools/
    base_data/
      __init__.py
      build_shards.py
      filter.py
      stream_pgn.py
```

### 2.3 Install + Path Setup
In the first code cell of your Kaggle notebook:
```python
!pip install -q zstandard python-chess pyarrow pandas pyyaml
import sys
sys.path.insert(0, "/kaggle/input/base-data-tools")
```

## Section 3: C1 Stream + Shard Build
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

## Section 4: C2-0 Evidence
```bash
%cd /kaggle/working
!python /kaggle/input/base-data-tools/scripts/train_v2.py \
    --shards-dir /kaggle/working/base_shards \
    --out-dir /kaggle/working/base_v1_bench \
    --time-50-batches --device cuda
# Owner copies stdout into evidence.
```

## Section 5: C2 Full Training
```bash
!python /kaggle/input/base-data-tools/scripts/train_v2.py \
    --shards-dir /kaggle/working/base_shards \
    --out-dir /kaggle/working/base_v1 \
    --epochs 4 --batch-size 1024 --lr 2e-3 --device cuda \
    --log-steps 50 --dump-config /kaggle/working/base_v1/config.json
# If session interrupts: rerun same command with --resume
```

## Section 6: Download Artifacts; Local Verification
```powershell
Get-FileHash .\base_v1.pt -Algorithm SHA256
# record SHA-256 in docs/progress/phaseC2_gates.md
```

## Section 7: LOCAL Gate Run (owner runs after download)
```powershell
python scripts/gates_c_a_c_b_c_c.py \
    --base-checkpoint ..\data\base\base_v1.pt \
    --base-shards-dir ..\data\base\shards \
    --out ..\docs\progress\phaseC2_gates.md
```
