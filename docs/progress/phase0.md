# Phase 0, Task 0.3 (Evidence Collection) - the_king_crusher

> **STATUS: UNVERIFIED until owner pastes and CI is green**

## STEP 1 - Current Model Status

### Command
```powershell
python -c "from app.ml import export; print(export.is_model_current('the_king_crusher'))"
```

### Output
```
True
```

*(Model is current; retraining skipped as per rules.)*

### Training Info Metadata & History Table
#### Command
```powershell
python -c "import json; d=json.load(open('../data/players/the_king_crusher/training_info.json', encoding='utf-8')); print('epochs_run:', d['epochs_run']); print('best_val_loss:', d['best_val_loss']); print('device:', d['device']); print('model_config:', d['model_config']); print('dataset_sha256:', d['dataset_sha256']); print('created_at:', d['created_at']); print(); print('| epoch | train_loss   | val_loss     | lr         |'); print('|-------|--------------|--------------|------------|'); [print('| %5d | %12.6f | %12.6f | %10.6f |' % (h['epoch'], h['train_loss'], h['val_loss'], h['lr'])) for h in d['history']]"
```

#### Output
```
epochs_run: 32
best_val_loss: 5.6509273258256325
device: cpu
model_config: {'channels': 64, 'num_res_blocks': 5, 'hidden_dim': 256, 'dropout': 0.3}
dataset_sha256: b976ac82a611024d10b4c2dc92a367ea82cc31ad762a3e3a71162517b5524f3c
created_at: 2026-10-10T19:42:28.111319+00:00

| epoch | train_loss   | val_loss     | lr         |
|-------|--------------|--------------|------------|
|     1 |     8.349857 |     7.816299 |   0.001000 |
|     2 |     7.959227 |     7.766684 |   0.001000 |
|     3 |     7.887342 |     7.771608 |   0.001000 |
|     4 |     7.835815 |     7.707266 |   0.001000 |
|     5 |     7.814080 |     7.788013 |   0.001000 |
|     6 |     7.793110 |     7.707825 |   0.001000 |
|     7 |     7.782624 |     7.713332 |   0.001000 |
|     8 |     7.767093 |     7.709496 |   0.001000 |
|     9 |     7.744661 |     7.676313 |   0.000500 |
|    10 |     7.720631 |     7.626051 |   0.000500 |
|    11 |     7.631117 |     7.553111 |   0.000500 |
|    12 |     7.579742 |     7.540419 |   0.000500 |
|    13 |     7.553199 |     7.470570 |   0.000500 |
|    14 |     7.533034 |     7.455213 |   0.000500 |
|    15 |     7.503459 |     7.477973 |   0.000500 |
|    16 |     7.470718 |     7.363785 |   0.000500 |
|    17 |     7.273025 |     7.085554 |   0.000500 |
|    18 |     6.940644 |     6.764424 |   0.000500 |
|    19 |     6.613735 |     6.442850 |   0.000500 |
|    20 |     6.282145 |     6.209118 |   0.000500 |
|    21 |     5.878661 |     5.881042 |   0.000500 |
|    22 |     5.533643 |     5.769650 |   0.000500 |
|    23 |     5.257944 |     5.712236 |   0.000500 |
|    24 |     5.040944 |     5.650927 |   0.000500 |
|    25 |     4.807796 |     5.753561 |   0.000500 |
|    26 |     4.605276 |     5.653233 |   0.000500 |
|    27 |     4.403513 |     5.732353 |   0.000500 |
|    28 |     4.233162 |     5.762979 |   0.000500 |
|    29 |     3.873864 |     5.927802 |   0.000250 |
|    30 |     3.704067 |     5.970940 |   0.000250 |
|    31 |     3.582316 |     6.120670 |   0.000250 |
|    32 |     3.476813 |     6.177011 |   0.000250 |
```

---

## STEP 2 - Dataset Report

### File: `data/players/the_king_crusher/dataset_report.md`
#### Command
```powershell
Get-Content data/players/the_king_crusher/dataset_report.md -Raw
```

#### Output
```markdown
# Dataset Report for the_king_crusher

**Model Tier**: full

## Overall Stats
- Total Games: 1200
- Total Positions: 50162
- Average Target Moves per Game: 41.8
- Positions with Clock Data: 100.0%

## Splits
- **train**: 40009 positions
- **val**: 5149 positions
- **test**: 5004 positions

## Distributions
### Color
- white: 26306
- black: 23856
### Speed Category
- blitz: 33018
- bullet: 15145
- rapid: 1999
### Moved Piece Type
- P: 13257
- R: 9015
- N: 7840
- K: 6718
- B: 6714
- Q: 6618

## Top Openings
### Top 10 First Moves (as White)
- e2e3: 315
- c2c3: 210
- g1f3: 41
- e2e4: 29
- d2d4: 9
- c2c4: 4
- b2b3: 2
- g2g3: 1
- b1c3: 1
### Top 10 Replies (as Black)
- c7c6: 323
- d7d5: 175
- g8f6: 32
- c7c5: 31
- e7e6: 21
- g7g6: 2
- e7e5: 2
- f7f5: 1
- b8c6: 1

## Ingestion
- Retries Used: 0
- Requests: 3
- Skipped Duplicates: 0

### Per-Bucket Requests
- last_6m: 1 requests
- 6_18m: 1 requests
- 18_36m: 1 requests

### Request Log Summary
- Total Requests in Log: 3
- [last_6m] status 200, 838 lines, max=1500
- [6_18m] status 200, 1206 lines, max=1500
- [18_36m] status 200, 1500 lines, max=1500

### Ingestion Drops
- no_pgn: 0
- unparseable_pgn: 0
- duplicate: 0
- variant: 0
- custom_start: 0
- bad_termination: 22
- unfinished: 0
- too_short: 66
- bad_date: 0
- player_not_found: 0
- bad_time_control: 0
- below_min_time_control: 143
- excluded_category: 0
- too_old: 0

## Extraction Drops
- illegal_move: 0
- no_target_moves: 0
- unparseable_pgn: 0
```

### File: `data/players/the_king_crusher/info.json`
#### Command
```powershell
Get-Content data/players/the_king_crusher/info.json -Raw
```

#### Output
```json
{
  "total_samples": 50162,
  "split_counts": {
    "train": 40009,
    "val": 5149,
    "test": 5004
  },
  "model_tier": "full",
  "schema_version": "v1",
  "build_date": "2026-10-08T15:57:56.680320+00:00",
  "drop_stats": {
    "unparseable_pgn": 0,
    "illegal_move": 0,
    "no_target_moves": 0
  },
  "retries_used": 0,
  "berserk_positions_fraction": 0.4012599178661138
}
```

---

## STEP 3 - Training Epoch Profiling (`backend/scripts/profile_epoch.py`)

### Command
```powershell
python scripts/profile_epoch.py the_king_crusher --profile-only
```

### Output
```
======================================================================
STEP 3: PROFILING TRAINING EPOCH FOR 'the_king_crusher'
Device: cpu
======================================================================
Total training positions: 40009
Total training batches in loader: 626 (batch_size=64)
Batches to profile: 200 of 626
----------------------------------------------------------------------
PROFILING RESULTS (200 batches):
Component                      | Mean (s/batch)   | Fraction of Total  | Total Time (s)
--------------------------------------------------------------------------------------
(a) data fetch                 | 0.060377         |           21.81% | 12.0753       
(b) forward + compute_loss     | 0.081545         |           29.46% | 16.3089       
(c) backward + optimizer.step  | 0.134909         |           48.73% | 26.9819       
--------------------------------------------------------------------------------------
Total (all components)         | 0.276831         |          100.00% | 55.3661       
======================================================================
```

---

## STEP 4 - BatchNorm Staleness Check (`backend/scripts/profile_epoch.py`)

### Command
```powershell
python scripts/profile_epoch.py the_king_crusher --bn-check-only
```

### Output
```
======================================================================
STEP 4: BATCHNORM STALENESS CHECK FOR 'the_king_crusher'
Device: cpu
======================================================================
Val positions: 5149 (81 batches)
Train positions: 40009 (626 batches)
Initial val loss (checkpoint BN stats): 5.650927
Put 11 BatchNorm modules in train() and reset running statistics.
Running one pass over the TRAIN loader under torch.no_grad() to recompute BN stats...
Pass over train loader finished in 80.09s.
----------------------------------------------------------------------
Initial val loss (checkpoint BN stats):          5.650927
Recomputed val loss (re-accumulated BN stats):   5.645783
Difference (recomputed - initial):               -0.005144 nats
Absolute difference:                             0.005144 nats
STATUS: Difference <= 0.5 nats indicates FRESH / STABLE BatchNorm statistics.
======================================================================
```

---

## STEP 5 - Test Evaluation

### Command
```powershell
python scripts/evaluate_test_split.py the_king_crusher
```

### Output
```
============================================================
EVALUATING MODEL ON HELD-OUT TEST SPLIT FOR 'the_king_crusher'
============================================================
Total test positions: 5004

--- Test Set Evaluation Results ---
Total Test Positions: 5004
Test Total Loss:      5.7920
Test From-Square Loss:2.3816
Test To-Square Loss:  3.1774
Test Promo Loss:      0.000000
Test Aux Loss:        0.2329
From-Square Top-1 Acc:44.66%
From-Square Top-3 Acc:68.39%
To-Square Top-1 Acc:  25.12%
To-Square Top-3 Acc:  40.63%
Exact Move Top-1 Acc: 19.00%
Masked From Top-1:    46.64%
Masked Exact Top-1:   25.02%

Saved test metrics to E:\dynamic_bots\data\players\the_king_crusher\test_evaluation_metrics.json
Saved training curves chart to E:\dynamic_bots\data\players\the_king_crusher\training_curves.png
Copied chart to backup: E:\dynamic_bots\data\players_backup\the_king_crusher\training_curves.png
```

---

## STEP 6 - Hardware Environment

### 1. Processor
#### Command
```powershell
Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors
```

#### Output
```
Name                                            NumberOfCores NumberOfLogicalProcessors
----                                            ------------- -------------------------
AMD Ryzen 5 7520U with Radeon Graphics                      4                         8
```

### 2. Memory
#### Command
```powershell
Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
```

#### Output
```
TotalPhysicalMemory
-------------------
        16407719936
```

### 3. Torch, Platform, CPU Count
#### Command
```powershell
python -c "import torch,platform,os; print(torch.__version__, torch.cuda.is_available(), platform.platform(), os.cpu_count())"
```

#### Output
```
2.11.0+cpu False Windows-11-10.0.26200-SP0 8
```

### 4. Python Version
#### Command
```powershell
python --version
```

#### Output
```
Python 3.13.1
```

### 5. Filtered Packages
#### Command
```powershell
pip list | Select-String -Pattern "^(Package|---|torch|numpy|pandas|pyarrow|python-chess|fastapi|scikit-learn)\b"
```

#### Output
```
Package                 Version
fastapi                 0.141.1
numpy                   2.2.4
pandas                  2.2.3
pyarrow                 25.0.1
python-chess            1.999
scikit-learn            1.9.0
torch                   2.11.0
```

---

## STEP 7 - Test Suite Count

### Command
```powershell
cd backend && python -m pytest -q
```

### Output
```
........................................................................ [ 61%]
..............................................                           [100%]
============================== warnings summary ===============================
C:\Users\jrabh\AppData\Local\Programs\Python\Python313\Lib\site-packages\fastapi\testclient.py:1
  C:\Users\jrabh\AppData\Local\Programs\Python\Python313\Lib\site-packages\fastapi\testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

C:\Users\jrabh\AppData\Local\Programs\Python\Python313\Lib\site-packages\starlette\testclient.py:53
  C:\Users\jrabh\AppData\Local\Programs\Python\Python313\Lib\site-packages\starlette\testclient.py:53: DeprecationWarning: The anyio.abc.BlockingPortal alias is deprecated, use anyio.from_thread.BlockingPortal instead.
    _PortalFactoryType = Callable[[], AbstractContextManager[anyio.abc.BlockingPortal]]

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
118 passed, 2 warnings in 13.72s
```

### Final Summary Line
```
118 passed, 2 warnings in 13.72s
```
