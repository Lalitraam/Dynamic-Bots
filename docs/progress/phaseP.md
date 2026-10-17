# Phase P Progress: Analysis, Error Modeling & Calibration

## 1. Analysis Pipeline & Complexity Features
- **Stockfish Wrapper:** Built `StockfishAnalyser` wrapping `SimpleEngine` with depth `STOCKFISH_DEPTH = 12` (`sf_16.1`), centipawn loss calculation, mate score conversion (±15000), sign adjustment for Black, and dead-lost threshold handling (`eval_before < -2000`).
- **Complexity Features:** Implemented `compute_features` extracting all 12 complexity features: `num_legal`, `forcing`, `policy_entropy`, `top_gap`, `n_plausible`, `material_imbalance`, `own_hanging`, `in_check`, `phase`, `clock_ratio`, `clock_abs`, and `opp_last_capture`.
- **Pipeline Scripts:**
  - `backend/scripts/analyse_games.py`: Extracts features and computes Stockfish centipawn loss from `samples_v3.parquet`.
  - `backend/scripts/train_error_model.py`: Fits `HistGradientBoostingClassifier(max_depth=4, max_iter=200, lr=0.05, class_weight='balanced')` and isotonic calibration layer.
  - Sourced true player median rating from `app.db.players.get_player_median_rating(username)`.

---

## 2. Pinned Ground-Truth Human Baseline
All human metrics are pinned to a single ground-truth evaluation on the 5,004 positions of `data/players/the_king_crusher/analysis_test.parquet` (Depth 12, version `sf_16.1`, recorded in `backend/data_baselines/human_baseline_the_king_crusher.json` on 2026-10-17):

- **Human ACPL:** **55.62 cp** (95% CI: `[50.19, 61.24]`)
- **Human $\ge 100$ cp Rate (Errors):** **13.49%** (95% CI: `[12.13%, 14.94%]`)
- **Human $\ge 300$ cp Rate (Blunders):** **3.86%** (95% CI: `[3.24%, 4.54%]`)
- **Test Sample Size:** 5,004 positions across 120 games.

---

## 3. Error Model Classification Performance & Gameplay Rejection
- **Test Set Metrics (`analysis_test.parquet`, 5,004 rows):**
  - **AUC-ROC:** 0.6586
  - **Brier Score:** 0.1135
- **Trigger Rate Clarification:** Offline tests showed an 11.2% activation rate ($u < p_{\text{err}}$). This 11.2% was strictly an error-model activation trigger rate and must not be conflated with move error or blunder rates.
- **Gameplay Decision:** While the classifier demonstrates raw predictive value for analytics dashboards, dynamic error-branch temperature flattening worsened bot ACPL (increasing average centipawn loss from ~89 cp to 130+ cp). Therefore, `ERROR_MODEL_ENABLED = False` is enforced for live gameplay.

---

## 4. Paired Replay Calibration & Diversity Trade-Off
Using cached Stockfish evaluations (depth 12, `sf_16.1`, 21,828 cache entries verified with 4-tuple keys `(fen, move_uci, depth, version)`), we performed paired game-level bootstrap analysis (2,000 resamples) across 5 seeds on the 5,004 test positions (Command: `python scripts/run_paired_significance.py`, Date: 2026-10-17):

| Config | ACPL (95% CI) | $\ge 300$ cp (%) (95% CI) | % Positions Stochastic | Uniq/Pos | $\Delta$ ACPL vs Greedy (95% CI) | $\Delta \ge 300$ cp vs Greedy (95% CI) | Ratio vs Human ACPL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **greedy** | 89.07 [81.85, 96.39] | 8.87% [7.84%, 9.96%] | 0.00% | 1.00 | 0.00 (ref) | 0.00% (ref) | 1.60x |
| **t=0.30_p=0.70** | 90.57 [83.52, 97.66] | 9.07% [8.05%, 10.08%] | 4.84% | 1.05 | +1.50 [+0.48, +2.65] | +0.20% [+0.05%, +0.35%] | 1.63x |
| **t=0.50_p=0.70** | 91.97 [84.88, 99.10] | 9.19% [8.20%, 10.24%] | 15.87% | 1.20 | +2.89 [+1.37, +4.39] | +0.32% [+0.10%, +0.54%] | 1.65x |
| **t=0.70_p=0.70** | **93.38 [86.48, 100.17]** | **9.34% [8.34%, 10.37%]** | **32.17%** | **1.48** | **+4.30 [+2.11, +6.41]** | **+0.47% [+0.15%, +0.77%]** | **1.68x** |

### Selected Configuration (Option B)
- **`OPTIMAL_T_BASE = 0.70`**
- **`OPTIMAL_TOP_P = 0.70`**
- **`OPTIMAL_ERROR_MULT = 1.0`**
- **`ERROR_MODEL_ENABLED = False`**

**Reason:** `t=0.30/p=0.70` behaves almost like greedy with only 4.84% stochastic positions, whereas `t=0.70/p=0.70` delivers meaningful variety (32.17% stochastic positions, 1.48 unique moves per position across 5 seeds) for an acceptable ACPL cost of +4.30 cp vs greedy.

---

## 5. Honest Status & Scientific Limitations
- **Best Available Sampling Setting:** This represents the best available sampling setting; bot ACPL is ~1.68x the human baseline (measured: 93.38 cp vs 55.62 cp human baseline from `backend/data_baselines/human_baseline_the_king_crusher.json`).
- **Sampling Tuning Is Exhausted:** Sampling temperature and top-p filtering cannot bridge the remaining 37.7 cp gap. The gap reflects base policy precision limitations, not inference parameters.
- **Open Follow-Up (Status: NOT STARTED):**
  - Training-side improvements: expanding fine-tuning training volume (Plan E11).
  - Objective function enhancement: centipawn-loss-weighted policy loss during training.
