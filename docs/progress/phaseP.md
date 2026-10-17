## 2026-10-18 — Phase P Implementation (Analysis & Error Model)

- **Stockfish Wrapper:** Built `StockfishAnalyser` wrapping `SimpleEngine` with configurable depth (`STOCKFISH_DEPTH = 12`), cp loss calculation, mate score conversion (±15000), cp sign flipping for Black, and dead-lost threshold handling (`eval_before < -2000`).
- **Complexity Features:** Implemented `compute_features` extracting all 12 complexity features: `num_legal`, `forcing`, `policy_entropy`, `top_gap`, `n_plausible`, `material_imbalance`, `own_hanging`, `in_check`, `phase`, `clock_ratio`, `clock_abs`, and `opp_last_capture`.
- **Error Model:** Implemented `ErrorModel` with `HistGradientBoostingClassifier(max_depth=4, max_iter=200, lr=0.05)` and `IsotonicRegression` calibration, including fallback rate (`ERROR_MODEL_FALLBACK_RATE = 0.08`).
- **Pipeline Scripts:**
  - `backend/scripts/analyse_games.py`: Analyzes positions from `samples_v3.parquet` using base model policy log-probs and Stockfish.
  - `backend/scripts/train_error_model.py`: Fits error model and saves to `data/players/<username>/models/v1/error_model.joblib`.
- **Tests & Verification:**
  - 14 tests across `test_stockfish.py`, `test_features.py`, and `test_error_model.py` passing.
  - Full test suite: 246 passed.

## 2026-10-18 — Tasks P.6 through P.9 Completion

- **Task P.6 (Analysis & Error Model Training):**
  - Generated `analysis_train.parquet` (5,000 rows) and `analysis_val.parquet` (1,000 rows) with Stockfish evaluations and 12 complexity features.
  - Trained `error_model.joblib` for `the_king_crusher` achieving 86.9% validation accuracy.
  - Top feature importances identified: `top_gap`, `num_legal`, `clock_abs`, `n_plausible`, and `phase`.
- **Task P.7 (Provider V3 Pattern-Conditioned Sampling):**
  - Integrated dynamic error model loading into `V3Provider`.
  - Implemented pattern-conditioned sampling: temperature doubled to $T_{\text{base}} \times 2.0$ when $u < p_{\text{err}}$, otherwise normal $T_{\text{base}}$.
  - Implemented Stockfish blunder cap with single resample if centipawn loss exceeds $\min(\text{player\_p99}, 600)$.
  - Added unit tests in `test_provider_v3_error.py` and `test_provider_v3.py`.
- **Task P.8 (E12 Experiment - Pattern vs Temperature-Only):**
  - Evaluated on 5,004 test split positions.
  - Confirmed 0.00 pt top-1 accuracy penalty (35.7% top-1, 2.4170 NLL).
  - Adopted pattern-conditioned sampling.
- **Task P.9 (E13 Experiment - Strength Calibration):**
  - Conducted 20-game self-play match: `the_king_crusher:v1` achieved 80.0% win rate against `base`.
  - Evaluated ACPL and blunder rate: bot achieved 112.0 ACPL with 10.3% blunder rate.

## 2026-10-18 — Error Model Calibration Fixes & Re-evaluation

- **Class Imbalance Fix:** Added `class_weight='balanced'` to `HistGradientBoostingClassifier` in `app/analysis/error_model.py`.
- **Retrained Error Model:**
  - Fit on 5,000 train rows and calibrated on 1,000 val rows with isotonic regression.
  - Validation accuracy: 62.8%; Mean predicted $p_{\text{err}} = 13.00\%$ (matching true human empirical error rate of 13.0%).
  - 76.5% of validation positions evaluate with $p_{\text{err}} > 0.10$.
  - Saved to `data/players/the_king_crusher/models/v1/error_model.joblib`.
- **E12 Re-run (5,004 test positions):**
  - Average predicted $p_{\text{err}}$: 8.0% for baseline (a) fallback vs 11.2% for pattern model (b).
  - Simulated error rate: 7.9% for (a) vs 11.2% for (b), aligning closely with human rate (13.0%).
- **E13 Re-run (20 self-play games):**
  - Bot achieved 75.0% win rate (15 wins, 4 losses, 1 draw) vs base.
  - Bot ACPL evaluated at 106.7 cp (delta = 53.6 cp from human real ACPL 53.2 cp).
  - Real blunder rate confirmed from `analysis_train.parquet` at 3.8% ($\ge 300$ cp) and real error rate at 13.0% ($\ge 100$ cp).
## 2026-10-18 — Clock Units Correction & Production Fixes

- **Clock Units Fix:**
  - In `backend/scripts/analyse_games.py`, corrected `clock_sec` extraction: `row["clock_before_move"]` is already in seconds, removed the redundant `/ 1000.0`.
  - Re-analyzed 5,000 train rows (`analysis_train.parquet`) and 1,000 val rows (`analysis_val.parquet`).
  - Retrained and saved `error_model.joblib`: Validation Accuracy = 63.50%, Mean $p_{\text{err}} = 13.00\%$, Median = 13.04%, Fraction $> 0.10$ = 66.0%.
- **V3Provider Inference Features:**
  - Evaluated legal move log-probabilities from model forward pass and passed `policy_log_probs` into `compute_features()`.
  - Calibrated error branch temperature multiplier from $2.0\times$ down to $1.3\times$ to avoid catastrophic blunders while injecting human-like noise.
- **GameManager & Production Hardening:**
  - Wired `StockfishAnalyser` (at `config.STOCKFISH_PATH`) into `V3Provider` instantiation in both `create_game` and `apply_move` to enforce blunder capping.
  - Sourced true player median rating from `samples_v3.parquet` / `info.json` rather than falling back to `config.DEFAULT_RATING`.
  - Handled `ThinkTimeModel(buckets=None)` default instantiation safely without exceptions.
  - Hardened secret validation in `manager.py` and `ws.py` to compare against `hash_secret(eff_secret)`.
  - Wrapped `provider.decide()` in `asyncio.to_thread()` to prevent event loop blocking.
- **Evaluation Cohort Leakage Fix:**
  - Updated `config/cohorts.yaml` to include all three fine-tuned players: `the_king_crusher`, `chesslavaplanet`, `Fritzi_2003` frozen at `2026-10-18`.
