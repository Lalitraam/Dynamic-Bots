# Decision D-06: Error Probability Model and Sampling Configuration

**Date:** 2026-10-18 (Updated 2026-10-17 with Final Significance & Diversity Results)  
**Status:** Approved  
**Approved by:** Sujit  

## Context

Phase P implements position-level blunder and error modeling to evaluate and calibrate human fidelity.
An accurate human-like chess bot must not blunder uniformly at random; rather, human error rates correlate with time pressure, position complexity, tactical sharpness, and material imbalances.

---

## Decisions

### 1. Centipawn Loss and Error Definition
- Ground truth evaluation is obtained using Stockfish 16+ via python-chess (`SimpleEngine`).
- Evaluation depth is standardized at `STOCKFISH_DEPTH = 12`.
- `cp_loss = max(0, eval_before_mover - eval_after_mover)`, clipped to `ERROR_MODEL_CP_CLIP = 1000`.
- An error is defined as `cp_loss >= ERROR_MODEL_MIN_CP_LOSS` (100 centipawns).
- Blunder threshold is defined as `ERROR_MODEL_BLUNDER_CP = 300` centipawns.
- Positions where the mover is already decisively lost (`eval_before_mover < -2000`) are excluded from error labeling (`cp_loss = 0.0, is_error = False`) to prevent meaningless blunders in lost positions from corrupting the model.

### 2. 12 Complexity Features
The model extracts 12 position- and situation-dependent features:
1. `num_legal`: Number of legal moves in the position.
2. `forcing`: Number of legal checks and captures.
3. `policy_entropy`: Shannon entropy computed from base model policy probabilities.
4. `top_gap`: Difference in log-probabilities between the top-1 and top-2 candidate moves.
5. `n_plausible`: Count of candidate moves with predicted probability $\ge 0.05$.
6. `material_imbalance`: Mover material minus opponent material ($P=1, N=3, B=3, R=5, Q=9$).
7. `own_hanging`: Number of mover pieces attacked by opponent and not defended.
8. `in_check`: Binary flag ($1.0$ if mover is currently in check, else $0.0$).
9. `phase`: Game phase ($0.0$ for opening, $1.0$ for middlegame, $2.0$ for endgame based on non-pawn material).
10. `clock_ratio`: Remaining clock divided by base time ($clock / base\_s$, or $-1.0$ if unknown).
11. `clock_abs`: Log-scaled absolute clock ($\ln(clock + 1)$, or $0.0$ if unknown).
12. `opp_last_capture`: Binary flag ($1.0$ if the opponent's preceding move was a capture, else $0.0$).

### 3. Model Architecture & Raw Classifier Quality
- **Classifier:** `HistGradientBoostingClassifier(max_depth=4, max_iter=200, learning_rate=0.05, class_weight='balanced', random_state=42)`.
- **Calibration Layer:** `IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)` fit on validation set predictions.
- **Classification Performance on Held-Out Test Split (`analysis_test.parquet`, 5,004 positions):**
  - **AUC-ROC:** 0.6586
  - **Brier Score:** 0.1135
- **Trigger Rate Clarification:** In offline tests, the error branch triggered at an 11.2% rate ($u < p_{\text{err}}$). This 11.2% was strictly an error-model activation trigger rate and must not be conflated with move error or blunder rates.

### 4. Offline Rejection of Error Branch Temperature Flattening
- While the classifier demonstrates raw predictive signal (AUC 0.6586, Brier 0.1135), flattening temperature via the error branch ($T = T_{\text{base}} \times \text{mult}$) severely degrades gameplay ACPL (increasing average centipawn loss from ~89 cp to 130+ cp).
- Consequently, the error model is disabled for live move selection (`ERROR_MODEL_ENABLED = False` in `config.py`).
- The model and complexity feature pipeline are retained solely for dashboard and post-game telemetry.

---

## Human Ground-Truth Baseline

All human metrics are pinned to a single ground-truth run on the 5,004 positions in `data/players/the_king_crusher/analysis_test.parquet` (Stockfish depth 12, version `sf_16.1`, recorded in `backend/data_baselines/human_baseline_the_king_crusher.json` on 2026-10-17):

- **Human ACPL:** **55.62 cp** (95% CI: `[50.19, 61.24]`)
- **Human $\ge 100$ cp Rate (Errors):** **13.49%** (95% CI: `[12.13%, 14.94%]`)
- **Human $\ge 300$ cp Rate (Blunders):** **3.86%** (95% CI: `[3.24%, 4.54%]`)
- **Test Sample Size:** 5,004 positions across 120 games.

---

## Strength vs Diversity Trade-Off

Using cached Stockfish evaluations (depth 12, `sf_16.1`, 21,828 verified cache entries), we evaluated 4 candidate configurations with 5 seeds each across all 5,004 test positions (Command: `python scripts/run_paired_significance.py`, Date: 2026-10-17). Paired differences and 95% confidence intervals were computed using 2,000 game-level bootstrap resamples.

| Config | ACPL (95% CI) | $\ge 300$ cp (%) (95% CI) | % Positions Stochastic | Uniq/Pos | $\Delta$ ACPL vs Greedy (95% CI) | $\Delta \ge 300$ cp vs Greedy (95% CI) | Ratio vs Human ACPL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **greedy** | 89.07 [81.85, 96.39] | 8.87% [7.84%, 9.96%] | 0.00% | 1.00 | 0.00 (ref) | 0.00% (ref) | 1.60x |
| **t=0.30_p=0.70** | 90.57 [83.52, 97.66] | 9.07% [8.05%, 10.08%] | 4.84% | 1.05 | +1.50 [+0.48, +2.65] | +0.20% [+0.05%, +0.35%] | 1.63x |
| **t=0.50_p=0.70** | 91.97 [84.88, 99.10] | 9.19% [8.20%, 10.24%] | 15.87% | 1.20 | +2.89 [+1.37, +4.39] | +0.32% [+0.10%, +0.54%] | 1.65x |
| **t=0.70_p=0.70** | **93.38 [86.48, 100.17]** | **9.34% [8.34%, 10.37%]** | **32.17%** | **1.48** | **+4.30 [+2.11, +6.41]** | **+0.47% [+0.15%, +0.77%]** | **1.68x** |

### Evaluation of Options

1. **Option A (`t=0.30_p=0.70`)**:
   - Yields only **4.84% stochastic positions** (95.16% of positions pick the exact same move across all 5 seeds).
   - While achieving slightly lower ACPL (90.57 cp vs 93.38 cp), it behaves almost indistinguishably from deterministic greedy in practice, resulting in robotic and repetitive play.
2. **Option B (`t=0.70_p=0.70`) [CHOSEN]**:
   - Yields **32.17% stochastic positions** and **1.48 unique moves per position**, restoring genuine human-like move unpredictability and opening variation.
   - The measured ACPL cost is modest: **+4.30 cp** vs greedy (`[+2.11, +6.41]`) and **+0.47%** blunder rate delta (`[+0.15%, +0.77%]`).

### Explicit Decision
We adopt **Option B**:
```python
OPTIMAL_T_BASE: float = 0.70
OPTIMAL_TOP_P: float = 0.70
OPTIMAL_ERROR_MULT: float = 1.0
ERROR_MODEL_ENABLED: bool = False
```

**Reason:** `t=0.30/p=0.70` behaves almost like greedy with only 4.84% stochastic positions, whereas `t=0.70/p=0.70` delivers meaningful variety (32.17% stochastic positions, 1.48 unique moves per position) for an acceptable ACPL cost of +4.30 cp vs greedy.

---

## Scientific Assessment & Limitations

- **Best Available Sampling Setting:** This represents the best available sampling setting; bot ACPL is ~1.68x the human baseline (measured: 93.38 cp vs 55.62 cp human baseline from `backend/data_baselines/human_baseline_the_king_crusher.json`).
- **Sampling Tuning Is Exhausted:** Extensive grid search over 29 configurations confirms that sampling temperature and truncation cannot close the remaining 37.7 cp ACPL gap (93.38 cp vs 55.62 cp). The remaining gap is a property of the learned policy network itself, not sampling hyperparameters.
- **Open Follow-Up (NOT Started):** Closing the residual gap requires training-side modifications—specifically, expanding fine-tuning data volume (Plan E11) or training with a centipawn-loss-weighted objective. This work is formally logged as NOT started.
