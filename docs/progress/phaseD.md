# Phase D Progress: Fine-Tuning Pipeline & Promotion Gate

**Date:** 2026-10-15  
**Status:** Complete  
**Branch:** `phase-b`

## Summary of Accomplishments

### 1. Additive Config Constants (`backend/app/config.py`)
- Fine-tuning hyperparameters:
  - `FT_MAX_MINUTES = 20`
  - `FT_MAX_EPOCHS = 8`
  - `FT_PATIENCE = 2`
  - `FT_BATCH_SIZE = 256`
  - `FT_LR_FULL = 3e-4`, `FT_LR_FREEZE = 5e-4`, `FT_LR_FILM_ONLY = 1e-3`
  - `FT_FREEZE_UNTIL_BLOCK = 3`
  - `FT_MIN_VAL_GAMES = 10`
- Calibration & gating constants:
  - `TEMPERATURE_GRID = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]`
  - `PROMOTION_GATE_MIN_GAMES = 20`
  - `PROMOTION_GATE_BOOTSTRAP_N = 2000`
  - `PROMOTION_GATE_SEED = 0`

### 2. Fine-Tuning Pipeline (`backend/app/ml/finetune.py`)
- `FineTuneVariant` enum: `FULL`, `FREEZE`, `FILM_ONLY`.
- `select_variant(n_games, n_val_games)`: Implements game-count tiering and validation override to `FILM_ONLY` when $N_{\text{val}} < 10$.
- `freeze_for_variant(model, variant)`: Manages layer-specific freezing with permanent `.eval()` enforcement for frozen BatchNorm layers.
- `set_frozen_bn_eval(model, variant)`: Helper invoked after every `model.train()` call to prevent BatchNorm running statistics from mutating on frozen blocks.
- `fine_tune(...)`: Full pipeline with early stopping, wall-clock timeout checks, best checkpoint restoration, and artifact generation.
- `ModelPredictor`: Predictor wrapper supporting temperature scaling for evaluation.

### 3. Temperature Fitting (`backend/app/ml/temperature_fit.py`)
- `fit_temperature(...)`: Evaluates validation NLL across the temperature grid and identifies the optimal temperature $T^*$.
- Exports calibration results to `temperature.json`.

### 4. Promotion Gate (`backend/app/ml/promotion_gate.py`)
- `paired_game_bootstrap(...)`: Resamples games with replacement ($B = 2000$, seed 0) to compute 95% confidence intervals on paired metric differences ($\Delta = \text{FT} - \text{Base}$). Enforces key alignment on `(game_id, ply)`.
- `should_promote(...)`: Dual-metric decision rule:
  - When $N_{\text{test\_games}} \ge 20$: Promotes FT iff $\text{CI}_{\text{nll}}.\text{hi} < 0$ and $\text{CI}_{\text{top1}}.\text{lo} > 0$.
  - When $N_{\text{test\_games}} < 20$: Falls back to requiring FT NLL < Base NLL across both validation split halves.

### 5. CLI Runner (`backend/scripts/finetune_runner.py`)
- End-to-end command line orchestrator executing data loading, variant selection, fine-tuning, temperature fitting, test evaluation, promotion gating, artifact export, and terminal summary reporting.

### 6. Architectural Decision Record (`docs/decisions/D-02-finetune-variants.md`)
- Formal documentation of variant specifications, BatchNorm preservation, temperature scaling, and gating rules.

### 7. Unit Tests (`backend/app/tests/`)
- `test_finetune.py`: 7 tests covering variant selection, parameter freezing sets, frozen BN stat stability, wall-clock timeout, and smoke fine-tuning.
- `test_promotion_gate.py`: 5 tests covering promotion, base retention, small sample handling, validation halves fallback, and paired bootstrap mismatch exceptions.
- `test_temperature_fit.py`: 2 tests covering NLL calibration improvement and grid argmin selection.
- Teeth checks verified:
  - Changing FREEZE cutoff $3 \to 0$ causes `test_3_freeze_variant_correct_set` to fail.
  - Removing BatchNorm `.eval()` re-application causes `test_5_frozen_bn_stats_do_not_update` to fail.
  - Setting `should_promote` to always return `promote_ft` causes `test_2_gate_keeps_base_when_ft_worse` to fail.

## 2026-10-16 — Reason String Fix + FREEZE Variant Test

- Fixed promotion_gate.py reason string (previously claimed top-1 improvement
  in a case where top-1 CI was negative).
- Added regression test test_6_reason_string_reflects_nll_only.
- Wrote docs/decisions/D-03-promotion-gate-criteria.md.
- Re-ran fine-tune for the_king_crusher with FREEZE variant. Full backup
  preserved at data/players/the_king_crusher/models/v1_full_backup/.
- FULL result (backup): top1=0.4710, nll=1.6673, promoted=ft
- FREEZE result: top1=0.4724, nll=1.6561, top3=0.7622, rank=2.9728, ece=0.0268, promoted=ft

## 2026-10-16 — Phase D Extended: chesslavaplanet + Fritzi_2003

### chesslavaplanet
- Games ingested: 1200
- Rows in samples_v3.parquet: 51632
- Train / Val / Test games: 960 / 120 / 120
- Player median rating: 2777
- Variant selected: full
- Best val NLL: 1.6884
- Test top-1: base=0.4494 -> ft=0.4673
- Test NLL: base=1.7891 -> ft=1.7029
- Test ECE: base=0.0492 -> ft=0.0220
- Promotion gate: promote_ft
- Wall clock: 15.36m

### Fritzi_2003
- Games ingested: 1200
- Rows in samples_v3.parquet: 54737
- Train / Val / Test games: 960 / 120 / 120
- Player median rating: 2904
- Variant selected: full
- Best val NLL: 1.6712
- Test top-1: base=0.4092 -> ft=0.4444
- Test NLL: base=1.8799 -> ft=1.7775
- Test ECE: base=0.0142 -> ft=0.0123
- Promotion gate: promote_ft
- Wall clock: 20.02m

### Milestone status
- M-PERSONAL requires 3 promoted players.
- the_king_crusher: promoted (FREEZE, 2026-10-15)
- chesslavaplanet: promoted (FULL, 2026-10-15)
- Fritzi_2003: promoted (FULL, 2026-10-15)


