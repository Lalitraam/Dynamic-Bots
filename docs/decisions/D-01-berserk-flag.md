# Decision D-01: Berserk Positions and Berserk Meta Feature Flag

**Date:** 2026-10-12  
**Status:** Approved (Option A)  
**Approved by:** Sujit  

AMENDS plan ECH-IMP-004: META_DIM 9 -> 10 in B.3, B.4, C2 FiLM MLP (meta 10->64->64) and Appendix B

## Context & Problem
In the dataset for `the_king_crusher`, a substantial proportion of games involve berserk play:
- Train split: 17479 / 40009 rows (43.69%), 413 / 960 games (43.02%)
- Val split: 2649 / 5149 rows (51.45%), 58 / 120 games (48.33%)
- Test split: 0 / 5004 rows (0.00%), 0 / 120 games (0.00%)

The split is purely chronological (train: 2024-01-01 15:59:00+00:00 to 2026-05-26 15:23:55+00:00; val: 2026-05-26 15:30:23+00:00 to 2026-06-25 18:02:45+00:00; test: 2026-06-25 18:03:59+00:00 to 2026-10-05 17:15:39+00:00). The test split has 0% berserk because the newest ~10% of games contain no berserk games (distribution shift: 2026-07 0/24, 2026-08 0/27, 2026-09 0/19, 2026-10 0/19 games berserk).

Empirical evidence collected across the dataset:
1. **Berserk removes the increment and halves the clock:**
   - 471 of 471 header-berserk games start with initial clock at exactly `base_seconds / 2`.
   - In header-berserk games with increment > 0, 0 of 5777 own-move transitions had $d = clk_{prev} - clk_{cur} < 0$, confirming that the increment is eliminated. In contrast, non-berserk increment games had 3059 of 7603 transitions with $d < 0$.
2. **Behavioral shift:**
   - Players move significantly faster when berserk (e.g. blitz berserk p95 thinking time is 5.00s [count: 18349, p50: 1.00s] vs 14.00s for non-berserk [count: 13910, p50: 2.00s]). Discarding berserk positions would discard valuable tactical data, while conflating berserk and non-berserk contexts would confuse time/move modeling.

## Decision (Option A)
1. **Keep Berserk Positions:** Berserk positions are retained in the training and validation sets to preserve sample size and opening/tactical repertoire coverage.
2. **Berserk Meta Feature Flag:**
   - Add a binary indicator `berserk` (1.0 if `target_berserk` else 0.0) as feature index 9 in the metadata vector.
   - Metadata vector dimensions: `META_DIM_V3 = 10`
     - Features 0-7: Original 8 v2 metadata features (`encode_meta`).
     - Feature 8: Legal move count feature $\min(\text{num\_legal\_moves} / 60.0, 1.0)$.
     - Feature 9: Berserk flag ($1.0$ if berserk else $0.0$).
3. **Inference & Evaluation:**
   - At play / inference time, the `berserk` flag is always set to `0.0`.
   - Gating, early stopping, and temperature fits evaluate exclusively on non-berserk positions (`berserk_filter="non_berserk"`).
4. **Cache Storage:**
   - The tensor cache stores `berserk.npy` (int8 array, 1 for berserk, 0 for non-berserk) to enable instantaneous sub-filtering and mmap slicing without reading parquet metadata.

## Consequence
Consequence: berserk was concentrated in earlier months, so val/test are non-representative for berserk; gates use non-berserk positions only.

