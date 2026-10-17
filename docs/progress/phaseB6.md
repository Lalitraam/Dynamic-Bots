# Phase B.6 Baseline Evaluation: `the_king_crusher`

- **Test split positions**: 5004
- **Unique games in test**: 120

## Overall Performance Comparison

| Predictor | top1 [lo95,hi95] | top3 | nll [lo95,hi95] | n_games | n_positions |
|:---|:---:|:---:|:---:|:---:|:---:|
| UniformLegal | 0.0538 [0.0491, 0.0588] | 0.1466 | 3.2802 [3.2351, 3.3267] | 120 | 5004 |
| PositionFrequency | 0.1116 [0.1016, 0.1225] | 0.2126 | 3.1645 [3.1218, 3.2121] | 120 | 5004 |
| CurrentModel | 0.2614 [0.2474, 0.2753] | 0.4359 | 3.0683 [2.9971, 3.1431] | 120 | 5004 |
| BookThenModel | 0.2590 [0.2448, 0.2733] | 0.4391 | 3.0777 [3.0045, 3.1553] | 120 | 5004 |

## Per-Phase Breakdown (top-1 / NLL)

| Predictor | Opening (ply<=20) | Middlegame (21-60) | Endgame (>60) |
|:---|:---:|:---:|:---:|
| UniformLegal | 0.0340 / 3.4559 | 0.0406 / 3.4969 | 0.0883 / 2.8353 |
| PositionFrequency | 0.2643 / 2.9950 | 0.0406 / 3.4969 | 0.0883 / 2.8353 |
| CurrentModel | 0.5231 / 1.8552 | 0.1710 / 3.6558 | 0.1775 / 3.2210 |
| BookThenModel | 0.5135 / 1.8927 | 0.1710 / 3.6558 | 0.1775 / 3.2210 |

## Calibration (ECE)

- **CurrentModel ECE (10 equal-width bins)**: 0.1275

## By Berserk Split (top-1 / NLL)

| Predictor | Normal (target_berserk=False) | Berserk (target_berserk=True) |
|:---|:---:|:---:|
| UniformLegal | 0.0538 / 3.2802 | 0.0000 / 0.0000 |
| PositionFrequency | 0.1116 / 3.1645 | 0.0000 / 0.0000 |
| CurrentModel | 0.2614 / 3.0683 | 0.0000 / 0.0000 |
| BookThenModel | 0.2590 / 3.0777 | 0.0000 / 0.0000 |

---

- `BOOK_EVAL_EPS = 0.001`
- training val loss is smoothed CE, not NLL; not comparable

Book sparsity: 196 / 4238 train positions have raw_visits >= 5.
BookThenModel top1 0.2590 vs CurrentModel 0.2614 on the v3 test split.
This is expected: the eval book is train-only, ply<=16, raw>=5, so it
covers few test positions and concentrates on the player's habitual
openings. The final MoveProvider (plan 6.5) BLENDS the book with policy
via alpha = n/(n+BOOK_ALPHA_K), so policy dominates where the book is
sparse. B.6's BookThenModel replacement was diagnostic only.
