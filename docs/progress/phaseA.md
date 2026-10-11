# Phase A Baseline Evaluation: `the_king_crusher`

- **Test split positions**: 5004
- **Unique games in test**: 120

## Overall Performance Comparison

| Predictor | top-1 (mean [lo95,hi95]) | top-3 | NLL (mean [lo95,hi95]) | n_games | n_positions |
|:---|:---:|:---:|:---:|:---:|:---:|
| UniformLegal | 0.0538 [0.0491, 0.0588] | 0.1466 | 3.2802 [3.2351, 3.3267] | 120 | 5004 |
| PositionFrequency | 0.1116 [0.1016, 0.1225] | 0.2126 | 3.1645 [3.1218, 3.2121] | 120 | 5004 |
| BookThenModel | 0.2593 [0.2450, 0.2738] | 0.4359 | 3.1092 [3.0329, 3.1890] | 120 | 5004 |
| CurrentModel | 0.2614 [0.2474, 0.2753] | 0.4359 | 3.0683 [2.9971, 3.1431] | 120 | 5004 |

## Per-Phase Breakdown (top-1 / NLL)

| Predictor | Opening (ply<=20) | Middlegame (21-60) | Endgame (>60) |
|:---|:---:|:---:|:---:|
| UniformLegal | 0.0340 / 3.4559 | 0.0406 / 3.4969 | 0.0883 / 2.8353 |
| PositionFrequency | 0.2643 / 2.9950 | 0.0406 / 3.4969 | 0.0883 / 2.8353 |
| BookThenModel | 0.5147 / 2.0183 | 0.1710 / 3.6558 | 0.1775 / 3.2210 |
| CurrentModel | 0.5231 / 1.8552 | 0.1710 / 3.6558 | 0.1775 / 3.2210 |

## Calibration (ECE)

- **CurrentModel ECE (10 equal-width bins)**: 0.1275

## By Berserk Split (top-1 / NLL)

| Predictor | Normal (target_berserk=False) | Berserk (target_berserk=True) |
|:---|:---:|:---:|
| UniformLegal | 0.0538 / 3.2802 | 0.0000 / 0.0000 |
| PositionFrequency | 0.1116 / 3.1645 | 0.0000 / 0.0000 |
| BookThenModel | 0.2593 / 3.1092 | 0.0000 / 0.0000 |
| CurrentModel | 0.2614 / 3.0683 | 0.0000 / 0.0000 |

## By Clock Ratio Split (top-1 / NLL)

| Predictor | <0.1 | 0.1-0.3 | >0.3 | Unknown |
|:---|:---:|:---:|:---:|:---:|
| UniformLegal | 0.1160 / 2.5916 | 0.0936 / 2.8709 | 0.0453 / 3.3706 | 0.0000 / 0.0000 |
| PositionFrequency | 0.1160 / 2.5916 | 0.1027 / 2.8584 | 0.1125 / 3.2354 | 0.0000 / 0.0000 |
| BookThenModel | 0.2269 / 2.8667 | 0.1517 / 3.3333 | 0.2747 / 3.0945 | 0.0000 / 0.0000 |
| CurrentModel | 0.2269 / 2.8667 | 0.1517 / 3.3343 | 0.2772 / 3.0460 | 0.0000 / 0.0000 |

---

- `BOOK_EVAL_EPS = 0.001`
- training val loss is smoothed CE, not NLL; not comparable
