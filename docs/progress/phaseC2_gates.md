# Phase C2/C3 Base Model Verification Gates

- **Base Checkpoint**: `..\data\base\base_v1_fast\base_v1.pt`
- **Base Shards Directory**: `..\data\base\shards\base_shards_clean2`
- **Test Positions**: 4658
- **Train Median Rating**: 1657

---

## Gate C-a: Base Test Evaluation

| Predictor | Top-1 (95% CI) | Top-3 (95% CI) | NLL (95% CI) |
| :--- | :--- | :--- | :--- |
| Base Policy (v1) | 0.4324 [0.4157, 0.4487] | 0.7095 [0.6915, 0.7281] | 1.8488 [1.7999, 1.9005] |
| PositionFrequency | 0.1037 [0.0958, 0.1120] | 0.2225 [0.2105, 0.2341] | 3.0935 [3.0499, 3.1410] |
| E2 Reference (500k) | *Not Available* | *Not Available* | *Not Available* |

- (top1_base - top1_posfreq) = 0.3286 (Criterion: >= 0.15): **PASS**
- Absolute top1_base = 0.4324 (Criterion: >= 0.40): **PASS**
- E2 reference not available -> Gate C-a is **INCONCLUSIVE**.

**Gate C-a Verdict**: `INCONCLUSIVE`

---

## Gate C-b: the_king_crusher v3 Test Comparison

| Metric | Base Policy | Current Player Model | Paired Delta (a - b) [95% CI] |
| :--- | :--- | :--- | :--- |
| Top-1 | 0.4836 | 0.2614 | 0.2222 [0.2072, 0.2366] |
| NLL | 1.6347 | 3.0683 | -1.4336 [-1.5027, -1.3667] |

- NLL delta CI upper (-1.3667) < 0: **PASS**
- Top-1 delta CI lower (0.2072) > 0: **PASS**

**Gate C-b Verdict**: `PASS`

---

## Gate C-c: Rating-Conditioning Sanity

- Sampled positions: 500
- Mean log-prob LL(r): -1.87090
- Mean log-prob LL(r+400): -1.89022
- Mean log-prob LL(r-400): -1.88742
- Perturbation +400 penalty [LL(r) - LL(r+400)]: +0.01932 (> 0: **PASS**)
- Perturbation -400 penalty [LL(r) - LL(r-400)]: +0.01652 (> 0: **PASS**)

**Gate C-c Verdict**: `PASS`
