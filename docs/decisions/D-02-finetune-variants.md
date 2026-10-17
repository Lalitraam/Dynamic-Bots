# Decision D-02: Fine-Tuning Variants and Promotion Gate Rules

**Date:** 2026-10-15  
**Status:** Approved  
**Approved by:** Architecture / Plan Section 9 (Phase D)

## Context & Motivation
Fine-tuning the frozen foundation model (`ChessPolicyNetV2`, base_v1) on individual player repertoires presents varying degrees of sample size scarcity and overfitting risk:
- Large sample regimes ($\ge 800$ games) offer sufficient signal to fine-tune the entire network without catastrophic forgetting.
- Medium sample regimes ($150 - 799$ games) risk degrading early feature detectors (stem and early residual blocks), requiring trunk freezing while adapting high-level representations.
- Small sample regimes ($50 - 149$ games, or $< 10$ validation games) cannot reliably update trunk weights without overfitting, requiring adaptation strictly via FiLM conditioning layers and head parameters.
- Very small regimes ($< 50$ games) lack statistical validity for fine-tuning and must be rejected.

Additionally, a principled gating mechanism is required before deploying fine-tuned weights: fine-tuned models must demonstrate statistically significant improvements over the base model, avoiding regressions caused by noise or sample variance.

---

## 1. Fine-Tuning Variants & Decision Logic

Variant selection is automated based on available game counts:

| Variant | Game Range | Validation Override | Frozen Submodules | Trainable Submodules | Initial LR |
|---|---|---|---|---|---|
| **FULL** | $N \ge 800$ | $N_{\text{val}} \ge 10$ | None | All parameters | $3 \times 10^{-4}$ |
| **FREEZE** | $150 \le N < 800$ | $N_{\text{val}} \ge 10$ | `stem_conv`, `stem_bn`, `meta_mlp`, `blocks[0..2]`, `film_generators[0..2]` | `blocks[3..5]`, `film_generators[3..5]`, `q_proj`, `k_proj`, `bias_pair`, `promo_head` | $5 \times 10^{-4}$ |
| **FILM_ONLY** | $50 \le N < 150$ | Or $N_{\text{val}} < 10$ | All trunk blocks, stem, meta_mlp, q_proj, k_proj | `film_generators[*]`, `bias_pair`, `promo_head` | $1 \times 10^{-3}$ |
| **REJECT** | $N < 50$ | Any | N/A (raises ValueError) | N/A | N/A |

### Validation Game Safeguard
If $N_{\text{val}} < 10$, the validation split contains insufficient games for reliable early stopping across a full ResNet trunk. In this case, the variant is automatically overridden to `FILM_ONLY` regardless of $N$.

### BatchNorm Handling Under Freezing
When layers are frozen in `FREEZE` or `FILM_ONLY` variants, setting `requires_grad = False` prevents gradient updates to affine weights, but standard PyTorch `BatchNorm2d` continues updating running mean and running variance if `model.train()` is called. To preserve the base model's calibrated representations:
1. All frozen BatchNorm layers are permanently set to `.eval()`.
2. This `.eval()` status is explicitly re-applied after any `model.train()` invocation at the beginning of every training epoch.

---

## 2. Temperature Calibration

Probability calibration is performed post-fine-tuning via grid search over:
$$\mathcal{T} = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]$$
The optimal temperature $T^*$ minimizes unregularized validation Negative Log-Likelihood (NLL with $\epsilon=0$ label smoothing):
$$T^* = \arg\min_{T \in \mathcal{T}} \text{NLL}_{\text{val}}(T)$$
Results are stored in `temperature.json` with keys `T_base`, `val_nll_before`, `val_nll_after`, and the full grid mapping.

---

## 3. Promotion Gate Architecture & Significance Testing

To protect against deploying degraded models, a rigorous promotion gate evaluates candidate fine-tuned models against the frozen base model on the unseen test split.

### Paired Game-Level Block Bootstrap
Because positions within the same game exhibit strong tactical autocorrelation, resampling is conducted strictly at the **game level**:
1. All positions in the test split are matched by `(game_id, ply)` between the base and fine-tuned models. Any mismatch raises an immediate error.
2. For each paired position $i$, the metric difference is computed:
   $$\Delta_i = m_i^{(\text{ft})} - m_i^{(\text{base})}$$
3. $G$ games are resampled with replacement $B = 2000$ times (fixed seed 0), pooling all positions for the sampled games in each replica.
4. Point estimates and 95% percentile confidence intervals $(\text{lo}_{95}, \text{hi}_{95})$ are derived.

### Decision Rules (Plan Section 9 D.3)

1. **Standard Regime ($N_{\text{test\_games}} \ge 20$):**
   A fine-tuned model is promoted (`promote_ft`) if and only if both conditions are met at the 95% confidence level:
   $$\text{CI}_{\text{nll}}.\text{hi} < 0 \quad \text{AND} \quad \text{CI}_{\text{top1}}.\text{lo} > 0$$
   - $\text{CI}_{\text{nll}}.\text{hi} < 0$: FT has strictly lower NLL than base with 95% confidence.
   - $\text{CI}_{\text{top1}}.\text{lo} > 0$: FT has strictly higher top-1 accuracy than base with 95% confidence.
   If either condition fails, the decision is `promote_base`.

2. **Small Sample Regime ($N_{\text{test\_games}} < 20$ with Validation Halves):**
   If fewer than 20 test games exist, the bootstrap CI cannot provide robust coverage. The validation games are split chronologically into two halves ($H_1, H_2$):
   - Promote FT if and only if:
     $$\text{NLL}_{\text{ft}}(H_1) < \text{NLL}_{\text{base}}(H_1) \quad \text{AND} \quad \text{NLL}_{\text{ft}}(H_2) < \text{NLL}_{\text{base}}(H_2)$$
   - Otherwise, `promote_base`.

3. **Insufficient Evidence:**
   If $N_{\text{test\_games}} < 20$ and validation split halves are unavailable, the gate returns `insufficient` and deployment defaults to base.

---

## 4. Deployment Artifacts

All final model assets are written to `data/players/<player_name>/models/v1/`:
- `weights.pt`: Saved **only** if the decision is `promote_ft`. If `promote_base`, `weights.pt` is omitted or removed so inference falls back directly to the frozen base checkpoint.
- `config.json`: Model architecture configuration.
- `temperature.json`: Optimal temperature $T^*$ and NLL calibration curve.
- `eval.json`: Comprehensive report containing `"promoted": "ft" | "base"`, gate decision, reason, point estimates, 95% CIs, and subgroup metrics.
- `model_card.md`: Human-readable summary table comparing Base vs FT performance.
