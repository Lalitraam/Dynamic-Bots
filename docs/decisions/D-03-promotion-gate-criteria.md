# Decision D-03: Promotion Gate Uses NLL-Only Criteria

**Date:** 2026-10-16
**Status:** Approved
**Approved by:** Sujit

## Context

Plan section 9 D.3 specifies the promotion gate rule as:

> Test split >= 20 games: paired game-level bootstrap on NLL, fine-tuned vs
> base; CI excluding zero in favour of fine-tuned -> promote fine-tuned,
> else promote base.

An earlier implementation attempt required BOTH NLL and top-1 to show
statistically significant improvement. That was stricter than the plan
spec and was not documented in a decision record. On 2026-10-16 it was
reverted to match the plan.

## Decision

The promotion gate uses NLL-only criteria:

- Promote fine-tuned if NLL 95% CI upper bound < 0.
- Otherwise, promote base.

The top-1 CI is still computed and reported in eval.json and the model
card, but it is NOT part of the promotion decision.

## Rationale

1. Plan-spec compliance. Section 7.4 requires a decision record for any
   change to gate thresholds. Reverting to spec is the conservative
   choice and needs no deviation record.

2. NLL captures probabilistic quality across all legal moves, not only
   the argmax. A model can improve NLL, top-3, mean rank and calibration
   without changing top-1. For the_king_crusher v1 (2026-10-16):

     metric    base     ft       delta
     --------  -------  -------  ------
     top1      0.4746   0.4710   -0.0036
     top3      0.7492   0.7632   +0.0140
     nll       1.7015   1.6673   -0.0342
     rank      3.0158   2.9924   -0.0234
     ece       0.0668   0.0215   -0.0454

   NLL, top-3, mean rank and ECE all improve substantially. Top-1 is flat.
   Under NLL-only criteria, this is a promoted improvement. Under strict
   (NLL + top-1) criteria, the base would be kept despite clear evidence
   of a better probability distribution.

3. The plan already provides a mechanism for top-1 sensitivity: the
   "cut order" in section 8.3 and the E6 personalisation matrix in
   section 11. Top-1 is not the sole arbiter of quality.

## Consequence

- This record supersedes the strict interpretation used in the
  2026-10-15 initial Phase D run.
- The reason string in promotion_gate.py has been corrected to reflect
  NLL-only criteria (see Task 2 of the Phase D follow-up prompt).
- The threshold change (removing top-1 from the AND clause) is
  documented here as required by plan section 7.4.
