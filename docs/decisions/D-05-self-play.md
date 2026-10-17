# Decision D-05: Self-Play Harness

**Date:** 2026-10-18
**Status:** Approved
**Approved by:** Sujit

## Context

Plan section 9 Phase SP defines the self-play harness needed for
strength evaluation (E13). The harness plays bot vs opponent with
simulated clocks and records games.

## Decisions

### Clock semantics
- No real waiting. Each move subtracts think_ms immediately.
- Increment is added after every successful move.
- Flag detected when clock_before_ms - think_ms <= 0.
- Clocks never allowed to go below 0 (clamp at 0 for reporting).

### Termination order
Checkmate > stalemate > insufficient material > fivefold repetition >
75-move rule > flag > cap. First true condition wins.

### Cap
SP_MAX_PLIES = 300. Cap triggers "cap" termination, result 1/2-1/2.

### Opponent
Base model (temperature-only), rating input = bot's median rating,
think-time from pooled base thinktime for that rating band.

### Determinism
Single seed drives both move sampling and think-time sampling.
Two runs with the same seed produce identical PGNs.

## Consequence
- E13 (strength calibration) depends on this harness.
- Phase P (analysis + error model) uses play_match for verification.
- No real waiting means a 20-game match finishes in <10 seconds.
