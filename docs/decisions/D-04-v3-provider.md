# Decision D-04: V3 Provider — Runtime Move Pipeline

**Date:** 2026-10-17
**Status:** Approved
**Approved by:** Sujit

## Context

Plan section 4.2 defines the runtime move pipeline as a five-step process
(Book, Policy, Sampling, Guardrail, Think time). Phase E implements the
first, second, third and fifth steps. The optional Stockfish guardrail
(step 4) is a no-op until Phase P and is behind a config flag (GUARD_ENABLED).

Plan section 9 Phase VS F0 freezes the interface that any provider must
implement: MoveContext (input), MoveDecision (output), MoveProvider (protocol).

## Decisions

### F0 interfaces
- `MoveContext` and `MoveDecision` are dataclasses with the exact fields
  specified in plan section 9 F0.
- `MoveProvider` is a `Protocol` with `decide(ctx, rng) -> MoveDecision`.
- These are frozen. Any change requires a HUMAN GATE.

### Sampling
- Temperature-only mode for v1. Pattern-conditioned sampling (P.4) is a
  no-op until E12 passes.
- Top-p = 0.95 (config.TOP_P).

### Think time
- Empirical sampling from the player's samples_v3.parquet.
- Bucket key: (speed, phase, clock_ratio, is_forced).
- Reservoir sampling caps each bucket at 500 values, seed 0 (deterministic).
- Fallback hierarchy: full bucket -> same speed+phase -> same speed -> pooled.
- Clamp: [THINK_FLOOR_MS, bot_clock_ms - THINK_MARGIN_MS].

### Model cache
- LRU with capacity 8, keyed by (username, version).
- Falls back to base_v1 if the version directory has no weights.pt.

### Telemetry
- Schema from Appendix A. Table `move_telemetry`.
- Purpose: operational monitoring + raw record for E12/E13 experiments.

### Guardrail
- GUARD_ENABLED = False by default.
- The guardrail code path is a no-op; Phase P implements it.

## Consequence

- The provider is testable standalone (no game server required).
- Phase F (game server) will call V3Provider.decide directly.
- The book (BookV2) is optional; if None, the pipeline skips step 1.
