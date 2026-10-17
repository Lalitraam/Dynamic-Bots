# Phase E Progress: V3 Provider & Runtime Move Pipeline

## 2026-10-17 — Phase E Initial Implementation

- F0 frozen interfaces defined in app/engine/provider.py.
- V3Provider implemented (temperature-only mode).
- ThinkTimeModel built from per-player samples_v3.parquet.
- ModelCache LRU(capacity=8) with base fallback.
- Telemetry schema defined (move_telemetry).
- Guardrail remains a no-op behind GUARD_ENABLED = False.
- All new tests pass; full suite count = 222 passed.

## 2026-10-17 — Follow-up: FT-through-provider test
- Added test_1_ft_model_loads_through_cache in test_provider_v3_ft.py.
- Confirms ModelCache loads a fine-tuned v1 checkpoint for the_king_crusher
  (not the base fallback).
- Confirms V3Provider returns model_version = "the_king_crusher:v1".

