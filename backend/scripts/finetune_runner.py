"""
Fine-tuning runner CLI script (Phase D).
Orchestrates variant selection, model training, temperature scaling,
dual-metric evaluation against base, and promotion gating.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

import pandas as pd
import torch

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app import config
from app.ml.evaluate import evaluate
from app.ml.finetune import (
    FineTuneVariant,
    ModelPredictor,
    fine_tune,
    fit_temperature,
    select_variant,
)
from app.ml.model_v2 import ChessPolicyNetV2
from app.ml.promotion_gate import should_promote


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fine-tuning and promotion gate.")
    parser.add_argument("player_name", type=str, help="Player directory name under data/players/")
    parser.add_argument(
        "--base-checkpoint",
        type=str,
        default=str(backend_dir.parent / "data" / "base" / "base_v1_fast" / "base_v1.pt"),
        help="Path to frozen base model checkpoint",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(backend_dir.parent / "data"),
        help="Path to data directory root",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device ('cpu' or 'cuda')")
    parser.add_argument(
        "--variant",
        type=str,
        default="auto",
        choices=["auto", "full", "freeze", "film_only"],
        help="Fine-tuning variant or auto",
    )
    parser.add_argument(
        "--max-minutes",
        type=int,
        default=getattr(config, "FT_MAX_MINUTES", 20),
        help="Maximum wall-clock time in minutes",
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)
    base_checkpoint_path = Path(args.base_checkpoint)
    player_dir = data_root / "players" / args.player_name
    parquet_path = player_dir / "samples_v3.parquet"

    print(f"============================================================")
    print(f"FINE-TUNING RUNNER: {args.player_name}")
    print(f"Data root:        {data_root}")
    print(f"Base checkpoint:  {base_checkpoint_path}")
    print(f"Device:           {args.device}")
    print(f"Max minutes:      {args.max_minutes}")
    print(f"============================================================")

    # 1. Load samples_v3.parquet
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing dataset at {parquet_path}")

    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    # 2. Count games per split (train/val/test)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    n_train_games = int(train_df["game_id"].nunique())
    n_val_games = int(val_df["game_id"].nunique())
    n_test_games = int(test_df["game_id"].nunique())
    n_total_games = int(df["game_id"].nunique())

    print(f"Game counts: Total={n_total_games} | Train={n_train_games} | Val={n_val_games} | Test={n_test_games}")
    print(f"Row counts:  Total={len(df)} | Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")

    # 3. Select variant
    if args.variant == "auto":
        variant = select_variant(n_total_games, n_val_games)
        print(f"Auto-selected variant: {variant.value.upper()} (n_games={n_total_games}, n_val={n_val_games})")
    else:
        variant = FineTuneVariant(args.variant)
        print(f"User-specified variant: {variant.value.upper()}")

    # 4 & 5. Run fine_tune
    out_dir = player_dir / "models" / "v1"
    staging_dir = out_dir / "_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    print("\nStarting fine_tune()...")
    ft_res = fine_tune(
        base_checkpoint_path=base_checkpoint_path,
        train_parquet=train_df,
        val_parquet=val_df,
        test_parquet=test_df,
        out_dir=staging_dir,
        variant=variant,
        max_minutes=args.max_minutes,
        device=args.device,
    )

    print(f"Fine-tuning complete: {ft_res['stop_reason']} after {ft_res['epochs_run']} epochs "
          f"(best epoch {ft_res['best_epoch']}, wall: {ft_res['wall_minutes']:.2f}m)")

    # 6. Fit temperature on val
    print("\nFitting temperature on validation split...")
    staged_ckpt = torch.load(staging_dir / "weights.pt", map_location=args.device, weights_only=False)
    ft_model = ChessPolicyNetV2.from_config(staged_ckpt["config"])
    ft_model.load_state_dict(staged_ckpt["state_dict"])
    ft_model.to(args.device)

    temp_res = fit_temperature(
        model=ft_model,
        val_parquet=val_df,
        out_path=staging_dir / "temperature.json",
        device=args.device,
    )
    T_best = float(temp_res["T_base"])
    print(f"Optimal Temperature: {T_best} (Val NLL: {temp_res['val_nll_before']:.4f} -> {temp_res['val_nll_after']:.4f})")

    # 7. Compute per-position metrics on test for BASE and FT
    print("\nEvaluating BASE and FT models on test split...")
    base_ckpt = torch.load(base_checkpoint_path, map_location="cpu", weights_only=False)
    base_cfg = base_ckpt.get(
        "config",
        {"channels": 96, "num_blocks": 6, "policy_dim": 32, "meta_dim": 10, "num_planes": 22},
    )
    base_model = ChessPolicyNetV2.from_config(base_cfg)
    base_model.load_state_dict(base_ckpt.get("state_dict", base_ckpt))
    base_model.to(args.device)

        # Use the player's own median rating, not the default 1500
    player_ratings = df["player_rating"].dropna().astype(int).tolist()
    player_median_rating = int(np.median(player_ratings)) if player_ratings else 1500
    print(f"Player median rating: {player_median_rating}")

    pred_base = ModelPredictor(
        base_model, device=args.device, temperature=1.0,
        median_rating=player_median_rating,
    )
    pred_ft = ModelPredictor(
        ft_model, device=args.device, temperature=T_best,
        median_rating=player_median_rating,
    )

    test_rows = test_df.to_dict(orient="records")
    eval_base = evaluate(pred_base, test_rows)
    eval_ft = evaluate(pred_ft, test_rows)

    # If small sample (< 20 test games), prepare validation halves
    val_h1_base, val_h1_ft = None, None
    val_h2_base, val_h2_ft = None, None
    if n_test_games < getattr(config, "PROMOTION_GATE_MIN_GAMES", 20):
        val_games = list(val_df["game_id"].unique())
        half_idx = len(val_games) // 2
        h1_gids = set(val_games[:half_idx])
        h2_gids = set(val_games[half_idx:])
        v1_rows = val_df[val_df["game_id"].isin(h1_gids)].to_dict(orient="records")
        v2_rows = val_df[val_df["game_id"].isin(h2_gids)].to_dict(orient="records")
        val_h1_base = evaluate(pred_base, v1_rows)
        val_h1_ft = evaluate(pred_ft, v1_rows)
        val_h2_base = evaluate(pred_base, v2_rows)
        val_h2_ft = evaluate(pred_ft, v2_rows)

    # 8. Call should_promote()
    gate_decision = should_promote(
        test_split_rows_base=eval_base["per_position"],
        test_split_rows_ft=eval_ft["per_position"],
        n_test_games=n_test_games,
        val_split_half1_base=val_h1_base,
        val_split_half1_ft=val_h1_ft,
        val_split_half2_base=val_h2_base,
        val_split_half2_ft=val_h2_ft,
    )

    decision = gate_decision["decision"]
    promoted = "ft" if decision == "promote_ft" else "base"
    print("\n" + "=" * 60)
    print(f"PROMOTION GATE DECISION: {decision.upper()} -> PROMOTED: {promoted.upper()}")
    print(f"Reason: {gate_decision['reason']}")
    print("=" * 60)

    # 9 & 10. Write final artifacts to <data-root>/players/<player_name>/models/v1/
    out_dir.mkdir(parents=True, exist_ok=True)

    if promoted == "ft":
        torch.save(staged_ckpt, out_dir / "weights.pt")
    else:
        # If promote_base: skip weights.pt; if present, remove it
        if (out_dir / "weights.pt").exists():
            (out_dir / "weights.pt").unlink()

    # Clean up staging directory
    import shutil
    shutil.rmtree(staging_dir, ignore_errors=True)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(ft_model.to_config(), f, indent=2)

    with open(out_dir / "temperature.json", "w", encoding="utf-8") as f:
        json.dump(temp_res, f, indent=2)

    eval_summary = {
        "promoted": promoted,
        "decision": decision,
        "reason": gate_decision["reason"],
        "delta_nll": gate_decision["delta_nll"],
        "ci_nll": list(gate_decision["ci_nll"]) if gate_decision["ci_nll"] else None,
        "delta_top1": gate_decision["delta_top1"],
        "ci_top1": list(gate_decision["ci_top1"]) if gate_decision["ci_top1"] else None,
        "n_test_games": n_test_games,
        "n_test_positions": len(test_rows),
        "base_metrics": {
            "top1": eval_base["top1"],
            "top3": eval_base["top3"],
            "nll": eval_base["nll"],
            "rank": eval_base["rank"],
            "ece": eval_base["ece"],
        },
        "ft_metrics": {
            "top1": eval_ft["top1"],
            "top3": eval_ft["top3"],
            "nll": eval_ft["nll"],
            "rank": eval_ft["rank"],
            "ece": eval_ft["ece"],
        },
        "fine_tune_log": ft_res,
    }

    with open(out_dir / "eval.json", "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    # Write model_card.md
    model_card_content = f"""# Model Card: {args.player_name} (v1)

## Summary
- **Player**: `{args.player_name}`
- **Promoted Model**: `{promoted}`
- **Decision**: `{decision}`
- **Variant**: `{variant.value}`
- **Optimal Temperature (T)**: `{T_best}`

## Promotion Gate
- **Decision**: `{decision}`
- **Reason**: {gate_decision['reason']}
- **Delta NLL (FT - Base)**: `{gate_decision['delta_nll']}` (95% CI: `{gate_decision['ci_nll']}`)
- **Delta Top-1 (FT - Base)**: `{gate_decision['delta_top1']}` (95% CI: `{gate_decision['ci_top1']}`)

## Test Metrics Comparison ({len(test_rows)} positions across {n_test_games} games)
| Metric | Frozen Base | Fine-Tuned (T={T_best}) | Delta |
|---|---|---|---|
| Top-1 Accuracy | {eval_base['top1']:.4f} | {eval_ft['top1']:.4f} | {eval_ft['top1'] - eval_base['top1']:+.4f} |
| Top-3 Accuracy | {eval_base['top3']:.4f} | {eval_ft['top3']:.4f} | {eval_ft['top3'] - eval_base['top3']:+.4f} |
| Negative Log-Likelihood (NLL) | {eval_base['nll']:.4f} | {eval_ft['nll']:.4f} | {eval_ft['nll'] - eval_base['nll']:+.4f} |
| Mean Rank | {eval_base['rank']:.4f} | {eval_ft['rank']:.4f} | {eval_ft['rank'] - eval_base['rank']:+.4f} |
| ECE (10 bins) | {eval_base['ece']:.4f} | {eval_ft['ece']:.4f} | {eval_ft['ece'] - eval_base['ece']:+.4f} |

## Training History
- **Epochs run**: {ft_res['epochs_run']} (Best epoch: {ft_res['best_epoch']})
- **Stop reason**: {ft_res['stop_reason']}
- **Wall clock**: {ft_res['wall_minutes']:.2f} minutes
"""
    with open(out_dir / "model_card.md", "w", encoding="utf-8") as f:
        f.write(model_card_content)

    # 11. Print summary table to stdout
    print("\n" + "=" * 70)
    print("FINAL SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Metric':<25} | {'Base Model':<12} | {'Fine-Tuned':<12} | {'Delta':<12}")
    print("-" * 70)
    print(f"{'Top-1 Accuracy':<25} | {eval_base['top1']:<12.4f} | {eval_ft['top1']:<12.4f} | {eval_ft['top1'] - eval_base['top1']:<+12.4f}")
    print(f"{'Top-3 Accuracy':<25} | {eval_base['top3']:<12.4f} | {eval_ft['top3']:<12.4f} | {eval_ft['top3'] - eval_base['top3']:<+12.4f}")
    print(f"{'NLL':<25} | {eval_base['nll']:<12.4f} | {eval_ft['nll']:<12.4f} | {eval_ft['nll'] - eval_base['nll']:<+12.4f}")
    print(f"{'Mean Rank':<25} | {eval_base['rank']:<12.4f} | {eval_ft['rank']:<12.4f} | {eval_ft['rank'] - eval_base['rank']:<+12.4f}")
    print(f"{'ECE':<25} | {eval_base['ece']:<12.4f} | {eval_ft['ece']:<12.4f} | {eval_ft['ece'] - eval_base['ece']:<+12.4f}")
    print("-" * 70)
    print(f"Promoted Model:   {promoted.upper()}")
    print(f"Artifacts saved:  {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
