"""
Evaluate trained model and baselines on the held-out test split.

Usage:
  python scripts/evaluate_test_split.py [username]            # Phase A evaluation (default)
  python scripts/evaluate_test_split.py [username] --legacy   # Legacy evaluation & curves
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app import config
from app.ml import export
from app.ml.baselines import (
    BOOK_EVAL_EPS,
    BookThenModel,
    CurrentModel,
    PositionFrequency,
    UniformLegal,
)
from app.ml.dataset import PlayerMoveDataset
from app.ml.evaluate import evaluate
from app.ml.losses import compute_loss
from app.services import storage
from app.services.opening_book import OpeningBook


def format_ci(mean: float, ci: Any) -> str:
    if isinstance(ci, tuple) and len(ci) == 2:
        return f"{mean:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"
    if isinstance(ci, dict) and ci.get("status") == "insufficient_evidence":
        return f"{mean:.4f} [insufficient evidence]"
    return f"{mean:.4f}"


def run_phase_a_evaluation(username: str = "the_king_crusher") -> str:
    p_dir = storage.player_dir(username, create=False)
    samples_path = p_dir / "samples.parquet"
    if not samples_path.exists():
        raise FileNotFoundError(f"Dataset not found at {samples_path}")

    # Load dataset splits
    table = pq.read_table(str(samples_path))
    df = table.to_pandas()

    test_df = df[df["split"] == "test"]
    train_df = df[df["split"] == "train"]

    test_rows = test_df.to_dict("records")
    train_rows = train_df.to_dict("records")

    # Load player model and opening book
    model = export.load_player_model(username, device="cpu")
    model.eval()
    book = OpeningBook(username)

    # Initialize baselines
    predictors = {
        "UniformLegal": UniformLegal(),
        "PositionFrequency": PositionFrequency(train_rows),
        "BookThenModel": BookThenModel(model, book, device="cpu"),
        "CurrentModel": CurrentModel(model, device="cpu"),
    }

    results: dict[str, dict[str, Any]] = {}
    for name, pred in predictors.items():
        results[name] = evaluate(pred, test_rows)

    # Build markdown report
    lines: list[str] = []
    lines.append(f"# Phase A Baseline Evaluation: `{username}`\n")
    lines.append(f"- **Test split positions**: {len(test_rows)}")
    lines.append(f"- **Unique games in test**: {results['UniformLegal']['n_games']}\n")

    lines.append("## Overall Performance Comparison\n")
    lines.append("| Predictor | top-1 (mean [lo95,hi95]) | top-3 | NLL (mean [lo95,hi95]) | n_games | n_positions |")
    lines.append("|:---|:---:|:---:|:---:|:---:|:---:|")

    for name in ["UniformLegal", "PositionFrequency", "BookThenModel", "CurrentModel"]:
        r = results[name]
        top1_str = format_ci(r["top1"], r["top1_ci"])
        top3_str = f"{r['top3']:.4f}"
        nll_str = format_ci(r["nll"], r["nll_ci"])
        lines.append(
            f"| {name} | {top1_str} | {top3_str} | {nll_str} | {r['n_games']} | {r['n_positions']} |"
        )

    lines.append("\n## Per-Phase Breakdown (top-1 / NLL)\n")
    lines.append("| Predictor | Opening (ply<=20) | Middlegame (21-60) | Endgame (>60) |")
    lines.append("|:---|:---:|:---:|:---:|")

    for name in ["UniformLegal", "PositionFrequency", "BookThenModel", "CurrentModel"]:
        bp = results[name]["by_phase"]
        p1 = f"{bp['ply<=20']['top1']:.4f} / {bp['ply<=20']['nll']:.4f}"
        p2 = f"{bp['21-60']['top1']:.4f} / {bp['21-60']['nll']:.4f}"
        p3 = f"{bp['>60']['top1']:.4f} / {bp['>60']['nll']:.4f}"
        lines.append(f"| {name} | {p1} | {p2} | {p3} |")

    lines.append("\n## Calibration (ECE)\n")
    cm_ece = results["CurrentModel"]["ece"]
    lines.append(f"- **CurrentModel ECE (10 equal-width bins)**: {cm_ece:.4f}")

    lines.append("\n## By Berserk Split (top-1 / NLL)\n")
    lines.append("| Predictor | Normal (target_berserk=False) | Berserk (target_berserk=True) |")
    lines.append("|:---|:---:|:---:|")

    for name in ["UniformLegal", "PositionFrequency", "BookThenModel", "CurrentModel"]:
        bb = results[name]["by_berserk"]
        b_false = f"{bb['false']['top1']:.4f} / {bb['false']['nll']:.4f}"
        b_true = f"{bb['true']['top1']:.4f} / {bb['true']['nll']:.4f}"
        lines.append(f"| {name} | {b_false} | {b_true} |")

    lines.append("\n## By Clock Ratio Split (top-1 / NLL)\n")
    lines.append("| Predictor | <0.1 | 0.1-0.3 | >0.3 | Unknown |")
    lines.append("|:---|:---:|:---:|:---:|:---:|")

    for name in ["UniformLegal", "PositionFrequency", "BookThenModel", "CurrentModel"]:
        bcr = results[name]["by_clock_ratio"]
        c1 = f"{bcr['<0.1']['top1']:.4f} / {bcr['<0.1']['nll']:.4f}"
        c2 = f"{bcr['0.1-0.3']['top1']:.4f} / {bcr['0.1-0.3']['nll']:.4f}"
        c3 = f"{bcr['>0.3']['top1']:.4f} / {bcr['>0.3']['nll']:.4f}"
        cu = f"{bcr['unknown']['top1']:.4f} / {bcr['unknown']['nll']:.4f}"
        lines.append(f"| {name} | {c1} | {c2} | {c3} | {cu} |")

    lines.append("\n---\n")
    lines.append(f"- `BOOK_EVAL_EPS = {BOOK_EVAL_EPS}`")
    lines.append("- training val loss is smoothed CE, not NLL; not comparable\n")

    report = "\n".join(lines)

    # Save to docs/progress/phaseA.md
    docs_path = backend_dir.parent / "docs" / "progress" / "phaseA.md"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(docs_path, "w", encoding="utf-8") as f:
        f.write(report)

    return report


def evaluate_test_set_legacy(username: str = "the_king_crusher") -> None:
    import matplotlib.pyplot as plt

    p_dir = storage.player_dir(username, create=False)
    samples_path = p_dir / "samples.parquet"

    print("=" * 60)
    print(f"EVALUATING MODEL ON HELD-OUT TEST SPLIT FOR '{username}' (LEGACY MODE)")
    print("=" * 60)

    if not export.is_model_current(username):
        print("!" * 70)
        print(f"WARNING: Model checkpoint for '{username}' is NOT current with samples.parquet!")
        print("!" * 70)

    # 1. Load Model
    model = export.load_player_model(username, device="cpu")
    model.eval()

    # 2. Test DataLoader
    test_dataset = PlayerMoveDataset(str(samples_path), split="test")
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    print(f"Total test positions: {len(test_dataset)}")

    total_loss_sum = 0.0
    from_loss_sum = 0.0
    to_loss_sum = 0.0
    promo_loss_sum = 0.0
    aux_loss_sum = 0.0
    num_batches = 0

    correct_from_top1 = 0
    correct_from_top3 = 0
    correct_to_top1 = 0
    correct_to_top3 = 0
    correct_exact_top1 = 0
    correct_masked_from_top1 = 0
    correct_masked_exact_top1 = 0

    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            outputs = model(board=batch["board"], meta=batch["meta"])
            losses = compute_loss(outputs, batch)

            total_loss_sum += losses["total"].item()
            from_loss_sum += losses["from"].item()
            to_loss_sum += losses["to"].item()
            promo_loss_sum += losses["promo"].item()
            aux_loss_sum += losses["aux"].item()
            num_batches += 1

            from_logits = outputs["from_sq_logits"]
            to_logits = outputs["to_sq_logits"]

            from_targets = batch["from_sq"]
            to_targets = batch["to_sq"]

            pred_from = from_logits.argmax(dim=-1)
            pred_to = to_logits.argmax(dim=-1)

            correct_from_top1 += (pred_from == from_targets).sum().item()
            correct_to_top1 += (pred_to == to_targets).sum().item()
            correct_exact_top1 += ((pred_from == from_targets) & (pred_to == to_targets)).sum().item()

            from_mask = batch["legal_mask"].any(dim=-1)
            masked_from_logits = from_logits.clone()
            masked_from_logits[~from_mask] = float("-inf")
            pred_from_masked = masked_from_logits.argmax(dim=-1)

            b_indices = torch.arange(len(from_targets), device=from_logits.device)
            to_mask = batch["legal_mask"][b_indices, pred_from_masked]
            masked_to_logits = to_logits.clone()
            masked_to_logits[~to_mask] = float("-inf")
            pred_to_masked = masked_to_logits.argmax(dim=-1)

            correct_masked_from_top1 += (pred_from_masked == from_targets).sum().item()
            correct_masked_exact_top1 += (
                (pred_from_masked == from_targets) & (pred_to_masked == to_targets)
            ).sum().item()

            _, top3_from = from_logits.topk(3, dim=-1)
            _, top3_to = to_logits.topk(3, dim=-1)

            correct_from_top3 += (top3_from == from_targets.unsqueeze(-1)).any(dim=-1).sum().item()
            correct_to_top3 += (top3_to == to_targets.unsqueeze(-1)).any(dim=-1).sum().item()

            total_samples += len(from_targets)

    metrics = {
        "test_total_loss": total_loss_sum / num_batches,
        "test_from_loss": from_loss_sum / num_batches,
        "test_to_loss": to_loss_sum / num_batches,
        "test_promo_loss": promo_loss_sum / num_batches,
        "test_aux_loss": aux_loss_sum / num_batches,
        "from_top1_acc": (correct_from_top1 / total_samples) * 100,
        "from_top3_acc": (correct_from_top3 / total_samples) * 100,
        "to_top1_acc": (correct_to_top1 / total_samples) * 100,
        "to_top3_acc": (correct_to_top3 / total_samples) * 100,
        "exact_move_top1_acc": (correct_exact_top1 / total_samples) * 100,
        "masked_from_top1": (correct_masked_from_top1 / total_samples) * 100,
        "masked_exact_top1": (correct_masked_exact_top1 / total_samples) * 100,
        "total_test_positions": total_samples,
    }

    print("\n--- Test Set Evaluation Results ---")
    print(f"Total Test Positions: {metrics['total_test_positions']}")
    print(f"Test Total Loss:      {metrics['test_total_loss']:.4f}")
    print(f"Test From-Square Loss:{metrics['test_from_loss']:.4f}")
    print(f"Test To-Square Loss:  {metrics['test_to_loss']:.4f}")
    print(f"Test Promo Loss:      {metrics['test_promo_loss']:.6f}")
    print(f"Test Aux Loss:        {metrics['test_aux_loss']:.4f}")
    print(f"From-Square Top-1 Acc:{metrics['from_top1_acc']:.2f}%")
    print(f"From-Square Top-3 Acc:{metrics['from_top3_acc']:.2f}%")
    print(f"To-Square Top-1 Acc:  {metrics['to_top1_acc']:.2f}%")
    print(f"To-Square Top-3 Acc:  {metrics['to_top3_acc']:.2f}%")
    print(f"Exact Move Top-1 Acc: {metrics['exact_move_top1_acc']:.2f}%")
    print(f"Masked From Top-1:    {metrics['masked_from_top1']:.2f}%")
    print(f"Masked Exact Top-1:   {metrics['masked_exact_top1']:.2f}%")

    test_metrics_path = p_dir / "test_evaluation_metrics.json"
    with open(test_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved test metrics to {test_metrics_path}")

    # Training Curves Plot
    training_info_path = p_dir / "training_info.json"
    with open(training_info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    history = info["history"]
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_from = [h["train_from_loss"] for h in history]
    train_to = [h["train_to_loss"] for h in history]
    lrs = [h["lr"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, train_loss, 'o-', color='#2563eb', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_loss, 's--', color='#dc2626', label='Val Loss', linewidth=2)

    valid_vals = [(h["epoch"], h["val_loss"]) for h in history if h.get("val_loss") is not None]
    if valid_vals:
        best_epoch, best_val = min(valid_vals, key=lambda x: x[1])
        axes[0].axvline(
            x=best_epoch,
            color='#16a34a',
            linestyle=':',
            label=f'Best Val (Epoch {best_epoch}, {best_val:.4f})',
        )

    axes[0].set_title('Total Loss Progression (Train vs Val)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    lines_labels = axes[0].get_legend_handles_labels()
    axes[0].legend(lines_labels[0], lines_labels[1], fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].plot(epochs, train_from, '^-', color='#7c3aed', label='From-Square Loss', linewidth=2)
    axes[1].plot(epochs, train_to, 'v-', color='#d97706', label='To-Square Loss', linewidth=2)
    axes[1].set_title('Move Component Losses (From vs To)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Loss', fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    axes[2].plot(epochs, lrs, 'd-', color='#059669', label='Learning Rate', linewidth=2)
    axes[2].set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Epoch', fontsize=11)
    axes[2].set_ylabel('LR', fontsize=11)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    chart_path = p_dir / "training_curves.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Saved training curves chart to {chart_path}")

    backup_chart = config.DATA_ROOT / "players_backup" / username / "training_curves.png"
    backup_chart.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(chart_path, backup_chart)
    print(f"Copied chart to backup: {backup_chart}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model on test split")
    parser.add_argument("username", nargs="?", default="the_king_crusher", help="Player username")
    parser.add_argument("--legacy", action="store_true", help="Run legacy evaluation script")
    args = parser.parse_args()

    if args.legacy:
        evaluate_test_set_legacy(args.username)
    else:
        report = run_phase_a_evaluation(args.username)
        print(report)


if __name__ == "__main__":
    main()
