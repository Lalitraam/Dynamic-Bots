"""
Evaluate trained The_King_Crusher model on the held-out test split (4,955 positions)
and generate training curve plots.
"""
import sys
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.ml.dataset import PlayerMoveDataset
from app.ml.model import ChessPolicyNet
from app.ml.losses import compute_loss
from app.ml import export
from app import config
from app.services import storage


def evaluate_test_set():
    username = "the_king_crusher"
    p_dir = storage.player_dir(username)
    model_path = p_dir / "model.pt"
    samples_path = p_dir / "samples.parquet"

    print("=" * 60)
    print("EVALUATING MODEL ON HELD-OUT TEST SPLIT")
    print("=" * 60)

    # 1. Load Model
    model = export.load_model(
        model_path,
        device="cpu",
        channels=config.MODEL_CHANNELS,
        num_res_blocks=config.MODEL_RES_BLOCKS,
        hidden_dim=config.MODEL_HIDDEN_DIM,
        dropout=config.MODEL_DROPOUT,
    )
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

            # Accuracy calculations
            from_logits = outputs["from_sq_logits"]  # (B, 64)
            to_logits = outputs["to_sq_logits"]      # (B, 64)

            from_targets = batch["from_sq"]       # (B,)
            to_targets = batch["to_sq"]           # (B,)

            # Top-1
            pred_from = from_logits.argmax(dim=-1)
            pred_to = to_logits.argmax(dim=-1)

            correct_from_top1 += (pred_from == from_targets).sum().item()
            correct_to_top1 += (pred_to == to_targets).sum().item()
            correct_exact_top1 += ((pred_from == from_targets) & (pred_to == to_targets)).sum().item()

            # Top-3
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

    # Save metrics to json
    test_metrics_path = p_dir / "test_evaluation_metrics.json"
    with open(test_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved test metrics to {test_metrics_path}")

    # 3. Generate Training Curves Plot
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

    # Plot 1: Total Train vs Val Loss
    axes[0].plot(epochs, train_loss, 'o-', color='#2563eb', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_loss, 's--', color='#dc2626', label='Val Loss', linewidth=2)
    axes[0].axvline(x=14, color='#16a34a', linestyle=':', label='Best Val (Epoch 14, 4.9756)')
    axes[0].set_title('Total Loss Progression (Train vs Val)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Plot 2: From vs To Component Losses
    axes[1].plot(epochs, train_from, '^-', color='#7c3aed', label='From-Square Loss', linewidth=2)
    axes[1].plot(epochs, train_to, 'v-', color='#d97706', label='To-Square Loss', linewidth=2)
    axes[1].set_title('Move Component Losses (From vs To)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Loss', fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # Plot 3: Learning Rate
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

    # Also copy chart to backup directory
    backup_chart = config.DATA_ROOT / "players_backup" / username / "training_curves.png"
    import shutil
    shutil.copy2(chart_path, backup_chart)
    print(f"Copied chart to backup: {backup_chart}")


if __name__ == "__main__":
    evaluate_test_set()
