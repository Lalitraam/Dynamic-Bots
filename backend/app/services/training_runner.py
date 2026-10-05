"""
Training runner service for chess player-cloning (Milestone 3).

Orchestrates:
1. Validating player ingestion and model_tier.
2. Loading datasets (train/val/test).
3. Training the policy network.
4. Saving artifacts: model.pt, training_info.json, training_report.md.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .. import config
from ..ml import export
from ..ml.dataset import PlayerMoveDataset
from ..ml.model import ChessPolicyNet
from ..ml.train import train_model
from . import storage

logger = logging.getLogger(__name__)


def train_player_model(
    username: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Train a CNN policy network for a player and persist model artifacts.

    Parameters
    ----------
    username : str
        Player's username.
    overrides : dict, optional
        Hyperparameter overrides (e.g. TRAIN_MAX_EPOCHS=2 for tests).

    Returns
    -------
    dict
        Summary dictionary matching training_info.json.
    """
    p_dir = storage.player_dir(username)
    info_path = p_dir / "info.json"

    # 1. Validate info.json and model_tier
    if not info_path.exists():
        raise FileNotFoundError(
            f"No ingestion data found for '{username}'. Run ingestion first."
        )

    with open(info_path, "r", encoding="utf-8") as f:
        player_info = json.load(f)

    model_tier = player_info.get("model_tier")
    if model_tier == "rejected":
        raise ValueError(
            f"Player '{username}' has model_tier 'rejected'. Training is not allowed."
        )

    # 2. Locate samples.parquet and construct datasets
    samples_path = p_dir / "samples.parquet"
    if not samples_path.exists():
        raise FileNotFoundError(
            f"samples.parquet not found for '{username}' in {p_dir}."
        )

    train_dataset = PlayerMoveDataset(str(samples_path), split="train")
    val_dataset = PlayerMoveDataset(str(samples_path), split="val")
    # test_dataset = PlayerMoveDataset(str(samples_path), split="test")

    batch_size = (overrides or {}).get("TRAIN_BATCH_SIZE", config.TRAIN_BATCH_SIZE)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # 3. Select compute device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training model for %s on device: %s", username, device)

    # 4. Initialize model and train
    model_kwargs = {
        "channels": (overrides or {}).get("MODEL_CHANNELS", config.MODEL_CHANNELS),
        "num_res_blocks": (overrides or {}).get("MODEL_RES_BLOCKS", config.MODEL_RES_BLOCKS),
        "hidden_dim": (overrides or {}).get("MODEL_HIDDEN_DIM", config.MODEL_HIDDEN_DIM),
        "dropout": (overrides or {}).get("MODEL_DROPOUT", config.MODEL_DROPOUT),
    }
    model = ChessPolicyNet(**model_kwargs).to(device)
    train_results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config_overrides=overrides,
    )

    # 5. Load best weights back into model
    best_state = train_results.get("best_state_dict")
    if best_state is not None:
        model.load_state_dict(best_state)

    # 6. Save model checkpoint
    model_pt_path = p_dir / "model.pt"
    export.save_model(model, model_pt_path)

    # 7. Assemble training_info and training_report
    now_iso = datetime.now(timezone.utc).isoformat()
    training_info: dict[str, Any] = {
        "schema_version": "1.0.0",
        "username": username,
        "model_tier": model_tier,
        "device": str(device),
        "created_at": now_iso,
        "epochs_run": train_results["epochs_run"],
        "best_val_loss": train_results["best_val_loss"],
        "history": train_results["history"],
        "hyperparameters": {
            "batch_size": batch_size,
            "lr": (overrides or {}).get("TRAIN_LR", config.TRAIN_LR),
            "weight_decay": (overrides or {}).get(
                "TRAIN_WEIGHT_DECAY", config.TRAIN_WEIGHT_DECAY
            ),
            "max_epochs": (overrides or {}).get(
                "TRAIN_MAX_EPOCHS", config.TRAIN_MAX_EPOCHS
            ),
        },
    }

    info_out_path = p_dir / "training_info.json"
    with open(info_out_path, "w", encoding="utf-8") as f:
        json.dump(training_info, f, indent=2)

    # Markdown report
    report_lines = [
        f"# Training Report: {username}",
        "",
        f"- **Date**: {now_iso}",
        f"- **Device**: {device}",
        f"- **Model Tier**: {model_tier}",
        f"- **Epochs Run**: {train_results['epochs_run']}",
        f"- **Best Val Loss**: {train_results['best_val_loss']:.4f}"
        if isinstance(train_results["best_val_loss"], float)
        else f"- **Best Val Loss**: {train_results['best_val_loss']}",
        "",
        "## Hyperparameters",
        f"- Batch size: {batch_size}",
        f"- Learning rate: {(overrides or {}).get('TRAIN_LR', config.TRAIN_LR)}",
        f"- Weight decay: {(overrides or {}).get('TRAIN_WEIGHT_DECAY', config.TRAIN_WEIGHT_DECAY)}",
        "",
        "## Epoch History",
    ]
    for h in train_results["history"]:
        vl_str = f"{h['val_loss']:.4f}" if h.get("val_loss") is not None else "N/A"
        report_lines.append(
            f"- Epoch {h['epoch']}: train_loss={h['train_loss']:.4f}, val_loss={vl_str}, lr={h['lr']:.6f}"
        )

    report_path = p_dir / "training_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    return training_info
