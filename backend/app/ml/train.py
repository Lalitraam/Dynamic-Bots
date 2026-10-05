"""
Training loop for chess policy network (Milestone 3).

Functions
---------
train_model(model, train_loader, val_loader, device, config_overrides=None) -> dict
"""

from __future__ import annotations

import copy
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .. import config
from .losses import compute_loss


def _get_cfg(key: str, default: Any, overrides: dict | None, *aliases: str) -> Any:
    """Retrieve hyperparameter from overrides or fall back to default."""
    if overrides:
        if key in overrides:
            return overrides[key]
        for a in aliases:
            if a in overrides:
                return overrides[a]
    return default


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device | str,
    config_overrides: dict | None = None,
) -> dict[str, Any]:
    """
    Train a ChessPolicyNet model using AdamW and ReduceLROnPlateau with early stopping.

    Parameters
    ----------
    model : nn.Module
        The policy network instance.
    train_loader : DataLoader
        DataLoader for the training split.
    val_loader : DataLoader
        DataLoader for the validation split (may be empty on small fixtures).
    device : torch.device or str
        Compute device ('cuda' or 'cpu').
    config_overrides : dict, optional
        Hyperparameter overrides (e.g. TRAIN_MAX_EPOCHS=2 for tests).

    Returns
    -------
    dict
        {
            "best_val_loss": float,
            "epochs_run": int,
            "history": list[dict],
            "best_state_dict": dict,
        }
    """
    # 1. Reproducibility seed
    seed = _get_cfg("TRAIN_SEED", config.TRAIN_SEED, config_overrides, "seed")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 2. Hyperparameters
    lr = _get_cfg("TRAIN_LR", config.TRAIN_LR, config_overrides, "lr")
    weight_decay = _get_cfg("TRAIN_WEIGHT_DECAY", config.TRAIN_WEIGHT_DECAY, config_overrides, "weight_decay")
    max_epochs = _get_cfg("TRAIN_MAX_EPOCHS", config.TRAIN_MAX_EPOCHS, config_overrides, "epochs", "max_epochs")
    early_stop_patience = _get_cfg(
        "TRAIN_EARLY_STOP_PATIENCE",
        config.TRAIN_EARLY_STOP_PATIENCE,
        config_overrides,
        "early_stop_patience",
    )
    lr_patience = _get_cfg("TRAIN_LR_PATIENCE", config.TRAIN_LR_PATIENCE, config_overrides, "lr_patience")
    lr_factor = _get_cfg("TRAIN_LR_FACTOR", config.TRAIN_LR_FACTOR, config_overrides, "lr_factor")

    # 3. Model setup
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=lr_patience, factor=lr_factor
    )

    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    best_state_dict = copy.deepcopy(model.state_dict())
    patience_counter = 0

    has_val = val_loader is not None and len(val_loader) > 0

    for epoch in range(1, max_epochs + 1):
        # --- Train epoch ---
        model.train()
        train_loss_sum = 0.0
        train_from_sum = 0.0
        train_to_sum = 0.0
        train_promo_sum = 0.0
        train_aux_sum = 0.0
        num_train_batches = 0

        for batch in train_loader:
            batch_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            outputs = model(board=batch_device["board"], meta=batch_device["meta"])
            losses = compute_loss(outputs, batch_device, config_overrides=config_overrides)
            losses["total"].backward()
            optimizer.step()

            train_loss_sum += losses["total"].item()
            train_from_sum += losses["from"].item()
            train_to_sum += losses["to"].item()
            train_promo_sum += losses["promo"].item()
            train_aux_sum += losses["aux"].item()
            num_train_batches += 1

        train_loss = (train_loss_sum / num_train_batches) if num_train_batches > 0 else 0.0
        train_from_loss = (train_from_sum / num_train_batches) if num_train_batches > 0 else 0.0
        train_to_loss = (train_to_sum / num_train_batches) if num_train_batches > 0 else 0.0
        train_promo_loss = (train_promo_sum / num_train_batches) if num_train_batches > 0 else 0.0
        train_aux_loss = (train_aux_sum / num_train_batches) if num_train_batches > 0 else 0.0

        current_lr = optimizer.param_groups[0]["lr"]

        # --- Validation epoch ---
        val_loss: float | None = None
        if has_val:
            model.eval()
            val_loss_sum = 0.0
            num_val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch_device = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                    outputs = model(board=batch_device["board"], meta=batch_device["meta"])
                    losses = compute_loss(outputs, batch_device, config_overrides=config_overrides)
                    val_loss_sum += losses["total"].item()
                    num_val_batches += 1

            val_loss = (val_loss_sum / num_val_batches) if num_val_batches > 0 else 0.0
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_from_loss": train_from_loss,
                "train_to_loss": train_to_loss,
                "train_promo_loss": train_promo_loss,
                "train_aux_loss": train_aux_loss,
                "val_loss": val_loss,
                "lr": current_lr,
            })

            if patience_counter >= early_stop_patience:
                break
        else:
            # Empty val_loader: skip early stopping and scheduler step, track best train loss
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_state_dict = copy.deepcopy(model.state_dict())

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_from_loss": train_from_loss,
                "train_to_loss": train_to_loss,
                "train_promo_loss": train_promo_loss,
                "train_aux_loss": train_aux_loss,
                "val_loss": None,
                "lr": current_lr,
                "warning": "val_loader was empty; skipped val loss and early stopping",
            })

    return {
        "best_val_loss": best_val_loss if has_val else best_train_loss,
        "epochs_run": len(history),
        "history": history,
        "best_state_dict": best_state_dict,
    }
