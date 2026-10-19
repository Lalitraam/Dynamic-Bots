"""
Temperature fitting module for fine-tuned chess imitation models.
Calibrates move prediction probabilities by grid-searching temperature T on validation NLL.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .. import config
from .losses_v2 import masked_policy_loss


def _prepare_val_dataset(val_parquet: Any) -> Dataset:
    """Helper to convert various input formats into a PyTorch Dataset."""
    if isinstance(val_parquet, Dataset):
        return val_parquet

    # Local import to prevent circular dependencies
    from scripts.train_v2 import ParquetV3Dataset

    if isinstance(val_parquet, list):
        return ParquetV3Dataset(val_parquet, history=True)

    if isinstance(val_parquet, pd.DataFrame):
        return ParquetV3Dataset(val_parquet.to_dict(orient="records"), history=True)

    p = Path(val_parquet)
    if p.is_dir():
        from scripts.train_v2 import load_dataset_from_shards
        return load_dataset_from_shards(p, split="val")

    df = pd.read_parquet(p)
    if "split" in df.columns:
        is_string_split = (
            df["split"].dtype == object
            or pd.api.types.is_string_dtype(df["split"])
            or (not df["split"].empty and isinstance(df["split"].dropna().iloc[0], str))
        )
        if is_string_split:
            df = df[df["split"] == "val"]
        else:
            df = df[df["split"] == 1]
    return ParquetV3Dataset(df.to_dict(orient="records"), history=True)


def fit_temperature(
    model: torch.nn.Module,
    val_parquet: Any,
    out_path: Path | str | None = None,
    device: str = "cpu",
    temperature_grid: Sequence[float] | None = None,
    batch_size: int = 256,
) -> dict[str, Any]:
    """
    Grid search over TEMPERATURE_GRID minimising validation NLL.

    Parameters
    ----------
    model : torch.nn.Module
        Trained or fine-tuned ChessPolicyNet model.
    val_parquet : Path | str | DataFrame | list | Dataset
        Validation data source.
    out_path : Path | str, optional
        Path to save temperature.json.
    device : str
        Execution device ('cpu' or 'cuda').
    temperature_grid : Sequence[float], optional
        Grid of temperatures to evaluate (defaults to config.TEMPERATURE_GRID).
    batch_size : int
        Batch size for forward evaluation.

    Returns
    -------
    dict
        {
            "T_base": float,
            "T_best": float,
            "val_nll_before": float,
            "val_nll_after": float,
            "grid": dict[str, float]
        }
    """
    if temperature_grid is None:
        temperature_grid = getattr(
            config,
            "TEMPERATURE_GRID",
            [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        )

    model.eval()
    dev = torch.device(device)
    model.to(dev)

    if isinstance(val_parquet, DataLoader):
        val_loader = val_parquet
    else:
        val_ds = _prepare_val_dataset(val_parquet)
        from scripts.train_v2 import collate_parquet_batch, ParquetV3Dataset
        collate_fn = collate_parquet_batch if isinstance(val_ds, ParquetV3Dataset) else None
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

    # 1. Collect model predictions and targets in memory for rapid grid search
    cached_batches: list[dict[str, torch.Tensor]] = []
    with torch.no_grad():
        for batch in val_loader:
            b_board = batch["board"].to(dev)
            b_meta = batch["meta"].to(dev)
            b_target = batch["target_move_idx"].to(dev)
            b_legal = batch["legal_mask"].to(dev)
            b_weight = batch["sample_weight"].to(dev)

            out = model(b_board, b_meta)
            logits = out["policy_logits"]

            cached_batches.append({
                "logits": logits,
                "legal_mask": b_legal,
                "target_move_idx": b_target,
                "sample_weight": b_weight,
            })

    # 2. Evaluate NLL over each temperature in the grid
    grid: dict[float, float] = {}
    for T in temperature_grid:
        T_float = float(T)
        total_loss = 0.0
        total_weight = 0.0

        for b in cached_batches:
            scaled_logits = b["logits"] / T_float
            loss = masked_policy_loss(
                policy_logits=scaled_logits,
                legal_mask=b["legal_mask"],
                target_move_idx=b["target_move_idx"],
                sample_weight=b["sample_weight"],
                label_smoothing=0.0,
            )
            w_sum = float(b["sample_weight"].sum().item())
            total_loss += float(loss.item()) * w_sum
            total_weight += w_sum

        avg_nll = total_loss / max(total_weight, 1e-8)
        grid[T_float] = avg_nll

    best_T = min(grid.keys(), key=lambda t: grid[t])
    val_nll_before = grid.get(1.0, grid[best_T])
    val_nll_after = grid[best_T]

    res = {
        "T_base": float(best_T),
        "T_best": float(best_T),
        "val_nll_before": float(val_nll_before),
        "val_nll_after": float(val_nll_after),
        "grid": {str(k): float(v) for k, v in grid.items()},
    }

    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)

    return res
