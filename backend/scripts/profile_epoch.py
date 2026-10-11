"""
Profile training epoch operations and verify BatchNorm statistics staleness.

Phase 0, Task 0.3 (evidence collection).

Usage:
    python scripts/profile_epoch.py [username] [--profile-only | --bn-check-only | --all]
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app import config
from app.ml import export
from app.ml.dataset import PlayerMoveDataset
from app.ml.losses import compute_loss
from app.services import storage


def run_profile_epoch(username: str = "the_king_crusher", target_batches: int = 200, device_str: str | None = None) -> dict:
    """
    STEP 3: Profile data fetch, forward+loss, and backward+optimizer.step separately.
    Works on a copy.deepcopy of the player model so checkpoint is never altered.
    """
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    p_dir = storage.player_dir(username, create=False)
    samples_path = p_dir / "samples.parquet"
    if not samples_path.exists():
        raise FileNotFoundError(f"samples.parquet not found for '{username}' in {p_dir}")

    print("=" * 70)
    print(f"STEP 3: PROFILING TRAINING EPOCH FOR '{username}'")
    print(f"Device: {device}")
    print("=" * 70)

    # 1. Load original model and create deepcopy
    orig_model = export.load_player_model(username, device=device)
    model = copy.deepcopy(orig_model)
    model.train()

    # 2. Setup optimizer exactly as training runner
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.TRAIN_LR,
        weight_decay=config.TRAIN_WEIGHT_DECAY,
    )

    # 3. Setup train DataLoader exactly as training runner
    train_dataset = PlayerMoveDataset(str(samples_path), split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    actual_batches = min(target_batches, len(train_loader))
    print(f"Total training positions: {len(train_dataset)}")
    print(f"Total training batches in loader: {len(train_loader)} (batch_size={config.TRAIN_BATCH_SIZE})")
    print(f"Batches to profile: {actual_batches} of {len(train_loader)}")
    print("-" * 70)

    data_fetch_times: list[float] = []
    forward_loss_times: list[float] = []
    backward_step_times: list[float] = []

    train_iter = iter(train_loader)

    for i in range(actual_batches):
        # (a) data fetch: time spent waiting on next(loader iterator)
        t0 = time.perf_counter()
        batch = next(train_iter)
        t1 = time.perf_counter()
        data_fetch_times.append(t1 - t0)

        # (b) forward + compute_loss
        batch_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        outputs = model(board=batch_device["board"], meta=batch_device["meta"])
        losses = compute_loss(outputs, batch_device)
        total_loss = losses["total"]
        t2 = time.perf_counter()
        forward_loss_times.append(t2 - t1)

        # (c) backward + optimizer.step
        total_loss.backward()
        optimizer.step()
        t3 = time.perf_counter()
        backward_step_times.append(t3 - t2)

    total_data_time = sum(data_fetch_times)
    total_fwd_time = sum(forward_loss_times)
    total_bwd_time = sum(backward_step_times)
    total_time = total_data_time + total_fwd_time + total_bwd_time

    mean_data_time = total_data_time / actual_batches
    mean_fwd_time = total_fwd_time / actual_batches
    mean_bwd_time = total_bwd_time / actual_batches
    mean_total_time = total_time / actual_batches

    frac_data = total_data_time / total_time if total_time > 0 else 0.0
    frac_fwd = total_fwd_time / total_time if total_time > 0 else 0.0
    frac_bwd = total_bwd_time / total_time if total_time > 0 else 0.0

    print("PROFILING RESULTS ({} batches):".format(actual_batches))
    print(f"{'Component':<30} | {'Mean (s/batch)':<16} | {'Fraction of Total':<18} | {'Total Time (s)':<14}")
    print("-" * 86)
    print(f"{'(a) data fetch':<30} | {mean_data_time:<16.6f} | {frac_data * 100:>15.2f}% | {total_data_time:<14.4f}")
    print(f"{'(b) forward + compute_loss':<30} | {mean_fwd_time:<16.6f} | {frac_fwd * 100:>15.2f}% | {total_fwd_time:<14.4f}")
    print(f"{'(c) backward + optimizer.step':<30} | {mean_bwd_time:<16.6f} | {frac_bwd * 100:>15.2f}% | {total_bwd_time:<14.4f}")
    print("-" * 86)
    print(f"{'Total (all components)':<30} | {mean_total_time:<16.6f} | {100.0:>15.2f}% | {total_time:<14.4f}")
    print("=" * 70)

    return {
        "batches_profiled": actual_batches,
        "mean_data_fetch_sec": mean_data_time,
        "mean_forward_loss_sec": mean_fwd_time,
        "mean_backward_step_sec": mean_bwd_time,
        "mean_total_sec": mean_total_time,
        "frac_data_fetch": frac_data,
        "frac_forward_loss": frac_fwd,
        "frac_backward_step": frac_bwd,
        "total_data_fetch_sec": total_data_time,
        "total_forward_loss_sec": total_fwd_time,
        "total_backward_step_sec": total_bwd_time,
        "total_sec": total_time,
    }


def check_batchnorm_staleness(username: str = "the_king_crusher", device_str: str | None = None) -> dict:
    """
    STEP 4: BatchNorm staleness check.
    On copy.deepcopy of best checkpoint:
    (i) eval() mode, compute mean val loss over val loader with compute_loss (weighted "total");
    (ii) put ONLY BatchNorm modules in train(), reset running stats (reset_running_stats()),
         do one pass over TRAIN loader under torch.no_grad(), return whole model to eval(), recompute val loss.
    Never saves the modified copy.
    """
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    p_dir = storage.player_dir(username, create=False)
    samples_path = p_dir / "samples.parquet"
    if not samples_path.exists():
        raise FileNotFoundError(f"samples.parquet not found for '{username}' in {p_dir}")

    print("=" * 70)
    print(f"STEP 4: BATCHNORM STALENESS CHECK FOR '{username}'")
    print(f"Device: {device}")
    print("=" * 70)

    # 1. Load original checkpoint and make deepcopy
    orig_model = export.load_player_model(username, device=device)
    model = copy.deepcopy(orig_model)

    val_dataset = PlayerMoveDataset(str(samples_path), split="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    train_dataset = PlayerMoveDataset(str(samples_path), split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    print(f"Val positions: {len(val_dataset)} ({len(val_loader)} batches)")
    print(f"Train positions: {len(train_dataset)} ({len(train_loader)} batches)")

    # (i) eval() mode: compute mean val loss over val loader
    model.eval()
    val_loss_sum_initial = 0.0
    num_val_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            batch_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            outputs = model(board=batch_device["board"], meta=batch_device["meta"])
            losses = compute_loss(outputs, batch_device)
            val_loss_sum_initial += losses["total"].item()
            num_val_batches += 1

    val_loss_initial = val_loss_sum_initial / num_val_batches if num_val_batches > 0 else 0.0
    print(f"Initial val loss (checkpoint BN stats): {val_loss_initial:.6f}")

    # (ii) Put ONLY BatchNorm modules in train(), reset running stats
    bn_count = 0
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.train()
            m.reset_running_stats()
            bn_count += 1
        else:
            m.eval()

    print(f"Put {bn_count} BatchNorm modules in train() and reset running statistics.")
    print("Running one pass over the TRAIN loader under torch.no_grad() to recompute BN stats...")

    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in train_loader:
            batch_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            model(board=batch_device["board"], meta=batch_device["meta"])
    t_pass = time.perf_counter() - t0
    print(f"Pass over train loader finished in {t_pass:.2f}s.")

    # Return whole model to eval() and recompute val loss
    model.eval()
    val_loss_sum_updated = 0.0
    num_val_batches_updated = 0
    with torch.no_grad():
        for batch in val_loader:
            batch_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            outputs = model(board=batch_device["board"], meta=batch_device["meta"])
            losses = compute_loss(outputs, batch_device)
            val_loss_sum_updated += losses["total"].item()
            num_val_batches_updated += 1

    val_loss_updated = val_loss_sum_updated / num_val_batches_updated if num_val_batches_updated > 0 else 0.0
    diff = val_loss_updated - val_loss_initial
    abs_diff = abs(diff)

    print("-" * 70)
    print(f"Initial val loss (checkpoint BN stats):          {val_loss_initial:.6f}")
    print(f"Recomputed val loss (re-accumulated BN stats):   {val_loss_updated:.6f}")
    print(f"Difference (recomputed - initial):               {diff:+.6f} nats")
    print(f"Absolute difference:                             {abs_diff:.6f} nats")
    if abs_diff > 0.5:
        print("WARNING: Difference > 0.5 nats indicates STALE BatchNorm statistics.")
    else:
        print("STATUS: Difference <= 0.5 nats indicates FRESH / STABLE BatchNorm statistics.")
    print("=" * 70)

    # Confirm that original checkpoint was never saved or overwritten
    del model
    del orig_model

    return {
        "val_loss_initial": val_loss_initial,
        "val_loss_updated": val_loss_updated,
        "difference": diff,
        "abs_difference": abs_diff,
        "stale_bn": abs_diff > 0.5,
    }


def main():
    parser = argparse.ArgumentParser(description="Profile epoch training operations & check BatchNorm staleness.")
    parser.add_argument("username", nargs="?", default="the_king_crusher", help="Player username")
    parser.add_argument("--profile-only", action="store_true", help="Only run Step 3 profiling")
    parser.add_argument("--bn-check-only", action="store_true", help="Only run Step 4 BatchNorm staleness check")
    parser.add_argument("--device", default=None, help="Compute device ('cpu' or 'cuda')")
    parser.add_argument("--batches", type=int, default=200, help="Number of batches to profile for Step 3")
    args = parser.parse_args()

    if args.profile_only:
        run_profile_epoch(args.username, target_batches=args.batches, device_str=args.device)
    elif args.bn_check_only:
        check_batchnorm_staleness(args.username, device_str=args.device)
    else:
        # Default: run both Step 3 and Step 4
        run_profile_epoch(args.username, target_batches=args.batches, device_str=args.device)
        print()
        check_batchnorm_staleness(args.username, device_str=args.device)


if __name__ == "__main__":
    main()
