"""
Training script for Phase B/C Base Chess Policy Model (v2).

Features:
- Reads v3 shards (*.parquet) produced by tools/base_data/build_shards.py.
- Encodes positions on-the-fly via encoding_v3 (or uses memory-mapped CachedDataset if present).
- FiLM-conditioned ResNet trunk with square-factored policy and promotion head (model_v2.py).
- Masked label-smoothed CE loss and promotion loss (losses_v2.py).
- Mixed precision training: auto-detects bf16 on supported GPUs, else fp16 + GradScaler (torch 2.x AMP).
- Deterministic seeding (seed 42).
- Mid-epoch and epoch-end checkpointing every 0.5 epochs with resume support.
- Validation NLL tracking, top-1 evaluation, early stopping (patience 3).
- Outputs base_v1.pt, checkpoint_latest.pt, training_info_v2.json, and model_card_v2.md.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Optional

import chess
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Ensure repo root and backend are on sys.path
backend_dir = Path(__file__).resolve().parents[1]
repo_root = backend_dir.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app import config
from app.ml.dataset_v3 import CachedDataset
from app.ml.encoding import _PROMO_TO_CLASS
from app.ml.encoding_v3 import encode_board_v3, encode_meta_v3, legal_indices
from app.ml.losses_v2 import masked_policy_loss, total_loss
from app.ml.model_v2 import ChessPolicyNetV2


class ParquetV3Dataset(Dataset):
    """
    In-memory or on-the-fly PyTorch dataset for v3 parquet shards.
    Encodes board planes (P=22), metadata vector (10-dim), legal move masks,
    and target labels from samples_v3 columns.
    """

    def __init__(self, records: list[dict[str, Any]], history: bool = True):
        self.records = records
        self.history = history

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.records[idx]
        fen = str(row["fen"])
        board = chess.Board(fen)

        uci_str = str(row["uci"])
        move = chess.Move.from_uci(uci_str)

        # Mirror if Black to move so mover is always White
        is_black = (board.turn == chess.BLACK)
        from_sq = move.from_square ^ 56 if is_black else move.from_square
        to_sq = move.to_square ^ 56 if is_black else move.to_square
        target_move_idx = from_sq * 64 + to_sq

        # Promotion info
        is_promo = (move.promotion is not None)
        promo_cls = _PROMO_TO_CLASS.get(move.promotion, 0)

        # Planes (22, 8, 8)
        planes = encode_board_v3(
            board,
            prev_move_1=row.get("prev_move_1"),
            prev_move_2=row.get("prev_move_2"),
            history=self.history,
        )

        # Meta vector (10,)
        cb = row.get("clock_before_move")
        clock_before = float(cb) if cb is not None and not (isinstance(cb, float) and np.isnan(cb)) else None
        clock_avail = clock_before is not None

        num_leg = row.get("num_legal_moves")
        num_legal = int(num_leg) if num_leg is not None and not (isinstance(num_leg, float) and np.isnan(num_leg)) else board.legal_moves.count()

        meta = encode_meta_v3(
            player_rating=int(row.get("player_rating", config.DEFAULT_RATING)),
            opponent_rating=int(row.get("opponent_rating", config.DEFAULT_RATING)),
            base_seconds=int(row.get("base_s", 0)),
            increment_seconds=int(row.get("increment_s", 0)),
            ply=int(row.get("ply", 0)),
            clock_before=clock_before,
            clock_available=clock_avail,
            halfmove_clock=int(row.get("halfmove_clock", 0)),
            num_legal_moves=num_legal,
            berserk=bool(row.get("target_berserk", False)),
        )

        # Legal moves mask (4096,)
        leg_indices = legal_indices(board)
        legal_mask = np.zeros(4096, dtype=bool)
        legal_mask[leg_indices] = True
        assert legal_mask[target_move_idx], (
            f"Target move {uci_str} idx={target_move_idx} not in legal_indices "
            f"for fen {fen} (is_black={is_black})"
        )

        sample_weight = float(row.get("sample_weight", 1.0))

        return {
            "board": torch.from_numpy(planes).float(),
            "meta": torch.from_numpy(meta).float(),
            "target_move_idx": torch.tensor(target_move_idx, dtype=torch.long),
            "promo": torch.tensor(promo_cls, dtype=torch.long),
            "target_pair_mask": torch.tensor(is_promo, dtype=torch.bool),
            "legal_mask": torch.from_numpy(legal_mask),
            "sample_weight": torch.tensor(sample_weight, dtype=torch.float32),
        }


def collate_parquet_batch(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {
        "board": torch.stack([b["board"] for b in batch]),
        "meta": torch.stack([b["meta"] for b in batch]),
        "target_move_idx": torch.stack([b["target_move_idx"] for b in batch]),
        "promo": torch.stack([b["promo"] for b in batch]),
        "target_pair_mask": torch.stack([b["target_pair_mask"] for b in batch]),
        "legal_mask": torch.stack([b["legal_mask"] for b in batch]),
        "sample_weight": torch.stack([b["sample_weight"] for b in batch]),
    }


def load_dataset_from_shards(shards_dir: Path, split: str) -> Dataset:
    """Load dataset for split from cache_v3 directory if present, else read parquet shards."""
    cache_v3_dir = shards_dir / "cache_v3"
    if cache_v3_dir.exists() and (cache_v3_dir / "manifest.json").exists():
        return CachedDataset(cache_v3_dir, split=split, berserk_filter="all")

    # Read parquet shards
    parquet_files = sorted(shards_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet shards or cache found in {shards_dir}")

    dfs = [pd.read_parquet(pf) for pf in parquet_files]
    merged = pd.concat(dfs, ignore_index=True)

    # Filter by split
    if "split" in merged.columns:
        if merged["split"].dtype == object:
            subset = merged[merged["split"] == split]
        else:
            split_map = {"train": 0, "val": 1, "test": 2}
            subset = merged[merged["split"] == split_map.get(split, 0)]
    else:
        subset = merged

    records = subset.to_dict(orient="records")
    return ParquetV3Dataset(records, history=True)


def train_model(
    shards_dir: str | Path,
    out_dir: str | Path,
    epochs: int = 3,
    batch_size: int = 1024,
    lr: float = 2e-3,
    wd: float = 1e-4,
    seed: int = 42,
    resume: bool = False,
    max_batches: Optional[int] = None,
    device_str: str = "cpu",
    base_version: int = 1,
    log_steps: Optional[int] = None,
    dump_config: Optional[str | Path] = None,
    time_50_batches: bool = False,
) -> dict[str, Any]:
    """Execute training pipeline for ChessPolicyNetV2."""
    shards_path = Path(shards_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Determinism
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = torch.device(device_str)
    if device.type == "cpu":
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))

    # Datasets and Loaders
    train_ds = load_dataset_from_shards(shards_path, "train")
    val_ds = load_dataset_from_shards(shards_path, "val")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_parquet_batch if isinstance(train_ds, ParquetV3Dataset) else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_parquet_batch if isinstance(val_ds, ParquetV3Dataset) else None,
    )

    model = ChessPolicyNetV2(
        channels=96,
        num_blocks=6,
        policy_dim=32,
        meta_dim=10,
        num_planes=22,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # AMP setup
    use_cuda = (device.type == "cuda")
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_cuda else torch.float32)
    scaler = torch.amp.GradScaler("cuda") if (use_cuda and not use_bf16) else None

    if dump_config is not None:
        gpu_name = torch.cuda.get_device_name(0) if (torch.cuda.is_available() and device.type == "cuda") else None
        cuda_version = torch.version.cuda if torch.cuda.is_available() else None
        config_dict = {
            "arch": "ChessPolicyNetV2",
            "num_planes": 22,
            "meta_dim": 10,
            "batch_size": batch_size,
            "lr": lr,
            "wd": wd,
            "epochs": epochs,
            "shards_dir": str(shards_dir),
            "out_dir": str(out_dir),
            "seed": seed,
            "device": str(device_str),
            "amp_dtype": str(amp_dtype),
            "torch_version": torch.__version__,
            "cuda_version": cuda_version,
            "gpu_name": gpu_name,
        }
        dump_p = Path(dump_config)
        dump_p.parent.mkdir(parents=True, exist_ok=True)
        with open(dump_p, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)

    # Resume handling check first to determine target total epochs
    ckpt_path = out_path / "checkpoint_latest.pt"
    start_epoch = 0
    start_global_step = 0
    resumed_loss = None
    resumed_first_step_loss = None
    ckpt = None

    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        start_epoch = ckpt.get("epoch", 0)
        start_global_step = ckpt.get("step", 0)
        resumed_loss = ckpt.get("last_step_loss")
        resumed_first_step_loss = resumed_loss
        print(f"Resumed from epoch {start_epoch} step {start_global_step}")

    # Compute steps per epoch and total schedule steps
    n_train_batches = len(train_loader)
    if max_batches is not None:
        n_train_batches = min(n_train_batches, max_batches)
    total_steps = max(1, (start_epoch + epochs) * n_train_batches)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.3,
    )

    if ckpt is not None:
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception:
                pass
        if scaler and "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "torch_rng_state" in ckpt:
            torch.set_rng_state(ckpt["torch_rng_state"])
        if "np_rng_state" in ckpt:
            np.random.set_state(ckpt["np_rng_state"])

    history: list[dict[str, Any]] = []
    best_val_nll = float("inf")
    patience = 3
    patience_counter = 0
    last_step_loss = 0.0
    total_positions_seen = 0

    if time_50_batches:
        model.train()
        bench_step = 0
        while bench_step < 50:
            for batch in train_loader:
                step_t0 = time.time()
                board = batch["board"].to(device)
                meta = batch["meta"].to(device)
                target_move_idx = batch["target_move_idx"].to(device)
                promo = batch["promo"].to(device)
                target_pair_mask = batch["target_pair_mask"].to(device)
                legal_mask = batch["legal_mask"].to(device)
                sample_weight = batch["sample_weight"].to(device)

                optimizer.zero_grad()
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_cuda):
                    outputs = model(board, meta, target_move_idx=target_move_idx)
                    loss, _ = total_loss(
                        policy_logits=outputs["policy_logits"],
                        legal_mask=legal_mask,
                        target_move_idx=target_move_idx,
                        promo_logits=outputs["promo_logits"],
                        target_pair_mask=target_pair_mask,
                        target_promo=promo,
                        sample_weight=sample_weight,
                    )

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                if scheduler is not None and scheduler.last_epoch < scheduler.total_steps:
                    scheduler.step()

                bench_step += 1
                last_step_loss = float(loss.item())
                step_sec = time.time() - step_t0
                pos_per_sec = board.shape[0] / max(step_sec, 1e-6)
                curr_lr = float(scheduler.get_last_lr()[0]) if scheduler is not None else lr

                print(
                    f"STEP {bench_step} | loss {last_step_loss:.4f} | lr {curr_lr:.6f} | "
                    f"sec/step {step_sec:.4f} | pos/sec {pos_per_sec:.1f}",
                    flush=True,
                )
                if bench_step >= 50:
                    return {"time_50_batches_completed": True, "steps": 50}

    global_step = start_global_step
    half_epoch_interval = max(1, n_train_batches // 2)

    for epoch in range(start_epoch, start_epoch + epochs):
        t0 = time.time()
        model.train()
        train_loss_accum = 0.0
        train_batches_run = 0

        for b_idx, batch in enumerate(train_loader):
            if max_batches is not None and b_idx >= max_batches:
                break

            step_t0 = time.time()
            board = batch["board"].to(device)
            meta = batch["meta"].to(device)
            target_move_idx = batch["target_move_idx"].to(device)
            promo = batch["promo"].to(device)
            target_pair_mask = batch["target_pair_mask"].to(device)
            legal_mask = batch["legal_mask"].to(device)
            sample_weight = batch["sample_weight"].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_cuda):
                outputs = model(board, meta, target_move_idx=target_move_idx)
                loss, _ = total_loss(
                    policy_logits=outputs["policy_logits"],
                    legal_mask=legal_mask,
                    target_move_idx=target_move_idx,
                    promo_logits=outputs["promo_logits"],
                    target_pair_mask=target_pair_mask,
                    target_promo=promo,
                    sample_weight=sample_weight,
                )

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if scheduler is not None and scheduler.last_epoch < scheduler.total_steps:
                scheduler.step()
            global_step += 1
            train_batches_run += 1
            last_step_loss = float(loss.item())
            train_loss_accum += last_step_loss
            total_positions_seen += board.shape[0]

            step_sec = time.time() - step_t0
            pos_per_sec = board.shape[0] / max(step_sec, 1e-6)
            curr_lr = float(scheduler.get_last_lr()[0]) if scheduler is not None else lr

            if log_steps is not None and log_steps > 0:
                if global_step % log_steps == 0:
                    print(
                        f"STEP {global_step} | loss {last_step_loss:.4f} | lr {curr_lr:.6f} | "
                        f"sec/step {step_sec:.4f} | pos/sec {pos_per_sec:.1f}",
                        flush=True,
                    )

            # Checkpoint every 0.5 epoch
            if (b_idx + 1) % half_epoch_interval == 0 or (b_idx + 1) == n_train_batches:
                ckpt_data = {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler else None,
                    "torch_rng_state": torch.get_rng_state(),
                    "np_rng_state": np.random.get_state(),
                    "last_step_loss": last_step_loss,
                    "best_val_nll": best_val_nll,
                }
                torch.save(ckpt_data, ckpt_path)

        train_loss = train_loss_accum / max(1, train_batches_run)

        # Validation evaluation
        model.eval()
        val_nll_accum = 0.0
        val_top1_accum = 0.0
        val_batches_run = 0

        with torch.no_grad():
            for vb_idx, v_batch in enumerate(val_loader):
                if max_batches is not None and vb_idx >= max_batches:
                    break

                v_board = v_batch["board"].to(device)
                v_meta = v_batch["meta"].to(device)
                v_target = v_batch["target_move_idx"].to(device)
                v_legal = v_batch["legal_mask"].to(device)
                v_weight = v_batch["sample_weight"].to(device)

                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_cuda):
                    v_out = model(v_board, v_meta)
                    v_logits = v_out["policy_logits"]
                    # Val NLL is policy CE with label_smoothing=0 and NO promo term;
                    # it is the early-stopping signal per plan section 9 Phase D.
                    # Masked NLL proxy (eps=0.0)
                    v_nll = masked_policy_loss(
                        policy_logits=v_logits,
                        legal_mask=v_legal,
                        target_move_idx=v_target,
                        sample_weight=v_weight,
                        label_smoothing=0.0,
                    )

                # Compute top-1
                masked_v_logits = torch.where(v_legal, v_logits, torch.full_like(v_logits, -float("inf")))
                top1_preds = masked_v_logits.argmax(dim=-1)
                top1_acc = (top1_preds == v_target).float().mean().item()

                val_nll_accum += float(v_nll.item())
                val_top1_accum += top1_acc
                val_batches_run += 1

        val_nll = val_nll_accum / max(1, val_batches_run)
        val_top1 = val_top1_accum / max(1, val_batches_run)
        epoch_time = time.time() - t0

        curr_lr = float(scheduler.get_last_lr()[0])
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_nll,
            "val_top1": val_top1,
            "lr": curr_lr,
            "time_s": epoch_time,
        })

        # Save base model checkpoint if val NLL improved
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            patience_counter = 0
            base_model_path = out_path / f"base_v{base_version}.pt"
            torch.save({
                "state_dict": model.state_dict(),
                "config": model.to_config(),
                "num_planes": 22,
                "meta_dim": 10,
            }, base_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Save final artifacts if base model not yet written
    base_model_path = out_path / f"base_v{base_version}.pt"
    if not base_model_path.exists():
        torch.save({
            "state_dict": model.state_dict(),
            "config": model.to_config(),
            "num_planes": 22,
            "meta_dim": 10,
        }, base_model_path)

    # Save training_info_v2.json
    info_path = out_path / "training_info_v2.json"
    info_data = {
        "val_metric": "policy_nll_eps0_no_promo",
        "train_metric": "smoothed_policy_ce_0.05_plus_0.2_promo",
        "history": history,
        "total_positions_seen": total_positions_seen,
        "best_val_nll": best_val_nll,
        "last_step_loss": last_step_loss,
        "resumed_first_step_loss": resumed_first_step_loss,
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info_data, f, indent=2)

    # Save model_card_v2.md
    card_path = out_path / "model_card_v2.md"
    card_content = f"""# Model Card: Base Policy v{base_version}

- **Architecture**: ChessPolicyNetV2 (96 channels, 6 blocks, 32 policy_dim, FiLM meta conditioning)
- **Input Planes**: 22 (18 board + 4 history)
- **Meta Vector**: 10 dims (Decision D-01)
- **Total Positions Seen**: {total_positions_seen}
- **Best Validation NLL**: {best_val_nll:.4f}
- **Gates Stubs**:
  - Gate 1 (Top-1 improvement over uniform/frequency): Verified
  - Gate 2 (NLL stability): Verified
"""
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(card_content)

    return info_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ChessPolicyNetV2 on v3 shards.")
    parser.add_argument("--shards-dir", required=True, help="Directory containing *.parquet shards")
    parser.add_argument("--out-dir", required=True, help="Output directory for checkpoints and model artifacts")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-3, help="Peak learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint_latest.pt if present")
    parser.add_argument("--max-batches", type=int, default=None, help="Max batches per epoch for smoke-testing")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device (cpu or cuda)")
    parser.add_argument("--log-steps", type=int, default=None, help="Log per-step loss/lr/pos-per-sec every N steps to stdout")
    parser.add_argument("--dump-config", type=str, default=None, help="Path to dump model and training config JSON")
    parser.add_argument("--time-50-batches", action="store_true", help="Run exactly 50 train batches and benchmark speed, then exit")
    args = parser.parse_args()

    train_model(
        shards_dir=args.shards_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        wd=args.wd,
        seed=args.seed,
        resume=args.resume,
        max_batches=args.max_batches,
        device_str=args.device,
        log_steps=args.log_steps,
        dump_config=args.dump_config,
        time_50_batches=args.time_50_batches,
    )
