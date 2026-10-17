"""
Fine-tuning module for ChessPolicyNetV2 (Phase D).
Implements variant selection, parameter freezing, end-to-end fine-tuning pipeline,
and temperature calibration.
"""
from __future__ import annotations

import copy
from enum import Enum
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .. import config
from .encoding_v3 import encode_board_v3, encode_meta_v3
from .losses_v2 import masked_policy_loss, total_loss
from .model_v2 import ChessPolicyNetV2
from .temperature_fit import fit_temperature


_PROMO_TO_CLASS = {
    None: 0,
    5: 0,  # QUEEN
    2: 1,  # KNIGHT
    3: 2,  # BISHOP
    4: 3,  # ROOK
}


class FineTuneVariant(Enum):
    FULL = "full"
    FREEZE = "freeze"
    FILM_ONLY = "film_only"


def select_variant(n_games: int, n_val_games: int) -> FineTuneVariant:
    """Determine fine-tuning variant based on available game counts.

    n_games >= 800  -> FULL
    150 <= n_games < 800 -> FREEZE
    50 <= n_games < 150 -> FILM_ONLY
    n_games < 50 -> raise ValueError
    If n_val_games < FT_MIN_VAL_GAMES -> override to FILM_ONLY.
    """
    if n_games < 50:
        raise ValueError(f"Game count {n_games} is below minimum requirement 50")

    min_val_games = getattr(config, "FT_MIN_VAL_GAMES", 10)
    if n_val_games < min_val_games:
        return FineTuneVariant.FILM_ONLY

    if n_games >= 800:
        return FineTuneVariant.FULL
    elif n_games >= 150:
        return FineTuneVariant.FREEZE
    else:
        return FineTuneVariant.FILM_ONLY


def set_frozen_bn_eval(model: ChessPolicyNetV2, variant: FineTuneVariant) -> None:
    """Set frozen BatchNorm layers to .eval() mode."""
    if variant == FineTuneVariant.FULL:
        return

    cutoff = getattr(config, "FT_FREEZE_UNTIL_BLOCK", 3)
    if variant == FineTuneVariant.FREEZE:
        if hasattr(model, "stem_bn"):
            model.stem_bn.eval()
        for i in range(min(cutoff, len(model.blocks))):
            model.blocks[i].bn1.eval()
            model.blocks[i].bn2.eval()
    elif variant == FineTuneVariant.FILM_ONLY:
        if hasattr(model, "stem_bn"):
            model.stem_bn.eval()
        for block in model.blocks:
            block.bn1.eval()
            block.bn2.eval()


def freeze_for_variant(model: ChessPolicyNetV2, variant: FineTuneVariant) -> list[str]:
    """Freeze parameters per variant. Returns frozen param names.

    FULL: nothing frozen.
    FREEZE: freeze stem_conv, stem_bn, blocks[0..2], film_generators[0..2], meta_mlp.
            Set frozen BatchNorm layers to .eval() permanently.
    FILM_ONLY: freeze everything except film_generators, bias_pair, promo_head.
    """
    # Reset all parameters to trainable first
    for p in model.parameters():
        p.requires_grad = True

    if variant == FineTuneVariant.FULL:
        return []

    frozen_names: list[str] = []
    cutoff = getattr(config, "FT_FREEZE_UNTIL_BLOCK", 3)

    if variant == FineTuneVariant.FREEZE:
        for name, param in model.named_parameters():
            is_frozen = False
            if (
                name.startswith("stem_conv.")
                or name.startswith("stem_bn.")
                or name.startswith("meta_mlp.")
            ):
                is_frozen = True
            else:
                for b_i in range(cutoff):
                    if name.startswith(f"blocks.{b_i}.") or name.startswith(f"film_generators.{b_i}."):
                        is_frozen = True
                        break

            if is_frozen:
                param.requires_grad = False
                frozen_names.append(name)
            else:
                param.requires_grad = True

    elif variant == FineTuneVariant.FILM_ONLY:
        for name, param in model.named_parameters():
            if (
                name.startswith("film_generators.")
                or name == "bias_pair"
                or name.startswith("promo_head.")
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False
                frozen_names.append(name)

    # Set frozen BatchNorm layers to .eval() permanently
    set_frozen_bn_eval(model, variant)
    return frozen_names


def _resolve_dataset(source: Any, split_name: str = "train") -> Dataset:
    """Load or wrap a dataset from a Path, DataFrame, list, or existing Dataset."""
    if isinstance(source, Dataset):
        return source

    from scripts.train_v2 import ParquetV3Dataset, load_dataset_from_shards

    if isinstance(source, list):
        return ParquetV3Dataset(source, history=True)

    if isinstance(source, pd.DataFrame):
        return ParquetV3Dataset(source.to_dict(orient="records"), history=True)

    p = Path(source)
    if p.is_dir():
        return load_dataset_from_shards(p, split=split_name)

    df = pd.read_parquet(p)
    if "split" in df.columns:
        if df["split"].dtype == object:
            df = df[df["split"] == split_name]
        else:
            split_map = {"train": 0, "val": 1, "test": 2}
            df = df[df["split"] == split_map.get(split_name, 0)]
    return ParquetV3Dataset(df.to_dict(orient="records"), history=True)


class ModelPredictor:
    """Predictor wrapper for evaluate() conforming to Predictor protocol."""

    def __init__(
        self,
        model: ChessPolicyNetV2,
        device: str = "cpu",
        temperature: float = 1.0,
        median_rating: int = 1500,
    ):
        self.model = model
        self.device = device
        self.temperature = max(1e-4, float(temperature))
        self.median_rating = median_rating
        self.model.eval()

    @torch.no_grad()
    def legal_log_probs(self, board: Any, ctx: dict) -> dict[str, float]:
        if not board.legal_moves:
            raise ValueError(f"No legal moves in position {board.fen()}")

        import chess
        is_black = (board.turn == chess.BLACK)
        planes_arr = encode_board_v3(
            board,
            prev_move_1=ctx.get("prev_move_1"),
            prev_move_2=ctx.get("prev_move_2"),
            history=True,
        )

        opp_rating = int(ctx.get("opponent_rating", config.DEFAULT_RATING))
        base_s = int(ctx.get("base_seconds", 0))
        inc_s = int(ctx.get("increment_seconds", 0))
        ply = int(ctx.get("ply", 0))
        clock_before = ctx.get("clock_before")
        clock_avail = bool(ctx.get("clock_available", False))
        halfmove = int(ctx.get("halfmove_clock", 0))

        meta_arr = encode_meta_v3(
            player_rating=self.median_rating,
            opponent_rating=opp_rating,
            base_seconds=base_s,
            increment_seconds=inc_s,
            ply=ply,
            clock_before=clock_before,
            clock_available=clock_avail,
            halfmove_clock=halfmove,
            num_legal_moves=board.legal_moves.count(),
            berserk=bool(ctx.get("target_berserk", False)),
        )

        board_t = torch.from_numpy(planes_arr).unsqueeze(0).float().to(self.device)
        meta_t = torch.from_numpy(meta_arr).unsqueeze(0).to(self.device)

        out = self.model(board_t, meta_t)
        policy_logits = (out["policy_logits"][0] / self.temperature).cpu().numpy()

        # Compute square embeddings for promo head
        x = F.relu(self.model.stem_bn(self.model.stem_conv(board_t)))
        meta_emb = self.model.meta_mlp(meta_t)
        for block, film in zip(self.model.blocks, self.model.film_generators):
            x = block(x)
            gb = film(meta_emb)
            gamma, beta = gb.chunk(2, dim=1)
            x = x * (1.0 + gamma.unsqueeze(-1).unsqueeze(-1)) + beta.unsqueeze(-1).unsqueeze(-1)
        sq = x.permute(0, 2, 3, 1).reshape(1, 64, self.model.channels)

        unnorm_logp: dict[str, float] = {}
        for m in board.legal_moves:
            f = m.from_square ^ 56 if is_black else m.from_square
            t = m.to_square ^ 56 if is_black else m.to_square
            idx = f * 64 + t

            if m.promotion is not None:
                promo_cls = _PROMO_TO_CLASS.get(m.promotion, 0)
                promo_in = torch.cat([sq[0, f], sq[0, t]], dim=-1).unsqueeze(0)
                promo_log = self.model.promo_head(promo_in) / self.temperature
                promo_logp = float(F.log_softmax(promo_log, dim=-1)[0, promo_cls].item())
            else:
                promo_logp = 0.0

            unnorm_logp[m.uci()] = float(policy_logits[idx]) + promo_logp

        max_val = max(unnorm_logp.values())
        exp_sum = sum(math.exp(v - max_val) for v in unnorm_logp.values())
        log_denom = max_val + math.log(exp_sum)

        return {uci: val - log_denom for uci, val in unnorm_logp.items()}


def fine_tune(
    base_checkpoint_path: Path,
    train_parquet: Any,
    val_parquet: Any,
    test_parquet: Any = None,
    out_dir: Path | None = None,
    variant: FineTuneVariant = FineTuneVariant.FULL,
    max_minutes: int = 20,
    device: str = "cpu",
    seed: int = 42,
) -> dict[str, Any]:
    """Run the fine-tuning pipeline end-to-end.

    Returns:
      {
        "variant": str,
        "best_epoch": int,
        "epochs_run": int,
        "stop_reason": "epoch_cap"|"patience"|"wall_clock",
        "history": [{"epoch":1, "train_loss":..., "val_nll":..., "val_top1":...}],
        "wall_minutes": float,
      }
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    t0 = time.time()
    max_seconds = max_minutes * 60.0

    # 1. Load base_v1 checkpoint and instantiate ChessPolicyNetV2
    base_p = Path(base_checkpoint_path)
    ckpt = torch.load(base_p, map_location="cpu", weights_only=False)
    cfg = ckpt.get(
        "config",
        {"channels": 96, "num_blocks": 6, "policy_dim": 32, "meta_dim": 10, "num_planes": 22},
    )
    model = ChessPolicyNetV2.from_config(cfg)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(dev)

    # 2. Load train/val datasets
    train_ds = _resolve_dataset(train_parquet, split_name="train")
    val_ds = _resolve_dataset(val_parquet, split_name="val")

    from scripts.train_v2 import collate_parquet_batch, ParquetV3Dataset
    collate_fn_train = collate_parquet_batch if isinstance(train_ds, ParquetV3Dataset) else None
    collate_fn_val = collate_parquet_batch if isinstance(val_ds, ParquetV3Dataset) else None

    batch_size = getattr(config, "FT_BATCH_SIZE", 256)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_train
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_val
    )

    # 3. Freeze per variant; set frozen BN to eval()
    freeze_for_variant(model, variant)

    # 4. AdamW lr per variant, wd=1e-4
    if variant == FineTuneVariant.FULL:
        lr = getattr(config, "FT_LR_FULL", 3e-4)
    elif variant == FineTuneVariant.FREEZE:
        lr = getattr(config, "FT_LR_FREEZE", 5e-4)
    elif variant == FineTuneVariant.FILM_ONLY:
        lr = getattr(config, "FT_LR_FILM_ONLY", 1e-3)
    else:
        lr = 3e-4

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

    # 5. Training loop constants
    max_epochs = getattr(config, "FT_MAX_EPOCHS", 8)
    patience = getattr(config, "FT_PATIENCE", 2)

    best_val_nll = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state_dict = copy.deepcopy(model.state_dict())
    stop_reason = "epoch_cap"
    history: list[dict[str, Any]] = []
    epochs_run = 0

    for epoch in range(1, max_epochs + 1):
        epochs_run = epoch
        model.train()
        # Set frozen BN to eval permanently, re-applied after model.train()
        set_frozen_bn_eval(model, variant)

        train_loss_accum = 0.0
        train_batches_run = 0
        wall_clock_hit = False

        for batch in train_loader:
            # 6. Wall-clock check every batch
            if time.time() - t0 > max_seconds:
                stop_reason = "wall_clock"
                wall_clock_hit = True
                break

            b_board = batch["board"].to(dev)
            b_meta = batch["meta"].to(dev)
            b_target = batch["target_move_idx"].to(dev)
            b_promo = batch["promo"].to(dev)
            b_target_pair = batch["target_pair_mask"].to(dev)
            b_legal = batch["legal_mask"].to(dev)
            b_weight = batch["sample_weight"].to(dev)

            optimizer.zero_grad()
            out = model(b_board, b_meta, target_move_idx=b_target)
            loss, _ = total_loss(
                policy_logits=out["policy_logits"],
                legal_mask=b_legal,
                target_move_idx=b_target,
                promo_logits=out["promo_logits"],
                target_pair_mask=b_target_pair,
                target_promo=b_promo,
                sample_weight=b_weight,
            )
            loss.backward()
            optimizer.step()

            train_loss_accum += float(loss.item())
            train_batches_run += 1

        if wall_clock_hit:
            break

        train_loss = train_loss_accum / max(1, train_batches_run)

        # Validation evaluation
        model.eval()
        val_nll_accum = 0.0
        val_top1_accum = 0.0
        val_weight_accum = 0.0

        with torch.no_grad():
            for v_batch in val_loader:
                v_board = v_batch["board"].to(dev)
                v_meta = v_batch["meta"].to(dev)
                v_target = v_batch["target_move_idx"].to(dev)
                v_legal = v_batch["legal_mask"].to(dev)
                v_weight = v_batch["sample_weight"].to(dev)

                v_out = model(v_board, v_meta)
                v_logits = v_out["policy_logits"]

                v_nll = masked_policy_loss(
                    policy_logits=v_logits,
                    legal_mask=v_legal,
                    target_move_idx=v_target,
                    sample_weight=v_weight,
                    label_smoothing=0.0,
                )
                masked_v_logits = torch.where(
                    v_legal, v_logits, torch.full_like(v_logits, -float("inf"))
                )
                top1_preds = masked_v_logits.argmax(dim=-1)
                top1_acc = (top1_preds == v_target).float().mean().item()

                w_sum = float(v_weight.sum().item())
                val_nll_accum += float(v_nll.item()) * w_sum
                val_top1_accum += top1_acc * w_sum
                val_weight_accum += w_sum

        val_nll = val_nll_accum / max(val_weight_accum, 1e-8)
        val_top1 = val_top1_accum / max(val_weight_accum, 1e-8)

        epoch_stats = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_nll": round(val_nll, 4),
            "val_top1": round(val_top1, 4),
        }
        history.append(epoch_stats)

        # 7. Early stop: patience = FT_PATIENCE on val_nll
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_epoch = epoch
            patience_counter = 0
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                stop_reason = "patience"
                break

    wall_minutes = round((time.time() - t0) / 60.0, 4)

    # Load best model weights
    model.load_state_dict(best_state_dict)

    result = {
        "variant": variant.value,
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "stop_reason": stop_reason,
        "history": history,
        "wall_minutes": wall_minutes,
    }

    # 8. Save weights.pt, config.json, training_log.json to out_dir
    if out_dir is not None:
        out_p = Path(out_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": model.to_config(),
                "variant": variant.value,
                "best_epoch": best_epoch,
            },
            out_p / "weights.pt",
        )

        with open(out_p / "config.json", "w", encoding="utf-8") as f:
            json.dump(model.to_config(), f, indent=2)

        with open(out_p / "training_log.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    return result
