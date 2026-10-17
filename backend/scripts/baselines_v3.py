"""
B.6: Baselines recomputed on v3 splits.

CLI:
    python scripts/baselines_v3.py the_king_crusher
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any

import chess
import numpy as np
import pyarrow.parquet as pq
import torch

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app import config
from app.ml import export
from app.ml.baselines import (
    BOOK_EVAL_EPS,
    CurrentModel,
    PositionFrequency,
    UniformLegal,
)
from app.ml.evaluate import evaluate, evaluate_cached
from app.ml.tensor_cache import cache_is_valid
from app.services import storage
from app.services.opening_book_v2 import BookV2, build_book_v2


class BookV2ThenModel:
    """
    Evaluator combining OpeningBook V2 with CurrentModel.

    If BookV2.query(board) is not None, uses:
      (1 - BOOK_EVAL_EPS) * p_book + BOOK_EVAL_EPS * p_uniform
    with BOOK_EVAL_EPS = 1e-3.
    Otherwise falls back to CurrentModel.
    """

    def __init__(self, model: CurrentModel, book: BookV2, eps: float = BOOK_EVAL_EPS) -> None:
        self.model = model
        self.book = book
        self.eps = eps

    def legal_log_probs(self, board: chess.Board, ctx: dict) -> dict[str, float]:
        if not board.legal_moves:
            raise ValueError(f"No legal moves in position {board.fen()}")

        book_probs = self.book.query(board)
        if book_probs is not None:
            legal_moves = [m.uci() for m in board.legal_moves]
            n_legal = len(legal_moves)
            p_uniform = 1.0 / n_legal
            eps = self.eps
            res: dict[str, float] = {}
            for uci in legal_moves:
                p_b = book_probs.get(uci, 0.0)
                p = (1.0 - eps) * p_b + eps * p_uniform
                res[uci] = math.log(p)
            return res

        return self.model.legal_log_probs(board, ctx)


def make_model_batch_logprob_fn(model: torch.nn.Module, device: str = "cpu"):
    """
    Creates a batch_logprob_fn for evaluate_cached wrapping a v2 ChessPolicyNet model.

    Slices planes[:, :18] and meta[:, :8] from v3 tensor cache inputs.
    """
    model.eval()

    def batch_logprob_fn(batch: dict[str, np.ndarray]) -> np.ndarray:
        planes = batch["planes"][:, :18].astype(np.float32)
        meta = batch["meta"][:, :8].astype(np.float32)
        legal_mask = batch["legal_mask"]
        B = planes.shape[0]

        board_tensor = torch.from_numpy(planes).to(device)
        meta_tensor = torch.from_numpy(meta).to(device)

        with torch.no_grad():
            outputs = model(board=board_tensor, meta=meta_tensor)
            from_logits = outputs["from_sq_logits"]
            to_logits = outputs["to_sq_logits"]

        out = np.full((B, 4096), -np.inf, dtype=np.float64)

        for b in range(B):
            mask_b = legal_mask[b].reshape(64, 64)
            from_legal = mask_b.any(axis=1)
            if not np.any(from_legal):
                continue

            sub_from = from_logits[b, from_legal]
            from_lp = torch.log_softmax(sub_from, dim=-1).cpu().numpy()

            from_indices = np.where(from_legal)[0]
            for f_idx, f_val in enumerate(from_indices):
                to_legal = mask_b[f_val]
                sub_to = to_logits[b, to_legal]
                to_lp = torch.log_softmax(sub_to, dim=-1).cpu().numpy()

                to_indices = np.where(to_legal)[0]
                f_logp = from_lp[f_idx]
                for t_idx, t_val in enumerate(to_indices):
                    out[b, f_val * 64 + t_val] = f_logp + to_lp[t_idx]

        return out

    return batch_logprob_fn


def format_ci(mean: float, ci: Any) -> str:
    if isinstance(ci, tuple) and len(ci) == 2:
        return f"{mean:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"
    if isinstance(ci, dict) and ci.get("status") == "insufficient_evidence":
        return f"{mean:.4f} [insufficient evidence]"
    return f"{mean:.4f}"


def run_baselines_v3(username: str = "the_king_crusher") -> str:
    p_dir = storage.player_dir(username, create=False)
    parquet_path = p_dir / "samples_v3.parquet"
    cache_dir = p_dir / "cache_v3"

    if not parquet_path.exists():
        raise FileNotFoundError(f"samples_v3.parquet not found at {parquet_path}")

    valid, reason = cache_is_valid(cache_dir, parquet_path)
    if not valid:
        raise ValueError(f"Cache v3 is invalid: {reason}")

    # Load samples
    table = pq.read_table(str(parquet_path))
    df = table.to_pandas()
    all_rows = df.to_dict(orient="records")
    train_rows = [r for r in all_rows if r.get("split") == "train"]
    test_rows = [r for r in all_rows if r.get("split") == "test"]

    # Load v2 model
    model = export.load_player_model(username, device="cpu")
    model.eval()

    # Build BookV2 in eval mode
    book_dict = build_book_v2(train_rows, mode="eval")
    book = BookV2.from_dict(book_dict)

    # 1. UniformLegal
    res_uniform = evaluate(UniformLegal(), test_rows)

    # 2. PositionFrequency
    pos_freq = PositionFrequency(train_rows)
    res_pos_freq = evaluate(pos_freq, test_rows)

    # 3. CurrentModel (via evaluate_cached)
    model_batch_fn = make_model_batch_logprob_fn(model, device="cpu")
    res_current_model = evaluate_cached(
        model_batch_fn,
        cache_dir,
        parquet_path,
        split="test",
        berserk_filter="all",
        batch_size=512,
    )

    # 4. BookThenModel (via evaluate on parquet rows)
    current_model_predictor = CurrentModel(model, device="cpu")
    book_then_model = BookV2ThenModel(current_model_predictor, book, eps=BOOK_EVAL_EPS)
    res_book = evaluate(book_then_model, test_rows)

    results = {
        "UniformLegal": res_uniform,
        "PositionFrequency": res_pos_freq,
        "CurrentModel": res_current_model,
        "BookThenModel": res_book,
    }

    # Generate markdown output
    lines: list[str] = []
    lines.append(f"# Phase B.6 Baseline Evaluation: `{username}`\n")
    lines.append(f"- **Test split positions**: {len(test_rows)}")
    lines.append(f"- **Unique games in test**: {res_uniform['n_games']}\n")

    lines.append("## Overall Performance Comparison\n")
    lines.append("| Predictor | top1 [lo95,hi95] | top3 | nll [lo95,hi95] | n_games | n_positions |")
    lines.append("|:---|:---:|:---:|:---:|:---:|:---:|")

    for name in ["UniformLegal", "PositionFrequency", "CurrentModel", "BookThenModel"]:
        r = results[name]
        top1_str = format_ci(r["top1"], r["top1_ci"])
        top3_str = f"{r['top3']:.4f}"
        nll_str = format_ci(r["nll"], r["nll_ci"])
        lines.append(f"| {name} | {top1_str} | {top3_str} | {nll_str} | {r['n_games']} | {r['n_positions']} |")

    lines.append("\n## Per-Phase Breakdown (top-1 / NLL)\n")
    lines.append("| Predictor | Opening (ply<=20) | Middlegame (21-60) | Endgame (>60) |")
    lines.append("|:---|:---:|:---:|:---:|")

    for name in ["UniformLegal", "PositionFrequency", "CurrentModel", "BookThenModel"]:
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

    for name in ["UniformLegal", "PositionFrequency", "CurrentModel", "BookThenModel"]:
        bb = results[name]["by_berserk"]
        b_false = f"{bb['false']['top1']:.4f} / {bb['false']['nll']:.4f}"
        b_true = f"{bb['true']['top1']:.4f} / {bb['true']['nll']:.4f}"
        lines.append(f"| {name} | {b_false} | {b_true} |")

    lines.append("\n---\n")
    lines.append(f"- `BOOK_EVAL_EPS = {BOOK_EVAL_EPS}`")
    lines.append("- training val loss is smoothed CE, not NLL; not comparable\n")

    report = "\n".join(lines)

    # Save identical output to docs/progress/phaseB6.md
    docs_path = backend_dir.parent / "docs" / "progress" / "phaseB6.md"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(docs_path, "w", encoding="utf-8") as f:
        f.write(report)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baselines on v3 test split.")
    parser.add_argument("username", nargs="?", default="the_king_crusher", help="Player username")
    args = parser.parse_args()

    report_str = run_baselines_v3(args.username)
    print(report_str)
