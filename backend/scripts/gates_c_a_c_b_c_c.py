"""
Evaluation gates C-a, C-b, C-c for Phase C2/C3 base model verification.

CLI:
    python scripts/gates_c_a_c_b_c_c.py \\
        --base-checkpoint data/base/base_v1.pt \\
        --base-shards-dir data/base/shards \\
        --out docs/progress/phaseC2_gates.md
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any, Optional

import chess
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

# Ensure backend and repo root are on sys.path
backend_dir = Path(__file__).resolve().parents[1]
repo_root = backend_dir.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app import config
from app.ml import export
from app.ml.baselines import PositionFrequency
from app.ml.encoding import _PROMO_TO_CLASS
from app.ml.encoding_v3 import encode_board_v3, encode_meta_v3
from app.ml.evaluate import evaluate, evaluate_cached, paired_bootstrap_by_game
from app.ml.model_v2 import ChessPolicyNetV2
from app.ml.tensor_cache import cache_is_valid
from app.services import storage
from scripts.baselines_v3 import make_model_batch_logprob_fn


class BasePolicyPredictor:
    """Predictor wrapping ChessPolicyNetV2 for evaluate()."""

    def __init__(self, model: ChessPolicyNetV2, median_rating: int, device: str = "cpu"):
        self.model = model
        self.median_rating = median_rating
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def legal_log_probs(self, board: chess.Board, ctx: dict) -> dict[str, float]:
        if not board.legal_moves:
            raise ValueError(f"No legal moves in position {board.fen()}")

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
            berserk=False,
        )

        board_t = torch.from_numpy(planes_arr).unsqueeze(0).float().to(self.device)
        meta_t = torch.from_numpy(meta_arr).unsqueeze(0).to(self.device)

        out = self.model(board_t, meta_t)
        policy_logits = out["policy_logits"][0].cpu().numpy()  # (4096,)

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
                promo_log = self.model.promo_head(promo_in)
                promo_logp = float(F.log_softmax(promo_log, dim=-1)[0, promo_cls].item())
            else:
                promo_logp = 0.0

            unnorm_logp[m.uci()] = float(policy_logits[idx]) + promo_logp

        max_val = max(unnorm_logp.values())
        exp_sum = sum(math.exp(v - max_val) for v in unnorm_logp.values())
        log_denom = max_val + math.log(exp_sum)

        return {uci: val - log_denom for uci, val in unnorm_logp.items()}


def make_base_model_batch_logprob_fn(model: ChessPolicyNetV2, device: str = "cpu"):
    """Creates batch_logprob_fn for evaluate_cached with berserk flag turned off."""
    model.eval()

    def batch_logprob_fn(batch: dict[str, np.ndarray]) -> np.ndarray:
        planes = batch["planes"].astype(np.float32)
        meta = batch["meta"].copy().astype(np.float32)
        meta[:, 9] = 0.0  # Berserk off (index 9 in v3 meta)
        legal_mask = batch["legal_mask"]
        B = planes.shape[0]

        board_t = torch.from_numpy(planes).float().to(device)
        meta_t = torch.from_numpy(meta).to(device)

        with torch.no_grad():
            out = model(board_t, meta_t)
            logits = out["policy_logits"].cpu().numpy()

        out_logp = np.full((B, 4096), -np.inf, dtype=np.float64)
        for b in range(B):
            leg = legal_mask[b]
            sub_logits = logits[b, leg]
            if len(sub_logits) == 0:
                continue
            max_l = np.max(sub_logits)
            exp_l = np.exp(sub_logits - max_l)
            sub_logp = (sub_logits - max_l) - np.log(np.sum(exp_l))
            out_logp[b, leg] = sub_logp

        return out_logp

    return batch_logprob_fn


def load_parquet_records(shard_files: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sf in shard_files:
        table = pq.read_table(sf)
        rows.extend(table.to_pandas().to_dict(orient="records"))
    return rows


def format_ci(mean: float, ci: Any) -> str:
    if isinstance(ci, (tuple, list)) and len(ci) == 2:
        return f"{mean:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"
    if isinstance(ci, dict) and ci.get("status") == "insufficient_evidence":
        return f"{mean:.4f} [insufficient evidence]"
    return f"{mean:.4f}"


def run_gates(
    base_checkpoint: Path,
    base_shards_dir: Path,
    out_path: Path,
    device: str = "cpu",
) -> str:
    # 0. File presence checks
    if not base_checkpoint.exists():
        print(f"MISSING: {base_checkpoint}", flush=True)
        sys.exit(1)

    if not base_shards_dir.exists():
        print(f"MISSING: {base_shards_dir}", flush=True)
        sys.exit(1)

    # Shard checks
    test_shards = sorted(base_shards_dir.glob("test_shard_*.parquet"))
    train_shards = sorted(base_shards_dir.glob("train_shard_*.parquet"))
    if not test_shards:
        # Check if shards contain split column
        all_shards = sorted(base_shards_dir.glob("*.parquet"))
        if not all_shards:
            print(f"MISSING: No parquet shards found in {base_shards_dir}", flush=True)
            sys.exit(1)
        all_rows = load_parquet_records(all_shards)
        train_rows = [r for r in all_rows if r.get("split") == "train"]
        test_rows = [r for r in all_rows if r.get("split") == "test"]
    else:
        train_rows = load_parquet_records(train_shards)
        test_rows = load_parquet_records(test_shards)

    if not test_rows:
        print(f"MISSING: No base_test positions found in {base_shards_dir}", flush=True)
        sys.exit(1)

    # 1. Load base_v1 model
    ckpt = torch.load(base_checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    base_model = ChessPolicyNetV2.from_config(cfg).to(device)
    base_model.load_state_dict(ckpt["state_dict"])
    base_model.eval()

    # Compute median player rating
    ratings = [int(r.get("player_rating", 1500)) for r in train_rows]
    median_rating = int(np.median(ratings)) if ratings else 1500

    report_lines: list[str] = [
        "# Phase C2/C3 Base Model Verification Gates",
        "",
        f"- **Base Checkpoint**: `{base_checkpoint}`",
        f"- **Base Shards Directory**: `{base_shards_dir}`",
        f"- **Test Positions**: {len(test_rows)}",
        f"- **Train Median Rating**: {median_rating}",
        "",
        "---",
        "",
    ]

    # =========================================================================
    # GATE C-a: base_test evaluation against baselines
    # =========================================================================
    report_lines.append("## Gate C-a: Base Test Evaluation")
    report_lines.append("")

    base_pred = BasePolicyPredictor(base_model, median_rating=median_rating, device=device)
    res_base = evaluate(base_pred, test_rows)

    posfreq_pred = PositionFrequency(train_rows)
    res_posfreq = evaluate(posfreq_pred, test_rows)

    # Check E2 reference
    e2_ref_path = repo_root / "data" / "base" / "e2_joint_500k.pt"
    e2_available = e2_ref_path.exists()
    res_e2: Optional[dict[str, Any]] = None

    if e2_available:
        try:
            e2_ckpt = torch.load(e2_ref_path, map_location=device, weights_only=False)
            e2_model = ChessPolicyNetV2.from_config(e2_ckpt["config"]).to(device)
            e2_model.load_state_dict(e2_ckpt["state_dict"])
            e2_model.eval()
            e2_pred = BasePolicyPredictor(e2_model, median_rating=median_rating, device=device)
            res_e2 = evaluate(e2_pred, test_rows)
        except Exception as ex:
            e2_available = False
            report_lines.append(f"*Note: Failed to evaluate E2 reference ({ex}).*")

    top1_base = res_base["top1"]
    top1_posfreq = res_posfreq["top1"]
    top1_margin = top1_base - top1_posfreq

    cond_margin = (top1_margin >= 0.15)
    cond_abs_top1 = (top1_base >= 0.40)

    report_lines.append("| Predictor | Top-1 (95% CI) | Top-3 (95% CI) | NLL (95% CI) |")
    report_lines.append("| :--- | :--- | :--- | :--- |")
    report_lines.append(f"| Base Policy (v1) | {format_ci(res_base['top1'], res_base['top1_ci'])} | {format_ci(res_base['top3'], res_base['top3_ci'])} | {format_ci(res_base['nll'], res_base['nll_ci'])} |")
    report_lines.append(f"| PositionFrequency | {format_ci(res_posfreq['top1'], res_posfreq['top1_ci'])} | {format_ci(res_posfreq['top3'], res_posfreq['top3_ci'])} | {format_ci(res_posfreq['nll'], res_posfreq['nll_ci'])} |")
    if e2_available and res_e2 is not None:
        report_lines.append(f"| E2 Reference (500k) | {format_ci(res_e2['top1'], res_e2['top1_ci'])} | {format_ci(res_e2['top3'], res_e2['top3_ci'])} | {format_ci(res_e2['nll'], res_e2['nll_ci'])} |")
    else:
        report_lines.append("| E2 Reference (500k) | *Not Available* | *Not Available* | *Not Available* |")

    report_lines.append("")
    report_lines.append(f"- (top1_base - top1_posfreq) = {top1_margin:.4f} (Criterion: >= 0.15): **{'PASS' if cond_margin else 'FAIL'}**")
    report_lines.append(f"- Absolute top1_base = {top1_base:.4f} (Criterion: >= 0.40): **{'PASS' if cond_abs_top1 else 'FAIL'}**")

    if e2_available and res_e2 is not None:
        nll_base_hi = res_base["nll_ci"][1] if isinstance(res_base["nll_ci"], (tuple, list)) else res_base["nll"]
        nll_e2_lo = res_e2["nll_ci"][0] if isinstance(res_e2["nll_ci"], (tuple, list)) else res_e2["nll"]
        cond_nll_e2 = (nll_base_hi < nll_e2_lo)
        report_lines.append(f"- NLL CI upper ({nll_base_hi:.4f}) < E2 CI lower ({nll_e2_lo:.4f}): **{'PASS' if cond_nll_e2 else 'FAIL'}**")
        ca_pass = cond_margin and cond_abs_top1 and cond_nll_e2
        ca_verdict = "PASS" if ca_pass else "FAIL"
    else:
        report_lines.append("- E2 reference not available -> Gate C-a is **INCONCLUSIVE**.")
        ca_verdict = "INCONCLUSIVE"

    report_lines.append(f"\n**Gate C-a Verdict**: `{ca_verdict}`\n")
    report_lines.append("---")
    report_lines.append("")

    # =========================================================================
    # GATE C-b: the_king_crusher v3 test comparison
    # =========================================================================
    report_lines.append("## Gate C-b: the_king_crusher v3 Test Comparison")
    report_lines.append("")

    p_dir = storage.player_dir("the_king_crusher", create=False)
    parquet_path = p_dir / "samples_v3.parquet"
    cache_dir = p_dir / "cache_v3"

    if not parquet_path.exists() or not cache_dir.exists():
        report_lines.append(f"MISSING player dataset/cache at {p_dir} -> Gate C-b INCONCLUSIVE.")
        cb_verdict = "INCONCLUSIVE"
    else:
        base_batch_fn = make_base_model_batch_logprob_fn(base_model, device=device)
        res_base_tkc = evaluate_cached(
            base_batch_fn,
            cache_dir,
            parquet_path,
            split="test",
            berserk_filter="all",
            batch_size=512,
        )

        curr_model = export.load_player_model("the_king_crusher", device=device)
        curr_batch_fn = make_model_batch_logprob_fn(curr_model, device=device)
        res_curr_tkc = evaluate_cached(
            curr_batch_fn,
            cache_dir,
            parquet_path,
            split="test",
            berserk_filter="all",
            batch_size=512,
        )

        # Paired bootstrap (a=base, b=current)
        delta_nll, lo_nll, hi_nll = paired_bootstrap_by_game(
            res_base_tkc["per_position"], res_curr_tkc["per_position"], "nll", n=2000, seed=0
        )
        delta_top1, lo_top1, hi_top1 = paired_bootstrap_by_game(
            res_base_tkc["per_position"], res_curr_tkc["per_position"], "top1", n=2000, seed=0
        )

        report_lines.append("| Metric | Base Policy | Current Player Model | Paired Delta (a - b) [95% CI] |")
        report_lines.append("| :--- | :--- | :--- | :--- |")
        report_lines.append(f"| Top-1 | {res_base_tkc['top1']:.4f} | {res_curr_tkc['top1']:.4f} | {format_ci(delta_top1, (lo_top1, hi_top1))} |")
        report_lines.append(f"| NLL | {res_base_tkc['nll']:.4f} | {res_curr_tkc['nll']:.4f} | {format_ci(delta_nll, (lo_nll, hi_nll))} |")

        cond_nll = isinstance(hi_nll, (float, int)) and (hi_nll < 0.0)
        cond_top1 = isinstance(lo_top1, (float, int)) and (lo_top1 > 0.0)

        report_lines.append("")
        report_lines.append(f"- NLL delta CI upper ({hi_nll:.4f}) < 0: **{'PASS' if cond_nll else 'FAIL'}**")
        report_lines.append(f"- Top-1 delta CI lower ({lo_top1:.4f}) > 0: **{'PASS' if cond_top1 else 'FAIL'}**")

        cb_pass = cond_nll and cond_top1
        cb_verdict = "PASS" if cb_pass else "FAIL"

    report_lines.append(f"\n**Gate C-b Verdict**: `{cb_verdict}`\n")
    report_lines.append("---")
    report_lines.append("")

    # =========================================================================
    # GATE C-c: rating-conditioning sanity
    # =========================================================================
    report_lines.append("## Gate C-c: Rating-Conditioning Sanity")
    report_lines.append("")

    n_sample = min(500, len(test_rows))
    rng = np.random.default_rng(0)
    sample_indices = rng.choice(len(test_rows), size=n_sample, replace=False)
    sample_rows = [test_rows[i] for i in sample_indices]

    ll_r_list: list[float] = []
    ll_plus_list: list[float] = []
    ll_minus_list: list[float] = []

    delta_rating_scaled = 400.0 / config.META_RATING_SCALE  # 1.0

    for r in sample_rows:
        board = chess.Board(str(r["fen"]))
        uci_move = chess.Move.from_uci(str(r["uci"]))
        is_black = (board.turn == chess.BLACK)
        f = uci_move.from_square ^ 56 if is_black else uci_move.from_square
        t = uci_move.to_square ^ 56 if is_black else uci_move.to_square
        target_idx = f * 64 + t

        planes = encode_board_v3(
            board,
            prev_move_1=r.get("prev_move_1"),
            prev_move_2=r.get("prev_move_2"),
            history=True,
        )
        meta_base = encode_meta_v3(
            player_rating=int(r.get("player_rating", 1500)),
            opponent_rating=int(r.get("opponent_rating", 1500)),
            base_seconds=int(r.get("base_s", 0)),
            increment_seconds=int(r.get("increment_s", 0)),
            ply=int(r.get("ply", 0)),
            clock_before=r.get("clock_before_move"),
            clock_available=(r.get("clock_before_move") is not None),
            halfmove_clock=int(r.get("halfmove_clock", 0)),
            num_legal_moves=int(r.get("num_legal_moves", board.legal_moves.count())),
            berserk=False,
        )

        meta_p = meta_base.copy()
        meta_p[0] += delta_rating_scaled

        meta_m = meta_base.copy()
        meta_m[0] -= delta_rating_scaled

        board_t = torch.from_numpy(planes).unsqueeze(0).float().to(device)
        leg_mask = np.zeros(4096, dtype=bool)
        for m in board.legal_moves:
            mf = m.from_square ^ 56 if is_black else m.from_square
            mt = m.to_square ^ 56 if is_black else m.to_square
            leg_mask[mf * 64 + mt] = True

        for meta_vec, store_list in [(meta_base, ll_r_list), (meta_p, ll_plus_list), (meta_m, ll_minus_list)]:
            m_t = torch.from_numpy(meta_vec).unsqueeze(0).to(device)
            with torch.no_grad():
                out = base_model(board_t, m_t)
                logits = out["policy_logits"][0].cpu().numpy()
            sub = logits[leg_mask]
            max_v = np.max(sub)
            logp_target = (logits[target_idx] - max_v) - np.log(np.sum(np.exp(sub - max_v)))
            store_list.append(float(logp_target))

    mean_ll_r = float(np.mean(ll_r_list))
    mean_ll_plus = float(np.mean(ll_plus_list))
    mean_ll_minus = float(np.mean(ll_minus_list))

    diff_plus = mean_ll_r - mean_ll_plus
    diff_minus = mean_ll_r - mean_ll_minus

    report_lines.append(f"- Sampled positions: {n_sample}")
    report_lines.append(f"- Mean log-prob LL(r): {mean_ll_r:.5f}")
    report_lines.append(f"- Mean log-prob LL(r+400): {mean_ll_plus:.5f}")
    report_lines.append(f"- Mean log-prob LL(r-400): {mean_ll_minus:.5f}")
    report_lines.append(f"- Perturbation +400 penalty [LL(r) - LL(r+400)]: {diff_plus:+.5f} (> 0: **{'PASS' if diff_plus > 0 else 'FAIL'}**)")
    report_lines.append(f"- Perturbation -400 penalty [LL(r) - LL(r-400)]: {diff_minus:+.5f} (> 0: **{'PASS' if diff_minus > 0 else 'FAIL'}**)")

    cc_pass = (diff_plus > 0.0) and (diff_minus > 0.0)
    cc_verdict = "PASS" if cc_pass else "FAIL"

    report_lines.append(f"\n**Gate C-c Verdict**: `{cc_verdict}`\n")

    report_content = "\n".join(report_lines)

    # Save to out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(report_content, flush=True)
    return report_content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Gates C-a, C-b, C-c on base policy checkpoint.")
    parser.add_argument("--base-checkpoint", type=str, required=True, help="Path to base_v1.pt")
    parser.add_argument("--base-shards-dir", type=str, required=True, help="Path to base shards directory")
    parser.add_argument("--out", type=str, required=True, help="Output markdown report path")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device (cpu or cuda)")
    args = parser.parse_args()

    run_gates(
        base_checkpoint=Path(args.base_checkpoint),
        base_shards_dir=Path(args.base_shards_dir),
        out_path=Path(args.out),
        device=args.device,
    )
