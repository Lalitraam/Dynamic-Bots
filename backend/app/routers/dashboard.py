"""Dashboard router (Phase J).

Provides analytics and telemetry for individual player bots:
- Model card metrics (eval.json)
- Opening repertoires (White first moves, Black replies)
- Time management by phase (opening, middlegame, endgame)
- Error rates by clock ratio buckets (<0.1, 0.1-0.3, >0.3)
- Player rating
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import chess
from fastapi import APIRouter, Depends, HTTPException, status
import pyarrow.parquet as pq

from app import config
from app.auth import get_current_user
from app.db.players import get_player_median_rating
from app.engine.think_time import _clock_ratio_bucket, _phase_from_ply

router = APIRouter(prefix="/api/bots", tags=["dashboard"])


@router.get("/{username}/dashboard")
def get_bot_dashboard(
    username: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Return dashboard analytics for a given player bot."""
    p_dir = Path(config.DATA_ROOT) / "players" / username
    if not p_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Bot '{username}' not found",
        )

    # 1. Model card
    eval_path = p_dir / "models" / "v1" / "eval.json"
    model_card: dict[str, Any] = {
        "version": "v1",
        "top1": None,
        "nll": None,
        "personalisation_status": "Not Evaluated",
        "personalized": False,
        "decision": None,
    }
    if eval_path.exists():
        try:
            with open(eval_path, "r", encoding="utf-8") as f:
                ev = json.load(f)
            decision = ev.get("decision", "")
            is_pers = (decision == "promote_ft")
            metrics = ev.get("ft_metrics", {}) if is_pers else ev.get("base_metrics", {})
            model_card = {
                "version": "v1",
                "top1": metrics.get("top1"),
                "nll": metrics.get("nll"),
                "personalisation_status": "Personalised" if is_pers else "Base Model",
                "personalized": is_pers,
                "decision": decision,
            }
        except Exception:
            pass

    # 2. Openings
    book_path = p_dir / "opening_book.json"
    openings: dict[str, list[dict[str, Any]]] = {"white": [], "black": []}
    if book_path.exists():
        try:
            with open(book_path, "r", encoding="utf-8") as f:
                book = json.load(f)

            # Top 5 White first moves from initial position
            start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
            w_moves = book.get(start_fen, {})
            w_total = sum(w_moves.values()) or 1
            white_top5 = sorted(
                [{"move": m, "count": c, "probability": c / w_total} for m, c in w_moves.items()],
                key=lambda x: x["probability"],
                reverse=True,
            )[:5]

            # Top 5 Black replies across 1-ply responses
            b0 = chess.Board()
            black_counts: dict[str, int] = {}
            for m in b0.legal_moves:
                b0.push(m)
                fen_key = " ".join(b0.fen().split()[:4])
                if fen_key in book:
                    for bm, cnt in book[fen_key].items():
                        black_counts[bm] = black_counts.get(bm, 0) + cnt
                b0.pop()

            b_total = sum(black_counts.values()) or 1
            black_top5 = sorted(
                [{"move": m, "count": c, "probability": c / b_total} for m, c in black_counts.items()],
                key=lambda x: x["probability"],
                reverse=True,
            )[:5]

            openings = {"white": white_top5, "black": black_top5}
        except Exception:
            pass

    # 3. Time management
    samples_path = p_dir / "samples_v3.parquet"
    time_management: dict[str, float] = {}
    if samples_path.exists():
        try:
            df_s = pq.read_table(samples_path, columns=["ply", "time_spent_s"]).to_pandas()
            df_s["phase"] = df_s["ply"].apply(_phase_from_ply)
            time_management = {
                str(k): float(v) for k, v in df_s.groupby("phase")["time_spent_s"].mean().items()
            }
        except Exception:
            pass

    # 4. Errors
    analysis_path = p_dir / "analysis_train.parquet"
    errors: list[dict[str, Any]] = []
    if analysis_path.exists():
        try:
            df_a = pq.read_table(analysis_path).to_pandas()
            df_a["is_err"] = (df_a["cp_loss"] >= 100).astype(int)
            if "clock_ratio" not in df_a.columns and "clock_before_move" in df_a.columns and "base_s" in df_a.columns:
                df_a["clock_ratio"] = df_a["clock_before_move"] / df_a["base_s"].replace(0, float("nan"))
            df_a["bucket"] = df_a["clock_ratio"].apply(_clock_ratio_bucket)
            grouped = df_a.groupby("bucket")["is_err"].agg(["count", "mean"]).to_dict("index")
            for b in ["<0.1", "0.1-0.3", ">0.3"]:
                if b in grouped:
                    errors.append({
                        "bucket": b,
                        "clock_ratio": b,
                        "error_rate": float(grouped[b]["mean"]),
                        "count": int(grouped[b]["count"]),
                    })
        except Exception:
            errors = []

    # 5. Rating
    rating = get_player_median_rating(username)

    return {
        "username": username,
        "rating": rating,
        "model_card": model_card,
        "openings": openings,
        "time_management": time_management,
        "errors": errors,
    }
