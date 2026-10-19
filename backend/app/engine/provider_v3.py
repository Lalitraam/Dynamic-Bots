"""V3 Move Provider implementation (Phase E)."""

from __future__ import annotations
import json
import math
from pathlib import Path
import random
from typing import Any, Literal
import chess
import numpy as np
import torch
import torch.nn.functional as F

from app import config
from app.analysis.error_model import ErrorModel
from app.analysis.features import compute_features
from app.analysis.stockfish import StockfishAnalyser, compute_cp_loss
from app.engine.model_cache import ModelCache
from app.engine.provider import MoveContext, MoveDecision, MoveProvider
from app.engine.think_time import ThinkTimeModel
from app.ml.encoding_v3 import encode_board_v3
from app.ml.encoding import encode_meta
from app.services.opening_book_v2 import BookV2
import time
start_t = time.perf_counter()

_PROMO_TO_CLASS = {chess.QUEEN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, None: 0}


def _build_board_planes(ctx: MoveContext) -> np.ndarray:
    """Build 18-plane board encoding from context (matches trained model)."""
    # Note: We intentionally use history=False to match the 18-plane input
    # expected by the model trained with BOARD_PLANES=18
    return encode_board_v3(
        ctx.board,
        prev_move_1=None,
        prev_move_2=None,
        history=False,
    )


def _build_meta(ctx: MoveContext, bot_rating_from_model: int) -> np.ndarray:
    """Build 8-dimensional metadata vector from context (matches trained model)."""
    rating = bot_rating_from_model if bot_rating_from_model is not None else ctx.bot_rating
    clock_before = (ctx.bot_clock_ms / 1000.0) if ctx.bot_clock_ms is not None else None
    clock_avail = bool(ctx.bot_clock_ms is not None and ctx.base_seconds > 0)
    return encode_meta(
        player_rating=rating,
        opponent_rating=ctx.opp_rating,
        base_seconds=ctx.base_seconds,
        increment_seconds=ctx.increment_seconds,
        ply=ctx.ply,
        clock_before=clock_before,
        clock_available=clock_avail,
        halfmove_clock=ctx.board.halfmove_clock,
    )


def legal_indices(board: chess.Board) -> np.ndarray:
    """Return sorted unique legal move indices (from*64 + to) in mover frame."""
    is_black = (board.turn == chess.BLACK)
    indices: set[int] = set()
    for move in board.legal_moves:
        f = (move.from_square ^ 56) if is_black else move.from_square
        t = (move.to_square ^ 56) if is_black else move.to_square
        indices.add(f * 64 + t)
    return np.array(sorted(indices), dtype=np.int16)


def _apply_top_k(arr: np.ndarray, k: int = getattr(config, "OPTIMAL_TOP_K", 3)) -> np.ndarray:
    """Apply top-k filtering to probability or log-probability array."""
    if len(arr) == 0:
        return arr
    k = max(1, min(k, len(arr)))
    is_prob = bool(np.all(arr >= 0) and not np.all(arr == 0))
    if is_prob:
        probs = arr.astype(float).copy()
        order = np.argsort(-probs)
        to_zero = order[k:]
        probs[to_zero] = 0.0
        return probs
    else:
        out = arr.astype(float).copy()
        finite = np.isfinite(out)
        if not np.any(finite):
            return out
        order = np.argsort(-out)
        to_inf = order[k:]
        out[to_inf] = -np.inf
        return out


def _apply_top_p(arr: np.ndarray, p: float = config.TOP_P) -> np.ndarray:
    """Apply top-p filtering to probability or log-probability array."""
    if len(arr) == 0:
        return arr
    is_prob = bool(np.all(arr >= 0) and not np.all(arr == 0))
    if is_prob:
        probs = arr.astype(float)
        order = np.argsort(-probs)
        sorted_p = probs[order]
        cumsum = np.cumsum(sorted_p)
        keep_mask = cumsum <= p
        keep_mask[0] = True  # Always keep at least the top-1
        to_zero = order[~keep_mask]
        out = probs.copy()
        out[to_zero] = 0.0
        return out
    else:
        out = arr.astype(float).copy()
        finite = np.isfinite(out)
        if not np.any(finite):
            return out
        max_lp = np.max(out[finite])
        probs = np.zeros_like(out)
        probs[finite] = np.exp(out[finite] - max_lp)
        s = probs.sum()
        if s > 0:
            probs = probs / s
        order = np.argsort(-probs)
        cumsum = np.cumsum(probs[order])
        keep_mask = cumsum <= p
        keep_mask[0] = True
        to_inf = order[~keep_mask]
        out[to_inf] = -np.inf
        return out


def _sample_from(arr: np.ndarray, rng: Any = None) -> int:
    """Sample an index from probabilities or log_probs (ignoring 0 / -inf)."""
    if len(arr) == 0:
        return 0
    if np.all(arr >= 0):
        probs = arr.astype(float)
        s = probs.sum()
        if s > 0:
            probs = probs / s
        else:
            probs = np.ones_like(probs) / len(probs)
    else:
        finite = np.isfinite(arr)
        if not np.any(finite):
            return 0
        max_v = np.max(arr[finite])
        probs = np.zeros_like(arr, dtype=float)
        probs[finite] = np.exp(arr[finite] - max_v)
        s = probs.sum()
        if s > 0:
            probs = probs / s
        else:
            probs[finite] = 1.0 / np.sum(finite)

    if rng is None:
        return int(np.random.choice(len(probs), p=probs))
    elif hasattr(rng, "choices"):
        return int(rng.choices(range(len(probs)), weights=probs.tolist(), k=1)[0])
    elif hasattr(rng, "choice"):
        return int(rng.choice(range(len(probs)), p=probs))
    else:
        return int(random.choices(range(len(probs)), weights=probs.tolist(), k=1)[0])


class V3Provider:
    """Runtime move provider implementing plan section 4.2 in temperature-only mode."""

    def __init__(
        self,
        username: str,
        model_version: int,
        model_cache: ModelCache,
        think_time_model: ThinkTimeModel,
        book: BookV2 | None = None,
        temperature: float = 1.0,
        data_root: Path = config.DATA_ROOT,
        error_model: ErrorModel | None = None,
        stockfish_analyser: StockfishAnalyser | None = None,
    ):
        self.username = username
        self.model_version = model_version
        self.model_cache = model_cache
        self.think_time_model = think_time_model
        self.book = book
        self.data_root = Path(data_root)
        self.stockfish_analyser = stockfish_analyser

        model_dir = self.data_root / "players" / username / "models" / f"v{model_version}"
        weights_file = model_dir / "weights.pt"
        self.is_base_fallback = not weights_file.exists()

        # Load error model if not provided
        self.error_model = error_model
        if self.error_model is None:
            em_path = model_dir / "error_model.joblib"
            if em_path.exists():
                try:
                    self.error_model = ErrorModel.load(em_path)
                except Exception:
                    self.error_model = None

        # Load temperature from temperature.json if exists and default temperature passed
        temp_file = model_dir / "temperature.json"
        if temperature == 1.0 and temp_file.exists():
            try:
                with open(temp_file, "r", encoding="utf-8") as f:
                    t_data = json.load(f)
                # Use the model's temperature if available, otherwise fallback to optimal
                self.temperature = max(1e-4, float(t_data.get("T_best", config.OPTIMAL_T_BASE)))
            except Exception:
                self.temperature = config.OPTIMAL_T_BASE
        elif temperature == "greedy":
            self.temperature = "greedy"
        else:
            self.temperature = max(1e-4, float(temperature))

        # Read median player rating and p99 cp loss from player info if present
        self.bot_rating_from_model: int = config.DEFAULT_RATING
        self.player_p99: float = 600.0
        info_file = self.data_root / "players" / username / "info.json"
        if info_file.exists():
            try:
                with open(info_file, "r", encoding="utf-8") as f:
                    info = json.load(f)
                    self.bot_rating_from_model = int(info.get("median_rating", info.get("player_rating", config.DEFAULT_RATING)))
                    self.player_p99 = float(info.get("p99_cp_loss", 600.0))
            except Exception:
                pass

        # If guard is enabled and no analyser provided, try to create one
        if config.GUARD_ENABLED and self.stockfish_analyser is None:
            try:
                self.stockfish_analyser = StockfishAnalyser(path=config.STOCKFISH_PATH)
            except Exception:
                self.stockfish_analyser = None

    def _compute_cp_loss(self, board: chess.Board, move: chess.Move) -> float:
        """Calculate centipawn loss of move using Stockfish if available, else return 0.0."""
        if self.stockfish_analyser is not None:
            try:
                loss, _ = compute_cp_loss(board, move, self.stockfish_analyser)
                return loss
            except Exception:
                return 0.0
        return 0.0

    def _is_move_safe(self, board: chess.Board, move: chess.Move) -> bool:
        """Return True if move is not a blunder according to Stockfish (loss <= GUARD_CP)."""
        if not config.GUARD_ENABLED or self.stockfish_analyser is None:
            return True
        loss = self._compute_cp_loss(board, move)
        return loss <= config.GUARD_CP

    def decide(self, ctx: MoveContext, rng: random.Random) -> MoveDecision:
        """Runtime move decision pipeline - FAKE THINK TIME DISABLED, ULTRA FAST."""
        start_t = time.perf_counter()
        legal_moves = list(ctx.board.legal_moves)
        if not legal_moves:
            raise ValueError(f"No legal moves in position {ctx.board.fen()}")

        if self.think_time_model is not None:
            think_ms = self.think_time_model.sample(ctx, rng)
        else:
            think_ms = config.THINK_FLOOR_MS

        # Always use opening book if available (fastest path)
        if (self.book is not None and
            ctx.ply <= config.BOOK_PLY_CUTOFF * 2):
            book_query = self.book.query(ctx.board)
            if book_query:
                # Try to get a safe move from the book (up to 3 attempts)
                moves = list(book_query.keys())
                weights = list(book_query.values())
                for _ in range(3):
                    if rng is None:
                        chosen_uci = random.choices(moves, weights=weights, k=1)[0]
                    elif hasattr(rng, "choices"):
                        chosen_uci = rng.choices(moves, weights=weights, k=1)[0]
                    elif hasattr(rng, "choice"):
                        chosen_uci = rng.choices(moves, weights=weights, k=1)[0]  # fallback
                    else:
                        chosen_uci = random.choices(moves, weights=weights, k=1)[0]

                    chosen_move = chess.Move.from_uci(chosen_uci)
                    if chosen_move in legal_moves and self._is_move_safe(ctx.board, chosen_move):
                        model_version_str = "base" if self.is_base_fallback else f"{self.username}:v{self.model_version}"
                        return MoveDecision(
                            move=chosen_move,
                            source="book",
                            think_ms=think_ms,
                            top5=[(chosen_move.uci(), 1.0)] + [(m.uci(), 0.0) for m in legal_moves[:4]],
                            p_err=0.0 if getattr(config, "ERROR_MODEL_ENABLED", False) else None,
                            model_version=model_version_str,
                        )
                # If we couldn't find a safe move in the book after 3 attempts, fall back to model

        # Fallback to model prediction (still very fast)
        model = self.model_cache.get(self.username, self.model_version)
        planes_arr = _build_board_planes(ctx)
        meta_arr = _build_meta(ctx, ctx.bot_rating if ctx.bot_rating is not None else self.bot_rating_from_model)

        board_t = torch.from_numpy(planes_arr).unsqueeze(0).float()
        meta_t = torch.from_numpy(meta_arr).unsqueeze(0).float()

        # Handle meta dimension mismatch (e.g. 8 features passed to 10-dim model)
        expected_meta_dim = getattr(model, "meta_dim", None)
        if not isinstance(expected_meta_dim, int):
            if hasattr(model, "meta_mlp"):
                try:
                    expected_meta_dim = int(model.meta_mlp[0].in_features)
                except Exception:
                    expected_meta_dim = None
            else:
                expected_meta_dim = None
        if isinstance(expected_meta_dim, int):
            if meta_t.shape[1] < expected_meta_dim:
                pad = torch.zeros(meta_t.shape[0], expected_meta_dim - meta_t.shape[1], device=meta_t.device, dtype=meta_t.dtype)
                meta_t = torch.cat([meta_t, pad], dim=1)
            elif meta_t.shape[1] > expected_meta_dim:
                meta_t = meta_t[:, :expected_meta_dim]

        with torch.no_grad():
            out = model(board_t, meta_t)

        is_black = (ctx.board.turn == chess.BLACK)
        move_scores = []
        if "policy_logits" in out:
            raw_logits = out["policy_logits"][0].cpu().numpy()
            for m in legal_moves:
                f = (m.from_square ^ 56) if is_black else m.from_square
                t = (m.to_square ^ 56) if is_black else m.to_square
                score = float(raw_logits[f * 64 + t])
                move_scores.append((m, score))
        elif "from_sq_logits" in out:
            from_logits = out["from_sq_logits"][0].cpu().numpy()
            to_logits = out["to_sq_logits"][0].cpu().numpy()
            for m in legal_moves:
                f = (m.from_square ^ 56) if is_black else m.from_square
                t = (m.to_square ^ 56) if is_black else m.to_square
                score = float(from_logits[f] + to_logits[t])
                move_scores.append((m, score))
        else:
            move_scores = [(m, 1.0) for m in legal_moves]

        # Apply temperature if not greedy (use optimal temperature from model or config)
        if isinstance(self.temperature, (int, float)) and self.temperature != "greedy":
            eff_temp = max(1e-4, float(self.temperature))
            move_scores = [(m, score / eff_temp) for m, score in move_scores]

        # Sort moves by model score descending to get top candidates
        sorted_move_scores = sorted(move_scores, key=lambda x: x[1], reverse=True)
        top5_moves = sorted_move_scores[:5]
        top5 = [(move.uci(), float(score)) for move, score in top5_moves]
        model_version_str = "base" if self.is_base_fallback else f"{self.username}:v{self.model_version}"

        # Evaluate top 5 model predictions with Stockfish if guardrail is enabled
        if config.GUARD_ENABLED and self.stockfish_analyser is not None and top5_moves:
            # First check candidate #1 (the model's top preferred human move)
            best_cand_move, _ = top5_moves[0]
            loss_1 = self._compute_cp_loss(ctx.board, best_cand_move)
            if loss_1 <= config.GUARD_CP:
                # Top human move is tactically sound (loss <= GUARD_CP)
                return MoveDecision(
                    move=best_cand_move,
                    source="policy",
                    think_ms=think_ms,
                    top5=top5,
                    p_err=0.0 if getattr(config, "ERROR_MODEL_ENABLED", False) else None,
                    model_version=model_version_str,
                )

            # Candidate #1 is a blunder (loss > GUARD_CP)!
            # Evaluate the rest of the top 5 candidates to find the highest-ranked safe human move
            for m, _ in top5_moves[1:]:
                loss = self._compute_cp_loss(ctx.board, m)
                if loss <= config.GUARD_CP:
                    return MoveDecision(
                        move=m,
                        source="policy",
                        think_ms=think_ms,
                        top5=top5,
                        p_err=0.0 if getattr(config, "ERROR_MODEL_ENABLED", False) else None,
                        model_version=model_version_str,
                    )

            # If all top model candidates are tactical blunders, query Stockfish for the best move
            try:
                with self.stockfish_analyser._lock:
                    if self.stockfish_analyser.engine is None:
                        self.stockfish_analyser._init_engine()
                    info = self.stockfish_analyser.engine.analyse(ctx.board, chess.engine.Limit(depth=self.stockfish_analyser.depth))
                    pv = info.get("pv")
                    if pv and len(pv) > 0 and pv[0] in ctx.board.legal_moves:
                        return MoveDecision(
                            move=pv[0],
                            source="guardrail_sf",
                            think_ms=think_ms,
                            top5=top5,
                            p_err=0.0 if getattr(config, "ERROR_MODEL_ENABLED", False) else None,
                            model_version=model_version_str,
                        )
            except Exception:
                pass

            # Fallback to top-ranked move if engine query failed
            return MoveDecision(
                move=best_cand_move,
                source="policy",
                think_ms=think_ms,
                top5=top5,
                p_err=0.0 if getattr(config, "ERROR_MODEL_ENABLED", False) else None,
                model_version=model_version_str,
            )

        # Fallback if Stockfish guard is disabled: take top predicted move
        chosen_move = top5_moves[0][0] if top5_moves else legal_moves[0]

        return MoveDecision(
            move=chosen_move,
            source="policy",
            think_ms=think_ms,
            top5=top5,
            p_err=0.0 if getattr(config, "ERROR_MODEL_ENABLED", False) else None,
            model_version=model_version_str,
        )