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
from app.ml.encoding_v3 import encode_board_v3, encode_meta_v3
from app.services.opening_book_v2 import BookV2

_PROMO_TO_CLASS = {chess.QUEEN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, None: 0}


def _build_board_planes(ctx: MoveContext) -> np.ndarray:
    """Build 22-plane board encoding from context."""
    prev_1 = ctx.prev_moves[-1] if len(ctx.prev_moves) >= 1 else None
    prev_2 = ctx.prev_moves[-2] if len(ctx.prev_moves) >= 2 else None
    return encode_board_v3(
        ctx.board,
        prev_move_1=prev_1,
        prev_move_2=prev_2,
        history=True,
    )


def _build_meta(ctx: MoveContext, bot_rating_from_model: int) -> np.ndarray:
    """Build 10-dimensional metadata vector from context."""
    rating = bot_rating_from_model if bot_rating_from_model is not None else ctx.bot_rating
    clock_before = (ctx.bot_clock_ms / 1000.0) if ctx.bot_clock_ms is not None else None
    clock_avail = bool(ctx.bot_clock_ms is not None and ctx.base_seconds > 0)
    return encode_meta_v3(
        player_rating=rating,
        opponent_rating=ctx.opp_rating,
        base_seconds=ctx.base_seconds,
        increment_seconds=ctx.increment_seconds,
        ply=ctx.ply,
        clock_before=clock_before,
        clock_available=clock_avail,
        halfmove_clock=ctx.board.halfmove_clock,
        num_legal_moves=ctx.board.legal_moves.count(),
        berserk=False,
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
                    temperature = float(t_data.get("T_best", t_data.get("optimal_temperature", 1.0)))
            except Exception:
                pass
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

    def _compute_cp_loss(self, board: chess.Board, move: chess.Move) -> float:
        """Calculate centipawn loss of move using Stockfish if available, else return 0.0."""
        if self.stockfish_analyser is not None:
            try:
                loss, _ = compute_cp_loss(board, move, self.stockfish_analyser)
                return loss
            except Exception:
                return 0.0
        return 0.0

    def decide(self, ctx: MoveContext, rng: random.Random) -> MoveDecision:
        """Runtime move decision pipeline."""
        legal_moves = list(ctx.board.legal_moves)
        if not legal_moves:
            raise ValueError(f"No legal moves in position {ctx.board.fen()}")

        think_ms = self.think_time_model.sample(ctx, rng)

        model = self.model_cache.get(self.username, self.model_version)
        planes_arr = _build_board_planes(ctx)
        meta_arr = _build_meta(ctx, ctx.bot_rating if ctx.bot_rating is not None else self.bot_rating_from_model)

        board_t = torch.from_numpy(planes_arr).unsqueeze(0).float()
        meta_t = torch.from_numpy(meta_arr).unsqueeze(0).float()

        with torch.no_grad():
            out = model(board_t, meta_t)
            raw_logits = out["policy_logits"][0].cpu().numpy()

        is_black = (ctx.board.turn == chess.BLACK)
        legal_lps: dict[str, float] = {}
        for m in legal_moves:
            f = (m.from_square ^ 56) if is_black else m.from_square
            t = (m.to_square ^ 56) if is_black else m.to_square
            legal_lps[m.uci()] = float(raw_logits[f * 64 + t])

        if legal_lps:
            max_l = max(legal_lps.values())
            sum_exp = sum(math.exp(v - max_l) for v in legal_lps.values())
            log_sum = max_l + math.log(max(1e-12, sum_exp))
            policy_lps = {u: (v - log_sum) for u, v in legal_lps.items()}
        else:
            policy_lps = None

        # Error probability model prediction and temperature determination
        clock_sec = (ctx.bot_clock_ms / 1000.0) if ctx.bot_clock_ms is not None else None
        if self.error_model is not None:
            feats = compute_features(
                board=ctx.board,
                clock_before=clock_sec,
                base_s=ctx.base_seconds,
                ply=ctx.ply,
                policy_log_probs=policy_lps,
            )
            p_err = float(self.error_model.predict(feats)[0])
        else:
            p_err = float(getattr(config, "ERROR_MODEL_FALLBACK_RATE", 0.08))

        u = rng.random()
        t_base = self.temperature
        if u < p_err:
            eff_temp = max(1e-4, t_base * 1.1)
        else:
            eff_temp = max(1e-4, t_base)

        policy_logits = raw_logits / eff_temp

        is_black = (ctx.board.turn == chess.BLACK)
        has_promo = any(m.promotion is not None for m in legal_moves)
        sq = None
        if has_promo:
            with torch.no_grad():
                x = F.relu(model.stem_bn(model.stem_conv(board_t)))
                meta_emb = model.meta_mlp(meta_t)
                for block, film in zip(model.blocks, model.film_generators):
                    x = block(x)
                    gb = film(meta_emb)
                    gamma, beta = gb.chunk(2, dim=1)
                    x = x * (1.0 + gamma.unsqueeze(-1).unsqueeze(-1)) + beta.unsqueeze(-1).unsqueeze(-1)
                sq = x.permute(0, 2, 3, 1).reshape(1, 64, model.channels)

        move_unnorm_logp: list[float] = []
        for m in legal_moves:
            f = (m.from_square ^ 56) if is_black else m.from_square
            t = (m.to_square ^ 56) if is_black else m.to_square
            idx = f * 64 + t
            promo_logp = 0.0
            if m.promotion is not None and sq is not None:
                promo_cls = _PROMO_TO_CLASS.get(m.promotion, 0)
                promo_in = torch.cat([sq[0, f], sq[0, t]], dim=-1).unsqueeze(0)
                promo_log = model.promo_head(promo_in) / eff_temp
                promo_logp = float(F.log_softmax(promo_log, dim=-1)[0, promo_cls].item())
            move_unnorm_logp.append(float(policy_logits[idx]) + promo_logp)

        lp_arr = np.array(move_unnorm_logp, dtype=float)
        max_lp = np.max(lp_arr)
        policy_probs = np.exp(lp_arr - max_lp)
        policy_probs = policy_probs / policy_probs.sum()

        is_book = False
        source: Literal["book", "policy", "guarded"] = "policy"
        final_probs = policy_probs

        if self.book is not None and ctx.ply <= config.BOOK_PLY_CUTOFF:
            raw_visits = self.book.total_visits(ctx.board)
            if raw_visits >= config.BOOK_MIN_VISITS:
                book_query = self.book.query(ctx.board)
                if book_query:
                    alpha = float(raw_visits) / float(raw_visits + config.BOOK_ALPHA_K)
                    blended = np.zeros(len(legal_moves), dtype=float)
                    for i, m in enumerate(legal_moves):
                        p_book = book_query.get(m.uci(), 0.0)
                        blended[i] = alpha * p_book + (1.0 - alpha) * policy_probs[i]
                    s = blended.sum()
                    if s > 0:
                        final_probs = blended / s
                        source = "book"
                        is_book = True

        if not is_book:
            filtered_probs = _apply_top_p(policy_probs, config.TOP_P)
            s = filtered_probs.sum()
            if s > 0:
                final_probs = filtered_probs / s
            else:
                final_probs = policy_probs
            source = "policy"

        sorted_order = np.argsort(-final_probs)
        top5 = [(legal_moves[i].uci(), float(final_probs[i])) for i in sorted_order[:5]]

        chosen_idx = _sample_from(final_probs, rng)
        chosen_move = legal_moves[chosen_idx]

        # If in error branch (u < p_err), cap blunder if Stockfish is enabled
        if u < p_err and len(legal_moves) > 1:
            loss = self._compute_cp_loss(ctx.board, chosen_move)
            cap = min(self.player_p99, 600.0)
            if loss > cap:
                # Blunder exceeded cap -> resample once excluding this move
                resample_probs = final_probs.copy()
                resample_probs[chosen_idx] = 0.0
                if resample_probs.sum() > 0:
                    chosen_idx = _sample_from(resample_probs, rng)
                    chosen_move = legal_moves[chosen_idx]

        model_version_str = "base" if self.is_base_fallback else f"{self.username}:v{self.model_version}"
        return MoveDecision(
            move=chosen_move,
            source=source,
            think_ms=think_ms,
            top5=top5,
            p_err=p_err,
            model_version=model_version_str,
        )
