"""
Baseline predictors for chess imitation learning evaluation.

Provides:
- UniformLegal: assigns 1/N probability to all N legal moves
- PositionFrequency: FEN-key empirical move distribution with Laplace smoothing
- CurrentModel: ChessPolicyNet evaluation without opening book (T=1, row context)
- BookThenModel: Combines OpeningBook (with epsilon smoothing) and CurrentModel
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import chess
import numpy as np
import torch
import torch.nn as nn

from . import encoding

BOOK_EVAL_EPS = 1e-3


class UniformLegal:
    """Assigns uniform probability to all legal moves in a position."""

    def legal_log_probs(self, board: chess.Board, ctx: dict) -> dict[str, float]:
        if not board.legal_moves:
            raise ValueError(f"No legal moves in position {board.fen()}")

        legal_moves = [m.uci() for m in board.legal_moves]
        n_legal = len(legal_moves)
        log_p = -math.log(n_legal)
        return {u: log_p for u in legal_moves}


class PositionFrequency:
    """
    Empirical move frequency baseline with Laplace (+1) smoothing over legal moves.

    Key is the first 4 fields of the FEN (piece placement, active color,
    castling rights, en passant square).
    Unseen positions default to uniform probability over legal moves.
    """

    def __init__(self, train_rows: Iterable[dict]) -> None:
        self.counts: dict[str, dict[str, int]] = {}
        for row in train_rows:
            fen = row["fen"]
            key = " ".join(fen.split()[:4])
            uci = row["uci"]
            if key not in self.counts:
                self.counts[key] = {}
            self.counts[key][uci] = self.counts[key].get(uci, 0) + 1

    def legal_log_probs(self, board: chess.Board, ctx: dict) -> dict[str, float]:
        if not board.legal_moves:
            raise ValueError(f"No legal moves in position {board.fen()}")

        legal_moves = [m.uci() for m in board.legal_moves]
        n_legal = len(legal_moves)

        key = " ".join(board.fen().split()[:4])
        if key not in self.counts:
            log_p = -math.log(n_legal)
            return {u: log_p for u in legal_moves}

        move_counts = self.counts[key]
        counts_per_move = [move_counts.get(u, 0) for u in legal_moves]
        total = sum(counts_per_move)
        denom = total + n_legal

        return {
            u: math.log((c + 1) / denom)
            for u, c in zip(legal_moves, counts_per_move)
        }


class CurrentModel:
    """
    Evaluates ChessPolicyNet with temperature 1, no opening book,
    and context features taken from the row.
    """

    def __init__(self, model: nn.Module, device: str = "cpu") -> None:
        self.model = model
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def legal_log_probs(self, board: chess.Board, ctx: dict) -> dict[str, float]:
        if not board.legal_moves:
            raise ValueError(f"No legal moves in position {board.fen()}")

        self.model.eval()
        board_arr = encoding.encode_board(board)
        meta_arr = encoding.encode_meta(
            player_rating=ctx["player_rating"],
            opponent_rating=ctx["opponent_rating"],
            base_seconds=ctx["base_seconds"],
            increment_seconds=ctx["increment_seconds"],
            ply=ctx["ply"],
            clock_before=ctx["clock_before"],
            clock_available=ctx["clock_available"],
            halfmove_clock=ctx["halfmove_clock"],
        )

        board_tensor = torch.from_numpy(board_arr).unsqueeze(0).to(self.device)
        meta_tensor = torch.from_numpy(meta_arr).unsqueeze(0).to(self.device)

        outputs = self.model(board=board_tensor, meta=meta_tensor)
        from_logits = outputs["from_sq_logits"][0].cpu()  # (64,)
        to_logits = outputs["to_sq_logits"][0].cpu()      # (64,)
        promo_logits = outputs["promo_logits"][0].cpu()    # (4,)

        mask = encoding.legal_move_mask(board)  # (64, 64) in mover frame
        from_mask = mask.any(axis=1)           # (64,) in mover frame

        legal_from_indices = np.where(from_mask)[0]
        sub_from = from_logits[legal_from_indices]
        from_log_probs = torch.log_softmax(sub_from, dim=-1).tolist()
        from_logp_dict = dict(zip(legal_from_indices, from_log_probs))

        move_log_probs: dict[str, float] = {}

        for f in legal_from_indices:
            f_logp = from_logp_dict[f]
            to_mask = mask[f, :]
            legal_to_indices = np.where(to_mask)[0]
            sub_to = to_logits[legal_to_indices]
            to_log_probs = torch.log_softmax(sub_to, dim=-1).tolist()
            to_logp_dict = dict(zip(legal_to_indices, to_log_probs))

            for t in legal_to_indices:
                t_logp = to_logp_dict[t]
                legal_promos = encoding.legal_promo_classes(board, int(f), int(t))
                if len(legal_promos) == 1:
                    promo_logp_dict = {legal_promos[0]: 0.0}
                else:
                    sub_promo = promo_logits[legal_promos]
                    promo_log_probs = torch.log_softmax(sub_promo, dim=-1).tolist()
                    promo_logp_dict = dict(zip(legal_promos, promo_log_probs))

                for pr in legal_promos:
                    pr_logp = promo_logp_dict[pr]
                    joint_logp = f_logp + t_logp + pr_logp
                    move = encoding.decode_move(board, int(f), int(t), int(pr))
                    move_log_probs[move.uci()] = float(joint_logp)

        return move_log_probs


class BookThenModel:
    """
    Evaluator combining an opening book with CurrentModel.

    If the opening book returns move probabilities for the board, interpolates:
      p = (1 - EPS) * p_book + EPS * p_uniform
    with EPS = BOOK_EVAL_EPS = 1e-3, ensuring finite NLL.
    Otherwise falls back to CurrentModel.
    """

    def __init__(self, model: Any, book: Any, device: str = "cpu") -> None:
        if isinstance(model, CurrentModel):
            self.model = model
        else:
            self.model = CurrentModel(model, device=device)
        self.book = book

    def legal_log_probs(self, board: chess.Board, ctx: dict) -> dict[str, float]:
        if not board.legal_moves:
            raise ValueError(f"No legal moves in position {board.fen()}")

        book_probs = self.book.query(board)
        if book_probs is not None:
            legal_moves = [m.uci() for m in board.legal_moves]
            n_legal = len(legal_moves)
            p_uniform = 1.0 / n_legal
            eps = BOOK_EVAL_EPS
            res: dict[str, float] = {}
            for uci in legal_moves:
                p_b = book_probs.get(uci, 0.0)
                p = (1.0 - eps) * p_b + eps * p_uniform
                res[uci] = math.log(p)
            return res

        return self.model.legal_log_probs(board, ctx)
