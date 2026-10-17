"""
Opening Book V2 service (B.5): builds and queries weighted opening book from samples_v3.

Features:
- Build modes: 'eval' (train split only) and 'release' (all splits).
- Ply cutoff at config.BOOK_PLY_CUTOFF (16 plies).
- Raw visit thresholding at config.BOOK_MIN_VISITS (5 visits).
- Proportional probabilities from sample_weight sum.
- Torch-free.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any, Iterable, Sequence

import chess

from .. import config


def _fen_key(board_or_fen: chess.Board | str) -> str:
    """Return the first 4 fields of the FEN (ignores halfmove and fullmove clocks)."""
    if isinstance(board_or_fen, chess.Board):
        fen = board_or_fen.fen()
    else:
        fen = str(board_or_fen)
    return " ".join(fen.strip().split()[:4])


def build_book_v2(
    rows: Iterable[dict[str, Any]],
    weights: Sequence[float] | None = None,
    mode: str = "eval",
) -> dict[str, Any]:
    """
    Build Opening Book V2 from sample rows.

    Parameters
    ----------
    rows : Iterable[dict[str, Any]]
        Iterable of sample row dicts with 'fen', 'uci', 'ply', 'split', 'sample_weight'.
    weights : Sequence[float] | None
        Optional per-row weights aligned to rows. Default: row["sample_weight"].
    mode : str
        'eval' (train split rows only) or 'release' (all splits).

    Returns
    -------
    dict[str, Any]
        Plain JSON-serializable book dictionary with 'meta' and 'positions'.
    """
    if mode not in ("eval", "release"):
        raise ValueError(f"Invalid mode '{mode}': must be 'eval' or 'release'")

    ply_cutoff = config.BOOK_PLY_CUTOFF
    min_visits = config.BOOK_MIN_VISITS

    positions: dict[str, dict[str, dict[str, float | int]]] = {}

    row_list = list(rows)
    if weights is not None:
        if len(weights) != len(row_list):
            raise ValueError(f"weights length ({len(weights)}) does not match rows ({len(row_list)})")
        weight_iter = weights
    else:
        weight_iter = [float(r.get("sample_weight", 1.0)) for r in row_list]

    for row, w_val in zip(row_list, weight_iter):
        if mode == "eval" and row.get("split") != "train":
            continue

        ply = int(row.get("ply", 0))
        if ply > ply_cutoff:
            continue

        key = _fen_key(row["fen"])
        uci = str(row["uci"])

        if key not in positions:
            positions[key] = {}

        if uci not in positions[key]:
            positions[key][uci] = {"w": 0.0, "n": 0}

        positions[key][uci]["w"] += float(w_val)
        positions[key][uci]["n"] += 1

    return {
        "meta": {
            "mode": mode,
            "ply_cutoff": ply_cutoff,
            "min_visits": min_visits,
        },
        "positions": positions,
    }


class BookV2:
    """Opening Book V2: loads, queries, and samples moves with visit thresholding and weighted counts."""

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self.meta: dict[str, Any] = data.get("meta", {}) if data else {}
        self.positions: dict[str, dict[str, dict[str, float | int]]] = (
            data.get("positions", {}) if data else {}
        )
        self.ply_cutoff: int = self.meta.get("ply_cutoff", config.BOOK_PLY_CUTOFF)
        self.min_visits: int = self.meta.get("min_visits", config.BOOK_MIN_VISITS)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BookV2:
        """Create a BookV2 instance from a dictionary."""
        return cls(d)

    def to_dict(self) -> dict[str, Any]:
        """Convert the book to a plain JSON-serializable dictionary."""
        return {
            "meta": {
                "mode": self.meta.get("mode", "eval"),
                "ply_cutoff": self.ply_cutoff,
                "min_visits": self.min_visits,
            },
            "positions": self.positions,
        }

    def save(self, path: Path | str) -> None:
        """Save the book to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> BookV2:
        """Load a BookV2 instance from a JSON file."""
        with open(Path(path), "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    def query(self, board: chess.Board) -> dict[str, float] | None:
        """
        Query move probability distribution for the given board position.

        Returns None if board.ply() > BOOK_PLY_CUTOFF, position not seen,
        or total RAW visits < BOOK_MIN_VISITS.
        Otherwise returns probabilities proportional to WEIGHTED counts (summing to 1).
        """
        if board.ply() > self.ply_cutoff:
            return None

        key = _fen_key(board)
        if key not in self.positions:
            return None

        moves = self.positions[key]
        raw_visits = sum(m["n"] for m in moves.values())
        if raw_visits < self.min_visits:
            return None

        total_w = sum(m["w"] for m in moves.values())
        if total_w <= 0.0:
            return {uci: m["n"] / raw_visits for uci, m in moves.items()}

        return {uci: m["w"] / total_w for uci, m in moves.items()}

    def total_visits(self, board: chess.Board) -> int:
        """Return total raw visits for the given board position (0 if missing or ply > cutoff)."""
        if board.ply() > self.ply_cutoff:
            return 0

        key = _fen_key(board)
        if key not in self.positions:
            return 0

        return int(sum(m["n"] for m in self.positions[key].values()))

    def sample(self, board: chess.Board, rng: Any = None) -> chess.Move | None:
        """
        Sample a move probabilistically from the book distribution.

        Returns None if the position is not in the book or below visit threshold.
        """
        probs = self.query(board)
        if not probs:
            return None

        moves = list(probs.keys())
        weights = list(probs.values())

        if rng is None:
            chosen_uci = random.choices(moves, weights=weights, k=1)[0]
        elif hasattr(rng, "choices"):
            chosen_uci = rng.choices(moves, weights=weights, k=1)[0]
        elif hasattr(rng, "choice"):
            chosen_uci = rng.choice(moves, p=weights)
        else:
            chosen_uci = random.choices(moves, weights=weights, k=1)[0]

        return chess.Move.from_uci(chosen_uci)
