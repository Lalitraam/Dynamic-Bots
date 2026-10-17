"""Self-play harness for evaluating engine strength and behavioral fidelity (Phase SP)."""

from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path
import random
from typing import Literal
import chess
import chess.pgn

from app import config
from app.engine.provider import MoveContext, MoveDecision, MoveProvider
from app.engine.think_time import ThinkTimeModel


@dataclass
class MoveRecord:
    uci: str
    by: Literal["bot", "opponent"]
    clock_before_ms: int
    think_ms: int
    p_err: float | None


@dataclass
class GameRecord:
    pgn: str
    result: Literal["1-0", "0-1", "1/2-1/2"]
    moves: list[MoveRecord]
    termination: str  # "checkmate", "stalemate", "insufficient_material",
    # "fivefold", "75move", "flag", "cap"
    bot_color: Literal["white", "black"]

    def to_dict(self) -> dict:
        return {
            "pgn": self.pgn,
            "result": self.result,
            "termination": self.termination,
            "bot_color": self.bot_color,
            "moves": [
                {
                    "uci": m.uci,
                    "by": m.by,
                    "clock_before_ms": m.clock_before_ms,
                    "think_ms": m.think_ms,
                    "p_err": m.p_err,
                }
                for m in self.moves
            ],
        }


def _build_move_context(
    board: chess.Board,
    prev_moves: list[str],
    ply: int,
    base_s: int,
    inc_s: int,
    bot_clock_ms: int,
    opp_clock_ms: int,
    bot_rating: int,
    opp_rating: int,
) -> MoveContext:
    """Construct MoveContext from game state."""
    return MoveContext(
        board=board.copy(),
        prev_moves=list(prev_moves),
        bot_rating=bot_rating,
        opp_rating=opp_rating,
        base_seconds=base_s,
        increment_seconds=inc_s,
        bot_clock_ms=bot_clock_ms,
        opp_clock_ms=opp_clock_ms,
        ply=ply,
    )


def _pgn_from_moves(
    moves: list[chess.Move],
    bot_color: Literal["white", "black"],
    result_str: str,
    white_name: str = "White",
    black_name: str = "Black",
) -> str:
    """Build a PGN string from the move list."""
    game = chess.pgn.Game()
    game.headers["Event"] = "Self-Play Match"
    game.headers["Site"] = "dynamic_bots"
    game.headers["Date"] = "????.??.??"
    game.headers["Round"] = "1"
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    game.headers["Result"] = result_str

    node = game
    board = chess.Board()
    for m in moves:
        if m in board.legal_moves:
            node = node.add_variation(m)
            board.push(m)
        else:
            break

    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    return game.accept(exporter)


def _play_single_game(
    bot: MoveProvider,
    opponent: MoveProvider,
    bot_color: Literal["white", "black"],
    base_s: int,
    inc_s: int,
    seed: int,
    bot_rating: int = config.DEFAULT_RATING,
    opp_rating: int = config.DEFAULT_RATING,
) -> GameRecord:
    """Play one game with simulated clocks."""
    rng = random.Random(seed)
    board = chess.Board()

    bot_is_white = (bot_color == "white")
    white_clock_ms = base_s * 1000
    black_clock_ms = base_s * 1000

    moves_record: list[MoveRecord] = []
    played_moves: list[chess.Move] = []
    prev_moves_uci: list[str] = []

    termination: str = "cap"
    result: Literal["1-0", "0-1", "1/2-1/2"] = "1/2-1/2"

    max_plies = getattr(config, "SP_MAX_PLIES", 300)

    while True:
        # Check termination conditions in priority order:
        # Checkmate > stalemate > insufficient material > fivefold repetition > 75-move > cap
        if board.is_checkmate():
            termination = "checkmate"
            # If it's White's turn and checkmate, Black won (0-1). If Black's turn, White won (1-0).
            result = "0-1" if board.turn == chess.WHITE else "1-0"
            break

        if board.is_stalemate():
            termination = "stalemate"
            result = "1/2-1/2"
            break

        if board.is_insufficient_material():
            termination = "insufficient_material"
            result = "1/2-1/2"
            break

        if board.is_fivefold_repetition():
            termination = "fivefold"
            result = "1/2-1/2"
            break

        if board.is_seventyfive_moves():
            termination = "75move"
            result = "1/2-1/2"
            break

        if board.ply() >= max_plies:
            termination = "cap"
            result = "1/2-1/2"
            break

        if not any(board.legal_moves):
            termination = "cap"
            result = "1/2-1/2"
            break

        # Next player turn
        is_bot_turn = (board.turn == chess.WHITE and bot_is_white) or (
            board.turn == chess.BLACK and not bot_is_white
        )
        mover = bot if is_bot_turn else opponent
        mover_by: Literal["bot", "opponent"] = "bot" if is_bot_turn else "opponent"

        mover_clock_before = white_clock_ms if board.turn == chess.WHITE else black_clock_ms
        bot_clock_ms = white_clock_ms if bot_is_white else black_clock_ms
        opp_clock_ms = black_clock_ms if bot_is_white else white_clock_ms

        ctx = _build_move_context(
            board=board,
            prev_moves=prev_moves_uci[-2:],
            ply=board.ply(),
            base_s=base_s,
            inc_s=inc_s,
            bot_clock_ms=bot_clock_ms,
            opp_clock_ms=opp_clock_ms,
            bot_rating=bot_rating,
            opp_rating=opp_rating,
        )

        decision = mover.decide(ctx, rng)
        think_ms = decision.think_ms

        # Clock deduction and flag check
        new_clock = mover_clock_before - think_ms
        record_clock_before = mover_clock_before

        if new_clock <= 0:
            # Mover ran out of time
            if board.turn == chess.WHITE:
                white_clock_ms = 0
            else:
                black_clock_ms = 0

            moves_record.append(
                MoveRecord(
                    uci=decision.move.uci(),
                    by=mover_by,
                    clock_before_ms=record_clock_before,
                    think_ms=think_ms,
                    p_err=decision.p_err,
                )
            )
            # If White flagged, Black wins (0-1). If Black flagged, White wins (1-0).
            result = "0-1" if board.turn == chess.WHITE else "1-0"
            termination = "flag"
            break

        # Successful move: apply increment
        new_clock += inc_s * 1000
        if board.turn == chess.WHITE:
            white_clock_ms = new_clock
        else:
            black_clock_ms = new_clock

        moves_record.append(
            MoveRecord(
                uci=decision.move.uci(),
                by=mover_by,
                clock_before_ms=record_clock_before,
                think_ms=think_ms,
                p_err=decision.p_err,
            )
        )
        prev_moves_uci.append(decision.move.uci())
        played_moves.append(decision.move)
        board.push(decision.move)

    white_name = "Bot" if bot_is_white else "Opponent"
    black_name = "Opponent" if bot_is_white else "Bot"
    pgn_str = _pgn_from_moves(
        played_moves, bot_color, result, white_name=white_name, black_name=black_name
    )

    return GameRecord(
        pgn=pgn_str,
        result=result,
        moves=moves_record,
        termination=termination,
        bot_color=bot_color,
    )


def play_match(
    bot: MoveProvider,
    opponent: MoveProvider,
    n_games: int,
    base_s: int,
    inc_s: int,
    seed: int,
    bot_colors: Literal["alternate", "white", "black"] = "alternate",
    bot_rating: int = config.DEFAULT_RATING,
    opp_rating: int = config.DEFAULT_RATING,
) -> list[GameRecord]:
    """Play n_games between bot and opponent with simulated clocks."""
    results: list[GameRecord] = []

    for i in range(n_games):
        if bot_colors == "alternate":
            color: Literal["white", "black"] = "white" if (i % 2 == 0) else "black"
        elif bot_colors == "white":
            color = "white"
        else:
            color = "black"

        game_seed = seed + i * 10007
        game = _play_single_game(
            bot=bot,
            opponent=opponent,
            bot_color=color,
            base_s=base_s,
            inc_s=inc_s,
            seed=game_seed,
            bot_rating=bot_rating,
            opp_rating=opp_rating,
        )
        results.append(game)

    return results


def load_pooled_think_time(
    json_path: Path | str,
    rating_band: int | None = None,
) -> ThinkTimeModel:
    """Load ThinkTimeModel from pooled_thinktime.json, optionally filtered by rating band."""
    with open(Path(json_path), "r", encoding="utf-8") as f:
        data = json.load(f)

    buckets_data = data.get("buckets", {})
    reconstructed: dict[tuple, list[float]] = {}

    target_band_str = str(rating_band) if rating_band is not None else None

    for raw_k, samples in buckets_data.items():
        parts = raw_k.split("|")
        if len(parts) != 5:
            continue
        band, speed, phase, ratio_b, forced_s = parts
        is_forced = (forced_s.lower() == "true")

        if target_band_str is not None and band != target_band_str:
            continue

        model_key = (speed, phase, ratio_b, is_forced)
        reconstructed.setdefault(model_key, []).extend(samples)

    if not reconstructed and target_band_str is not None:
        for raw_k, samples in buckets_data.items():
            parts = raw_k.split("|")
            if len(parts) == 5:
                band, speed, phase, ratio_b, forced_s = parts
                is_forced = (forced_s.lower() == "true")
                model_key = (speed, phase, ratio_b, is_forced)
                reconstructed.setdefault(model_key, []).extend(samples)

    return ThinkTimeModel(reconstructed)
