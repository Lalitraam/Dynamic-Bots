"""
Base dataset shard builder.
Streams monthly PGN archives, applies filters, decontaminates evaluation cohort,
samples 50% of plies >= 4, computes metadata and berserk flags, and writes v3 shards.
"""
from __future__ import annotations

import calendar
from collections import defaultdict
import datetime
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import chess
import chess.pgn
import pandas as pd

from .stream_pgn import stream_games
from .filter import passes_filters, is_decontaminated, parse_tc, classify_tc

SHARD_SIZE_DEFAULT = 250_000


def get_rating_band(rating: int, band_size: int = 200) -> str:
    """Group rating into fixed-width interval [low, high)."""
    low = (rating // band_size) * band_size
    high = low + band_size
    return f"{low}-{high}"


def detect_berserk(game: chess.pgn.Game, color: chess.Color, base_s: float, inc_s: float) -> bool:
    """
    Berserk heuristic:
    1. Headers first (WhiteBerserk / BlackBerserk).
    2. Else mark a side berserked when its first %clk <= base/2 + increment + 1.
    """
    tag_name = "WhiteBerserk" if color == chess.WHITE else "BlackBerserk"
    raw_header = game.headers.get(tag_name)
    if raw_header is not None:
        val = str(raw_header).strip().lower()
        return val in ("1", "true")

    # Search for first %clk annotated for this color
    board = game.board()
    first_clk = None
    for node in game.mainline():
        mover = board.turn
        if mover == color:
            clk = node.clock()
            if clk is not None:
                first_clk = clk
                break
        board.push(node.move)

    if first_clk is not None:
        threshold = (base_s / 2.0) + inc_s + 1.0
        return first_clk <= threshold

    return False


def _parse_game_date(game: chess.pgn.Game) -> Optional[datetime.date]:
    """Parse game UTCDate or Date header into datetime.date."""
    raw = game.headers.get("UTCDate") or game.headers.get("Date")
    if not raw or raw.startswith("?"):
        return None
    raw = raw.replace(".", "-")
    try:
        parts = raw.split("-")
        return datetime.date(int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        return None


def _get_holdout_dates_for_month(month_spec: Any, games: Optional[List[chess.pgn.Game]] = None) -> Tuple[Set[datetime.date], Set[datetime.date]]:
    """
    Return (val_dates, test_dates) for the last month.
    Final 3 days -> base_test; 3 days before -> base_val.
    """
    m_str = str(month_spec)
    match = re.search(r"(\d{4})[-_](\d{2})", m_str)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        max_day = calendar.monthrange(year, month)[1]
        end_d = datetime.date(year, month, max_day)
    elif games:
        dates = [_parse_game_date(g) for g in games]
        valid_dates = [d for d in dates if d is not None]
        end_d = max(valid_dates) if valid_dates else datetime.date(2026, 1, 31)
    else:
        end_d = datetime.date(2026, 1, 31)

    test_dates = {end_d - datetime.timedelta(days=i) for i in range(3)}
    val_dates = {end_d - datetime.timedelta(days=i) for i in range(3, 6)}
    return val_dates, test_dates


def build_shards(
    months: Union[Iterable[Any], Any],
    out_dir: Union[str, Path],
    config: Optional[Any] = None,
    cohort: Optional[Iterable[str]] = None,
) -> dict:
    """
    Stream games across months, apply filtering & decontamination,
    sample 50% of plies >= 4, and write v3 shards.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(months, (str, Path)) or not isinstance(months, (list, tuple)):
        months_list = [months]
    else:
        months_list = list(months)

    if not months_list:
        raise ValueError("months list cannot be empty")

    shard_size = getattr(config, "SHARD_SIZE_V3", SHARD_SIZE_DEFAULT) if config else SHARD_SIZE_DEFAULT
    if isinstance(config, dict) and "shard_size" in config:
        shard_size = config["shard_size"]

    band_cap = getattr(config, "BAND_CAP", None) if config else None
    if isinstance(config, dict) and "band_cap" in config:
        band_cap = config["band_cap"]

    # Load default cohort if None
    if cohort is None:
        cohort = ["the_king_crusher"]

    # Pre-determine holdout dates for the final month
    last_month_spec = months_list[-1]
    val_dates, test_dates = _get_holdout_dates_for_month(last_month_spec)

    total_games_streamed = 0
    passed_games_count = 0
    decontaminated_games_count = 0

    split_game_ids: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}
    split_buffers: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    split_shard_counters: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    split_position_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    cell_counts: Dict[str, int] = defaultdict(int)
    shards_written: List[str] = []

    def _flush_buffer(split_name: str, force: bool = False):
        buf = split_buffers[split_name]
        if not buf:
            return
        if force or len(buf) >= shard_size:
            chunk = buf[:shard_size] if not force else buf
            split_buffers[split_name] = buf[shard_size:] if not force else []
            idx = split_shard_counters[split_name]
            split_shard_counters[split_name] += 1
            filename = f"{split_name}_shard_{idx:04d}.parquet"
            target_path = out_dir / filename
            df = pd.DataFrame(chunk)
            df.to_parquet(target_path, index=False)
            shards_written.append(str(target_path))

    for m_idx, month_item in enumerate(months_list):
        is_last_month = (m_idx == len(months_list) - 1)
        
        # Stream games from file/URL
        for game in stream_games(month_item):
            total_games_streamed += 1

            if not passes_filters(game, config):
                continue
            passed_games_count += 1

            if not is_decontaminated(game, cohort):
                continue
            decontaminated_games_count += 1

            # Game ID
            site = game.headers.get("Site", "")
            game_id = site.rstrip("/").split("/")[-1] if "/" in site else f"g_{total_games_streamed}"
            if not game_id:
                game_id = f"g_{total_games_streamed}"

            # Determine split
            g_date = _parse_game_date(game)
            if is_last_month and g_date is not None:
                if g_date in test_dates:
                    split = "test"
                elif g_date in val_dates:
                    split = "val"
                else:
                    split = "train"
            else:
                split = "train"

            split_game_ids[split].add(game_id)

            # Metadata from headers
            base_s, inc_s = parse_tc(game.headers.get("TimeControl", "180+0")) or (180, 0)
            category = classify_tc(base_s, inc_s)
            is_fast = category in ("bullet", "blitz")
            time_control_str = game.headers.get("TimeControl", f"{base_s}+{inc_s}")
            white_rating = int(game.headers.get("WhiteElo", "1500"))
            black_rating = int(game.headers.get("BlackElo", "1500"))
            termination = game.headers.get("Termination", "Normal")
            game_result_raw = game.headers.get("Result", "*")

            white_berserk = detect_berserk(game, chess.WHITE, base_s, inc_s)
            black_berserk = detect_berserk(game, chess.BLACK, base_s, inc_s)

            # Parse mainline plies
            board = game.board()
            moves_history: List[str] = []
            last_clock = {chess.WHITE: float(base_s), chess.BLACK: float(base_s)}
            prev_had_clk = {chess.WHITE: False, chess.BLACK: False}
            moves_count = {chess.WHITE: 0, chess.BLACK: 0}

            ply = 0
            for node in game.mainline():
                move = node.move
                mover_color = board.turn
                player_color_str = "white" if mover_color == chess.WHITE else "black"
                mover_rating = white_rating if mover_color == chess.WHITE else black_rating
                opp_rating = black_rating if mover_color == chess.WHITE else white_rating
                mover_berserk = white_berserk if mover_color == chess.WHITE else black_berserk

                effective_increment = 0.0 if mover_berserk else float(inc_s)
                clock_before = last_clock[mover_color]
                node_clk = node.clock()
                clock_after = float(node_clk) if node_clk is not None else None

                if moves_count[mover_color] == 0:
                    time_spent_s = None
                elif (not prev_had_clk[mover_color]) or clock_before is None or clock_after is None:
                    time_spent_s = None
                else:
                    time_spent_s = max(0.0, float(clock_before - clock_after + effective_increment))

                # Game result score for mover
                if game_result_raw == "1-0":
                    score = 1.0 if mover_color == chess.WHITE else 0.0
                elif game_result_raw == "0-1":
                    score = 0.0 if mover_color == chess.WHITE else 1.0
                elif game_result_raw == "1/2-1/2":
                    score = 0.5
                else:
                    score = None

                band_str = get_rating_band(mover_rating)
                cell_key = f"{band_str}|{category}"

                # Sample 50% of plies >= 4 from both sides
                # Plies >= 4: ply 4 (W), ply 5 (B), ply 6 (W), ply 7 (B)...
                # Taking (ply // 2) % 2 == 0 gives exactly 50% deterministic alternate moves per side
                if ply >= 4 and ((ply // 2) % 2 == 0):
                    # Band balancing check if band_cap is set (stratified by band x speed)
                    if band_cap is None or cell_counts[cell_key] < band_cap:
                        row = {
                            "game_id": game_id,
                            "ply": ply,
                            "fen": board.fen(),
                            "uci": move.uci(),
                            "date": str(g_date) if g_date else None,
                            "category": category,
                            "is_fast": is_fast,
                            "time_control": time_control_str,
                            "player_rating": mover_rating,
                            "opponent_rating": opp_rating,
                            "color": player_color_str,
                            "termination": termination,
                            "clock_before_move": clock_before,
                            "halfmove_clock": board.halfmove_clock,
                            "sample_weight": 1.0,
                            "split": split,
                            "schema_version": "v3",
                            "target_berserk": mover_berserk,
                            "prev_move_1": moves_history[-1] if len(moves_history) >= 1 else None,
                            "prev_move_2": moves_history[-2] if len(moves_history) >= 2 else None,
                            "clock_after_move": clock_after,
                            "time_spent_s": time_spent_s,
                            "increment_s": inc_s,
                            "base_s": base_s,
                            "created_at_ms": None,
                            "game_result_for_player": score,
                            "num_legal_moves": board.legal_moves.count(),
                        }
                        split_buffers[split].append(row)
                        split_position_counts[split] += 1
                        cell_counts[cell_key] += 1
                        _flush_buffer(split, force=False)

                # Step forward
                if clock_after is not None:
                    last_clock[mover_color] = clock_after
                prev_had_clk[mover_color] = (clock_after is not None)
                moves_count[mover_color] += 1
                moves_history.append(move.uci())
                board.push(move)
                ply += 1

    # Flush remaining buffers
    for split_name in ("train", "val", "test"):
        _flush_buffer(split_name, force=True)

    # Assert game-level disjointness across splits
    assert split_game_ids["train"].isdisjoint(split_game_ids["val"]), "Train and Val splits share games!"
    assert split_game_ids["train"].isdisjoint(split_game_ids["test"]), "Train and Test splits share games!"
    assert split_game_ids["val"].isdisjoint(split_game_ids["test"]), "Val and Test splits share games!"

    return {
        "total_games_streamed": total_games_streamed,
        "games_passed_filters": passed_games_count,
        "games_decontaminated": decontaminated_games_count,
        "split_game_counts": {
            "train": len(split_game_ids["train"]),
            "val": len(split_game_ids["val"]),
            "test": len(split_game_ids["test"]),
        },
        "split_position_counts": split_position_counts,
        "cell_counts": dict(cell_counts),
        "shards_written": shards_written,
    }
