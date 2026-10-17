"""
Game filtering and cohort decontamination for base dataset construction.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional
import chess
import chess.pgn

VALID_TERMINATIONS = {"Normal", "Time forfeit"}
VALID_CATEGORIES = {"bullet", "blitz", "rapid"}


def parse_tc(tc_str: str) -> Optional[tuple[int, int]]:
    """Parse TimeControl header into (base_s, inc_s)."""
    if not tc_str or "+" not in tc_str:
        return None
    parts = tc_str.strip().split("+")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def classify_tc(base_s: int, inc_s: int) -> str:
    """Classify time control by Lichess estimated time = base + 40 * inc."""
    estimated = base_s + 40 * inc_s
    if estimated < 30:
        return "ultrabullet"
    elif estimated < 180:
        return "bullet"
    elif estimated < 480:
        return "blitz"
    elif estimated < 1500:
        return "rapid"
    else:
        return "classical"


def passes_filters(game: chess.pgn.Game, config: Optional[Any] = None) -> bool:
    """
    Check if a game passes base dataset acceptance criteria:
      1. Standard variant
      2. Event starts with "Rated"
      3. No custom FEN / SetUp != "1"
      4. classify_time_control in {bullet, blitz, rapid}
      5. base >= 60s
      6. Termination in {Normal, Time forfeit}
      7. >= 10 plies
      8. Both player ratings present and numeric
      9. %clk present on at least 4 of the mover's moves for both sides
    """
    headers = game.headers

    # 1. Standard variant
    variant = headers.get("Variant", "Standard").strip()
    if variant.lower() != "standard":
        return False

    # 2. Event starts with "Rated"
    event = headers.get("Event", "").strip()
    if not event.startswith("Rated"):
        return False

    # 3. No custom FEN
    setup = headers.get("SetUp", "0").strip()
    fen = headers.get("FEN")
    if setup == "1" or (fen is not None and fen.strip() != chess.STARTING_FEN):
        return False

    # 4 & 5. TimeControl check: base >= 60 and category in {bullet, blitz, rapid}
    tc_raw = headers.get("TimeControl", "")
    parsed_tc = parse_tc(tc_raw)
    if parsed_tc is None:
        return False
    base_s, inc_s = parsed_tc
    min_base = getattr(config, "MIN_BASE_SECONDS", 60) if config else 60
    if base_s < min_base:
        return False

    category = classify_tc(base_s, inc_s)
    if category not in VALID_CATEGORIES:
        return False

    # 6. Termination in {Normal, Time forfeit}
    termination = headers.get("Termination", "").strip()
    if termination not in VALID_TERMINATIONS:
        return False

    # 7. >= 10 plies
    plies_count = sum(1 for _ in game.mainline_moves())
    min_plies = getattr(config, "MIN_PLIES", 10) if config else 10
    if plies_count < min_plies:
        return False

    # 8. Both ratings present and numeric
    white_elo = headers.get("WhiteElo", "").strip()
    black_elo = headers.get("BlackElo", "").strip()
    if not white_elo.isdigit() or not black_elo.isdigit():
        return False

    # 9. %clk present on at least 4 of the mover's moves for both sides
    white_clk_count = 0
    black_clk_count = 0
    turn = chess.WHITE

    for node in game.mainline():
        has_clk = (node.clock() is not None) or ("%clk" in (node.comment or ""))
        if has_clk:
            if turn == chess.WHITE:
                white_clk_count += 1
            else:
                black_clk_count += 1
        turn = not turn

    min_clk_moves = 4
    if white_clk_count < min_clk_moves or black_clk_count < min_clk_moves:
        return False

    return True


def is_decontaminated(game: chess.pgn.Game, cohort: Optional[Iterable[str]]) -> bool:
    """
    Returns False if either player's name is in the cohort set (case-insensitive exact match).
    Returns True otherwise.
    """
    if not cohort:
        return True

    cohort_set = {str(name).strip().lower() for name in cohort if str(name).strip()}
    if not cohort_set:
        return True

    white = game.headers.get("White", "").strip().lower()
    black = game.headers.get("Black", "").strip().lower()

    if white in cohort_set or black in cohort_set:
        return False

    return True
