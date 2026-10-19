"""
Extractor service: turns saved games.jsonl into un-split sample rows.

Uses `storage.load_games()` to read the saved games.
Replays the PGN and extracts one sample row before every move made by the target player.
"""

import io
from datetime import datetime, timezone
import chess
import chess.pgn
from typing import Any

from . import storage
from . import classifier


def extract_samples(username: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Extract supervised learning samples from the saved games.jsonl.

    Parameters
    ----------
    username : str
        The player's username.

    Returns
    -------
    tuple
        (samples, drop_stats)
        samples : list of dict, where each dict is a row ready for Parquet/Dataset.
        drop_stats : dict tracking reasons games were dropped during extraction.
    """
    games_data = storage.load_games(username)

    samples = []
    drop_stats = {
        "unparseable_pgn": 0,
        "illegal_move": 0,
        "no_target_moves": 0,
    }

    for record in games_data:
        game_id = record["game_id"]
        pgn = record["pgn"]
        player_color_str = record["player_color"]
        is_player_white = (player_color_str == "white")

        # Re-parse the game
        try:
            game = chess.pgn.read_game(io.StringIO(pgn))
        except Exception:
            game = None

        if game is None or game.errors:
            drop_stats["unparseable_pgn"] += 1
            continue

        # Get base time and increment for clock logic
        tc_raw = record["time_control"]
        parsed_tc = classifier.parse_time_control(tc_raw)
        if parsed_tc is None:
            # Should have been dropped in Milestone 1, but be safe
            base_seconds = 0.0
            base_s = 0
            increment_s = 0
        else:
            base_seconds = float(parsed_tc[0])
            base_s = int(parsed_tc[0])
            increment_s = int(parsed_tc[1])

        # Game UTC timestamp in ms
        dt = record["date"]
        if isinstance(dt, str):
            dt = datetime.fromisoformat(dt)
        if hasattr(dt, "tzinfo") and dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        created_at_ms = int(dt.timestamp() * 1000)

        board = game.board()
        
        # Check Berserk headers
        headers = game.headers
        white_berserk = (headers.get("WhiteBerserk", "").strip().lower() == "true")
        black_berserk = (headers.get("BlackBerserk", "").strip().lower() == "true")
        target_berserk = white_berserk if is_player_white else black_berserk

        # Game result for target player (1.0 win, 0.5 draw, 0.0 loss)
        result_header = headers.get("Result", "")
        if result_header == "1-0":
            game_result_for_player = 1.0 if is_player_white else 0.0
        elif result_header == "0-1":
            game_result_for_player = 0.0 if is_player_white else 1.0
        elif result_header == "1/2-1/2":
            game_result_for_player = 0.5
        else:
            game_result_for_player = 0.5

        # If a player berserked, their initial clock is base_seconds / 2 (opponent berserk does not change target clock)
        white_initial = (base_seconds / 2.0) if white_berserk else base_seconds
        black_initial = (base_seconds / 2.0) if black_berserk else base_seconds

        # Track the player's clock from their PREVIOUS move.
        # For their very first move, the clock is the initial time (if the game has clocks at all).
        # We need to know if the game has ANY clock annotations.
        has_clocks = False
        node_ptr = game
        while node_ptr.variations:
            node_ptr = node_ptr.variation(0)
            if node_ptr.clock() is not None:
                has_clocks = True
                break
        
        # We store the clock *after* each move.
        # last_clock[White] is the clock after White's last move.
        last_clock = {
            chess.WHITE: white_initial if has_clocks else None,
            chess.BLACK: black_initial if has_clocks else None,
        }

        game_rows = []
        ply = 0
        node = game
        game_has_target_moves = False
        moves_history = []
        target_moves_count = 0
        target_prev_move_had_clk = False

        try:
            while node.variations:
                next_node = node.variation(0)
                move = next_node.move

                # Is it the target player's turn to move?
                is_target_turn = (board.turn == chess.WHITE and is_player_white) or \
                                 (board.turn == chess.BLACK and not is_player_white)

                if is_target_turn:
                    game_has_target_moves = True
                    clock_before_move = last_clock[board.turn]

                    prev_move_1 = moves_history[-1] if len(moves_history) >= 1 else None
                    prev_move_2 = moves_history[-2] if len(moves_history) >= 2 else None
                    num_legal_moves = board.legal_moves.count()

                    node_clock = next_node.clock()
                    clock_after_move = float(node_clock) if node_clock is not None else None

                    # Rule for time_spent_s:
                    # 1. Berserk increment: When target berserks, increment is removed.
                    #    Evidence: 0 of 5777 own-move transitions had d < 0 in header-berserk games
                    #    with increment; 3059 of 7603 in non-berserk games.
                    #    effective_increment = 0 if target_berserk else increment_s.
                    # 2. Stale clock: If the target's previous move lacked a %clk tag,
                    #    clock_before_move is a carried-forward estimate, so time_spent_s cannot be
                    #    accurately computed and must be None. clock_before_move itself retains its
                    #    carry-forward behaviour unchanged.
                    # 3. First move: null (no elapsed time / increment applied prior to move 1).
                    # 4. Missing %clk: null (if clock_after_move is None or clock_before_move is None).
                    effective_increment = 0 if target_berserk else increment_s

                    if target_moves_count == 0:
                        time_spent_s = None
                    elif (not target_prev_move_had_clk) or clock_before_move is None or clock_after_move is None:
                        time_spent_s = None
                    else:
                        time_spent_s = max(0.0, float(clock_before_move - clock_after_move + effective_increment))

                    target_prev_move_had_clk = (node_clock is not None)
                    target_moves_count += 1
                    
                    row = {
                        "game_id": game_id,
                        "ply": ply,
                        "fen": board.fen(),
                        "uci": move.uci(),
                        "date": record["date"],  # datetime object
                        "category": record["category"],
                        "is_fast": record["is_fast"],
                        "time_control": record["time_control"],
                        "player_rating": record["rating"],
                        "opponent_rating": record["opponent_rating"],
                        "color": player_color_str,
                        "termination": record["termination"],
                        "clock_before_move": clock_before_move,
                        "halfmove_clock": board.halfmove_clock,
                        "sample_weight": 1.0,
                        "target_berserk": target_berserk,
                        "prev_move_1": prev_move_1,
                        "prev_move_2": prev_move_2,
                        "clock_after_move": clock_after_move,
                        "time_spent_s": time_spent_s,
                        "increment_s": increment_s,
                        "base_s": base_s,
                        "created_at_ms": created_at_ms,
                        "game_result_for_player": game_result_for_player,
                        "num_legal_moves": num_legal_moves,
                    }
                    game_rows.append(row)

                # Push the move
                board.push(move)
                moves_history.append(move.uci())
                
                # Update the clock for the player who just moved
                # Use the clock attached to the node we just entered
                node_clock = next_node.clock()
                if node_clock is not None:
                    last_clock[not board.turn] = node_clock
                else:
                    # Decision (Milestone 2): when a game HAS clock annotations overall but one
                    # specific move's %clk tag is absent, we carry the last-known clock value
                    # forward unchanged (last_clock is NOT modified here) and still treat that
                    # reading as clock_available=True for feature purposes.
                    #
                    # Rationale: a slightly stale clock estimate is genuinely different from a
                    # game that has NO clock data at all (which sets clock_available=False for
                    # every move).  A carried-forward value gives the model a real, if slightly
                    # inaccurate, signal rather than forcing the fixed 0.5 default.
                    #
                    # Implementation: last_clock[color] is simply left unchanged, so the next
                    # time that player's turn comes around clock_before_move will hold the last
                    # annotated value.  clock_available is True (set at sample-row build time
                    # when last_clock[board.turn] is not None).
                    pass  # last_clock carries forward — no update needed

                node = next_node
                ply += 1

        except (ValueError, AssertionError):
            # Illegal move encountered: discard partial rows for this game
            drop_stats["illegal_move"] += 1
            continue

        if not game_has_target_moves:
            drop_stats["no_target_moves"] += 1

        # Quality filtering: remove games with extreme time anomalies or obvious blunders
        filtered_game_rows = []
        for row in game_rows:
            # Skip games where player lost on time or made premature resignations
            if row.get('termination') in ['timeforfeit', 'resigned']:
                continue
            # Skip games with abnormally low move times (premoves) or extremely high times
            # Only check if we have time_spent_s data
            time_spent = row.get('time_spent_s')
            if time_spent is not None:
                # Skip moves with less than 50ms (likely premoves/accidental clicks)
                # or more than 300 seconds (5 minutes per move - likely disconnected)
                if time_spent < 0.05 or time_spent > 300:
                    continue
            filtered_game_rows.append(row)

        samples.extend(filtered_game_rows)

        # Track filtered games for debugging
        if len(filtered_game_rows) < len(game_rows):
            drop_stats["quality_filtered"] = drop_stats.get("quality_filtered", 0) + (len(game_rows) - len(filtered_game_rows))

    return samples, drop_stats

