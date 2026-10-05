"""
Extractor service: turns saved games.jsonl into un-split sample rows.

Uses `storage.load_games()` to read the saved games.
Replays the PGN and extracts one sample row before every move made by the target player.
"""

import io
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

        # Get base time for clock logic
        tc_raw = record["time_control"]
        parsed_tc = classifier.parse_time_control(tc_raw)
        if parsed_tc is None:
            # Should have been dropped in Milestone 1, but be safe
            base_seconds = 0.0
        else:
            base_seconds = float(parsed_tc[0])

        board = game.board()
        
        # Track the player's clock from their PREVIOUS move.
        # For their very first move, the clock is the base time (if the game has clocks at all).
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
            chess.WHITE: base_seconds if has_clocks else None,
            chess.BLACK: base_seconds if has_clocks else None,
        }

        ply = 0
        node = game
        game_has_target_moves = False

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
                    }
                    samples.append(row)

                # Push the move
                board.push(move)
                
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

        except ValueError:
            # Illegal move encountered
            drop_stats["illegal_move"] += 1
            # We already appended the valid samples up to this point; we could keep them or drop them.
            # Keeping the valid prefix is fine.
            continue

        if not game_has_target_moves:
            drop_stats["no_target_moves"] += 1

    return samples, drop_stats
