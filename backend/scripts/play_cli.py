"""
Play against a trained bot in the terminal (Milestone 4).

Talks to the running API, so start the server first:

    uvicorn app.main:app --reload

then, in a second terminal (venv active, inside backend/):

    python scripts/play_cli.py the_king_crusher
    python scripts/play_cli.py the_king_crusher --color black --temperature 0.6

Type moves in UCI notation: e2e4, g1f3, e7e8q (the last letter picks the
promotion piece). Type q to quit.
"""
import argparse
import sys

import chess
import httpx

QUIT_WORDS = {"q", "quit", "exit"}


def _detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail", response.text)
    except Exception:  # noqa: BLE001
        return response.text
    return detail if isinstance(detail, str) else str(detail)


def render_board(fen: str, human_color: str) -> str:
    """ASCII board with coordinates, from the human's point of view."""
    rows = str(chess.Board(fen)).split("\n")          # rank 8 first
    files = "a b c d e f g h"
    if human_color == "black":
        rows = [" ".join(reversed(r.split(" "))) for r in reversed(rows)]
        ranks = list(range(1, 9))
        files = "h g f e d c b a"
    else:
        ranks = list(range(8, 0, -1))
    lines = [f"{rank}  {row}" for rank, row in zip(ranks, rows)]
    return "\n".join(lines + ["", "   " + files])


def play(client, username, color="white", temperature=1.0,
         input_fn=input, print_fn=print):
    """Run one game. Returns the last game state (or None if it never started)."""
    resp = client.post(
        "/api/play/sessions",
        json={"username": username, "human_color": color, "temperature": temperature},
    )
    if resp.status_code != 201:
        print_fn(f"Could not start a game: {resp.status_code} - {_detail(resp)}")
        return None

    state = resp.json()
    print_fn(f"\nYou are {state['human_color']} against {state['username']}'s bot.")
    if state["last_bot_move"]:
        print_fn(f"Bot opens with {state['last_bot_move']['san']}.")

    while True:
        print_fn("\n" + render_board(state["fen"], state["human_color"]))
        if state["status"] == "finished":
            winner = state["winner"] or "nobody"
            print_fn(
                f"\nGame over: {state['result']} ({state['termination']}). Winner: {winner}."
            )
            return state
        if state["in_check"]:
            print_fn("You are in check!")

        move = input_fn("\nYour move (e.g. e2e4, q to quit): ").strip()
        if move.lower() in QUIT_WORDS:
            print_fn("Bye!")
            return state

        resp = client.post(
            f"/api/play/sessions/{state['session_id']}/moves", json={"move": move}
        )
        if resp.status_code == 200:
            state = resp.json()
            bot_move = state["last_bot_move"]
            if bot_move and bot_move["ply"] == len(state["moves"]):
                print_fn(f"Bot plays {bot_move['san']}.")
        else:
            print_fn(f"  ! {_detail(resp)}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Play against a trained chess bot.")
    parser.add_argument("username", help="Lichess username whose bot you want to play")
    parser.add_argument("--color", choices=["white", "black", "random"], default="white")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="0.1 = sticks to the player's favourite moves, 2.0 = more random")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args(argv)

    try:
        with httpx.Client(base_url=args.url, timeout=120) as client:
            play(client, args.username, args.color, args.temperature)
    except httpx.ConnectError:
        print(f"Could not reach {args.url}. Is the server running "
              f"(uvicorn app.main:app --reload)?")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
