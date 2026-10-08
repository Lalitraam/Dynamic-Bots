"""
Tests for game sessions (Milestone 4, Phase 3).
Pure python-chess + fake movers: no model, no torch, no HTTP.
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import chess
import pytest

from app import config
from app.services.game_sessions import (
    BotMoveError,
    GameOverError,
    GameSession,
    IllegalMoveError,
    InvalidMoveFormatError,
    NotYourTurnError,
    SessionLimitError,
    SessionManager,
    SessionNotFoundError,
)
from app.services.model_cache import LoadedBot

# Positions used below (all verified with python-chess)
FEN_AFTER_F3_E5 = "rnbqkbnr/pppp1ppp/8/4p3/8/5P2/PPPPP1PP/RNBQKBNR w KQkq e6 0 2"
FEN_FOOLS_MATE_BLACK_TO_MOVE = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq g3 0 2"
FEN_PROMOTION = "8/P7/8/8/8/8/k6K/8 w - - 0 1"
FEN_STALEMATE_IN_ONE = "k7/8/2K5/8/8/4Q3/8/8 w - - 0 1"


def make_bot(name="botplayer"):
    return LoadedBot(name, object(), object(), (1, 1), time.time())


class FirstLegalMover:
    """Deterministic bot: always plays the first legal move. Records its calls."""

    def __init__(self):
        self.calls = []

    def __call__(self, bot, board, meta, temperature):
        self.calls.append({"fen": board.fen(), "meta": dict(meta), "temperature": temperature})
        return next(iter(board.legal_moves))


class ScriptedMover:
    """Bot that plays a fixed list of UCI moves."""

    def __init__(self, ucis):
        self.ucis = list(ucis)

    def __call__(self, bot, board, meta, temperature):
        return chess.Move.from_uci(self.ucis.pop(0))


def failing_mover(bot, board, meta, temperature):
    raise RuntimeError("model exploded")


def illegal_mover(bot, board, meta, temperature):
    return chess.Move.from_uci("a1a8")      # never legal from these positions


def new_manager(mover=None, **kw):
    return SessionManager(mover=mover or FirstLegalMover(), **kw)


def create(manager, human_color=chess.WHITE, board=None, **kw):
    return manager.create(
        username="botplayer",
        bot=make_bot(),
        human_color=human_color,
        temperature=kw.pop("temperature", 1.0),
        board=board,
        **kw,
    )


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------

def test_create_as_white_human_moves_first():
    session = create(new_manager(), chess.WHITE)
    snap = session.snapshot()
    assert snap["status"] == "active"
    assert snap["turn"] == "white" and snap["human_to_move"] is True
    assert snap["moves"] == [] and snap["last_bot_move"] is None
    assert len(snap["legal_moves"]) == 20
    assert snap["human_color"] == "white" and snap["bot_color"] == "black"


def test_create_as_black_bot_opens_immediately():
    mover = FirstLegalMover()
    session = create(new_manager(mover), chess.BLACK, human_rating=1700, bot_rating=2100,
                     temperature=0.7)
    snap = session.snapshot()
    assert len(snap["moves"]) == 1 and snap["moves"][0]["by"] == "bot"
    assert snap["turn"] == "black" and snap["human_to_move"] is True
    # The mover got the bot's perspective of the metadata and the temperature.
    call = mover.calls[0]
    assert call["meta"] == {"player_rating": 2100, "opponent_rating": 1700}
    assert call["temperature"] == 0.7


def test_bot_failure_when_opening_registers_nothing():
    manager = new_manager(failing_mover)
    with pytest.raises(BotMoveError):
        create(manager, chess.BLACK)
    assert len(manager) == 0


# ---------------------------------------------------------------------------
# A full turn
# ---------------------------------------------------------------------------

def test_play_turn_applies_human_then_bot():
    manager = new_manager()
    session = create(manager)
    snap = session.play_turn("e2e4", manager.mover)

    assert [m["by"] for m in snap["moves"]] == ["human", "bot"]
    assert snap["moves"][0]["uci"] == "e2e4" and snap["moves"][0]["san"] == "e4"
    assert snap["moves"][0]["ply"] == 1 and snap["moves"][1]["ply"] == 2
    assert snap["turn"] == "white" and snap["human_to_move"] is True
    assert snap["last_bot_move"] == snap["moves"][1]


def test_uci_input_is_trimmed_and_case_insensitive():
    manager = new_manager()
    session = create(manager)
    snap = session.play_turn("  E2E4 ", manager.mover)
    assert snap["moves"][0]["uci"] == "e2e4"


@pytest.mark.parametrize("bad", ["", "e2", "e2e4e5", "zzzz", "0000", "e2e9", "e7e8k", 42, None])
def test_malformed_moves_rejected_and_board_unchanged(bad):
    manager = new_manager()
    session = create(manager)
    before = session.board.fen()
    with pytest.raises(InvalidMoveFormatError):
        session.play_turn(bad, manager.mover)
    assert session.board.fen() == before and session.moves == []


@pytest.mark.parametrize("illegal", ["e2e5", "e1e2", "a1a3", "e7e5", "e2e4q"])
def test_illegal_moves_rejected_and_board_unchanged(illegal):
    manager = new_manager()
    session = create(manager)
    before = session.board.fen()
    with pytest.raises(IllegalMoveError):
        session.play_turn(illegal, manager.mover)
    assert session.board.fen() == before and session.moves == []


def test_promotion_requires_a_piece_letter():
    manager = new_manager()
    session = create(manager, board=chess.Board(FEN_PROMOTION))
    with pytest.raises(IllegalMoveError, match="Promotion piece required"):
        session.play_turn("a7a8", manager.mover)
    snap = session.play_turn("a7a8q", manager.mover)
    assert snap["moves"][0]["san"].startswith("a8=Q")   # also gives check


def test_not_your_turn():
    # Built directly: a position where it is the bot's turn and the bot has not moved.
    session = GameSession(
        "s1", "botplayer", make_bot(), chess.WHITE, 1.0, 1500, 1500,
        board=chess.Board(FEN_FOOLS_MATE_BLACK_TO_MOVE),
    )
    with pytest.raises(NotYourTurnError):
        session.play_turn("a2a3", FirstLegalMover())


# ---------------------------------------------------------------------------
# Game over
# ---------------------------------------------------------------------------

def test_human_checkmate_ends_game_without_bot_reply():
    manager = new_manager()
    session = create(manager, chess.BLACK, board=chess.Board(FEN_FOOLS_MATE_BLACK_TO_MOVE))
    snap = session.play_turn("d8h4", manager.mover)
    assert snap["status"] == "finished"
    assert snap["result"] == "0-1" and snap["winner"] == "black"
    assert snap["termination"] == "checkmate"
    assert [m["by"] for m in snap["moves"]] == ["human"]      # bot did not move
    assert snap["legal_moves"] == [] and snap["human_to_move"] is False
    with pytest.raises(GameOverError):
        session.play_turn("a2a3", manager.mover)


def test_bot_checkmate_is_reported():
    manager = new_manager(ScriptedMover(["d8h4"]))
    session = create(manager, chess.WHITE, board=chess.Board(FEN_AFTER_F3_E5))
    snap = session.play_turn("g2g4", manager.mover)
    assert snap["status"] == "finished"
    assert snap["winner"] == "black" and snap["termination"] == "checkmate"
    assert snap["last_bot_move"]["san"] == "Qh4#"


def test_stalemate_is_a_draw():
    manager = new_manager()
    session = create(manager, board=chess.Board(FEN_STALEMATE_IN_ONE))
    snap = session.play_turn("e3b6", manager.mover)
    assert snap["status"] == "finished"
    assert snap["result"] == "1/2-1/2" and snap["winner"] is None
    assert snap["termination"] == "stalemate"


def test_threefold_repetition_ends_the_game():
    manager = new_manager(ScriptedMover(["g8f6", "f6g8", "g8f6", "f6g8"]))
    session = create(manager)
    snap = None
    for uci in ["g1f3", "f3g1", "g1f3", "f3g1"]:
        snap = session.play_turn(uci, manager.mover)
    assert snap["status"] == "finished"
    assert snap["result"] == "1/2-1/2"
    assert snap["termination"] == "threefold_repetition"


# ---------------------------------------------------------------------------
# Bot failures roll the human move back
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_mover", [failing_mover, illegal_mover])
def test_bot_failure_rolls_back_and_allows_retry(bad_mover):
    manager = new_manager(bad_mover)
    session = create(manager)
    start_fen = session.board.fen()

    with pytest.raises(BotMoveError):
        session.play_turn("e2e4", manager.mover)
    assert session.board.fen() == start_fen and session.moves == []

    snap = session.play_turn("e2e4", FirstLegalMover())       # same move works now
    assert [m["by"] for m in snap["moves"]] == ["human", "bot"]


# ---------------------------------------------------------------------------
# Manager: lookup, expiry, cap
# ---------------------------------------------------------------------------

def test_unique_ids_and_lookup():
    manager = new_manager()
    a, b = create(manager), create(manager)
    assert a.session_id != b.session_id
    assert manager.get(a.session_id) is a
    assert len(manager) == 2


def test_unknown_session_raises():
    with pytest.raises(SessionNotFoundError):
        new_manager().get("does-not-exist")


def test_idle_sessions_expire_and_activity_keeps_them_alive():
    now = [0.0]
    manager = new_manager(clock=lambda: now[0], ttl_seconds=100)
    keep, drop = create(manager), create(manager)

    now[0] = 80
    manager.get(keep.session_id)            # activity refreshes the timer
    now[0] = 150                            # drop is idle for 150 > 100; keep for 70
    assert manager.get(keep.session_id) is keep
    with pytest.raises(SessionNotFoundError):
        manager.get(drop.session_id)
    assert len(manager) == 1


def test_expired_sessions_are_purged_on_create():
    now = [0.0]
    manager = new_manager(clock=lambda: now[0], ttl_seconds=10)
    create(manager)
    now[0] = 50
    create(manager)
    assert len(manager) == 1


def test_cap_rejects_new_games_when_all_active():
    manager = new_manager(max_sessions=2)
    create(manager)
    create(manager)
    with pytest.raises(SessionLimitError):
        create(manager)


def test_cap_evicts_oldest_finished_game():
    now = [0.0]
    manager = new_manager(clock=lambda: now[0], max_sessions=2)
    old_done = create(manager, chess.BLACK, board=chess.Board(FEN_FOOLS_MATE_BLACK_TO_MOVE))
    now[0] = 1
    old_done.play_turn("d8h4", manager.mover)                 # finished
    now[0] = 2
    active = create(manager)
    now[0] = 3
    newest = create(manager)                                  # evicts the finished game
    assert len(manager) == 2
    with pytest.raises(SessionNotFoundError):
        manager.get(old_done.session_id)
    assert manager.get(active.session_id) is active
    assert manager.get(newest.session_id) is newest


def test_defaults_come_from_config(monkeypatch):
    monkeypatch.setattr(config, "SESSION_MAX_COUNT", 1)
    monkeypatch.setattr(config, "SESSION_IDLE_TTL_SECONDS", 5)
    manager = SessionManager(mover=FirstLegalMover())
    assert manager.max_sessions == 1 and manager.ttl_seconds == 5


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

def test_simultaneous_identical_moves_apply_exactly_once():
    def slow_mover(bot, board, meta, temperature):
        time.sleep(0.15)
        return next(iter(board.legal_moves))

    manager = new_manager(slow_mover)
    session = create(manager)
    barrier = threading.Barrier(2)

    def attempt(_):
        barrier.wait()
        try:
            session.play_turn("e2e4", manager.mover)
            return "ok"
        except IllegalMoveError:
            return "illegal"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = sorted(pool.map(attempt, range(2)))

    assert results == ["illegal", "ok"]
    assert len(session.moves) == 2            # one human move + one bot reply
