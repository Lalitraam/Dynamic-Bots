"""Tests for GameManager and WebSocket gameplay (Phase F)."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app import config
from app.db.migrate import get_db, run_migrations
from app.db.users import create_invite_code
from app.game.manager import GameManager
from app.main import app
from app.routers import bots


@pytest.fixture(autouse=True)
def test_db_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolate tests into a temporary database."""
    test_db = tmp_path / "test_gm.db"
    monkeypatch.setattr(config, "DB_PATH", test_db)
    run_migrations(test_db)


def setup_user_and_bot(username: str = "the_king_crusher") -> int:
    """Helper to insert user and player records into DB."""
    conn = get_db()
    with conn:
        cursor = conn.execute(
            "INSERT INTO users (email, password_hash, role) VALUES (?, ?, ?)",
            (f"user_{username}@example.com", "dummyhash", "user"),
        )
        user_id = cursor.lastrowid
        conn.execute(
            "INSERT OR IGNORE INTO players (username, status) VALUES (?, 'ready')",
            (username,),
        )
        conn.execute(
            "INSERT OR IGNORE INTO user_bots (user_id, username) VALUES (?, ?)",
            (user_id, username),
        )
    conn.close()
    return user_id


@pytest.mark.asyncio
async def test_create_game_and_legal_move():
    """Test creating a game, applying a legal human move, and receiving bot counter-move."""
    user_id = setup_user_and_bot("the_king_crusher")
    gm = GameManager(data_root=config.DATA_ROOT)
    t0 = 1000.0

    state = await gm.create_game(
        game_id="game_1",
        player_username="the_king_crusher",
        human_color="white",
        base_s=180,
        inc_s=2,
        secret="secret123",
        owner_id=user_id,
        now=t0,
    )
    assert state.game_id == "game_1"
    assert state.status == "active"
    assert len(state.moves) == 0

    # Human plays e2e4
    state = await gm.apply_move(
        game_id="game_1",
        uci="e2e4",
        secret="secret123",
        now=t0 + 2.0,
        client_ply=1,
    )

    # 2 plies played (Human + Bot)
    assert len(state.moves) == 2
    assert state.moves[0]["by"] == "human"
    assert state.moves[0]["uci"] == "e2e4"
    assert state.moves[0]["san"] == "e4"

    assert state.moves[1]["by"] == "bot"
    assert state.moves[1]["elapsed_ms"] is not None
    assert state.status == "active"


@pytest.mark.asyncio
async def test_idempotent_duplicate_moves():
    """Test idempotent duplicate move is ignored if client_ply has already been applied."""
    user_id = setup_user_and_bot("the_king_crusher")
    gm = GameManager(data_root=config.DATA_ROOT)
    t0 = 1000.0

    await gm.create_game(
        game_id="game_dup",
        player_username="the_king_crusher",
        human_color="white",
        base_s=180,
        inc_s=2,
        secret="secret_dup",
        owner_id=user_id,
        now=t0,
    )

    # First attempt: ply 1
    state1 = await gm.apply_move(
        game_id="game_dup",
        uci="e2e4",
        secret="secret_dup",
        now=t0 + 1.0,
        client_ply=1,
    )
    assert len(state1.moves) == 2

    # Duplicate network retry with client_ply=1
    state2 = await gm.apply_move(
        game_id="game_dup",
        uci="e2e4",
        secret="secret_dup",
        now=t0 + 2.0,
        client_ply=1,
    )
    # Move count and board position unchanged
    assert len(state2.moves) == 2
    assert state2.board.fen() == state1.board.fen()


@pytest.mark.asyncio
async def test_human_timeout_flagging():
    """Test human timeout causes time forfeit and completes game."""
    user_id = setup_user_and_bot("the_king_crusher")
    gm = GameManager(data_root=config.DATA_ROOT)
    t0 = 1000.0

    await gm.create_game(
        game_id="game_timeout",
        player_username="the_king_crusher",
        human_color="white",
        base_s=5,
        inc_s=1,
        secret="secret_to",
        owner_id=user_id,
        now=t0,
    )

    # Move 1: white e4 at t0 + 1.0 -> White has 6000ms, started['white']=True
    state = await gm.apply_move(
        game_id="game_timeout",
        uci="e2e4",
        secret="secret_to",
        now=t0 + 1.0,
        client_ply=1,
    )
    assert state.status == "active"

    # White thinks for 15 seconds before playing move 2 (having only ~6s)
    flagged_state = await gm.apply_move(
        game_id="game_timeout",
        uci="g1f3",
        secret="secret_to",
        now=t0 + 16.0,
        client_ply=3,
    )

    assert flagged_state.status == "completed"
    assert flagged_state.result == "0-1"
    assert flagged_state.termination == "time forfeit"
    assert flagged_state.clock.remaining_ms["white"] == 0


def test_websocket_flow(monkeypatch: pytest.MonkeyPatch):
    """Test full WebSocket flow: handshake hello, snapshot, move, and resignation."""
    monkeypatch.setattr(bots, "check_lichess_user_exists", lambda u: True)
    client = TestClient(app)
    create_invite_code("INVITE-WS")
    client.post(
        "/api/auth/register",
        json={"email": "ws_player@example.com", "password": "Password123!", "invite_code": "INVITE-WS"},
    )
    login_resp = client.post(
        "/api/auth/login",
        json={"email": "ws_player@example.com", "password": "Password123!"},
    )
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Add bot to user account so player and user_bot exist
    client.post("/api/bots", json={"username": "the_king_crusher"}, headers=headers)

    create_resp = client.post(
        "/api/games",
        json={"username": "the_king_crusher", "human_color": "white", "base_s": 180, "inc_s": 2},
        headers=headers,
    )
    assert create_resp.status_code == 201
    game_data = create_resp.json()
    game_id = game_data["game_id"]
    secret = game_data["secret"]

    with client.websocket_connect(f"/ws/{game_id}") as ws:
        # Send hello handshake
        ws.send_json({"t": "hello", "secret": secret})
        snap1 = ws.receive_json()
        assert snap1["t"] == "snapshot"
        assert snap1["game_id"] == game_id
        assert snap1["status"] == "active"
        assert len(snap1["moves"]) == 0

        # Send legal move
        ws.send_json({"t": "move", "uci": "e2e4", "n": 1})
        snap2 = ws.receive_json()
        assert snap2["t"] == "snapshot"
        assert len(snap2["moves"]) == 2
        assert snap2["moves"][0]["uci"] == "e2e4"
        assert snap2["moves"][1]["by"] == "bot"

        # Send resignation
        ws.send_json({"t": "resign"})
        snap3 = ws.receive_json()
        assert snap3["t"] == "snapshot"
        assert snap3["status"] == "completed"
        assert snap3["result"] == "0-1"
        assert snap3["termination"] == "resignation"


def test_think_time_model_default_init():
    from app.engine.think_time import ThinkTimeModel
    t = ThinkTimeModel()
    assert t.buckets == {}

