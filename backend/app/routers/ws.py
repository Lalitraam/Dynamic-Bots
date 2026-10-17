"""WebSocket router for live game sessions (Phase F)."""

from __future__ import annotations
from typing import Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

from app.game.manager import game_manager, hash_secret

router = APIRouter(tags=["websocket"])

# Active connections: game_id -> set of active WebSockets
_active_connections: dict[str, set[WebSocket]] = {}


async def broadcast_to_game(game_id: str, payload: dict[str, Any]) -> None:
    """Broadcast JSON payload to all active WebSocket connections for game_id."""
    sockets = list(_active_connections.get(game_id, set()))
    for ws in sockets:
        try:
            await ws.send_json(payload)
        except Exception:
            _active_connections.get(game_id, set()).discard(ws)


async def handle_game_ws(websocket: WebSocket, game_id: str) -> None:
    """Handle live WebSocket connection for a game session."""
    # 1. Accept connection
    await websocket.accept()

    # 2. Wait for {"t": "hello", "secret": "..."}
    try:
        hello_msg = await websocket.receive_json()
    except Exception:
        await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
        return

    if not isinstance(hello_msg, dict) or hello_msg.get("t") != "hello":
        await websocket.send_json({"t": "error", "error": "Expected handshake {'t': 'hello', 'secret': '...'}"})
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    secret = hello_msg.get("secret")
    if not secret:
        await websocket.send_json({"t": "error", "error": "Missing secret"})
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # 3. Validate game and secret
    game_state = await game_manager.get_or_restore_game(game_id)
    if not game_state:
        await websocket.send_json({"t": "error", "error": f"Game '{game_id}' not found"})
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    if hash_secret(secret) != game_state.secret_hash:
        await websocket.send_json({"t": "error", "error": "Invalid game secret"})
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # 4. Register connection and send initial snapshot
    if game_id not in _active_connections:
        _active_connections[game_id] = set()
    _active_connections[game_id].add(websocket)

    try:
        snapshot = game_state.to_snapshot()
        await websocket.send_json({"t": "snapshot", **snapshot})

        # 5. Message loop: wait for {"t": "move", "uci": "...", "n": ply} or {"t": "resign"}
        while True:
            msg = await websocket.receive_json()
            if not isinstance(msg, dict):
                continue

            msg_type = msg.get("t")
            if msg_type == "move":
                uci = msg.get("uci")
                client_n = msg.get("n")
                if not uci:
                    await websocket.send_json({"t": "error", "error": "Missing 'uci' field"})
                    continue

                try:
                    updated_state = await game_manager.apply_move(
                        game_id=game_id,
                        uci=uci,
                        secret=secret,
                        client_ply=client_n,
                    )
                    snap = updated_state.to_snapshot()
                    await broadcast_to_game(game_id, {"t": "snapshot", **snap})
                except Exception as e:
                    await websocket.send_json({"t": "error", "error": str(e)})

            elif msg_type == "resign":
                try:
                    updated_state = await game_manager.resign_game(game_id=game_id, secret=secret)
                    snap = updated_state.to_snapshot()
                    await broadcast_to_game(game_id, {"t": "snapshot", **snap})
                except Exception as e:
                    await websocket.send_json({"t": "error", "error": str(e)})

            elif msg_type == "ping":
                await websocket.send_json({"t": "pong"})

            else:
                await websocket.send_json({"t": "error", "error": f"Unknown message type '{msg_type}'"})

    except WebSocketDisconnect:
        pass
    finally:
        if game_id in _active_connections:
            _active_connections[game_id].discard(websocket)
            if not _active_connections[game_id]:
                del _active_connections[game_id]


@router.websocket("/ws/{game_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str):
    """FastAPI WebSocket endpoint for real-time game moves and clock updates."""
    await handle_game_ws(websocket, game_id)


@router.websocket("/api/ws/{game_id}")
async def api_websocket_endpoint(websocket: WebSocket, game_id: str):
    """FastAPI WebSocket endpoint under /api prefix."""
    await handle_game_ws(websocket, game_id)
