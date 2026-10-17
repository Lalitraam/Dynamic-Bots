"""GameManager managing in-memory active games, locking, and engine execution (Phase F)."""

from __future__ import annotations
import asyncio
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any
import chess
import pyarrow.parquet as pq

from app import config
from app.analysis.stockfish import StockfishAnalyser
from app.db.games import add_move as db_add_move, add_telemetry, create_game as db_create_game, get_game as db_get_game, get_moves as db_get_moves, update_game as db_update_game
from app.engine.model_cache import ModelCache
from app.engine.provider import MoveContext
from app.engine.provider_v3 import V3Provider
from app.engine.think_time import ThinkTimeModel
from app.game.state import GameState


def hash_secret(secret: str) -> str:
    """Return SHA-256 hash of raw game secret."""
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


class GameManager:
    """In-memory active games manager with concurrency locks and V3Provider integration."""

    def __init__(self, data_root: Path | str = config.DATA_ROOT):
        self.data_root = Path(data_root)
        self.games: dict[str, GameState] = {}
        self.locks: dict[str, asyncio.Lock] = {}
        self.subscribers: dict[str, set[Any]] = {}  # game_id -> set of WebSocket connections
        self.provider_cache = ModelCache(capacity=getattr(config, "MODEL_CACHE_SIZE", 8), data_root=self.data_root)
        self._stockfish: StockfishAnalyser | None = None

    def _get_player_rating(self, username: str) -> int:
        player_dir = self.data_root / "players" / username
        info_file = player_dir / "info.json"
        if info_file.exists():
            try:
                with open(info_file, "r", encoding="utf-8") as f:
                    info = json.load(f)
                    if "median_rating" in info:
                        return int(info["median_rating"])
                    if "player_rating" in info:
                        return int(info["player_rating"])
            except Exception:
                pass
        for s_name in ("samples_v3.parquet", "samples.parquet"):
            sp = player_dir / s_name
            if sp.exists():
                try:
                    df = pq.read_table(sp, columns=["player_rating"]).to_pandas()
                    if "player_rating" in df.columns and len(df) > 0:
                        return int(df["player_rating"].median())
                except Exception:
                    pass
        return config.DEFAULT_RATING

    def _get_stockfish_analyser(self) -> StockfishAnalyser | None:
        sf_path = getattr(config, "STOCKFISH_PATH", None)
        if sf_path and Path(sf_path).exists():
            if self._stockfish is None or self._stockfish.engine is None:
                try:
                    self._stockfish = StockfishAnalyser(path=sf_path, depth=getattr(config, "STOCKFISH_DEPTH", 8))
                except Exception:
                    self._stockfish = None
            return self._stockfish
        return None

    def _get_think_time_model(self, username: str) -> ThinkTimeModel:
        player_dir = self.data_root / "players" / username
        samples_path = player_dir / "samples_v3.parquet"
        if not samples_path.exists():
            samples_path = player_dir / "samples.parquet"
        if samples_path.exists():
            try:
                return ThinkTimeModel.from_parquet(samples_path)
            except Exception:
                pass
        return ThinkTimeModel()

    def get_lock(self, game_id: str) -> asyncio.Lock:
        if game_id not in self.locks:
            self.locks[game_id] = asyncio.Lock()
        return self.locks[game_id]

    async def create_game(
        self,
        game_id: str,
        player_username: str,
        human_color: str,
        base_s: int,
        inc_s: int,
        secret: str,
        owner_id: int | None = None,
        now: float | None = None,
    ) -> GameState:
        """Create and initialize a new GameState in memory and persist in DB."""
        secret_h = hash_secret(secret)
        game_state = GameState(
            game_id=game_id,
            player_username=player_username,
            human_color=human_color,
            base_s=base_s,
            inc_s=inc_s,
            secret_hash=secret_h,
            owner_id=owner_id,
        )

        db_create_game(
            game_id=game_id,
            owner_id=owner_id or 1,
            player_username=player_username,
            model_version=f"{player_username}:v1",
            human_color=human_color,
            base_s=base_s,
            inc_s=inc_s,
            secret_hash=secret_h,
        )

        self.games[game_id] = game_state
        cur_now = now if now is not None else time.time()
        game_state.clock.start(cur_now)

        # If human plays Black, bot plays move 1 (White)
        if game_state.human_color == "black":
            stockfish = self._get_stockfish_analyser()
            provider = V3Provider(
                username=player_username,
                model_version=1,
                model_cache=self.provider_cache,
                think_time_model=self._get_think_time_model(player_username),
                temperature=1.0,
                data_root=self.data_root,
                stockfish_analyser=stockfish,
            )
            bot_rating = self._get_player_rating(player_username)
            ctx = MoveContext(
                board=game_state.board,
                prev_moves=[],
                bot_rating=bot_rating,
                opp_rating=config.DEFAULT_RATING,
                base_seconds=base_s,
                increment_seconds=inc_s,
                bot_clock_ms=base_s * 1000,
                opp_clock_ms=base_s * 1000,
                ply=0,
            )
            rng = random.Random()
            t0 = time.perf_counter()
            decision = await asyncio.to_thread(provider.decide, ctx, rng)
            infer_ms = (time.perf_counter() - t0) * 1000.0

            san_str = game_state.board.san(decision.move)
            game_state.clock.on_move("white", cur_now)
            move_dict = {
                "ply": 1,
                "uci": decision.move.uci(),
                "san": san_str,
                "by": "bot",
                "clock_after_ms": game_state.clock.remaining_ms["white"],
                "elapsed_ms": int(decision.think_ms),
            }
            game_state.moves.append(move_dict)
            game_state.board.push(decision.move)

            db_add_move(
                game_id=game_id,
                ply=1,
                uci=decision.move.uci(),
                san=san_str,
                by="bot",
                clock_after_ms=game_state.clock.remaining_ms["white"],
                elapsed_ms=int(decision.think_ms),
            )
            top1_p = decision.top5[0][1] if decision.top5 else None
            add_telemetry(
                game_id=game_id,
                ply=1,
                source=decision.source,
                top1_prob=top1_p,
                p_err=decision.p_err,
                infer_ms=infer_ms,
                think_ms=decision.think_ms,
                model_version=decision.model_version,
            )

        return game_state

    async def get_or_restore_game(self, game_id: str) -> GameState | None:
        """Fetch game from memory, or restore from database if missing."""
        if game_id in self.games:
            return self.games[game_id]

        row = db_get_game(game_id)
        if not row:
            return None

        game_state = GameState(
            game_id=game_id,
            player_username=row["player_username"],
            human_color=row["human_color"],
            base_s=row["base_s"],
            inc_s=row["inc_s"],
            secret_hash=row["secret_hash"],
            owner_id=row["owner_id"],
        )
        game_state.status = row["status"]
        game_state.result = row["result"]
        game_state.termination = row["termination"]

        # Reconstruct board and moves
        moves = db_get_moves(game_id)
        for m in moves:
            game_state.moves.append(dict(m))
            game_state.board.push(chess.Move.from_uci(m["uci"]))

        self.games[game_id] = game_state
        return game_state

    async def apply_move(
        self,
        game_id: str,
        uci: str,
        secret: str | None = None,
        now: float | None = None,
        client_ply: int | None = None,
        secret_hash: str | None = None,
    ) -> GameState:
        """Validate human move, update clock, execute bot move, and update DB."""
        eff_secret = secret if secret is not None else secret_hash
        if not eff_secret:
            raise PermissionError("Missing game secret")

        lock = self.get_lock(game_id)
        async with lock:
            game_state = await self.get_or_restore_game(game_id)
            if not game_state:
                raise ValueError(f"Game '{game_id}' not found")

            # 1. Validate secret
            if hash_secret(eff_secret) != game_state.secret_hash:
                raise PermissionError("Invalid game secret")

            # 2. Check if game is already finished
            if game_state.status != "active":
                return game_state

            # 3. Check for idempotent duplicate move
            if client_ply is not None and client_ply <= len(game_state.moves):
                return game_state

            cur_now = now if now is not None else time.time()
            if game_state.clock.last_tick is None:
                game_state.clock.start(cur_now)

            # 4. Parse & validate human move
            try:
                human_move = chess.Move.from_uci(uci.strip())
            except Exception:
                raise ValueError(f"Invalid UCI move '{uci}'")

            if human_move not in game_state.board.legal_moves:
                raise ValueError(f"Illegal move '{uci}' in position {game_state.board.fen()}")

            # 5. Authoritative clock update for human
            ok = game_state.clock.on_move(game_state.human_color, cur_now)
            if not ok:
                # Human flagged on time
                game_state.status = "completed"
                game_state.result = "0-1" if game_state.human_color == "white" else "1-0"
                game_state.termination = "time forfeit"
                db_update_game(
                    game_id,
                    status=game_state.status,
                    result=game_state.result,
                    termination=game_state.termination,
                )
                return game_state

            human_ply = len(game_state.moves) + 1
            human_san = game_state.board.san(human_move)
            move_rec = {
                "ply": human_ply,
                "uci": human_move.uci(),
                "san": human_san,
                "by": "human",
                "clock_after_ms": game_state.clock.remaining_ms[game_state.human_color],
            }
            game_state.moves.append(move_rec)
            game_state.board.push(human_move)

            db_add_move(
                game_id=game_id,
                ply=human_ply,
                uci=human_move.uci(),
                san=human_san,
                by="human",
                clock_after_ms=game_state.clock.remaining_ms[game_state.human_color],
            )

            # 6. Check if game ended after human move
            if game_state.board.is_game_over():
                game_state.status = "completed"
                game_state.result = game_state.board.result()
                game_state.termination = "normal"
                db_update_game(
                    game_id,
                    status=game_state.status,
                    result=game_state.result,
                    termination=game_state.termination,
                )
                return game_state

            # 7. Bot counter-move
            bot_ply = human_ply + 1
            stockfish = self._get_stockfish_analyser()
            provider = V3Provider(
                username=game_state.player_username,
                model_version=1,
                model_cache=self.provider_cache,
                think_time_model=self._get_think_time_model(game_state.player_username),
                temperature=1.0,
                data_root=self.data_root,
                stockfish_analyser=stockfish,
            )

            bot_rating = self._get_player_rating(game_state.player_username)
            prev_moves = [m["uci"] for m in game_state.moves[-3:]]
            ctx = MoveContext(
                board=game_state.board,
                prev_moves=prev_moves,
                bot_rating=bot_rating,
                opp_rating=config.DEFAULT_RATING,
                base_seconds=game_state.base_s,
                increment_seconds=game_state.inc_s,
                bot_clock_ms=game_state.clock.remaining_ms[game_state.bot_color],
                opp_clock_ms=game_state.clock.remaining_ms[game_state.human_color],
                ply=bot_ply,
            )

            rng = random.Random()
            t0 = time.perf_counter()
            decision = await asyncio.to_thread(provider.decide, ctx, rng)
            infer_ms = (time.perf_counter() - t0) * 1000.0

            # Clock deduction for bot
            bot_now = cur_now + (decision.think_ms / 1000.0)
            bot_ok = game_state.clock.on_move(game_state.bot_color, bot_now)
            if not bot_ok:
                game_state.status = "completed"
                game_state.result = "1-0" if game_state.bot_color == "black" else "0-1"
                game_state.termination = "time forfeit"
                db_update_game(
                    game_id,
                    status=game_state.status,
                    result=game_state.result,
                    termination=game_state.termination,
                )
                return game_state

            bot_san = game_state.board.san(decision.move)
            bot_rec = {
                "ply": bot_ply,
                "uci": decision.move.uci(),
                "san": bot_san,
                "by": "bot",
                "clock_after_ms": game_state.clock.remaining_ms[game_state.bot_color],
                "elapsed_ms": int(decision.think_ms),
            }
            game_state.moves.append(bot_rec)
            game_state.board.push(decision.move)

            db_add_move(
                game_id=game_id,
                ply=bot_ply,
                uci=decision.move.uci(),
                san=bot_san,
                by="bot",
                clock_after_ms=game_state.clock.remaining_ms[game_state.bot_color],
                elapsed_ms=int(decision.think_ms),
            )
            top1_p = decision.top5[0][1] if decision.top5 else None
            add_telemetry(
                game_id=game_id,
                ply=bot_ply,
                source=decision.source,
                top1_prob=top1_p,
                p_err=decision.p_err,
                infer_ms=infer_ms,
                think_ms=decision.think_ms,
                model_version=decision.model_version,
            )

            if game_state.board.is_game_over():
                game_state.status = "completed"
                game_state.result = game_state.board.result()
                game_state.termination = "normal"
                db_update_game(
                    game_id,
                    status=game_state.status,
                    result=game_state.result,
                    termination=game_state.termination,
                )

            return game_state

    async def resign_game(self, game_id: str, secret: str) -> GameState:
        """Handle resignation by human player."""
        lock = self.get_lock(game_id)
        async with lock:
            game_state = await self.get_or_restore_game(game_id)
            if not game_state:
                raise ValueError(f"Game '{game_id}' not found")
            if hash_secret(secret) != game_state.secret_hash:
                raise PermissionError("Invalid game secret")

            if game_state.status == "active":
                game_state.status = "completed"
                game_state.result = "0-1" if game_state.human_color == "white" else "1-0"
                game_state.termination = "resignation"
                db_update_game(
                    game_id,
                    status=game_state.status,
                    result=game_state.result,
                    termination=game_state.termination,
                )
            return game_state


# Global GameManager singleton
game_manager = GameManager()
