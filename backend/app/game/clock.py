"""Authoritative game clock management (Phase F)."""

from __future__ import annotations


class GameClock:
    """Pure, injectable authoritative chess clock with increment and Lichess move-1 rule."""

    def __init__(self, base_s: int, inc_s: int):
        self.base_ms = int(base_s * 1000)
        self.inc_ms = int(inc_s * 1000)
        self.remaining_ms = {"white": self.base_ms, "black": self.base_ms}
        self.running_side: str = "white"
        self.last_tick: float | None = None
        self.started = {"white": False, "black": False}

    def start(self, now: float) -> None:
        """Start or reset the clock ticker at current timestamp."""
        self.last_tick = float(now)

    def on_move(self, side: str, now: float) -> bool:
        """Handle move completion for side.
        
        Returns True if successful, False if flagged (time ran out).
        """
        side = side.lower().strip()
        now = float(now)

        if self.last_tick is None:
            self.last_tick = now

        # 1. Calculate elapsed and deduct only if this side already started
        if self.started[side]:
            elapsed_ms = max(0, int((now - self.last_tick) * 1000))
            self.remaining_ms[side] -= elapsed_ms
            if self.remaining_ms[side] <= 0:
                self.remaining_ms[side] = 0
                return False

        # 2. Add increment
        self.remaining_ms[side] += self.inc_ms

        # 3. Mark started for this side
        self.started[side] = True

        # 4. Switch running side and reset ticker
        self.running_side = "black" if side == "white" else "white"
        self.last_tick = now
        return True

    def get_remaining(self, now: float) -> dict[str, int]:
        """Return current clocks, deducting elapsed time for the running side."""
        now = float(now)
        out = dict(self.remaining_ms)

        if self.last_tick is not None and self.started[self.running_side]:
            elapsed_ms = max(0, int((now - self.last_tick) * 1000))
            out[self.running_side] = max(0, out[self.running_side] - elapsed_ms)

        return out
