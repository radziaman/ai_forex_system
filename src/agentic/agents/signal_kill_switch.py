"""
Signal Kill Switch — monitors and handles the trading kill switch.

When kill switch is active, all signals are blocked.
Automatically requests release after cooldown when drawdown recovers.
"""

from __future__ import annotations
import time
from typing import Optional, Callable, Awaitable
from loguru import logger

from agentic.core.agent_message import (
    MessageType,
    MessagePriority,
)


class KillSwitchMonitor:
    """Monitors kill switch state in world state.

    When kill switch is active, all signals are blocked.
    Automatically requests release after cooldown when drawdown recovers.
    """

    def __init__(self, send_fn: Callable[..., Awaitable[None]]):
        self._send_fn = send_fn
        self._active_since: Optional[float] = None
        self._alerted: bool = False

    def is_active(self, world_state_getter: Callable) -> bool:
        """Check if kill switch is active in world state.

        Args:
            world_state_getter: A callable like agent.get_world(key, default).

        Returns:
            True if kill switch is active.
        """
        active = bool(world_state_getter("risk.kill_switch", False))
        if active and self._active_since is None:
            self._active_since = time.time()
        return active

    async def handle(self, world_state_getter: Callable):
        """Handle active kill switch.

        Logs warning, alerts Telegram, and attempts recovery after 30 min
        if drawdown has recovered below 3%.

        Args:
            world_state_getter: A callable like agent.get_world(key, default).
        """
        if not self._alerted:
            logger.warning("Kill switch active — all signals blocked")
            await self._send_fn(
                MessageType.RISK_ALERT,
                payload={
                    "type": "kill_switch",
                    "reason": "Kill switch active — all signals blocked",
                },
                priority=MessagePriority.CRITICAL,
            )
            self._alerted = True
        else:
            elapsed = time.time() - self._active_since
            if elapsed > 30 * 60:
                drawdown = world_state_getter("risk.drawdown", 0.0)
                if drawdown < 0.03:
                    logger.info(
                        "Kill switch recovery: drawdown recovered, "
                        "attempting release"
                    )
                    self._active_since = None
                    self._alerted = False
                    # Signal will set world state for release request
                    return "request_release"
                else:
                    logger.warning(
                        f"Kill switch still active after {elapsed / 60:.0f}min, "
                        f"drawdown={drawdown:.1%}"
                    )
        return None

    def clear(self):
        """Reset state when kill switch clears."""
        self._active_since = None
        self._alerted = False

    @property
    def is_active_since(self) -> Optional[float]:
        """Return the time when kill switch was activated, if active."""
        return self._active_since

    @property
    def was_alerted(self) -> bool:
        """Return whether the alert has been sent."""
        return self._alerted
