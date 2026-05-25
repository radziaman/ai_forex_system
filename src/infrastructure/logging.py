"""
Professional logging utilities for production trading systems.

Designed for the RTS Agentic FX System Elite — but the patterns are universal.

Features:
  - ``log_once``: Rate-limited logging — identical messages suppressed within a
    configurable time window. Prevents alert fatigue from repeated errors.
  - ``PeriodicCounter``: Aggregate high-frequency events (ticks, trades, errors)
    into periodic summaries. One line per minute replaces hundreds of individual
    debug lines — the standard pattern at TwoSigma, Renaissance, and Jane Street.
  - ``SystemHeartbeat``: Periodic structured snapshot of system state. Emits a
    single JSON line every N seconds with health, PnL, ticks/s, agents alive.
    Designed for ingestion by log aggregators (ELK, Grafana, Datadog).

Usage:

    from infrastructure.logging import log_once, PeriodicCounter

    # Warn about connection loss at most once per hour
    log_once(logger.warning, "Connection lost", key="conn_lost", throttle=3600)

    # Count ticks per minute instead of logging each one
    tick_counter = PeriodicCounter("tick", interval=60, logger=logger)
    ...
    tick_counter.tick()  # on every tick ingestion
    # Auto-emits: "PeriodicCounter[tick]: 1,245 events in 60.0s (20.8/s)"

    # Structured heartbeat on the orchestrator
    from infrastructure.logging import SystemHeartbeat
    heartbeat = SystemHeartbeat(interval=300, logger=logger)
    heartbeat.emit(ws_snapshot)  # structured JSON with health, PnL, ticks
"""

from __future__ import annotations

import time
import json
import threading
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass, field

from loguru import logger as _base_logger


# ──────────────────────────────────────────────────────────────────────────────
# Rate-limited logger
# ──────────────────────────────────────────────────────────────────────────────


class RateLimiter:
    """Tracks last-emitted timestamps for rate-limited logging.

    Thread-safe. Used internally by ``log_once`` — you shouldn't need to
    instantiate this directly.
    """

    def __init__(self):
        self._last: Dict[str, float] = {}
        self._lock = threading.Lock()

    def can_emit(self, key: str, throttle: float = 300.0) -> bool:
        """Return True if ``throttle`` seconds have elapsed since last emit."""
        now = time.time()
        with self._lock:
            last = self._last.get(key, 0.0)
            if now - last >= throttle:
                self._last[key] = now
                return True
            return False


# Global rate-limiter instance — shared across all agents
_global_rate_limiter = RateLimiter()


def log_once(
    log_fn: Callable,
    message: str,
    key: Optional[str] = None,
    throttle: float = 300.0,
    **extra,
):
    """Log *message* via *log_fn* at most once per *throttle* seconds.

    Uses a global rate-limiter keyed by ``key`` (defaults to ``message``).
    Any ``extra`` keyword arguments are appended as structured fields.

    Examples
    --------
    >>> log_once(logger.warning, "Connection lost", key="conn_lost", throttle=60)
    >>> log_once(logger.info, "Trade rejected", symbol="EURUSD", reason="max_dd")
    """
    if key is None:
        key = message
    if not _global_rate_limiter.can_emit(key, throttle):
        return
    if extra:
        structured = f"{message} | {json.dumps(extra)}"
    else:
        structured = message
    log_fn(structured)


# ──────────────────────────────────────────────────────────────────────────────
# Periodic counter — aggregate high-frequency events into summaries
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class PeriodicCounter:
    """Aggregate event counts into periodic summary log lines.

    Instead of logging every event (e.g., every tick), increment the counter
    and a summary is automatically emitted every *interval* seconds.

    Parameters
    ----------
    name : str
        Human-readable name for the event stream (e.g., ``"tick"``, ``"trade"``).
    interval : float
        Seconds between summary emissions (default 60).
    logger : Any
        Logger to emit summaries on (default ``loguru.logger``).
    level : str
        Log level for summaries (default ``"INFO"``).

    Example
    -------
    >>> ticks = PeriodicCounter("tick", interval=60)
    >>> ticks.tick()   # called 1,245 times in 60 seconds
    >>> ticks.tick()
    # At 60s: "PeriodicCounter[tick]: 1,245 events in 60.0s (20.8/s)"
    """

    name: str
    interval: float = 60.0
    logger: Any = _base_logger
    level: str = "INFO"

    _count: int = 0
    _last_reset: float = field(default_factory=time.time)

    def tick(self, n: int = 1):
        """Record *n* events. Emits summary if *interval* has elapsed."""
        self._count += n
        elapsed = time.time() - self._last_reset
        if elapsed >= self.interval:
            rate = self._count / max(elapsed, 0.001)
            getattr(self.logger, self.level.lower())(
                f"PeriodicCounter[{self.name}]: {self._count:,} events "
                f"in {elapsed:.1f}s ({rate:.1f}/s)"
            )
            self._count = 0
            self._last_reset = time.time()

    def reset(self):
        """Manually reset counter without emitting."""
        self._count = 0
        self._last_reset = time.time()

    @property
    def rate(self) -> float:
        """Current event rate since last reset (events/second)."""
        elapsed = time.time() - self._last_reset
        return self._count / max(elapsed, 0.001)


# ──────────────────────────────────────────────────────────────────────────────
# Structured system heartbeat
# ──────────────────────────────────────────────────────────────────────────────


class SystemHeartbeat:
    """Periodic structured snapshot of system state.

    Designed to be called from the orchestrator or monitoring agent. Emits a
    single JSON line every *interval* seconds with health, PnL, ticks/s,
    agents alive, and connection status. Compatible with structured log
    ingestion (ELK, Grafana Loki, Datadog).

    Parameters
    ----------
    interval : float
        Seconds between heartbeats (default 300 = 5 minutes).
    logger : Any
        Logger to emit on (default ``loguru.logger``).
    """

    def __init__(self, interval: float = 300.0, logger: Any = _base_logger):
        self.interval = interval
        self.logger = logger
        self._last = 0.0

    def emit(self, world_state_getter: Callable[[str, Any], Any]) -> bool:
        """Emit a heartbeat line if *interval* has elapsed. Returns True if
        emitted."""
        now = time.time()
        if now - self._last < self.interval:
            return False
        self._last = now

        try:
            snapshot = {
                "event": "heartbeat",
                "timestamp": now,
                "health": world_state_getter("master.system_health", 1.0),
                "system_health": world_state_getter("system.health_score", 1.0),
                "agents_alive": (
                    len(world_state_getter("master.domain_health", {}))
                    if world_state_getter("master.domain_health", {})
                    else 0
                ),
                "regime": world_state_getter("regime.current", "?"),
                "positions": world_state_getter("account.open_positions", 0),
                "balance": (
                    world_state_getter("account.balance", 0)
                    if world_state_getter("execution.connected", False)
                    else 0
                ),
                "equity": (
                    world_state_getter("account.equity", 0)
                    if world_state_getter("execution.connected", False)
                    else 0
                ),
                "mode": world_state_getter("config.mode", "paper"),
                "connected": bool(world_state_getter("execution.connected", False)),
                "tick_rate": world_state_getter("data.tick_rate", 0),
                "sharpe": world_state_getter("performance.stats", {}).get("sharpe", 0),
                "total_pnl": world_state_getter("performance.stats", {}).get(
                    "total_pnl", 0
                ),
                "total_trades": world_state_getter("performance.stats", {}).get(
                    "total_trades", 0
                ),
            }
            self.logger.info(f"Heartbeat | {json.dumps(snapshot, default=str)}")
        except Exception as e:
            self.logger.warning(f"Heartbeat failed: {e}")

        return True
