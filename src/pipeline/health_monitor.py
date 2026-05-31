"""HealthMonitor — self-healing module for pipeline lifecycle.

Monitors module heartbeats, detects crashes, and emits recovery events.
Each pipeline module should emit 'module_heartbeat' events periodically.
HealthMonitor detects missed heartbeats and auto-recovers.

Events consumed:
    module_heartbeat -> {module: str, status: str, timestamp: float}

Events emitted:
    module_heartbeat_timeout -> {module: str, missed: int}
    module_crashed -> {module: str, reason: str, recovery_attempt: int}
    module_recovered -> {module: str, recovery_attempt: int}
"""

import asyncio
import time
from typing import Dict, Optional
from loguru import logger


class HealthMonitor:
    """Monitors pipeline module health and auto-recovers crashed modules.

    Each pipeline module emits 'module_heartbeat' events periodically.
    If a module misses N consecutive heartbeats, HealthMonitor emits
    'module_crashed' and attempts to recover it by emitting
    'module_recover_{module_name}' events that the module can subscribe to.
    """

    def __init__(
        self,
        event_bus,
        check_interval: float = 10.0,
        heartbeat_timeout: float = 30.0,
        max_failures: int = 3,
        max_recovery_attempts: int = 5,
        cooldown_after_recovery: float = 300.0,
    ):
        self._bus = event_bus
        self._check_interval = check_interval
        self._heartbeat_timeout = heartbeat_timeout
        self._max_failures = max_failures
        self._max_recovery_attempts = max_recovery_attempts
        self._cooldown_after_recovery = cooldown_after_recovery

        # module_name -> last_heartbeat_timestamp
        self._heartbeats: Dict[str, float] = {}
        # module_name -> consecutive_missed_count
        self._missed_counts: Dict[str, int] = {}
        # module_name -> total_recovery_attempts
        self._recovery_attempts: Dict[str, int] = {}
        # module_name -> timestamp of last recovery (for cooldown)
        self._last_recovery: Dict[str, float] = {}
        # module_name -> status string
        self._statuses: Dict[str, str] = {}
        # modules in cooldown after recovery
        self._in_cooldown: Dict[str, float] = {}

        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Subscribe to heartbeats and start health check loop."""
        self._bus.on("module_heartbeat", self._on_heartbeat)
        self._running = True
        self._task = asyncio.create_task(self._health_check_loop())
        logger.info(
            f"HealthMonitor started: check_interval={self._check_interval}s, "
            f"heartbeat_timeout={self._heartbeat_timeout}s, "
            f"max_failures={self._max_failures}"
        )

    async def stop(self):
        """Stop health monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        self._bus.off("module_heartbeat", self._on_heartbeat)
        logger.info("HealthMonitor stopped")

    def register_module(self, module_name: str, initial_status: str = "starting"):
        """Register a module for health monitoring."""
        now = time.time()
        self._heartbeats[module_name] = now
        self._statuses[module_name] = initial_status
        self._missed_counts[module_name] = 0
        self._recovery_attempts[module_name] = 0
        logger.info(
            f"HealthMonitor: registered module '{module_name}' ({initial_status})"
        )

    async def _on_heartbeat(self, **data):
        """Handle module heartbeat event."""
        module = data.get("module")
        status = data.get("status", "running")
        if module:
            self._heartbeats[module] = time.time()
            self._missed_counts[module] = 0  # Reset missed count on heartbeat
            self._statuses[module] = status

    async def _health_check_loop(self):
        """Periodic health check loop."""
        while self._running:
            await asyncio.sleep(self._check_interval)
            await self._check_modules()

    async def _check_modules(self):
        """Check each registered module for missed heartbeats."""
        now = time.time()
        for module, last_heartbeat in list(self._heartbeats.items()):
            # Skip modules in cooldown after recovery
            if module in self._in_cooldown:
                if now < self._in_cooldown[module]:
                    continue
                else:
                    del self._in_cooldown[module]

            elapsed = now - last_heartbeat
            if elapsed > self._heartbeat_timeout:
                self._missed_counts[module] = self._missed_counts.get(module, 0) + 1
                missed = self._missed_counts[module]

                logger.warning(
                    f"HealthMonitor: module '{module}' missed heartbeat "
                    f"({missed}/{self._max_failures}, {elapsed:.0f}s since last)"
                )

                # Emit timeout warning
                await self._bus.emit(
                    "module_heartbeat_timeout",
                    module=module,
                    missed=missed,
                    elapsed=elapsed,
                )

                # If max failures exceeded, declare crashed and attempt recovery
                if missed >= self._max_failures:
                    await self._recover_module(module)
            else:
                # Reset missed count if heartbeat received within window
                self._missed_counts[module] = 0

    async def _recover_module(self, module: str):
        """Attempt to recover a crashed module."""
        attempts = self._recovery_attempts.get(module, 0) + 1
        self._recovery_attempts[module] = attempts

        if attempts > self._max_recovery_attempts:
            logger.error(
                f"HealthMonitor: module '{module}' exceeded max recovery attempts "
                f"({attempts}/{self._max_recovery_attempts}) — giving up"
            )
            await self._bus.emit(
                "module_crashed",
                module=module,
                reason="max_recovery_attempts_exceeded",
                recovery_attempt=attempts,
            )
            return

        # Emit crash event
        await self._bus.emit(
            "module_crashed",
            module=module,
            reason="heartbeat_timeout",
            recovery_attempt=attempts,
        )

        # Emit recovery signal (module should subscribe to this)
        await self._bus.emit(
            f"module_recover_{module}",
            module=module,
            recovery_attempt=attempts,
        )

        # Put module in cooldown so health check loop skips it
        # (do not reset heartbeat — module stays unhealthy until it sends a new one)
        self._in_cooldown[module] = time.time() + self._cooldown_after_recovery
        self._missed_counts[module] = 0
        self._last_recovery[module] = time.time()

        logger.info(
            f"HealthMonitor: recovery signal sent for '{module}' "
            f"(attempt {attempts}/{self._max_recovery_attempts})"
        )

    def emit_heartbeat(self, module: str, status: str = "running"):
        """Convenience method for modules to emit their own heartbeat."""
        asyncio.ensure_future(
            self._bus.emit("module_heartbeat", module=module, status=status)
        )

    def get_health_report(self) -> Dict:
        """Return a snapshot of all module health statuses."""
        now = time.time()
        report = {}
        for module in self._heartbeats:
            elapsed = now - self._heartbeats[module]
            report[module] = {
                "status": self._statuses.get(module, "unknown"),
                "last_heartbeat_ago": round(elapsed, 1),
                "missed_count": self._missed_counts.get(module, 0),
                "recovery_attempts": self._recovery_attempts.get(module, 0),
                "healthy": elapsed < self._heartbeat_timeout,
            }
        return report
