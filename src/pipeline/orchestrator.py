"""Orchestrator — manages pipeline lifecycle, health monitoring, and coordination.

Replaces: MasterAgent + ConnectionAgent + MonitoringAgent + SystemHealthAgent.

Responsibilities:
- Start/stop all pipeline modules
- Run health checks every 60s
- Log system status
- Handle graceful shutdown

Simpler than the old 20-agent system: no agent consciousness, no emotions,
no cycle loops, no agent communication protocol.
"""

import asyncio
import time
from typing import Dict, Any
from loguru import logger

from .pipeline_context import PipelineContext


class Orchestrator:
    """Manages the pipeline lifecycle and health monitoring."""

    def __init__(
        self,
        ctx: PipelineContext,
    ):
        self.ctx = ctx
        self.config = ctx.config
        self.secrets = ctx.secrets
        self.bus = ctx.bus
        self._running = False
        self._modules: Dict[str, Any] = {}
        self._health_score = 1.0
        self._started_at = 0.0

    def register_module(self, name: str, module: Any) -> None:
        """Register a pipeline module for lifecycle management."""
        self._modules[name] = module

    async def start(self) -> None:
        """Start all modules and begin health monitoring."""
        self._started_at = time.time()
        self._running = True
        for name, module in self._modules.items():
            if hasattr(module, "start"):
                try:
                    await module.start()
                except Exception:
                    logger.exception(f"[orchestrator] Failed to start {name}")
                    continue
            logger.info(f"[orchestrator] {name} started")
        logger.info(f"[orchestrator] All {len(self._modules)} modules running")
        # Start health check loop
        asyncio.create_task(self._health_loop())

    async def stop(self) -> None:
        """Graceful shutdown of all modules."""
        self._running = False
        for name, module in reversed(list(self._modules.items())):
            if hasattr(module, "stop"):
                try:
                    await module.stop()
                except Exception:
                    logger.exception(f"[orchestrator] Error stopping {name}")
        logger.info("[orchestrator] All modules stopped")

    async def _health_loop(self) -> None:
        """Periodic health check every 60s. Also emits heartbeats for HealthMonitor."""
        while self._running:
            await asyncio.sleep(60)
            # Emit heartbeats for all registered modules (consumed by HealthMonitor)
            for name in list(self._modules.keys()):
                await self.bus.emit("module_heartbeat", module=name, status="running")

            alive_count = 0
            for name, module in self._modules.items():
                if hasattr(module, "is_alive"):
                    try:
                        if module.is_alive:
                            alive_count += 1
                    except Exception:
                        logger.exception(
                            f"[orchestrator] is_alive check failed for {name}"
                        )
            self._health_score = alive_count / max(len(self._modules), 1)
            uptime = time.time() - self._started_at
            logger.info(
                f"[orchestrator] Health: {alive_count}/{len(self._modules)} "
                f"modules alive | Uptime: {uptime:.0f}s | "
                f"Score: {self._health_score:.2f}"
            )
            await self.bus.emit(
                "health_check",
                health_score=self._health_score,
                uptime=uptime,
                modules_alive=alive_count,
                modules_total=len(self._modules),
            )

    @property
    def uptime(self) -> float:
        """Seconds since orchestrator started."""
        return time.time() - self._started_at if self._started_at else 0.0

    @property
    def is_alive(self) -> bool:
        """Whether the orchestrator is currently running."""
        return self._running

    @property
    def health_score(self) -> float:
        """Ratio of alive modules to total registered modules."""
        return self._health_score
