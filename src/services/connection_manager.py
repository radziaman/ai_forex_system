"""Broker connection manager with auto-reconnect, heartbeat, and health monitoring."""

import asyncio
import time
from typing import Optional, Callable, Awaitable
from loguru import logger


class ConnectionManager:
    """
    Monitors broker connection health, auto-reconnects with exponential backoff.
    Re-subscribes to market data after successful reconnect.
    """

    def __init__(
        self,
        client,
        symbols: list,
        max_retries: int = 10,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        heartbeat_interval: float = 30.0,
    ):
        self._client = client
        self._symbols = symbols
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._heartbeat_interval = heartbeat_interval
        self._running = False
        self._connected = False
        self._reconnect_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._on_reconnect: Optional[Callable[[], Awaitable[None]]] = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    def set_on_reconnect(self, callback: Callable[[], Awaitable[None]]):
        self._on_reconnect = callback

    async def start(self):
        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    async def stop(self):
        self._running = False
        if self._reconnect_task:
            self._reconnect_task.cancel()
        if self._heartbeat_task:
            self._heartbeat_task.cancel()

    async def report_disconnected(self):
        self._connected = False
        if self._running:
            logger.warning("Broker connection lost — auto-reconnect enabled")

    async def _heartbeat_loop(self):
        while self._running:
            await asyncio.sleep(self._heartbeat_interval)
            if self._connected:
                try:
                    info = await asyncio.wait_for(
                        self._client.get_account_info(), timeout=5.0
                    )
                    if info is None:
                        await self.report_disconnected()
                except (asyncio.TimeoutError, Exception) as e:
                    logger.warning(f"Heartbeat failed: {e}")
                    await self.report_disconnected()

    async def _reconnect_loop(self):
        retries = 0
        while self._running:
            if self._connected:
                retries = 0
                await asyncio.sleep(1)
                continue

            delay = min(self._base_delay * (2 ** retries), self._max_delay)
            retries += 1

            if retries > self._max_retries:
                logger.error(f"Max reconnection retries ({self._max_retries}) reached")
                await asyncio.sleep(self._max_delay)
                retries = self._max_retries
                continue

            logger.info(f"Reconnecting in {delay:.0f}s (attempt {retries}/{self._max_retries})")
            await asyncio.sleep(delay)

            try:
                success = await self._client.start()
                if success:
                    self._connected = True
                    retries = 0
                    logger.success(f"Reconnected to broker (attempt {retries})")

                    if self._on_reconnect:
                        await self._on_reconnect()
                else:
                    logger.warning(f"Reconnect attempt {retries} failed")
            except Exception as e:
                logger.warning(f"Reconnect error: {e}")
