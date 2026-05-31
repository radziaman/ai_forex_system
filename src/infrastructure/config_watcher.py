"""
Config file watcher that emits 'config_changed' events on the EventBus
when config.yaml is modified.

Uses polling (file mtime) rather than inotify/watchdog to avoid
external dependencies. Polls every N seconds (default: 10).
"""

from __future__ import annotations

import os
import asyncio
from typing import Any, Optional

import yaml
from loguru import logger


class ConfigWatcher:
    """Polls config.yaml for modifications and emits config_changed events."""

    def __init__(
        self,
        event_bus: Any,
        config_path: str = "config.yaml",
        poll_interval: float = 10.0,
        config: Optional[Any] = None,
    ):
        """
        Args:
            event_bus: EventBus instance to emit events on.
            config_path: Path to the YAML config file to watch.
            poll_interval: Seconds between polls.
            config: Optional AppConfig instance to reload on change.
        """
        self._bus = event_bus
        self._config_path = config_path
        self._poll_interval = poll_interval
        self._config = config
        self._last_mtime: float = 0.0
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start polling for config changes."""
        self._last_mtime = self._get_mtime()
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(
            f"ConfigWatcher started: polling {self._config_path} "
            f"every {self._poll_interval}s"
        )

    async def stop(self) -> None:
        """Stop polling."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("ConfigWatcher stopped")

    def _get_mtime(self) -> float:
        try:
            return os.path.getmtime(self._config_path)
        except OSError:
            return 0.0

    async def _poll_loop(self) -> None:
        """Poll for config changes and emit config_changed event."""
        while self._running:
            await asyncio.sleep(self._poll_interval)
            current_mtime = self._get_mtime()
            if current_mtime > self._last_mtime:
                self._last_mtime = current_mtime
                await self._on_config_changed()

    async def _on_config_changed(self) -> None:
        """Handle config file change — reload and emit event."""
        try:
            with open(self._config_path, "r") as f:
                new_config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"ConfigWatcher: failed to reload config: {e}")
            return

        if new_config is None:
            logger.warning(f"ConfigWatcher: {self._config_path} is empty — skipping")
            return

        logger.info("Config file modified — emitting config_changed")

        # If we have an AppConfig instance, reload it in-place
        if self._config is not None and hasattr(self._config, "reload"):
            try:
                self._config.reload(self._config_path)
            except Exception as e:
                logger.warning(f"ConfigWatcher: failed to update AppConfig: {e}")

        # Emit raw config dict so subscribers can read what they need
        await self._bus.emit(
            "config_changed",
            config=new_config,
            path=self._config_path,
        )
