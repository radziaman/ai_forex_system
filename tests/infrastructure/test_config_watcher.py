"""Tests for ConfigWatcher hot-reload functionality."""

import os
import tempfile
import asyncio
import pytest
from pipeline.event_bus import EventBus
from infrastructure.config import AppConfig
from infrastructure.config_watcher import ConfigWatcher


class TestConfigWatcher:
    @pytest.mark.asyncio
    async def test_detects_file_change(self):
        """ConfigWatcher should emit config_changed when file is modified."""
        # Write initial config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("key: value\n")
            tmp_path = f.name

        bus = EventBus()
        results = []

        async def capture(**data):
            results.append(data)

        bus.on("config_changed", capture)

        watcher = ConfigWatcher(bus, config_path=tmp_path, poll_interval=0.1)
        await watcher.start()

        # Let the watcher record initial mtime
        await asyncio.sleep(0.2)

        # Modify the file
        with open(tmp_path, "w") as f:
            f.write("key: new_value\n")

        await asyncio.sleep(0.3)
        await watcher.stop()

        assert len(results) >= 1
        assert results[0]["config"]["key"] == "new_value"

        os.unlink(tmp_path)

    @pytest.mark.asyncio
    async def test_no_change_no_event(self):
        """ConfigWatcher should NOT emit when file is unchanged."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("key: value\n")
            tmp_path = f.name

        bus = EventBus()
        results = []

        async def capture(**data):
            results.append(data)

        bus.on("config_changed", capture)

        watcher = ConfigWatcher(bus, config_path=tmp_path, poll_interval=0.1)
        await watcher.start()
        await asyncio.sleep(0.3)
        await watcher.stop()

        assert len(results) == 0
        os.unlink(tmp_path)

    @pytest.mark.asyncio
    async def test_missing_file_does_not_crash(self):
        """ConfigWatcher should handle missing file gracefully."""
        bus = EventBus()
        watcher = ConfigWatcher(
            bus, config_path="/nonexistent/config.yaml", poll_interval=0.1
        )
        await watcher.start()
        await asyncio.sleep(0.3)
        await watcher.stop()
        # No crash = pass

    @pytest.mark.asyncio
    async def test_reloads_appconfig_in_place(self):
        """ConfigWatcher should call AppConfig.reload when config changes."""

        # Create an AppConfig from a yaml file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("trading:\n  max_risk_per_trade: 0.02\n")
            tmp_path = f.name

        config = AppConfig.from_yaml(tmp_path)
        assert config.trading.max_risk_per_trade == 0.02

        bus = EventBus()
        results = []

        async def capture(**data):
            results.append(data)

        bus.on("config_changed", capture)

        watcher = ConfigWatcher(
            bus, config_path=tmp_path, poll_interval=0.1, config=config
        )
        await watcher.start()

        await asyncio.sleep(0.2)

        # Modify the file with new risk value
        with open(tmp_path, "w") as f:
            f.write("trading:\n  max_risk_per_trade: 0.05\n")

        await asyncio.sleep(0.3)
        await watcher.stop()

        # AppConfig should be updated in-place
        assert config.trading.max_risk_per_trade == 0.05

        os.unlink(tmp_path)
