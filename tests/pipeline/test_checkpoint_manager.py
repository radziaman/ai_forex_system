"""Tests for CheckpointManager."""
import asyncio
import tempfile
import pytest
from pipeline.event_bus import EventBus


class TestCheckpointManager:
    @pytest.mark.asyncio
    async def test_tracks_open_positions(self):
        """Position opened/closed events should update tracking."""
        from pipeline.checkpoint_manager import CheckpointManager

        bus = EventBus()
        mgr = CheckpointManager(
            bus, checkpoint_dir=tempfile.mkdtemp(), save_interval=9999
        )
        await mgr.start()

        await bus.emit(
            "position_opened",
            position={
                "position_id": 1,
                "symbol": "EURUSD",
                "direction": "BUY",
                "volume": 0.01,
                "entry_price": 1.12,
            },
        )
        assert mgr.open_position_count == 1

        await bus.emit("position_closed", position={"position_id": 1})
        assert mgr.open_position_count == 0

        await mgr.stop()

    @pytest.mark.asyncio
    async def test_saves_and_loads_positions(self):
        """Positions should survive save+load cycle."""
        from pipeline.checkpoint_manager import CheckpointManager

        tmpdir = tempfile.mkdtemp()
        bus = EventBus()

        # Save positions
        mgr = CheckpointManager(
            bus,
            checkpoint_dir=tmpdir,
            save_interval=9999,
            auto_load=False,
        )
        await mgr.start()
        await bus.emit(
            "position_opened",
            position={
                "position_id": 42,
                "symbol": "GBPUSD",
                "direction": "SELL",
                "volume": 0.02,
                "entry_price": 1.25,
            },
        )
        await mgr._save_all()
        await mgr.stop()

        # Load in new instance
        mgr2 = CheckpointManager(
            bus,
            checkpoint_dir=tmpdir,
            save_interval=9999,
            auto_load=True,
        )
        await mgr2._load_state()
        assert mgr2.open_position_count == 1
        positions = mgr2.get_open_positions()
        assert positions[0]["symbol"] == "GBPUSD"
        await mgr2.stop()

    @pytest.mark.asyncio
    async def test_save_checkpoint_event_triggers_save(self):
        """Emit save_checkpoint should trigger a save."""
        from pipeline.checkpoint_manager import CheckpointManager

        tmpdir = tempfile.mkdtemp()
        bus = EventBus()

        results = []
        bus.on("checkpoint_saved", lambda **d: results.append(d))

        mgr = CheckpointManager(
            bus,
            checkpoint_dir=tmpdir,
            save_interval=9999,
            auto_load=False,
        )
        await mgr.start()
        await bus.emit("save_checkpoint")
        await asyncio.sleep(0.1)
        await mgr.stop()

        assert len(results) >= 1
        assert "open_positions" in results[0]
