"""Tests for the --mode live code path without requiring cTrader credentials."""

import asyncio
import sys
import os
from unittest.mock import MagicMock

import pytest

# Add src to path (same pattern as other test files)
_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _src not in sys.path:
    sys.path.insert(0, _src)


class TestExecutionManagerLiveMode:
    """ExecutionManager should instantiate cleanly in live mode even without
    cTrader credentials."""

    def test_live_mode_imports_resolve(self):
        """Verify live-mode import paths exist."""
        from execution.engine import ExecutionEngine
        from execution.execution_quality import ExecutionQualityTracker
        from execution.position_reconciler import PositionReconciler
        from execution.broker_health import BrokerHealthMonitor
        from execution.is_execution import ISExecutionEngine

        assert ExecutionEngine is not None
        assert ExecutionQualityTracker is not None
        assert PositionReconciler is not None
        assert BrokerHealthMonitor is not None
        assert ISExecutionEngine is not None

    def test_paper_mode_default(self):
        """Default mode should be paper."""
        from pipeline.execution_manager import ExecutionManager
        from pipeline.pipeline_context import PipelineContext
        from pipeline.event_bus import EventBus
        from infrastructure.config import AppConfig

        config = AppConfig.from_yaml("config.yaml")
        ctx = PipelineContext(config=config, secrets=MagicMock(), bus=EventBus())
        mgr = ExecutionManager(ctx, mode="paper")
        assert mgr.mode == "paper"

    def test_live_mode_instantiation(self):
        """ExecutionManager should instantiate in live mode without crashing."""
        from pipeline.execution_manager import ExecutionManager
        from pipeline.pipeline_context import PipelineContext
        from pipeline.event_bus import EventBus
        from infrastructure.config import AppConfig

        config = AppConfig.from_yaml("config.yaml")
        ctx = PipelineContext(config=config, secrets=MagicMock(), bus=EventBus())
        mgr = ExecutionManager(ctx, mode="live")
        assert mgr.mode == "live"
        # Should not crash even without cTrader credentials
        # ExecutionEngine is loaded lazily, so just constructing is safe


class TestManagedPosition:
    """ManagedPosition dataclass should work correctly."""

    def test_managed_position_create(self):
        from pipeline.execution_manager import ManagedPosition

        pos = ManagedPosition(
            position_id=1,
            symbol="EURUSD",
            direction="BUY",
            volume=0.01,
            entry_price=1.1200,
            sl=1.1150,
            tp=1.1300,
            timestamp=1000.0,
        )
        assert pos.position_id == 1
        assert pos.symbol == "EURUSD"


class TestLiveEventFlow:
    """Event flow should work regardless of mode."""

    @pytest.mark.asyncio
    async def test_risk_approved_triggers_execution(self):
        """Pushing risk_approved event should not crash."""
        from pipeline.event_bus import EventBus
        from pipeline.execution_manager import ExecutionManager
        from pipeline.pipeline_context import PipelineContext
        from infrastructure.config import AppConfig
        from unittest.mock import MagicMock

        bus = EventBus()
        config = AppConfig.from_yaml("config.yaml")
        ctx = PipelineContext(config=config, secrets=MagicMock(), bus=bus)

        mgr = ExecutionManager(ctx, mode="paper")
        await mgr.start()

        results = []
        bus.on("position_opened", lambda **d: results.append(d))

        # Emit a risk_approved event
        await bus.emit(
            "risk_approved",
            signal={
                "symbol": "EURUSD",
                "direction": "BUY",
                "confidence": 0.7,
                "price": 1.1200,
                "expert_outputs": {},
            },
            volume=0.01,
            sl_price=1.1150,
            tp_price=1.1300,
        )

        await asyncio.sleep(0.05)
        await mgr.stop()

        # Just verify no crash

    @pytest.mark.asyncio
    async def test_rejects_null_signal_gracefully(self):
        """Null/invalid signals should be handled without crash."""
        from pipeline.event_bus import EventBus
        from pipeline.execution_manager import ExecutionManager
        from pipeline.pipeline_context import PipelineContext
        from infrastructure.config import AppConfig
        from unittest.mock import MagicMock

        bus = EventBus()
        config = AppConfig.from_yaml("config.yaml")
        ctx = PipelineContext(config=config, secrets=MagicMock(), bus=bus)

        mgr = ExecutionManager(ctx, mode="live")
        await mgr.start()

        # Emit with missing data
        await bus.emit(
            "risk_approved", signal=None, volume=None, sl_price=None, tp_price=None
        )
        await asyncio.sleep(0.05)
        await mgr.stop()
        # No crash = pass
