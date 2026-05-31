"""Tests for AttributionManager."""

import pytest
from pipeline.event_bus import EventBus
from pipeline.attribution_manager import AttributionManager


class TestAttributionManager:
    """Test suite for AttributionManager."""

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Starting and stopping should not raise."""
        bus = EventBus()
        mgr = AttributionManager(bus)
        await mgr.start()
        await mgr.stop()

    @pytest.mark.asyncio
    async def test_trade_attributed_event_on_position_closed(self):
        """position_closed event should produce trade_attributed."""
        bus = EventBus()
        mgr = AttributionManager(bus)
        await mgr.start()

        results = []

        async def _collect(**d):
            results.append(d)

        bus.on("trade_attributed", _collect)

        await bus.emit(
            "position_closed",
            position={
                "pnl": 50.0,
                "expected_pnl": 45.0,
                "fill_price": 1.1000,
                "signal_price": 1.0995,
                "strategy": "test_expert",
                "symbol": "EURUSD",
                "direction": "BUY",
                "market_return": 0.001,
            },
        )

        assert len(results) == 1
        attr = results[0]["attribution"]
        assert attr["total_pnl"] == 50.0
        assert attr["alpha_signal"] == 45.0
        assert "execution_quality" in attr
        assert "slippage" in attr

        await mgr.stop()

    @pytest.mark.asyncio
    async def test_attribution_report_event(self):
        """get_report should return per-strategy summary."""
        bus = EventBus()
        mgr = AttributionManager(bus)
        await mgr.start()

        await bus.emit(
            "position_closed",
            position={
                "pnl": 100.0,
                "expected_pnl": 80.0,
                "fill_price": 1.1000,
                "signal_price": 1.0995,
                "strategy": "test_expert",
                "symbol": "EURUSD",
                "direction": "BUY",
            },
        )

        report = mgr.get_report()
        assert "test_expert" in report
        assert report["test_expert"]["trades"] == 1

        await mgr.stop()

    @pytest.mark.asyncio
    async def test_alpha_decay_triggers_strategy_disable(self):
        """After enough losing trades, should_disable should trigger."""
        bus = EventBus()
        mgr = AttributionManager(bus, luck_window=5)
        await mgr.start()

        disable_events = []

        async def _collect(**d):
            disable_events.append(d)

        bus.on("strategy_disable", _collect)

        # 25 consecutive losing trades with negative expected_pnl
        for _ in range(25):
            await bus.emit(
                "position_closed",
                position={
                    "pnl": -10.0,
                    "expected_pnl": -5.0,
                    "fill_price": 1.1000,
                    "signal_price": 1.1000,
                    "strategy": "bad_expert",
                    "symbol": "EURUSD",
                    "direction": "SELL",
                },
            )

        # should_disable requires min_trades=20 AND all alpha_signal <= 0
        assert len(disable_events) >= 1, "Expected at least one strategy_disable event"
        assert disable_events[-1]["strategy"] == "bad_expert"
        assert disable_events[-1]["reason"] == "alpha_decay"

        await mgr.stop()

    @pytest.mark.asyncio
    async def test_emit_report_publishes_event(self):
        """emit_report should publish attribution_report event."""
        bus = EventBus()
        mgr = AttributionManager(bus)
        await mgr.start()

        await bus.emit(
            "position_closed",
            position={
                "pnl": 50.0,
                "expected_pnl": 40.0,
                "fill_price": 1.1000,
                "signal_price": 1.1005,
                "strategy": "alpha_strat",
                "symbol": "GBPUSD",
                "direction": "BUY",
            },
        )

        reports = []

        async def _collect(**d):
            reports.append(d)

        bus.on("attribution_report", _collect)

        await mgr.emit_report()

        assert len(reports) == 1
        report = reports[0]["report"]
        assert "alpha_strat" in report

        await mgr.stop()

    @pytest.mark.asyncio
    async def test_execution_result_does_not_crash(self):
        """execution_result event should not raise."""
        bus = EventBus()
        mgr = AttributionManager(bus)
        await mgr.start()

        await bus.emit(
            "execution_result",
            result={
                "slippage": 0.5,
                "symbol": "EURUSD",
                "filled": True,
            },
        )

        await mgr.stop()

    @pytest.mark.asyncio
    async def test_flat_position_data(self):
        """position_closed with flat fields (no 'position' key) should still work."""
        bus = EventBus()
        mgr = AttributionManager(bus)
        await mgr.start()

        results = []

        async def _collect(**d):
            results.append(d)

        bus.on("trade_attributed", _collect)

        # Emit with flat data instead of nested 'position' key
        await bus.emit(
            "position_closed",
            pnl=25.0,
            expected_pnl=20.0,
            fill_price=1.1000,
            signal_price=1.0990,
            strategy="flat_test",
            symbol="USDJPY",
            direction="SELL",
        )

        assert len(results) == 1
        attr = results[0]["attribution"]
        assert attr["total_pnl"] == 25.0
        assert attr["alpha_signal"] == 20.0

        await mgr.stop()
