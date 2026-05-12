"""Integration tests for critical system flows."""

import asyncio
import pytest
import numpy as np
import os
import tempfile
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock


class TestMasterAIOrchestrator:
    @pytest.fixture
    def orchestrator(self):
        from ai.master_orchestrator import (
            MasterAIOrchestrator,
            SystemState,
            EnhancementStatus,
        )

        ma = MasterAIOrchestrator(initial_balance=100000.0)
        return ma

    def test_initial_state(self, orchestrator):
        assert orchestrator.system_state.value == "optimal"
        assert orchestrator.system_pnl == 0.0
        assert orchestrator.system_sharpe == 0.0
        assert len(orchestrator.enhancements) == len(orchestrator.ENHANCEMENTS)
        assert orchestrator.epsilon == 0.3

    def test_safety_critical_always_active(self, orchestrator):
        for enh in orchestrator.SAFETY_CRITICAL:
            assert enh in orchestrator.enhancements

    def test_on_trade_result(self, orchestrator):
        trade = {
            "symbol": "EURUSD",
            "pnl": 50.0,
            "direction": "BUY",
            "entry": 1.12,
            "exit": 1.13,
        }
        orchestrator.on_trade_result(trade)
        assert len(orchestrator.trade_history) == 1
        assert orchestrator.system_pnl == 50.0

    def test_record_trade_updates_metrics(self, orchestrator):
        for i in range(10):
            orchestrator.record_trade(
                {"symbol": "EURUSD", "pnl": 10.0 if i % 2 == 0 else -5.0}
            )
        assert orchestrator.system_pnl == 25.0
        assert len(orchestrator.trade_history) == 10

    def test_check_enhancement_health(self, orchestrator):
        health = orchestrator._check_enhancement_health()
        assert isinstance(health, dict)
        assert len(health) == len(orchestrator.ENHANCEMENTS)
        for k, v in health.items():
            assert 0.0 <= v <= 1.0

    def test_calculate_system_performance_no_trades(self, orchestrator):
        perf = orchestrator._calculate_system_performance()
        assert perf == 0.0

    def test_calculate_system_performance_with_trades(self, orchestrator):
        for i in range(10):
            orchestrator.record_trade({"symbol": "EURUSD", "pnl": 10.0})
        perf = orchestrator._calculate_system_performance()
        assert perf != 0.0

    def test_assess_market_conditions(self, orchestrator):
        assert orchestrator._assess_market_conditions({}) == "unknown"
        assert (
            orchestrator._assess_market_conditions(
                {"EURUSD": {"atr": 0.001, "price": 1.12}}
            )
            == "normal"
        )
        assert (
            orchestrator._assess_market_conditions(
                {"EURUSD": {"atr": 0.1, "price": 1.12}}
            )
            == "volatile"
        )

    def test_state_transitions(self, orchestrator):
        from ai.master_orchestrator import SystemDecision

        # Start optimal
        assert orchestrator.system_state.value == "optimal"
        # Halt
        orchestrator._update_system_state(SystemDecision(action="halt", reason="test"))
        assert orchestrator.system_state.value == "halted"
        # Continue from halted -> recovering
        orchestrator._update_system_state(
            SystemDecision(action="continue", reason="test")
        )
        assert orchestrator.system_state.value == "recovering"
        # Continue from recovering -> optimal
        orchestrator._update_system_state(
            SystemDecision(action="continue", reason="test")
        )
        assert orchestrator.system_state.value == "optimal"


class TestCircuitBreaker:
    @pytest.fixture
    def cb(self):
        from risk.circuit_breaker import CircuitBreaker

        return CircuitBreaker(
            price_velocity_threshold=0.005,
            spread_multiplier_threshold=5.0,
            volume_spike_multiplier=10.0,
            cooldown_seconds=0,  # No cooldown for tests
        )

    def test_healthy_market(self, cb):
        should_halt, reason, snap = cb.check_market_health(
            "EURUSD",
            {
                "bid": 1.1200,
                "ask": 1.1201,
                "volume": 1000,
            },
        )
        assert not should_halt
        assert snap.is_healthy

    def test_price_velocity_trigger(self, cb):
        # Simulate flash crash with successive ticks
        cb.price_history["EURUSD"] = [
            1.12,
            1.12,
            1.12,
            1.12,
            1.12,
            1.12,
            1.12,
            1.12,
            1.10,
        ]  # ~1.8% drop
        cb.spread_history["EURUSD"] = [0.0001] * 9
        cb.volume_history["EURUSD"] = [1000] * 9
        cb.volatility_history["EURUSD"] = [0.001] * 20
        cb._update_normal_levels("EURUSD")

        should_halt, reason, snap = cb.check_market_health(
            "EURUSD",
            {
                "bid": 1.10,
                "ask": 1.1001,
                "volume": 10000,
                "price": 1.10,
            },
        )
        # Should detect velocity > 0.005
        assert should_halt

    def test_liquidity_drought(self, cb):
        cb.normal_spreads["EURUSD"] = 0.0001
        should_halt, reason, snap = cb.check_market_health(
            "EURUSD",
            {
                "bid": 1.12,
                "ask": 1.13,
                "volume": 1000,
                "price": 1.125,
            },
        )
        # Spread = 0.01 / 0.0001 = 100x normal
        assert should_halt

    def test_get_snapshot(self, cb):
        snap = cb.get_snapshot()
        assert snap is not None

    def test_force_resume(self, cb):
        cb.last_halt_time["EURUSD"] = 100.0
        cb.force_resume("EURUSD")
        assert "EURUSD" not in cb.last_halt_time

    @pytest.mark.asyncio
    async def test_api_health_tracking(self, cb):
        cb.update_api_health("ctrader", True)
        assert cb.api_health["ctrader"]["status"] == "healthy"
        for _ in range(3):
            cb.update_api_health("ctrader", False)
        assert cb.api_health["ctrader"]["status"] == "unhealthy"


class TestExecutionEngine:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.on_market_data = None
        client.on_order_update = None
        return client

    @pytest.fixture
    def mock_risk(self):
        risk = MagicMock()
        risk.kill_switch_triggered = False
        risk.mode = "PAPER"
        return risk

    @pytest.fixture
    def mock_data(self):
        data = MagicMock()
        data.get_price.return_value = 1.12
        return data

    @pytest.fixture
    def engine(self, mock_client, mock_risk, mock_data):
        from execution.engine import ExecutionEngine

        eng = ExecutionEngine(mock_client, mock_risk, mock_data)
        return eng

    def test_default_price_for_symbols(self, engine):
        assert engine._get_default_price("EURUSD") == 1.12
        assert engine._get_default_price("XAUUSD") == 2000.0
        assert engine._get_default_price("BTCUSD") == 45000.0
        assert engine._get_default_price("UNKNOWN") == 1.12

    def test_simulate_open_uses_symbol_price(self, engine):
        trade = engine._simulate_open("XAUUSD", "BUY", 1000, 1990, 2010, "test")
        assert trade.symbol == "XAUUSD"
        assert trade.entry_price == 1.12  # Mock returns 1.12

    def test_pnl_usd_jpy(self, engine):
        from execution.engine import TradeRecord

        trade = TradeRecord(
            timestamp=100,
            symbol="USDJPY",
            direction="BUY",
            volume=100000,
            entry_price=150.0,
        )
        pnl = engine._pnl_usd(150.0, 151.0, "BUY", 100000, "USDJPY")
        assert pnl > 0

    def test_get_open_positions_returns_dicts(self, engine):
        trade = engine._simulate_open("EURUSD", "BUY", 1000, 1.11, 1.14, "test")
        positions = engine.get_open_positions()
        assert len(positions) == 1
        assert positions[0]["symbol"] == "EURUSD"
        assert positions[0]["direction"] == "BUY"

    def test_close_position_removes_from_open(self, engine):
        trade = engine._simulate_open("EURUSD", "BUY", 1000, 1.11, 1.14, "test")
        pid = trade.position_id
        pnl = engine._pnl_usd(trade.entry_price, 1.15, "BUY", 1000, "EURUSD")
        result = engine._simulate_close(trade, "Take Profit", 1.15, pnl)
        assert result
        assert pid not in engine.open_positions


class TestEventBus:
    @pytest.fixture
    def event_bus(self):
        from infrastructure.event_bus import TradingEventBus, EventType

        eb = TradingEventBus()
        return eb

    def test_subscribe(self, event_bus):
        from infrastructure.event_bus import EventType

        received = []

        def handler(event):
            received.append(event)

        from infrastructure.event_bus import Event

        event_bus.subscribe(EventType.TICK, handler)
        asyncio.run(event_bus._handle_event(Event(EventType.TICK, {"price": 1.12})))
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_async_subscribe(self, event_bus):
        from infrastructure.event_bus import EventType

        received = []

        async def handler(event):
            received.append(event)

        event_bus.subscribe(EventType.TRADE_SIGNAL, handler)
        await event_bus.start()
        await event_bus.emit(EventType.TRADE_SIGNAL, {"signal": "BUY"})
        await event_bus._handle_event(event_bus._event_history[-1])
        await event_bus.stop()

    def test_get_stats(self, event_bus):
        from infrastructure.event_bus import EventType

        event_bus._event_stats[EventType.TICK] = 5
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        event_bus._event_queue = asyncio.Queue()
        stats = event_bus.get_stats()
        assert stats["total_events"] == 5
        loop.close()
