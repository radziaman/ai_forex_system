#!/usr/bin/env python3
"""
Tests for Execution Engine - order execution and position management.
"""
import os
import sys
import time
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from execution.engine import ExecutionEngine, TradeRecord
import asyncio


class MockCtraderClient:
    """Mock cTrader client for testing."""

    def __init__(self):
        self.on_market_data = None
        self.on_order_update = None
        self.orders = {}
        self._order_counter = 0

    async def place_order(self, order):
        self._order_counter += 1
        result = MagicMock()
        result.status = "FILLED"
        result.order_id = self._order_counter
        result.filled_price = order.price
        return result

    def get_account_info(self):
        acc = MagicMock()
        acc.balance = 100_000
        acc.equity = 100_000
        acc.margin = 0
        acc.free_margin = 100_000
        acc.currency = "USD"
        return acc


class MockRiskManager:
    """Mock RiskManager for testing."""

    def __init__(self):
        self.kill_switch_triggered = False
        self.mode = "PAPER"
        self.trade_history = []
        self.open_positions = {}

    def record_trade(self, trade, exit_price, pnl):
        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.status = "CLOSED"
        self.trade_history.append(trade)

    def update_trailing_stops(self, prices):
        pass


class MockDataManager:
    """Mock DataManager for testing."""

    def __init__(self):
        self.latest_snapshot = MagicMock()
        self.latest_snapshot.bid = 1.1200
        self.latest_snapshot.ask = 1.1202

    def get_price(self, symbol, timeframe):
        return 1.1200


@pytest.fixture
def engine():
    client = MockCtraderClient()
    risk = MockRiskManager()
    data = MockDataManager()
    return ExecutionEngine(client, risk, data)


@pytest.fixture
def sample_trade():
    return TradeRecord(
        timestamp=time.time(),
        symbol="EURUSD",
        direction="BUY",
        volume=0.1,
        entry_price=1.1200,
        sl=1.1150,
        tp=1.1300,
    )


class TestExecutionEngine:
    """Test cases for ExecutionEngine."""

    def test_initialization(self, engine):
        assert engine.client is not None
        assert engine.risk is not None
        assert engine.data is not None
        assert len(engine.open_positions) == 0
        assert engine.total_trades == 0

    @pytest.mark.asyncio
    async def test_open_position_paper_mode(self, engine):
        engine.risk.mode = "PAPER"
        trade = await engine.open_position(
            symbol="EURUSD",
            direction="BUY",
            volume=0.1,
            sl=1.1150,
            tp=1.1300,
            reason="Test signal"
        )
        assert trade is not None
        assert trade.symbol == "EURUSD"
        assert trade.direction == "BUY"
        assert trade.volume == 0.1
        assert trade.status == "OPEN"
        assert len(engine.open_positions) == 1

    @pytest.mark.asyncio
    async def test_close_position_paper_mode(self, engine):
        engine.risk.mode = "PAPER"
        trade = await engine.open_position(
            symbol="EURUSD", direction="BUY", volume=0.1, sl=1.1150, tp=1.1300
        )
        position_id = trade.position_id
        result = await engine.close_position(position_id, reason="Test close")
        assert result is True
        assert position_id not in engine.open_positions
        assert len(engine.trade_history) == 1

    @pytest.mark.asyncio
    async def test_close_nonexistent_position(self, engine):
        result = await engine.close_position(999)
        assert result is False

    @pytest.mark.asyncio
    async def test_kill_switch_prevents_trading(self, engine):
        engine.risk.kill_switch_triggered = True
        trade = await engine.open_position(
            symbol="EURUSD", direction="BUY", volume=0.1, sl=1.1150, tp=1.1300
        )
        assert trade is None

    def test_calculate_pnl_buy(self, engine):
        trade = TradeRecord(
            timestamp=time.time(),
            symbol="EURUSD", direction="BUY", entry_price=1.1200,
            volume=100_000
        )
        pnl = engine._calculate_pnl(trade, exit_price=1.1250)
        assert pnl > 0

    def test_calculate_pnl_sell(self, engine):
        trade = TradeRecord(
            timestamp=time.time(),
            symbol="EURUSD", direction="SELL", entry_price=1.1200,
            volume=100_000
        )
        pnl = engine._calculate_pnl(trade, exit_price=1.1150)
        assert pnl > 0

    def test_calculate_pnl_no_change(self, engine):
        trade = TradeRecord(
            timestamp=time.time(),
            symbol="EURUSD", direction="BUY", entry_price=1.1200,
            volume=100_000
        )
        pnl = engine._calculate_pnl(trade, exit_price=1.1200)
        assert pnl == 0.0

    def test_get_symbol_id(self, engine):
        assert engine._get_symbol_id("EURUSD") == 1
        assert engine._get_symbol_id("GBPUSD") == 2
        assert engine._get_symbol_id("EURJPY") == 3
        assert engine._get_symbol_id("USDJPY") == 4
        assert engine._get_symbol_id("AUDUSD") == 5
        assert engine._get_symbol_id("UNKNOWN") == 1

    def test_get_open_positions(self, engine):
        engine.risk.mode = "PAPER"
        import asyncio
        trade = asyncio.get_event_loop().run_until_complete(
            engine.open_position("EURUSD", "BUY", 0.1, 1.1150, 1.1300)
        )
        positions = engine.get_open_positions()
        assert len(positions) == 1
        assert positions[0]["symbol"] == "EURUSD"
        assert positions[0]["direction"] == "BUY"

    def test_get_trade_history(self, engine):
        assert len(engine.get_trade_history()) == 0

    @pytest.mark.asyncio
    async def test_close_all_positions(self, engine):
        engine.risk.mode = "PAPER"
        await engine.open_position("EURUSD", "BUY", 0.1, 1.1150, 1.1300)
        await engine.open_position("GBPUSD", "SELL", 0.1, 1.3150, 1.3050)
        assert len(engine.open_positions) == 2
        await engine.close_all_positions(reason="Test close all")
        assert len(engine.open_positions) == 0
        assert len(engine.trade_history) == 2

    def test_on_market_data_sl_trigger(self, engine):
        engine.risk.mode = "PAPER"
        import asyncio
        trade = asyncio.get_event_loop().run_until_complete(
            engine.open_position("EURUSD", "BUY", 0.1, 1.1150, 1.1300)
        )
        mock_depth = MagicMock()
        mock_depth.symbol = "EURUSD"
        mock_depth.bid = 1.1140
        mock_depth.ask = 1.1142
        with patch.object(engine, 'close_position', new_callable=AsyncMock) as mock_close:
            asyncio.get_event_loop().run_until_complete(engine._on_market_data(mock_depth))
            mock_close.assert_called_once()

    def test_on_order_update_filled(self, engine):
        result = MagicMock()
        result.status = "FILLED"
        result.order_id = 123
        with patch('loguru.logger.debug') as mock_log:
            engine._on_order_update(result)
            mock_log.assert_called_once()

    def test_on_order_update_rejected(self, engine):
        result = MagicMock()
        result.status = "REJECTED"
        result.error = "Insufficient margin"
        with patch('loguru.logger.error') as mock_log:
            engine._on_order_update(result)
            mock_log.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
