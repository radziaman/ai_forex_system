"""Tests for extracted services — position persistence."""

import pytest
import tempfile
from unittest.mock import MagicMock


class TestPositionPersistence:
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield tmp

    def test_save_load_positions(self, temp_dir):
        from services.position_persistence import PositionPersistence

        pp = PositionPersistence(base_path=temp_dir)
        positions = [
            {"position_id": 1, "symbol": "EURUSD", "direction": "BUY", "volume": 1000},
            {"position_id": 2, "symbol": "GBPUSD", "direction": "SELL", "volume": 2000},
        ]
        assert pp.save_positions(positions)
        loaded = pp.load_positions()
        assert len(loaded) == 2
        assert loaded[0]["symbol"] == "EURUSD"

    def test_save_load_trade_history(self, temp_dir):
        from services.position_persistence import PositionPersistence

        pp = PositionPersistence(base_path=temp_dir)
        trades = [
            {"symbol": "EURUSD", "pnl": 50.0, "direction": "BUY"},
            {"symbol": "GBPUSD", "pnl": -30.0, "direction": "SELL"},
        ]
        assert pp.save_trade_history(trades)
        loaded = pp.load_trade_history()
        assert len(loaded) == 2

    def test_save_load_risk_state(self, temp_dir):
        from services.position_persistence import PositionPersistence

        risk = MagicMock()
        risk.daily_pnl = 150.0
        risk.daily_trades = 3
        risk.consecutive_losses = 0
        risk.total_trades = 50
        risk.wins = 30
        risk.losses = 20
        risk.peak_balance = 105000.0
        risk.kill_switch_triggered = False
        risk.initial_balance = 100000.0

        pp = PositionPersistence(base_path=temp_dir)
        assert pp.save_risk_state(risk)

        risk2 = MagicMock()
        risk2.initial_balance = 100000.0
        assert pp.load_risk_state(risk2)
        assert risk2.daily_pnl == 150.0
        assert risk2.total_trades == 50

    def test_load_empty_returns_defaults(self, temp_dir):
        from services.position_persistence import PositionPersistence

        pp = PositionPersistence(base_path=temp_dir)
        assert pp.load_positions() == []
        assert pp.load_trade_history() == []

    def test_save_all(self, temp_dir):
        from services.position_persistence import PositionPersistence

        exec_engine = MagicMock()
        exec_engine.get_open_positions.return_value = [
            {"position_id": 1, "symbol": "EURUSD", "direction": "BUY", "volume": 1000},
        ]
        exec_engine.get_trade_history.return_value = [
            {"symbol": "EURUSD", "pnl": 50.0, "direction": "BUY"},
        ]
        risk = MagicMock()
        risk.daily_pnl = 0.0

        pp = PositionPersistence(base_path=temp_dir)
        assert pp.save_all(exec_engine, risk)
        assert len(pp.load_positions()) == 1
        assert len(pp.load_trade_history()) == 1
