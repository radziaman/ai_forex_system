"""Tests for extracted services."""
import pytest
import numpy as np
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock


class TestRiskService:
    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.trading.max_risk_per_trade = 0.02
        config.trading.max_drawdown = 0.10
        config.trading.max_margin_usage = 0.80
        config.trading.sl_atr_multiplier = 2.0
        config.trading.commission_per_lot = 7.0
        return config

    def test_initialization(self, mock_config):
        from services.risk_service import RiskService
        svc = RiskService(mock_config, "paper", 100000.0)
        assert svc.manager is not None
        assert svc.manager.initial_balance == 100000.0
        assert svc.manager.mode == "PAPER"
        assert len(svc.circuit_breakers) > 0

    def test_kelly_sizing(self, mock_config):
        from services.risk_service import RiskService
        svc = RiskService(mock_config, "paper", 100000.0)
        size = svc.calculate_kelly_size(100000.0, 1.12, 0.001, 0.7)
        assert size >= 0
        assert size <= 100000 * 0.8 / 1.12

    def test_pre_trade_check_passes_by_default(self, mock_config):
        from services.risk_service import RiskService
        svc = RiskService(mock_config, "paper", 100000.0)
        approved, reason = svc.pre_trade_check({"balance": 100000, "equity": 100000, "margin": 0})
        assert approved
        assert reason == "OK"

    def test_pre_trade_check_kill_switch(self, mock_config):
        from services.risk_service import RiskService
        svc = RiskService(mock_config, "paper", 100000.0)
        svc.manager.kill_switch_triggered = True
        approved, reason = svc.pre_trade_check({"balance": 100000, "equity": 100000, "margin": 0})
        assert not approved
        assert "Kill switch" in reason

    def test_circuit_breaker_creation(self, mock_config):
        from services.risk_service import RiskService
        svc = RiskService(mock_config, "paper", 100000.0)
        assert "EURUSD" in svc.circuit_breakers
        assert "BTCUSD" in svc.circuit_breakers

    def test_sl_tp_calculation(self, mock_config):
        from services.risk_service import RiskService
        svc = RiskService(mock_config, "paper", 100000.0)
        sl, tp = svc.calculate_sl_tp("BUY", 1.12, 0.001, {"sl_atr": 2.0, "tp_atr": 4.0})
        assert sl < 1.12 < tp
        assert abs(sl - (1.12 - 0.001 * 2.0)) < 1e-6
        assert abs(tp - (1.12 + 0.001 * 4.0)) < 1e-6


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
