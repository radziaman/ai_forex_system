#!/usr/bin/env python3
"""
Tests for Risk Management - RiskManager and TrailingStopManager.
"""
import os
import sys
import numpy as np
import pytest
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from risk.manager import (
    RiskManager, RiskParameters, TradeRecord,
    TrailingStopManager, TRADE_MODE_PAPER, TRADE_MODE_LIVE
)


class TestRiskParameters:
    """Test RiskParameters dataclass."""

    def test_default_values(self):
        params = RiskParameters()
        assert params.max_risk_per_trade == 0.02
        assert params.max_drawdown == 0.10
        assert params.kelly_fraction == 0.25
        assert params.sl_atr_multiplier == 2.0
        assert params.tp_atr_multiplier == 4.0

    def test_custom_values(self):
        params = RiskParameters(max_risk_per_trade=0.05, max_drawdown=0.15)
        assert params.max_risk_per_trade == 0.05
        assert params.max_drawdown == 0.15


class TestRiskManager:
    """Test cases for RiskManager."""

    @pytest.fixture
    def params(self):
        return RiskParameters(
            max_risk_per_trade=0.02,
            max_drawdown=0.10,
            kelly_fraction=0.25,
            sl_atr_multiplier=2.0,
            tp_atr_multiplier=4.0,
        )

    @pytest.fixture
    def risk_manager(self, params):
        return RiskManager(params, initial_balance=100_000.0)

    def test_initial_state(self, risk_manager):
        assert risk_manager.initial_balance == 100_000.0
        assert risk_manager.mode == TRADE_MODE_PAPER
        assert risk_manager.kill_switch_triggered is False
        assert risk_manager.total_trades == 0
        assert risk_manager.wins == 0
        assert risk_manager.losses == 0

    def test_calculate_kelly_size(self, risk_manager):
        balance = 100_000.0
        price = 1.12
        atr = 0.005
        confidence = 0.8
        size = risk_manager.calculate_kelly_size(balance, price, atr, confidence)
        assert size > 0
        assert size <= balance * risk_manager.params.max_margin_usage / price

    def test_kelly_size_zero_atr(self, risk_manager):
        size = risk_manager.calculate_kelly_size(100_000, 1.12, 0, 0.8)
        assert size > 0

    def test_calculate_atr_sl_tp(self, risk_manager):
        entry = 1.1200
        atr = 0.005
        sl, tp = risk_manager.calculate_atr_sl_tp(entry, atr)
        assert sl < entry
        assert tp > entry
        assert abs(sl - (entry - atr * 2.0)) < 1e-6
        assert abs(tp - (entry + atr * 4.0)) < 1e-6

    def test_var_insufficient_history(self, risk_manager):
        var = risk_manager.var(confidence=0.95)
        assert var == risk_manager.initial_balance * 0.02

    def test_var_with_history(self, risk_manager):
        prices = [100 - i * 0.5 for i in range(100)]  # Decreasing prices = negative returns
        for p in prices:
            risk_manager.update_price_history(p)
        var = risk_manager.var(confidence=0.95)
        assert var < 0

    def test_cvar(self, risk_manager):
        prices = [100 - i * 0.5 for i in range(100)]  # Decreasing prices = negative returns
        for p in prices:
            risk_manager.update_price_history(p)
        cvar = risk_manager.cvar(confidence=0.95)
        assert cvar < 0

    def test_get_win_rate_empty(self, risk_manager):
        assert risk_manager.get_win_rate() == 0.0

    def test_get_win_rate_with_trades(self, risk_manager):
        risk_manager.wins = 6
        risk_manager.losses = 4
        assert abs(risk_manager.get_win_rate() - 0.6) < 1e-6

    def test_pre_trade_checks_pass(self, risk_manager):
        approved, reason = risk_manager.pre_trade_checks(
            balance=100_000, equity=100_000, margin=5000, current_pnl=0
        )
        assert approved is True
        assert reason == "OK"

    def test_pre_trade_checks_kill_switch(self, risk_manager):
        risk_manager.kill_switch_triggered = True
        approved, reason = risk_manager.pre_trade_checks(
            balance=100_000, equity=100_000, margin=5000, current_pnl=0
        )
        assert approved is False
        assert "Kill switch" in reason

    def test_pre_trade_checks_drawdown(self, risk_manager):
        approved, reason = risk_manager.pre_trade_checks(
            balance=89_000, equity=89_000, margin=5000, current_pnl=-11_000
        )
        assert approved is False
        assert "drawdown" in reason.lower()

    def test_pre_trade_checks_daily_loss(self, risk_manager):
        risk_manager.daily_pnl = -6000
        approved, reason = risk_manager.pre_trade_checks(
            balance=100_000, equity=100_000, margin=5000, current_pnl=-6000
        )
        assert approved is False
        assert "Daily loss" in reason

    def test_pre_trade_checks_consecutive_losses(self, risk_manager):
        risk_manager.consecutive_losses = 5
        approved, reason = risk_manager.pre_trade_checks(
            balance=100_000, equity=100_000, margin=5000, current_pnl=0
        )
        assert approved is False
        assert "consecutive" in reason.lower()

    def test_record_trade_win(self, risk_manager):
        trade = TradeRecord(symbol="EURUSD", direction="BUY", entry_price=1.12)
        risk_manager.record_trade(trade, exit_price=1.13, pnl=100)
        assert risk_manager.wins == 1
        assert risk_manager.losses == 0
        assert risk_manager.total_trades == 1
        assert risk_manager.consecutive_losses == 0

    def test_record_trade_loss(self, risk_manager):
        trade = TradeRecord(symbol="EURUSD", direction="BUY", entry_price=1.12)
        risk_manager.record_trade(trade, exit_price=1.11, pnl=-100)
        assert risk_manager.wins == 0
        assert risk_manager.losses == 1
        assert risk_manager.consecutive_losses == 1

    def test_check_correlation_no_matrix(self, risk_manager):
        assert risk_manager.check_correlation("EURUSD", ["GBPUSD"]) is True

    def test_update_price_history(self, risk_manager):
        risk_manager.update_price_history(1.12)
        risk_manager.update_price_history(1.13)
        assert len(risk_manager._price_history) == 2

    def test_reset_daily_stats(self, risk_manager):
        risk_manager.daily_pnl = -1000
        risk_manager.daily_trades = 5
        risk_manager.consecutive_losses = 3
        risk_manager.reset_daily_stats(100_000)
        assert risk_manager.daily_pnl == 0.0
        assert risk_manager.daily_trades == 0
        assert risk_manager.consecutive_losses == 0


class TestTrailingStopManager:
    """Test cases for TrailingStopManager."""

    @pytest.fixture
    def tsm(self):
        return TrailingStopManager(tp_pcts=[0.01, 0.02, 0.03], trail_atr_mult=2.0)

    def test_initial_state(self, tsm):
        assert tsm.tp_levels == [0.01, 0.02, 0.03]
        assert tsm.trail_mult == 2.0
        assert all(not hit for hit in tsm.tp_hit)

    def test_no_tp_hit(self, tsm):
        entry = 1.1200
        current = 1.1205
        atr = 0.005
        sl = tsm.update(entry, current, atr, direction="long")
        assert sl is None
        assert all(not hit for hit in tsm.tp_hit)

    def test_first_tp_hit(self, tsm):
        entry = 1.1200
        # Need > 1% gain to trigger first TP level (0.01)
        current = entry * 1.011  # 1.1% gain, above 0.01 threshold
        atr = 0.005
        sl = tsm.update(entry, current, atr, direction="long")
        assert sl is not None
        assert tsm.tp_hit[0] is True
        assert sl == entry

    def test_partial_close_sizes(self, tsm):
        sizes = tsm.partial_close_sizes()
        assert sizes == [0.3, 0.3, 0.4]
        assert sum(sizes) == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
