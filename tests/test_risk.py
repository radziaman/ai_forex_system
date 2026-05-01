"""Tests for risk management module"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rts_ai_fx.risk import RiskManager, TrailingStopManager

from rts_ai_fx.risk import RiskManager, TrailingStopManager


class TestRiskManager:
    """Test risk management functionality"""

    def test_position_size_calculation(self):
        """Test position size calculation"""
        rm = RiskManager(account_balance=10000, max_risk_pct=0.02)
        size = rm.calculate_position_size(entry=1.1000, stop_loss=1.0950)

        assert size > 0
        # Max is balance * 0.1 = 1000
        assert size <= 1000

    def test_dynamic_sl_tp(self):
        """Test ATR-based SL/TP calculation"""
        rm = RiskManager(account_balance=10000)
        entry = 1.1000
        atr = 0.0010

        sl, tp = rm.calculate_dynamic_sl_tp(entry, atr, "long")
        assert sl < entry
        assert tp > entry

        sl, tp = rm.calculate_dynamic_sl_tp(entry, atr, "short")
        assert sl > entry
        assert tp < entry

    def test_daily_drawdown_check(self):
        """Test daily drawdown protection"""
        rm = RiskManager(account_balance=10000, max_daily_drawdown=0.05)

        assert rm.check_daily_drawdown(-400) is True
        assert rm.check_daily_drawdown(-600) is False

    def test_correlation_filter(self):
        """Test correlation filtering"""
        rm = RiskManager(account_balance=10000, max_open_positions=5)

        assert rm.check_correlation("EURUSD", ["GBPUSD", "USDJPY"]) is True
        assert (
            rm.check_correlation(
                "EURUSD", ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD"]
            )
            is False
        )


class TestTrailingStopManager:
    """Test trailing stop management"""

    def test_trailing_stop_update(self):
        """Test trailing stop updates at TP levels"""
        tsm = TrailingStopManager(tp1_pct=0.01, tp2_pct=0.02, tp3_pct=0.03)
        entry = 1.1000
        atr = 0.0010

        # pnl_pct = 0.0018 < 0.01 (tp1) - should NOT trigger
        new_sl = tsm.update_trailing_stop(entry, 1.1020, atr, "long")
        assert new_sl is None

        # pnl_pct = 0.02 > 0.01 (tp1) - SHOULD trigger
        new_sl = tsm.update_trailing_stop(entry, 1.1220, atr, "long")
        assert new_sl is not None
        assert new_sl == entry
        assert new_sl == entry
