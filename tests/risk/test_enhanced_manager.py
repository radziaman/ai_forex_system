"""Tests for EnhancedRiskManager."""

import numpy as np
import pytest

from risk.enhanced_manager import EnhancedRiskManager
from risk.manager import RiskParameters, TradeRecord


def _make_trade(position_id=1, entry=1.10, direction="BUY", sl=1.09):
    """Helper to create a trade-like object for tests."""
    t = TradeRecord(
        timestamp=0.0,
        symbol="EURUSD",
        direction=direction,
        volume=1.0,
        entry_price=entry,
        status="OPEN",
    )
    t.position_id = position_id
    t.sl = sl
    t.tp = 1.12
    return t


def test_cvar_adjusted_kelly_size():
    params = RiskParameters()
    mgr = EnhancedRiskManager(params, initial_balance=100_000.0)
    # Seed price history so VaR/CVaR are stable
    price = 1.10
    for i in range(100):
        p = price + np.sin(i / 10) * 0.01
        mgr.update_price_history(p, "EURUSD")

    size = mgr.calculate_cvar_kelly_size(
        balance=100_000.0,
        price=price,
        atr=0.005,
        confidence=0.95,
        symbol="EURUSD",
        open_positions=[],
    )
    assert 0 < size < 100_000.0 / price


def test_drawdown_circuit_breaker():
    params = RiskParameters(max_drawdown=0.05)
    mgr = EnhancedRiskManager(params, initial_balance=100_000.0)
    # Peak is 100k, equity drops to 90k -> 10% drawdown
    approved, reason = mgr.pre_trade_checks(
        balance=100_000.0,
        equity=90_000.0,
        margin=0.0,
        current_pnl=-10_000.0,
    )
    assert approved is False
    assert "drawdown" in reason.lower()
    assert mgr.kill_switch_triggered is True


def test_mae_mfe_tracking():
    params = RiskParameters()
    mgr = EnhancedRiskManager(params, initial_balance=100_000.0)
    trade = _make_trade(position_id=1, entry=1.10, direction="BUY", sl=1.09)
    mgr.record_trade_open(trade)

    # Price goes up to 1.12 (MFE +0.02) then down to 1.08 (MAE +0.02)
    mgr.update_trade_mae_mfe(1, 1.11)
    mgr.update_trade_mae_mfe(1, 1.12)
    mgr.update_trade_mae_mfe(1, 1.08)

    summary = mgr.get_mae_mfe_summary()
    assert 1 in summary
    assert summary[1]["mfe"] == pytest.approx(0.02, abs=1e-9)
    assert summary[1]["mae"] == pytest.approx(0.02, abs=1e-9)

    # Close the trade
    trade.status = "OPEN"
    mgr.record_trade_close(trade, exit_price=1.09, pnl=-100.0)
    assert 1 not in mgr.get_mae_mfe_summary()


def test_stress_test_flash_crash():
    params = RiskParameters()
    mgr = EnhancedRiskManager(params, initial_balance=100_000.0)
    result = mgr.stress_test_flash_crash("EURUSD", current_price=1.10)
    assert result["max_loss_pct"] > 0
    assert result["scenario"] == "flash_crash"
    assert len(result["shocks"]) > 0


def test_rolling_correlation():
    params = RiskParameters()
    mgr = EnhancedRiskManager(params, initial_balance=100_000.0)
    np.random.seed(42)
    base = 1.10
    for i in range(50):
        p1 = base + np.random.normal(0, 0.001)
        p2 = base + np.random.normal(0, 0.001)
        mgr.update_price_history(p1, "EURUSD")
        mgr.update_price_history(p2, "GBPUSD")

    corr = mgr._rolling_correlation()
    assert not corr.empty
    assert "EURUSD" in corr.columns
    assert "GBPUSD" in corr.columns
    assert -1.0 <= corr.loc["EURUSD", "GBPUSD"] <= 1.0
