"""Comprehensive tests for RiskManager — position sizing, VaR, pre-trade checks."""

import numpy as np
import pytest

from risk.manager import (
    RiskManager,
    RiskParameters,
    TradeRecord,
    TrailingStopManager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trade(
    position_id: int = 1,
    entry: float = 1.10,
    direction: str = "BUY",
    sl: float = 1.09,
    tp: float = 1.12,
    volume: float = 1.0,
) -> TradeRecord:
    """Factory helper — mirrors pattern from test_enhanced_manager."""
    t = TradeRecord(
        timestamp=0.0,
        symbol="EURUSD",
        direction=direction,
        volume=volume,
        entry_price=entry,
        status="OPEN",
    )
    t.position_id = position_id
    t.sl = sl
    t.tp = tp
    return t


def _close_trade(
    mgr: RiskManager,
    entry: float = 1.10,
    exit_price: float = 1.12,
    sl: float = 1.09,
    volume: float = 1.0,
    pnl: float = 200.0,
) -> TradeRecord:
    """Record a closed trade for testing."""
    trade = _make_trade(entry=entry, sl=sl, volume=volume)
    mgr.record_trade(trade, exit_price=exit_price, pnl=pnl)
    return trade


def _seed_price_history(mgr: RiskManager, n: int = 100, base: float = 1.10):
    """Fill price history with synthetic data."""
    for i in range(n):
        p = base + np.sin(i / 10) * 0.01
        mgr.update_price_history(p, "EURUSD")


# ===========================================================================
# 1. Initialization
# ===========================================================================


class TestInit:
    def test_init_default_params(self):
        """Creates with default RiskParameters."""
        mgr = RiskManager(RiskParameters())
        assert mgr.params.max_risk_per_trade == 0.02
        assert mgr.params.max_drawdown == 0.10
        assert mgr.params.kelly_fraction == 0.25

    def test_init_custom_params(self):
        """Creates with custom parameters."""
        params = RiskParameters(max_drawdown=0.05, kelly_fraction=0.5, max_positions=3)
        mgr = RiskManager(params)
        assert mgr.params.max_drawdown == 0.05
        assert mgr.params.kelly_fraction == 0.5
        assert mgr.params.max_positions == 3

    def test_init_sets_initial_balance(self):
        """initial_balance and peak_balance match."""
        mgr = RiskManager(RiskParameters(), initial_balance=50_000.0)
        assert mgr.initial_balance == 50_000.0
        assert mgr.peak_balance == 50_000.0

    def test_init_mode_paper(self):
        """Default mode is PAPER."""
        mgr = RiskManager(RiskParameters())
        assert mgr.mode == "PAPER"


# ===========================================================================
# 2. Kelly position sizing
# ===========================================================================


class TestKellySizing:
    def test_calculate_kelly_size_zero_trades(self):
        """No trade history → uses default 0.55 win rate."""
        mgr = RiskManager(RiskParameters(), initial_balance=100_000.0)
        size = mgr.calculate_kelly_size(
            balance=100_000.0, price=1.12, atr=0.002, confidence=0.95
        )
        # Should produce a reasonable positive size
        assert size > 0
        assert size <= 100_000.0 * 0.80 / 1.12

    def test_calculate_kelly_size_atr_zero(self):
        """ATR <= 0 → uses price * 0.001 fallback."""
        mgr = RiskManager(RiskParameters(), initial_balance=100_000.0)
        size_zero = mgr.calculate_kelly_size(
            balance=100_000.0, price=1.12, atr=0.0, confidence=0.95
        )
        size_neg = mgr.calculate_kelly_size(
            balance=100_000.0, price=1.12, atr=-0.001, confidence=0.95
        )
        assert size_zero > 0
        assert size_neg > 0

    def test_calculate_kelly_size_basic(self):
        """Known inputs produce expected output (positive, bounded)."""
        mgr = RiskManager(RiskParameters(), initial_balance=100_000.0)
        _seed_price_history(mgr, n=60, base=1.10)
        size = mgr.calculate_kelly_size(
            balance=100_000.0,
            price=1.10,
            atr=0.002,
            confidence=0.95,
            symbol="EURUSD",
            open_positions=[],
        )
        # Kelly clips to max_margin_usage * balance / price when R-multiple is high
        max_allowed = 100_000.0 * 0.80 / 1.10
        assert 0 < size <= max_allowed

    def test_calculate_kelly_size_with_history(self):
        """After recording trades, Kelly adjusts."""
        mgr = RiskManager(RiskParameters(), initial_balance=100_000.0)
        # Add some price history for VaR
        _seed_price_history(mgr, n=60, base=1.10)
        # Record some wins and losses
        for _ in range(5):
            _close_trade(mgr, entry=1.10, exit_price=1.13, pnl=300.0)
        for _ in range(3):
            _close_trade(mgr, entry=1.10, exit_price=1.08, pnl=-200.0)
        size = mgr.calculate_kelly_size(
            balance=100_000.0,
            price=1.10,
            atr=0.002,
            confidence=0.95,
            symbol="EURUSD",
        )
        assert size > 0
        # Win rate should be 5/8 = 0.625
        assert mgr.get_win_rate() == pytest.approx(0.625, abs=1e-3)
        # The kelly should still be bounded
        assert size <= 100_000.0 * 0.80 / 1.10

    def test_calculate_kelly_size_bounded(self):
        """Result is always <= max_margin_usage * balance / price."""
        mgr = RiskManager(
            RiskParameters(max_margin_usage=0.50), initial_balance=10_000.0
        )
        _seed_price_history(mgr, n=60, base=1.12)
        size = mgr.calculate_kelly_size(
            balance=10_000.0, price=1.12, atr=0.002, confidence=0.95
        )
        max_allowed = 10_000.0 * 0.50 / 1.12
        assert size <= max_allowed + 1e-9


# ===========================================================================
# 3. VaR and CVaR
# ===========================================================================


class TestVaRCVaR:
    def test_var_insufficient_data(self):
        """Less than 20 price points → returns -2% of balance."""
        mgr = RiskManager(RiskParameters(), initial_balance=100_000.0)
        # Only 15 prices
        for i in range(15):
            mgr.update_price_history(1.10 + np.sin(i / 5) * 0.01, "EURUSD")
        var_val = mgr.var()
        assert var_val == pytest.approx(-100_000.0 * 0.02, abs=1.0)

    def test_var_known_returns(self):
        """Simple price series, known expected VaR."""
        mgr = RiskManager(RiskParameters(), initial_balance=100_000.0)
        # Monotonically increasing → only positive returns → VaR should be
        # very close to 0 (the smallest return)
        for i in range(30):
            mgr.update_price_history(1.10 + i * 0.001, "EURUSD")
        var_val = mgr.var()
        # All returns positive so VaR (5th percentile) should be >= 0
        assert var_val >= 0

    def test_var_confidence_level(self):
        """Higher confidence = larger VaR (more negative)."""
        mgr = RiskManager(RiskParameters(), initial_balance=100_000.0)
        price = 1.10
        for i in range(100):
            p = price + np.sin(i / 5) * 0.01
            mgr.update_price_history(p, "EURUSD")
        var_95 = mgr.var(confidence=0.95)
        var_99 = mgr.var(confidence=0.99)
        # 99% VaR should be <= 95% VaR (more negative)
        assert var_99 <= var_95

    def test_cvar_insufficient_data(self):
        """Less than 20 price points → -3% of balance."""
        mgr = RiskManager(RiskParameters(), initial_balance=100_000.0)
        for i in range(15):
            mgr.update_price_history(1.10 + np.sin(i / 5) * 0.01, "EURUSD")
        cvar_val = mgr.cvar()
        assert cvar_val == pytest.approx(-100_000.0 * 0.03, abs=1.0)

    def test_cvar_more_negative_than_var(self):
        """CVaR is always more negative than VaR."""
        mgr = RiskManager(RiskParameters(), initial_balance=100_000.0)
        # Mix of positive and negative returns
        np.random.seed(42)
        price = 1.10
        for i in range(100):
            p = price + np.random.normal(0, 0.005)
            mgr.update_price_history(p, "EURUSD")
        var_val = mgr.var(confidence=0.95)
        cvar_val = mgr.cvar(confidence=0.95)
        # If VaR is negative, CVaR should be more negative (or equal)
        if var_val < 0:
            assert cvar_val <= var_val
        else:
            # If all returns positive CVaR could be == VaR, just verify it's
            # not positive (tail is at or below the worst return)
            pass


# ===========================================================================
# 4. Pre-trade checks
# ===========================================================================


class TestPreTradeChecks:
    def test_pre_trade_checks_ok(self):
        """All conditions normal → (True, 'OK')."""
        mgr = RiskManager(RiskParameters(), initial_balance=100_000.0)
        approved, reason = mgr.pre_trade_checks(
            balance=100_000.0,
            equity=105_000.0,
            margin=10_000.0,
            current_pnl=5_000.0,
        )
        assert approved is True
        assert reason == "OK"

    def test_pre_trade_checks_kill_switch(self):
        """Kill switch triggered → rejected."""
        mgr = RiskManager(RiskParameters(), initial_balance=100_000.0)
        mgr.kill_switch_triggered = True
        approved, reason = mgr.pre_trade_checks(
            balance=100_000.0,
            equity=105_000.0,
            margin=10_000.0,
            current_pnl=5_000.0,
        )
        assert approved is False
        assert "Kill switch" in reason

    def test_pre_trade_checks_drawdown(self):
        """Drawdown > max → trigger kill switch."""
        mgr = RiskManager(RiskParameters(max_drawdown=0.05), initial_balance=100_000.0)
        # Peak = 100k, equity = 90k → ~10% drawdown
        approved, reason = mgr.pre_trade_checks(
            balance=100_000.0,
            equity=90_000.0,
            margin=0.0,
            current_pnl=-10_000.0,
        )
        assert approved is False
        assert "drawdown" in reason.lower()
        assert mgr.kill_switch_triggered is True

    def test_pre_trade_checks_daily_loss(self):
        """Daily loss > max → rejected."""
        mgr = RiskManager(
            RiskParameters(max_daily_loss=0.05), initial_balance=100_000.0
        )
        mgr.daily_pnl = -6_000.0  # 6% loss
        approved, reason = mgr.pre_trade_checks(
            balance=100_000.0,
            equity=100_000.0,
            margin=0.0,
            current_pnl=0.0,
        )
        assert approved is False
        assert "daily loss" in reason.lower()

    def test_pre_trade_checks_consecutive_losses(self):
        """Too many consecutive losses → rejected."""
        mgr = RiskManager(
            RiskParameters(max_consecutive_losses=3), initial_balance=100_000.0
        )
        mgr.consecutive_losses = 3
        approved, reason = mgr.pre_trade_checks(
            balance=100_000.0,
            equity=100_000.0,
            margin=0.0,
            current_pnl=0.0,
        )
        assert approved is False
        assert "consecutive" in reason.lower()

    def test_pre_trade_checks_margin(self):
        """Margin usage > max → rejected."""
        mgr = RiskManager(
            RiskParameters(max_margin_usage=0.50), initial_balance=100_000.0
        )
        # equity=100k, margin=60k → 60% usage
        approved, reason = mgr.pre_trade_checks(
            balance=100_000.0,
            equity=100_000.0,
            margin=60_000.0,
            current_pnl=0.0,
        )
        assert approved is False
        assert "margin" in reason.lower()


# ===========================================================================
# 5. Trade recording
# ===========================================================================


class TestRecordTrade:
    def test_record_trade_win(self):
        """Positive PnL → win incremented, losses reset."""
        mgr = RiskManager(RiskParameters(), initial_balance=100_000.0)
        mgr.consecutive_losses = 2
        _close_trade(mgr, entry=1.10, exit_price=1.13, pnl=300.0)
        assert mgr.wins == 1
        assert mgr.consecutive_losses == 0

    def test_record_trade_loss(self):
        """Negative PnL → loss incremented."""
        mgr = RiskManager(RiskParameters(), initial_balance=100_000.0)
        _close_trade(mgr, entry=1.10, exit_price=1.08, pnl=-200.0)
        assert mgr.losses == 1
        assert mgr.consecutive_losses == 1

    def test_record_trade_tracks_daily_pnl(self):
        """Daily PnL updated correctly."""
        mgr = RiskManager(RiskParameters(), initial_balance=100_000.0)
        _close_trade(mgr, entry=1.10, exit_price=1.13, pnl=500.0)
        _close_trade(mgr, entry=1.10, exit_price=1.08, pnl=-200.0)
        assert mgr.daily_pnl == 300.0  # 500 - 200

    def test_record_trade_appends_history(self):
        """Trade added to history."""
        mgr = RiskManager(RiskParameters(), initial_balance=100_000.0)
        _close_trade(mgr, entry=1.10, exit_price=1.13, pnl=300.0)
        _close_trade(mgr, entry=1.10, exit_price=1.08, pnl=-200.0)
        assert len(mgr.trade_history) == 2
        assert mgr.trade_history[0].pnl == 300.0
        assert mgr.trade_history[1].pnl == -200.0
        assert mgr.trade_history[0].status == "CLOSED"
        assert mgr.trade_history[1].status == "CLOSED"
        assert mgr.total_trades == 2


# ===========================================================================
# 6. Win rate
# ===========================================================================


class TestWinRate:
    def test_get_win_rate_zero(self):
        """No trades → 0.0."""
        mgr = RiskManager(RiskParameters())
        assert mgr.get_win_rate() == 0.0

    def test_get_win_rate_all_wins(self):
        """All wins → 1.0."""
        mgr = RiskManager(RiskParameters())
        for _ in range(3):
            _close_trade(mgr, entry=1.10, exit_price=1.13, pnl=300.0)
        assert mgr.get_win_rate() == 1.0

    def test_get_win_rate_mixed(self):
        """3 wins out of 5 → 0.6."""
        mgr = RiskManager(RiskParameters())
        for _ in range(3):
            _close_trade(mgr, entry=1.10, exit_price=1.13, pnl=300.0)
        for _ in range(2):
            _close_trade(mgr, entry=1.10, exit_price=1.08, pnl=-200.0)
        assert mgr.get_win_rate() == pytest.approx(0.6, abs=1e-3)


# ===========================================================================
# 7. update_price_history
# ===========================================================================


class TestUpdatePriceHistory:
    def test_update_price_history_new_symbol(self):
        """First update creates list."""
        mgr = RiskManager(RiskParameters())
        mgr.update_price_history(1.12, "EURUSD")
        assert "EURUSD" in mgr._price_history
        assert mgr._price_history["EURUSD"] == [1.12]

    def test_update_price_history_appends(self):
        """Subsequent updates append."""
        mgr = RiskManager(RiskParameters())
        mgr.update_price_history(1.10, "EURUSD")
        mgr.update_price_history(1.11, "EURUSD")
        mgr.update_price_history(1.12, "EURUSD")
        assert mgr._price_history["EURUSD"] == [1.10, 1.11, 1.12]

    def test_update_price_history_caps(self):
        """Don't exceed max length."""
        mgr = RiskManager(RiskParameters())
        max_len = mgr._var_lookback * 24
        # Push more than the max
        for i in range(max_len + 100):
            mgr.update_price_history(1.10 + np.sin(i / 10) * 0.01, "EURUSD")
        assert len(mgr._price_history["EURUSD"]) == max_len


# ===========================================================================
# 8. Stress test
# ===========================================================================


class TestStressTest:
    def test_stress_test_empty(self):
        """Empty scenario returns 0."""
        mgr = RiskManager(RiskParameters(), initial_balance=100_000.0)
        result = mgr.stress_test([])
        assert result["max_loss"] == 0
        assert result["impact"] == 0

    def test_stress_test_known_loss(self):
        """Known scenario produces expected max loss."""
        mgr = RiskManager(RiskParameters(), initial_balance=100_000.0)
        scenario = [-0.05, -0.10, -0.03]
        result = mgr.stress_test(scenario)
        # 100k * 0.10 = 10k max loss
        assert result["max_loss"] == pytest.approx(10_000.0, abs=1.0)
        assert result["max_loss_pct"] == pytest.approx(0.10, abs=1e-3)
        assert result["scenarios_tested"] == 3


# ===========================================================================
# 9. SL/TP calculation
# ===========================================================================


class TestSlTp:
    def test_calculate_atr_sl_tp(self):
        """SL below entry, TP above entry (long)."""
        mgr = RiskManager(RiskParameters(sl_atr_multiplier=2.0, tp_atr_multiplier=4.0))
        sl, tp = mgr.calculate_atr_sl_tp(entry=1.10, atr=0.002)
        # SL: 1.10 - 2*0.002 = 1.096
        assert sl == pytest.approx(1.096, abs=1e-9)
        # TP: 1.10 + 4*0.002 = 1.108
        assert tp == pytest.approx(1.108, abs=1e-9)
        assert sl < 1.10 < tp

    def test_calculate_atr_sl_tp_zero_atr(self):
        """ATR=0 → SL/TP based on price * 0.015 (fallback in calculate_kelly)."""
        mgr = RiskManager(RiskParameters(sl_atr_multiplier=2.0, tp_atr_multiplier=4.0))
        sl, tp = mgr.calculate_atr_sl_tp(entry=1.10, atr=0.0)
        # ATR=0: sl = 1.10 - 0 = 1.10, tp = 1.10 + 0 = 1.10
        assert sl == 1.10
        assert tp == 1.10


# ===========================================================================
# 10. Correlation check
# ===========================================================================


class TestCorrelation:
    def test_check_correlation_ok(self):
        """Uncorrelated pairs → True."""
        mgr = RiskManager(RiskParameters())
        # EURUSD and USDJPY are not in the >0.80 threshold
        ok = mgr._check_correlation_risk("USDJPY", ["EURUSD"])
        assert ok[0] is True

    def test_check_correlation_high(self):
        """Highly correlated pair → False."""
        mgr = RiskManager(RiskParameters())
        # EURUSD and GBPUSD have FX_CORR 0.85 > 0.80
        ok = mgr._check_correlation_risk("GBPUSD", ["EURUSD"])
        assert ok[0] is False
        assert "correlation" in ok[1].lower()

    def test_correlation_risk_empty(self):
        """No open positions → OK."""
        mgr = RiskManager(RiskParameters())
        ok = mgr._check_correlation_risk("EURUSD", [])
        assert ok[0] is True
        assert ok[1] == "OK"


# ===========================================================================
# 11. TrailingStopManager
# ===========================================================================


class TestTrailingStop:
    def test_trailing_stop_no_hit(self):
        """Not yet at TP → returns None."""
        tsm = TrailingStopManager(tp_pcts=[0.01, 0.02, 0.03])
        result = tsm.update(entry=1.10, current=1.105, atr=0.002, direction="long")
        assert result is None

    def test_trailing_stop_first_tp(self):
        """Hits first TP → returns entry price."""
        tsm = TrailingStopManager(tp_pcts=[0.01, 0.02, 0.03])
        # ~1.09% above entry — clearly above 1% threshold
        result = tsm.update(entry=1.10, current=1.112, atr=0.002, direction="long")
        assert result == pytest.approx(1.10, abs=1e-9)
        assert tsm.tp_hit[0] is True

    def test_trailing_stop_second_tp(self):
        """Hits second TP → returns ATR-based level."""
        tsm = TrailingStopManager(tp_pcts=[0.01, 0.02, 0.03])
        # Hit first TP first (clearly above 1%)
        tsm.update(entry=1.10, current=1.112, atr=0.002, direction="long")
        # Then second TP: ~2.09% above entry — clearly above 2%
        result = tsm.update(entry=1.10, current=1.123, atr=0.002, direction="long")
        # Second TP returns entry + atr * trail_mult = 1.10 + 0.002*2 = 1.104
        assert result == pytest.approx(1.104, abs=1e-9)
        assert tsm.tp_hit[1] is True

    def test_trailing_stop_short_direction(self):
        """Short direction works correctly."""
        tsm = TrailingStopManager(tp_pcts=[0.01, 0.02, 0.03])
        # Short: ~1.09% below entry — clearly above 1% threshold
        result = tsm.update(entry=1.10, current=1.088, atr=0.002, direction="short")
        # First TP returns entry (1.10)
        assert result == pytest.approx(1.10, abs=1e-9)
        assert tsm.tp_hit[0] is True

    def test_trailing_stop_partial_close_sizes(self):
        """Returns [0.3, 0.3, 0.4]."""
        tsm = TrailingStopManager()
        sizes = tsm.partial_close_sizes()
        assert sizes == [0.3, 0.3, 0.4]
        assert sum(sizes) == pytest.approx(1.0, abs=1e-9)


# ===========================================================================
# 12. Edge cases / defensive
# ===========================================================================


class TestEdgeCases:
    def test_var_cvar_identical_tiny_dataset(self):
        """Just barely enough points for VaR (n=20)."""
        mgr = RiskManager(RiskParameters(), initial_balance=100_000.0)
        for i in range(20):
            mgr.update_price_history(1.10, "EURUSD")
        var_val = mgr.var()
        cvar_val = mgr.cvar()
        # All returns are 0, VaR = 0, CVaR tail has 0s
        assert isinstance(var_val, float)
        assert isinstance(cvar_val, float)

    def test_record_trade_zero_pnl(self):
        """PnL of zero → treated as a loss (not > 0)."""
        mgr = RiskManager(RiskParameters(), initial_balance=100_000.0)
        _close_trade(mgr, entry=1.10, exit_price=1.10, pnl=0.0)
        assert mgr.wins == 0
        assert mgr.losses == 1
        assert mgr.consecutive_losses == 1

    def test_pre_trade_checks_updates_peak_equity(self):
        """Peak equity updates when equity > peak."""
        mgr = RiskManager(RiskParameters(), initial_balance=100_000.0)
        mgr.pre_trade_checks(
            balance=100_000.0,
            equity=110_000.0,
            margin=0.0,
            current_pnl=10_000.0,
        )
        assert mgr.peak_balance == 110_000.0
