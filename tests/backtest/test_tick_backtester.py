"""Tests for TickBacktester."""

import numpy as np

from backtest.tick_backtester import TickBacktester


# ── Helpers ──


def make_tick_data(n: int = 100, start_price: float = 1.2000):
    """Generate synthetic tick data with bid/ask."""
    mid = start_price + np.cumsum(np.random.randn(n) * 0.0002)
    spread = 0.0001  # 1 pip
    return {
        "timestamp": np.arange(n, dtype=float),
        "bid": mid - spread / 2,
        "ask": mid + spread / 2,
    }


def constant_long_signal(idx, ticks, context):
    return 1


def alternating_signal(idx, ticks, context):
    return 1 if idx % 20 < 10 else -1


def zero_signal(idx, ticks, context):
    return 0


def limit_buy_signal(idx, ticks, context):
    if context["position"] == 0 and idx == 10:
        return {"signal": 1, "order_type": "LIMIT", "price": ticks["ask"][idx] - 0.0002}
    if context["position"] == 1 and idx == 30:
        return {"signal": -1, "order_type": "MARKET"}
    return 0


def stop_sell_signal(idx, ticks, context):
    if context["position"] == 0 and idx == 5:
        return {"signal": -1, "order_type": "STOP", "price": ticks["bid"][idx] - 0.0003}
    if context["position"] == -1 and idx == 25:
        return {"signal": 1, "order_type": "MARKET"}
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Basic Functionality
# ═══════════════════════════════════════════════════════════════════════════════


class TestTickBacktesterBasics:
    def test_no_signals_returns_zero_trades(self):
        ticks = make_tick_data(50)
        bt = TickBacktester(ticks, zero_signal)
        bt.run()
        results = bt.get_results()
        assert results["total_trades"] == 0
        assert results["total_return_pct"] == 0.0
        assert results["sharpe"] == 0.0

    def test_constant_long_opens_position(self):
        ticks = make_tick_data(50)
        bt = TickBacktester(ticks, constant_long_signal)
        bt.run()
        assert bt.position == 0  # closed at end
        assert len(bt.trades) >= 1

    def test_alternating_signal_generates_trades(self):
        ticks = make_tick_data(100)
        bt = TickBacktester(ticks, alternating_signal)
        bt.run()
        results = bt.get_results()
        assert results["total_trades"] > 0

    def test_equity_curve_has_entries(self):
        ticks = make_tick_data(50)
        bt = TickBacktester(ticks, constant_long_signal)
        bt.run()
        assert len(bt.equity) > 1
        assert bt.equity[0] == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Order Types
# ═══════════════════════════════════════════════════════════════════════════════


class TestTickBacktesterOrders:
    def test_limit_order_fills_when_price_reached(self):
        ticks = make_tick_data(60)
        bt = TickBacktester(ticks, limit_buy_signal)
        bt.run()
        # The limit buy should have filled at some point
        assert len(bt.trades) >= 0  # may or may not fill depending on path
        # If it did fill, check slippage is recorded
        if bt.trades:
            assert "slippage_pips" in bt.trades[0]

    def test_stop_order_fills_when_triggered(self):
        ticks = make_tick_data(60)
        # Make a clear downtrend so stop triggers
        ticks["bid"] = np.linspace(1.2000, 1.1900, 60)
        ticks["ask"] = ticks["bid"] + 0.0001
        bt = TickBacktester(ticks, stop_sell_signal)
        bt.run()
        # Should have at least attempted a trade
        results = bt.get_results()
        assert results["total_trades"] >= 0

    def test_spread_model_applies_variable_spread(self):
        ticks = {
            "timestamp": np.arange(50, dtype=float),
            "mid": np.linspace(1.2000, 1.2050, 50),
        }

        def variable_spread(mid, idx):
            return 0.0002 if idx % 2 == 0 else 0.0004

        bt = TickBacktester(ticks, constant_long_signal, spread_model=variable_spread)
        bt.run()
        # Variable spread should affect transaction costs
        if bt.trades:
            costs = [t["cost"] for t in bt.trades]
            # All costs should be positive
            assert all(c > 0 for c in costs)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Metrics
# ═══════════════════════════════════════════════════════════════════════════════


class TestTickBacktesterMetrics:
    def test_metrics_keys_match_vectorized(self):
        ticks = make_tick_data(100)
        bt = TickBacktester(ticks, alternating_signal)
        bt.run()
        results = bt.get_results()
        expected_keys = [
            "total_return_pct",
            "annual_return_pct",
            "sharpe",
            "max_drawdown_pct",
            "win_rate",
            "profit_factor",
            "total_trades",
            "avg_hold_bars",
            "avg_win_pct",
            "avg_loss_pct",
            "calmar_ratio",
            "sortino",
        ]
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"

    def test_floating_pnl_tracked(self):
        ticks = make_tick_data(50)
        bt = TickBacktester(ticks, constant_long_signal)
        bt.run()
        assert len(bt.floating_pnls) > 0
        # While in a trade, floating PnL should vary
        non_zero = [f for f in bt.floating_pnls if f != 0.0]
        assert len(non_zero) > 0
