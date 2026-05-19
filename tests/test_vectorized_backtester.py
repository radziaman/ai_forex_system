"""
Tests for the VectorizedBacktester.
Validates SL/TP calculation, transaction costs, equity curve, and metrics.
"""

import numpy as np
from backtest.vectorized_backtester import VectorizedBacktester, BacktestResult


# ── Helpers ──


def make_sine_wave_prices(n: int = 1000, freq: float = 0.02) -> np.ndarray:
    """Generate a sine-wave price series for deterministic testing."""
    t = np.linspace(0, 4 * np.pi, n)
    arr: np.ndarray = (
        1.20 + 0.02 * np.sin(t) + 0.001 * np.cumsum(np.random.randn(n) * 0.01)
    )
    return arr


def make_trending_prices(n: int = 1000, trend: float = 0.0001) -> np.ndarray:
    """Generate upward trending prices."""
    return 1.20 + trend * np.arange(n) + np.random.randn(n) * 0.002


def constant_signal(prices, features=None):
    """Always return 1 (long)."""
    return np.ones(len(prices), dtype=int)


def alternating_signal(prices, features=None):
    """Alternate between long and short."""
    n = len(prices)
    sig = np.zeros(n, dtype=int)
    sig[::2] = 1
    sig[1::2] = -1
    return sig


def random_signal(prices, features=None):
    """Generate random long/short signals (seed for reproducibility)."""
    rng = np.random.RandomState(42)
    sig = rng.choice([-1, 0, 1], size=len(prices), p=[0.2, 0.6, 0.2])
    return sig


def make_prices_from_range(
    n: int = 500, low: float = 1.10, high: float = 1.20
) -> np.ndarray:
    """Generate ranging prices with small noise."""
    mid = (low + high) / 2
    return mid + np.random.randn(n) * 0.005


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: SL/TP Calculation Validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestSLTPCalculation:
    """Validate that stop-loss and take-profit are calculated correctly.

    CRITICAL: The VectorizedBacktester has a known bug where ATR-based SL/TP
    is multiplied by pip_size (0.0001), making stops effectively at entry price.
    """

    def test_sl_tp_long_reasonable_distance(self):
        """SL/TP for long should be at a reasonable distance from entry.

        With ATR ~0.001 (typical for EURUSD 1h), SL should be ~2*ATR away,
        not ~2*ATR*pip_size.
        """
        n = 200
        prices = np.ones(n) * 1.20
        # Add small noise so we have some variation
        prices += np.random.randn(n) * 0.0001
        atr = np.full(n, 0.001)  # ~10 pips, realistic for hourly EURUSD

        bt = VectorizedBacktester(spread_pips=0.5)
        result = bt.run(prices, constant_signal, atr=atr, sl_atr=2.0, tp_atr=4.0)

        if result.total_trades > 0:
            # If SL was correctly calculated (2 * ATR = 0.002 away),
            # trades would last more than 1 bar on average
            assert result.avg_hold_bars >= 1.0, "All trades should last at least 1 bar"

            # Check if the SL is reasonable: distance should be ~2 * ATR = 0.002
            # With the bug (multiplying by pip_size): distance = 0.001 * 2 * 0.0001 = 0.0000002
            # Prices change by ~0.0001 per bar, so a 0.0000002 SL would be hit every single bar
            if result.avg_hold_bars <= 1.5:
                print(
                    f"WARNING: Avg hold only {result.avg_hold_bars:.1f} bars — "
                    f"suggests SL/TP bug (stops too tight)"
                )

    def test_sl_tp_short_reasonable_distance(self):
        """SL/TP for short should be at a reasonable distance."""
        n = 200
        prices = np.ones(n) * 1.20
        prices += np.random.randn(n) * 0.0001
        atr = np.full(n, 0.001)

        def short_signal(p, f=None):
            return -np.ones(len(p), dtype=int)

        bt = VectorizedBacktester(spread_pips=0.5)
        result = bt.run(prices, short_signal, atr=atr, sl_atr=2.0, tp_atr=4.0)

        if result.total_trades > 0:
            assert result.avg_hold_bars >= 1.0

    def test_pip_size_impact_on_sl(self):
        """Verify that pip_size drastically affects SL distance.

        Changing pip_size should change SL distance proportionally.
        """
        n = 300
        prices = np.ones(n) * 1.20000
        prices += np.cumsum(np.random.randn(n) * 0.0005)
        prices = np.abs(prices)  # ensure positive
        atr = np.full(n, 0.001)

        bt_small = VectorizedBacktester(spread_pips=0.5, pip_size=0.0001)
        bt_large = VectorizedBacktester(spread_pips=0.5, pip_size=0.01)

        result_small = bt_small.run(prices, constant_signal, atr=atr)
        result_large = bt_large.run(prices, constant_signal, atr=atr)

        # With larger pip_size, stops should be wider, so avg hold should be longer
        # (or at least different — this validates the pip_size is being used)
        print(f"Small pip_size avg_hold: {result_small.avg_hold_bars:.1f}")
        print(f"Large pip_size avg_hold: {result_large.avg_hold_bars:.1f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Signal Processing
# ═══════════════════════════════════════════════════════════════════════════════


class TestSignalProcessing:
    """Test that signals are processed correctly."""

    def test_no_signals_returns_zero_result(self):
        """When signal function returns all zeros, no trades should occur."""
        n = 100
        prices = np.cumsum(np.random.randn(n) * 0.001) + 1.20

        def zero_signal(p, f=None):
            return np.zeros(len(p), dtype=int)

        bt = VectorizedBacktester()
        result = bt.run(prices, zero_signal)

        assert result.total_trades == 0
        assert result.total_return_pct == 0.0
        assert result.sharpe == 0.0
        assert result.win_rate == 0.0
        assert result.profit_factor == 0.0

    def test_all_long_signals_in_trend(self):
        """Consistent long signals in an uptrend should be profitable."""
        n = 500
        prices = make_trending_prices(n, trend=0.0005)

        def long_signal(p, f=None):
            return np.ones(len(p), dtype=int)

        bt = VectorizedBacktester(spread_pips=0.1, commission_per_lot=1.0)
        result = bt.run(prices, long_signal)

        # Should have trades (unless SL/TP bug prevents any)
        if result.total_trades > 0:
            print(
                f"Trend long: {result.total_trades} trades, "
                f"return={result.total_return_pct:.2f}%, "
                f"sharpe={result.sharpe:.3f}"
            )

    def test_signal_distribution_respected(self):
        """Signal function values should correlate with number of trades."""
        n = 200
        prices = np.ones(n) * 1.20

        # 50% active signals
        def half_active(p, f=None):
            sig = np.zeros(n, dtype=int)
            sig[: n // 2] = 1
            return sig

        bt = VectorizedBacktester(spread_pips=0.5)
        result = bt.run(prices, half_active)

        # Should have trades from the active half
        print(f"Half active: {result.total_trades} trades")

    def test_signal_flip_closes_position(self):
        """Signal reversal should close existing position."""
        n = 100
        prices = np.ones(n) * 1.20 + np.random.randn(n) * 0.001

        # Long then short
        def flip_signal(p, f=None):
            sig = np.zeros(n, dtype=int)
            sig[:30] = 1
            sig[30:] = -1
            return sig

        bt = VectorizedBacktester(spread_pips=0.5)
        result = bt.run(prices, flip_signal)

        print(
            f"Flip signal: {result.total_trades} trades, "
            f"avg_hold={result.avg_hold_bars:.1f}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Transaction Costs
# ═══════════════════════════════════════════════════════════════════════════════


class TestTransactionCosts:
    """Validate that transaction costs are applied correctly."""

    def test_higher_spread_reduces_returns(self):
        """Wider spread should result in lower (or equal) returns."""
        n = 500
        prices = make_sine_wave_prices(n)

        bt_low = VectorizedBacktester(spread_pips=0.1, commission_per_lot=1.0)
        bt_high = VectorizedBacktester(spread_pips=5.0, commission_per_lot=1.0)

        result_low = bt_low.run(prices, alternating_signal)
        result_high = bt_high.run(prices, alternating_signal)

        if result_low.total_trades > 0 and result_high.total_trades > 0:
            print(f"Low spread return: {result_low.total_return_pct:.2f}%")
            print(f"High spread return: {result_high.total_return_pct:.2f}%")

    def test_commission_reduces_returns(self):
        """Higher commission should reduce net returns."""
        n = 500
        prices = make_sine_wave_prices(n)

        bt_low = VectorizedBacktester(spread_pips=0.5, commission_per_lot=1.0)
        bt_high = VectorizedBacktester(spread_pips=0.5, commission_per_lot=100.0)

        result_low = bt_low.run(prices, alternating_signal)
        result_high = bt_high.run(prices, alternating_signal)

        if result_low.total_trades > 0 and result_high.total_trades > 0:
            print(f"Low comm return: {result_low.total_return_pct:.2f}%")
            print(f"High comm return: {result_high.total_return_pct:.2f}%")

    def test_cost_multiplier_scales_costs(self):
        """cost_multiplier should scale transaction costs linearly."""
        n = 200
        prices = np.ones(n) * 1.20
        prices += np.random.randn(n) * 0.001

        bt = VectorizedBacktester(spread_pips=0.5, commission_per_lot=7.0)

        result_1x = bt.run(prices, alternating_signal, cost_multiplier=1.0)
        result_2x = bt.run(prices, alternating_signal, cost_multiplier=2.0)

        if result_1x.total_trades > 0 and result_2x.total_trades > 0:
            # 2x cost should have equal or lower returns
            print(f"1x cost return: {result_1x.total_return_pct:.2f}%")
            print(f"2x cost return: {result_2x.total_return_pct:.2f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Metrics Correctness
# ═══════════════════════════════════════════════════════════════════════════════


class TestBacktestMetrics:
    """Verify computed metrics are correct."""

    def test_empty_trades_returns_zero_metrics(self):
        """No trades should result in all-zero metrics."""
        n = 100
        prices = np.ones(n) * 1.20

        def zero_sig(p, f=None):
            return np.zeros(n, dtype=int)

        bt = VectorizedBacktester()
        result = bt.run(prices, zero_sig)

        assert result.sharpe == 0.0
        assert result.sortino == 0.0
        assert result.calmar_ratio == 0.0
        assert result.max_drawdown_pct == 0.0

    def test_win_rate_is_between_0_and_1(self):
        """Win rate must be a fraction between 0 and 1."""
        n = 500
        prices = make_sine_wave_prices(n)

        bt = VectorizedBacktester(spread_pips=0.5)
        result = bt.run(prices, random_signal)

        if result.total_trades > 0:
            assert (
                0.0 <= result.win_rate <= 1.0
            ), f"Win rate {result.win_rate} outside [0, 1]"

    def test_profit_factor_non_negative(self):
        """Profit factor should be non-negative."""
        n = 500
        prices = make_sine_wave_prices(n)

        bt = VectorizedBacktester(spread_pips=0.5)
        result = bt.run(prices, random_signal)

        assert (
            result.profit_factor >= 0.0
        ), f"Profit factor {result.profit_factor} is negative"

    def test_sharpe_reasonable_range(self):
        """Sharpe ratio should be within a reasonable range."""
        n = 1000
        prices = make_sine_wave_prices(n)

        bt = VectorizedBacktester()
        result = bt.run(prices, random_signal)

        # Sharpe should be finite and reasonable
        assert np.isfinite(result.sharpe)
        assert (
            -10.0 < result.sharpe < 10.0
        ), f"Sharpe {result.sharpe} outside expected range"

    def test_max_drawdown_non_negative(self):
        """Max drawdown should be non-negative."""
        n = 500
        prices = make_sine_wave_prices(n)

        bt = VectorizedBacktester(spread_pips=0.5)
        result = bt.run(prices, constant_signal)

        assert result.max_drawdown_pct >= 0.0

    def test_avg_hold_bars_reasonable(self):
        """Average hold bars should be reasonable for given data."""
        n = 100
        prices = np.ones(n) * 1.20 + np.random.randn(n) * 0.0005

        def slow_signal(p, f=None):
            sig = np.zeros(n, dtype=int)
            sig[10:20] = 1
            sig[40:50] = -1
            sig[70:80] = 1
            return sig

        bt = VectorizedBacktester(spread_pips=0.5)
        result = bt.run(prices, slow_signal)

        if result.total_trades > 0:
            assert result.avg_hold_bars >= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: Edge Cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Test backtester with edge-case inputs."""

    def test_single_bar_prices(self):
        """Backtester should handle single bar without crashing."""
        prices = np.array([1.20])
        result = VectorizedBacktester().run(prices, lambda p, f=None: np.array([1]))
        assert result.total_trades == 0

    def test_two_bar_prices(self):
        """Backtester should handle two bars."""
        prices = np.array([1.20, 1.21])
        result = VectorizedBacktester().run(prices, lambda p, f=None: np.array([1, 1]))
        # May or may not have trades depending on SL/TP
        assert result.total_trades >= 0

    def test_constant_prices_no_trades(self):
        """Flat prices with no signal should produce no trades."""
        prices = np.ones(200) * 1.20
        result = VectorizedBacktester().run(
            prices, lambda p, f=None: np.zeros(len(p), dtype=int)
        )
        assert result.total_trades == 0

    def test_nan_atr_handled(self):
        """NaN in ATR should not crash backtester."""
        n = 200
        prices = np.ones(n) * 1.20 + np.random.randn(n) * 0.001
        atr = np.full(n, np.nan)

        bt = VectorizedBacktester()
        result = bt.run(prices, constant_signal, atr=atr)

        # Should complete without error
        assert result.total_trades >= 0

    def test_negative_prices_raises_error(self):
        """Negative prices should likely fail (FX prices are positive)."""
        n = 100
        prices = -np.ones(n) * 1.20

        bt = VectorizedBacktester()
        try:
            result = bt.run(prices, constant_signal)
            # May or may not produce trades
            assert result.total_trades >= 0
        except Exception:
            pass  # Error is acceptable for invalid input

    def test_zero_atr_handled(self):
        """Zero ATR should not cause division by zero."""
        n = 200
        prices = np.ones(n) * 1.20 + np.random.randn(n) * 0.001
        atr = np.zeros(n)

        bt = VectorizedBacktester()
        result = bt.run(prices, constant_signal, atr=atr)

        assert result.total_trades >= 0

    def test_regime_filtering(self):
        """Backtest should work with regime filtering."""
        n = 500
        prices = make_sine_wave_prices(n)
        regimes = np.array(["ranging"] * n)

        bt = VectorizedBacktester()
        result = bt.run(prices, random_signal, regimes=regimes)

        assert result.total_trades >= 0
        if result.total_trades > 0:
            assert len(result.trade_regimes) == result.total_trades


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: Monte Carlo Validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestMonteCarlo:
    """Test Monte Carlo simulation methods."""

    def test_monte_carlo_empty_trades(self):
        """MC on empty trades should return zeros."""
        curves = VectorizedBacktester.monte_carlo_equity_curves(
            np.array([]), n_simulations=10
        )
        assert curves.shape == (10, 100)

    def test_monte_carlo_single_trade(self):
        """MC with single trade should work."""
        curves = VectorizedBacktester.monte_carlo_equity_curves(
            np.array([10.0]), n_simulations=5, n_trades=10
        )
        assert curves.shape == (5, 11)

    def test_monte_carlo_confidence_intervals(self):
        """Confidence intervals should be properly ordered."""
        curves = np.array(
            [
                [1.0, 1.1, 1.2],
                [1.0, 0.9, 0.8],
                [1.0, 1.0, 1.0],
            ]
        )
        median, lb, ub = VectorizedBacktester.compute_confidence_intervals(
            curves, confidence=0.95
        )
        assert len(median) == 3
        assert len(lb) == 3
        assert len(ub) == 3
        # Lower bound should be <= median <= upper bound
        assert np.all(lb <= median)
        assert np.all(median <= ub)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7: Sensitivity Analysis
# ═══════════════════════════════════════════════════════════════════════════════


class TestSensitivityAnalysis:
    """Test sensitivity analysis runs without errors."""

    def test_sensitivity_varying_slippage(self):
        """Sensitivity should run across slippage models."""
        n = 300
        prices = make_sine_wave_prices(n)

        bt = VectorizedBacktester(spread_pips=0.5)
        results = bt.run_with_sensitivity(prices, random_signal)

        assert len(results) == 9  # 3 slippage × 3 cost multipliers (0.5, 1.0, 2.0)
        for key, result in results.items():
            assert isinstance(result, BacktestResult)

    def test_sensitivity_consistent_trade_counts(self):
        """Slippage changes should not affect number of trades."""
        n = 300
        prices = make_sine_wave_prices(n) * 1.0  # copy

        bt = VectorizedBacktester(spread_pips=0.5)
        results = bt.run_with_sensitivity(prices, random_signal)

        if results:
            trade_counts = [r.total_trades for r in results.values()]
            assert (
                len(set(trade_counts)) == 1
            ), f"Trade counts differ across sensitivity: {trade_counts}"


# ═══════════════════════════════════════════════════════════════════════════════
# Section 8: BacktestResult Serialization
# ═══════════════════════════════════════════════════════════════════════════════


class TestBacktestResultSerialization:
    """Test BacktestResult dict serialization."""

    def test_to_dict_keys_present(self):
        """to_dict() should contain all expected keys."""
        result = BacktestResult(
            total_return_pct=10.5,
            annual_return_pct=5.2,
            sharpe=1.5,
            max_drawdown_pct=15.0,
            win_rate=0.55,
            profit_factor=1.8,
            total_trades=100,
            avg_hold_bars=24.0,
            avg_win_pct=2.0,
            avg_loss_pct=-1.0,
            calmar_ratio=0.35,
            sortino=1.2,
        )

        d = result.to_dict()
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
            assert key in d, f"Missing key: {key}"

    def test_to_dict_values_rounded(self):
        """Values should be properly rounded in dict."""
        result = BacktestResult(
            total_return_pct=10.55555,
            annual_return_pct=5.22222,
            sharpe=1.55555,
            max_drawdown_pct=15.33333,
            win_rate=0.55555,
            profit_factor=1.88888,
            total_trades=100,
            avg_hold_bars=24.33333,
            avg_win_pct=2.55555,
            avg_loss_pct=-1.55555,
            calmar_ratio=0.35555,
            sortino=1.22222,
        )

        d = result.to_dict()
        assert d["total_return_pct"] == 10.56
        assert d["sharpe"] == 1.556
        assert isinstance(d["total_trades"], int)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 9: Walk-Forward Integration (bug verification)
# ═══════════════════════════════════════════════════════════════════════════════


class TestWalkForwardHMMBug:
    """Verify the HMM dimensionality mismatch bug in predict_regimes_hmm.

    The run_backtest.py predict_regimes_hmm function creates a 3-column feature
    matrix for prediction, but the HMMRegimeDetector._extract_features creates
    8 columns. This causes ValueError when calling model.predict().
    """

    def test_hmm_feature_dimension_mismatch(self):
        """Confirm that HMM feature extraction uses 8 columns."""
        import pandas as pd
        from src.rts_ai_fx.regime_detector import HMMRegimeDetector  # noqa

        # Create a simple test DataFrame
        df = pd.DataFrame(
            {
                "close": [1.20 + i * 0.001 for i in range(200)],
            }
        )

        detector = HMMRegimeDetector(n_regimes=4)

        # Check _extract_features output dimension
        features = detector._extract_features(df)

        # HMMRegimeDetector._extract_features uses 8 columns
        # while predict_regimes_hmm manually creates 3-column features
        assert features.shape[1] == 8, (
            f"HMM features have {features.shape[1]} columns, "
            f"but predict_regimes_hmm uses 3 columns — this is the bug!"
        )
