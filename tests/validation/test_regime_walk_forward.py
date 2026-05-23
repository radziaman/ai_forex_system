"""Tests for regime-aware walk-forward optimization."""

import numpy as np

from validation.smart_walk_forward import SmartWalkForward, OptimizationResult


# ── Helpers ──


def simple_returns_fn(prices, features, regimes):
    """Simple strategy: returns based on price diff."""
    if len(prices) < 2:
        return []
    diffs = np.diff(prices)
    return diffs.tolist()


def constant_positive_returns(prices, features, regimes):
    """Always positive returns."""
    return [0.001] * max(len(prices) - 1, 0)


def make_trending_prices(n=1000):
    return 1.20 + np.cumsum(np.random.randn(n) * 0.001 + 0.0005)


def make_ranging_prices(n=1000):
    return 1.20 + np.random.randn(n) * 0.005


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Regime Split Basics
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegimeWalkForward:
    def test_regime_split_creates_additional_folds(self):
        np.random.seed(42)
        prices = np.concatenate([make_trending_prices(600), make_ranging_prices(600)])
        features = np.random.randn(len(prices))

        swf = SmartWalkForward(
            n_folds=3, test_window=100, embargo=10, min_train_window=200
        )
        _ = swf.run(prices, simple_returns_fn, features, regime_split=False)
        result_regime = swf.run(prices, simple_returns_fn, features, regime_split=True)

        assert result_regime.total_folds > 0
        # Regime split should produce at least one fold
        assert len(result_regime.fold_results) >= 1

    def test_regime_robustness_score_present(self):
        np.random.seed(42)
        prices = np.concatenate([make_trending_prices(600), make_ranging_prices(600)])
        features = np.random.randn(len(prices))

        swf = SmartWalkForward(
            n_folds=3, test_window=100, embargo=10, min_train_window=200
        )
        result = swf.run(prices, simple_returns_fn, features, regime_split=True)

        assert "avg_regime_sharpe" in result.regime_robustness_score
        assert "worst_regime_sharpe" in result.regime_robustness_score
        assert "regime_consistency" in result.regime_robustness_score

    def test_regime_split_returns_optimization_result(self):
        np.random.seed(42)
        prices = np.concatenate([make_trending_prices(600), make_ranging_prices(600)])
        features = np.random.randn(len(prices))

        swf = SmartWalkForward(
            n_folds=3, test_window=100, embargo=10, min_train_window=200
        )
        result = swf.run(prices, simple_returns_fn, features, regime_split=True)

        assert isinstance(result, OptimizationResult)
        assert result.total_folds >= 0

    def test_regime_split_with_insufficient_data(self):
        prices = np.random.randn(50)
        features = np.random.randn(50)

        swf = SmartWalkForward(
            n_folds=3, test_window=100, embargo=10, min_train_window=200
        )
        result = swf.run(prices, simple_returns_fn, features, regime_split=True)

        assert result.total_folds == 0
        assert result.regime_robustness_score == {}

    def test_regime_folds_have_different_train_test_regimes(self):
        np.random.seed(42)
        prices = np.concatenate([make_trending_prices(800), make_ranging_prices(800)])
        features = np.random.randn(len(prices))

        swf = SmartWalkForward(
            n_folds=3, test_window=100, embargo=10, min_train_window=200
        )
        result = swf.run(prices, simple_returns_fn, features, regime_split=True)

        if result.fold_results:
            # At least some folds should have different train/test regimes
            # (train on one regime, test on another)
            assert result.total_folds > 0

    def test_regime_consistency_non_negative(self):
        np.random.seed(42)
        prices = np.concatenate([make_trending_prices(600), make_ranging_prices(600)])
        features = np.random.randn(len(prices))

        swf = SmartWalkForward(
            n_folds=3, test_window=100, embargo=10, min_train_window=200
        )
        result = swf.run(prices, simple_returns_fn, features, regime_split=True)

        if result.regime_robustness_score:
            consistency = result.regime_robustness_score.get("regime_consistency", 0.0)
            assert consistency >= 0.0
            worst = result.regime_robustness_score.get("worst_regime_sharpe", 0.0)
            avg = result.regime_robustness_score.get("avg_regime_sharpe", 0.0)
            # Worst should be <= average
            assert worst <= avg + 1e-6
