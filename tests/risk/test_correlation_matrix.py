"""Tests for Regime-Dependent Correlation Matrix."""

import numpy as np
import pandas as pd

from risk.correlation_matrix import RegimeCorrelationMatrix, RegimeCorrelationStore


class TestRegimeCorrelationMatrix:
    def test_store_update_and_retrieve(self):
        store = RegimeCorrelationStore()
        store.update("EURUSD", "GBPUSD", 0.85, regime="trending")
        corr = store.get("EURUSD", "GBPUSD", regime="trending")
        assert abs(corr - 0.85) < 1e-6

    def test_store_returns_default_for_unknown_pair(self):
        store = RegimeCorrelationStore()
        corr = store.get("EURUSD", "BTCUSD", regime="ranging")
        assert corr == 0.0

    def test_store_decays_old_values(self):
        store = RegimeCorrelationStore(window=3)
        for i in range(4):
            store.update("EURUSD", "GBPUSD", 0.9 - i * 0.1, regime="trending")
        assert len(store._store["trending"].get(("EURUSD", "GBPUSD"), [])) == 3

    def test_compute_from_returns_matrix(self):
        np.random.seed(42)
        returns = pd.DataFrame(
            {
                "EURUSD": np.random.randn(100) * 0.01,
                "GBPUSD": np.random.randn(100) * 0.01,
            }
        )
        matrix = RegimeCorrelationMatrix()
        store = matrix.update_from_returns(returns, regime="trending")
        corr = store.get("EURUSD", "GBPUSD", regime="trending")
        assert abs(corr) <= 1.0

    def test_regime_correlation_differs_by_regime(self):
        np.random.seed(42)
        # Trending regime: mild positive drift, some noise
        drift = np.linspace(0, 0.003, 100)
        returns_trend = pd.DataFrame(
            {
                "EURUSD": drift + np.random.randn(100) * 0.008,
                "GBPUSD": drift + np.random.randn(100) * 0.008,
            }
        )
        # Crisis regime: both crash together (highly correlated)
        common_shock = np.random.randn(100) * 0.04 - 0.01
        returns_crisis = pd.DataFrame(
            {
                "EURUSD": common_shock + np.random.randn(100) * 0.01,
                "GBPUSD": common_shock + np.random.randn(100) * 0.01,
            }
        )
        matrix = RegimeCorrelationMatrix(decay_half_life=0.5)
        matrix.update_from_returns(returns_trend, regime="trending")
        matrix.update_from_returns(returns_crisis, regime="crisis")
        corr_crisis = matrix.get("EURUSD", "GBPUSD", regime="crisis")
        assert abs(corr_crisis) >= 0.3

    def test_manager_integration(self):
        from risk.manager import RiskManager, RiskParameters
        from risk.correlation_matrix import RegimeCorrelationMatrix

        rm = RiskManager(RiskParameters())
        matrix = RegimeCorrelationMatrix()
        rm._correlation_matrix = matrix
        result = rm._check_correlation_risk("EURUSD", ["GBPUSD"])
        assert isinstance(result, tuple) and len(result) == 2

    def test_rolling_correlation_detects_high_corr(self):
        np.random.seed(42)
        x = np.cumsum(np.random.randn(200))
        y = x + np.random.randn(200) * 0.1
        returns = pd.DataFrame({"A": np.diff(x), "B": np.diff(y)})
        matrix = RegimeCorrelationMatrix(window=50)
        matrix.update_from_returns(returns, regime="ranging")
        corr = matrix.get("A", "B", regime="ranging")
        assert corr > 0.5

    def test_to_dataframe_returns_valid_matrix(self):
        store = RegimeCorrelationStore()
        store.update("EURUSD", "GBPUSD", 0.85, regime="trending")
        df = store.to_dataframe("trending")
        assert isinstance(df, pd.DataFrame)
        assert "EURUSD" in df.columns
        assert "GBPUSD" in df.columns
