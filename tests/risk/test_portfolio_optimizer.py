"""Tests for portfolio optimizer — mean-variance, risk parity, HRP."""

import numpy as np
import pytest

from risk.portfolio_optimizer import (
    PortfolioWeights,
    PortfolioOptimizer,
    compute_efficient_frontier,
    hrp_optimize,
    mean_variance_optimize,
    risk_parity_optimize,
)


@pytest.fixture
def sample_returns():
    """Four-asset return series for general testing."""
    np.random.seed(42)
    n_obs = 100
    return {
        "EURUSD": np.random.normal(0.0005, 0.01, n_obs),
        "GBPUSD": np.random.normal(0.0003, 0.008, n_obs),
        "USDJPY": np.random.normal(0.0004, 0.009, n_obs),
        "XAUUSD": np.random.normal(0.001, 0.015, n_obs),
    }


@pytest.fixture
def two_asset_returns():
    """Two uncorrelated assets with different variances for risk parity testing."""
    np.random.seed(123)
    n_obs = 5000
    return {
        "LOW_VOL": np.random.normal(0.001, 0.01, n_obs),
        "HIGH_VOL": np.random.normal(0.001, 0.02, n_obs),
    }


class TestMeanVariance:
    """Tests for mean_variance_optimize."""

    def test_mean_variance_returns_weights_sum_to_one(self, sample_returns):
        weights = mean_variance_optimize(sample_returns, risk_aversion=1.0)
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, expected 1.0"

    def test_mean_variance_no_assets_returns_empty(self):
        weights = mean_variance_optimize({}, risk_aversion=1.0)
        assert weights == {}


class TestRiskParity:
    """Tests for risk_parity_optimize."""

    def test_risk_parity_equal_risk_contribution(self, two_asset_returns):
        weights = risk_parity_optimize(two_asset_returns, max_iter=500, tol=1e-8)

        df = __import__("pandas").DataFrame(two_asset_returns)
        cov = df.cov().values
        cols = list(two_asset_returns.keys())
        w = np.array([weights[c] for c in cols])
        sigma = np.sqrt(w @ cov @ w)
        rc = w * (cov @ w) / sigma

        # Risk contributions should be roughly equal
        assert (
            abs(rc[0] - rc[1]) < 0.02
        ), f"Risk contributions differ: {rc[0]} vs {rc[1]}"


class TestHRP:
    """Tests for hrp_optimize."""

    def test_hrp_clustering_based_weights(self, sample_returns):
        weights = hrp_optimize(sample_returns)
        assert all(v >= 0 for v in weights.values()), "HRP weights must be non-negative"
        assert abs(sum(weights.values()) - 1.0) < 1e-6, "HRP weights must sum to 1"
        assert len(weights) == len(sample_returns)


class TestEfficientFrontier:
    """Tests for compute_efficient_frontier."""

    def test_efficient_frontier_returns_frontier(self, sample_returns):
        frontier = compute_efficient_frontier(sample_returns, n_points=20)
        assert len(frontier) == 20
        for point in frontier:
            assert "return" in point
            assert "volatility" in point
            assert "weights" in point

    def test_efficient_frontier_increasing_volatility(self, sample_returns):
        frontier = compute_efficient_frontier(sample_returns, n_points=20)
        vols = [p["volatility"] for p in frontier]
        # Each subsequent vol should be >= previous (within tolerance)
        for i in range(1, len(vols)):
            assert (
                vols[i] >= vols[i - 1] - 1e-10
            ), f"Volatility decreased at index {i}: {vols[i - 1]} -> {vols[i]}"


class TestPortfolioOptimizer:
    """Tests for PortfolioOptimizer class."""

    def test_portfolio_optimizer_full_pipeline(self, sample_returns):
        opt = PortfolioOptimizer()
        result = opt.optimize(sample_returns, method="hrp")
        assert isinstance(result, PortfolioWeights)
        # HRP should produce weights for all assets
        assert len(result.weights) == len(sample_returns)
        assert result.method == "hrp"
        # expected_return and expected_volatility should be populated
        assert isinstance(result.expected_return, float)
        assert isinstance(result.expected_volatility, float)

    def test_portfolio_optimizer_target_vol(self, sample_returns):
        opt = PortfolioOptimizer(target_volatility=0.20)
        result = opt.optimize(sample_returns, method="mean_variance")
        # Expected volatility should be close to target (after scaling)
        assert abs(result.expected_volatility - 0.20) < 0.01, (
            f"Expected vol {result.expected_volatility:.4f}, " f"target 0.20"
        )

    def test_portfolio_optimizer_all_methods(self, sample_returns):
        """Verify all three methods can run through the optimizer."""
        opt = PortfolioOptimizer()
        for method in ("mean_variance", "risk_parity", "hrp"):
            result = opt.optimize(sample_returns, method=method)
            assert result.method == method
            assert len(result.weights) == len(sample_returns)

    def test_portfolio_optimizer_invalid_method(self, sample_returns):
        opt = PortfolioOptimizer()
        with pytest.raises(ValueError):
            opt.optimize(sample_returns, method="invalid")

    def test_portfolio_optimizer_empty_returns(self):
        opt = PortfolioOptimizer()
        result = opt.optimize({}, method="hrp")
        assert result.weights == {}
        assert result.expected_return == 0.0
        assert result.expected_volatility == 0.0
