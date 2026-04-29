"""Tests for feature engineering module"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_forex_system.features import FeatureEngineer

from ai_forex_system.features import FeatureEngineer


class TestFeatureEngineer:
    """Test feature engineering functionality"""

    @pytest.fixture  # noqa: ANN001
    def sample_data(self):
        """Create sample OHLCV data"""
        dates = pd.date_range("2025-01-01", periods=100, freq="1h")
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "open": 1.1000 + np.random.randn(100) * 0.01,
                "high": 1.1050 + np.random.randn(100) * 0.01,
                "low": 1.0950 + np.random.randn(100) * 0.01,
                "close": 1.1000 + np.random.randn(100) * 0.01,
                "volume": np.random.randint(100, 1000, 100),
            },
            index=dates,
        )
        return data

    def test_price_features(self, sample_data):
        """Test price feature generation"""
        fe = FeatureEngineer()
        result = fe.add_price_features(sample_data.copy())

        assert "body_size" in result.columns
        assert "upper_shadow" in result.columns
        assert "lower_shadow" in result.columns
        assert "range" in result.columns
        assert not result["body_size"].isnull().all()

    def test_momentum_indicators(self, sample_data):
        """Test momentum indicator calculation"""
        fe = FeatureEngineer()
        result = fe.add_momentum_indicators(sample_data.copy())

        assert "rsi_14" in result.columns
        assert "macd" in result.columns
        assert "momentum_3" in result.columns
        # Drop NaN values (first 14 rows for RSI)
        valid_rsi = result["rsi_14"].dropna()
        assert len(valid_rsi) > 0
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()

    def test_volatility_features(self, sample_data):
        """Test volatility feature generation"""
        fe = FeatureEngineer()
        result = fe.add_volatility_features(sample_data.copy())

        assert "atr_14" in result.columns
        assert "yang_zhang_vol" in result.columns
        assert not result["atr_14"].isnull().all()

    def test_generate_all_features(self, sample_data):
        """Test complete feature generation"""
        fe = FeatureEngineer()
        result = fe.generate_all_features(sample_data.copy())

        feature_count = len(result.columns)
        assert feature_count >= 20, f"Expected 20+ features, got {feature_count}"
        assert not result.empty
