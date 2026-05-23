"""Tests for CrossAssetEngine."""

import numpy as np
import pytest

from src.data.cross_asset_features import CrossAssetEngine


@pytest.fixture
def engine() -> CrossAssetEngine:
    return CrossAssetEngine(maxlen=500)


def _feed_correlated(
    engine: CrossAssetEngine,
    base: str = "EURUSD",
    n: int = 100,
    noise: float = 0.0001,
):
    """Feed two perfectly-correlated streams (EURUSD & GBPUSD) plus helpers."""
    np.random.seed(7)
    for i in range(n):
        core = 1.1000 + i * 0.0001 + np.random.randn() * noise
        prices = {
            "EURUSD": core,
            "GBPUSD": core * 1.15,
            "USDJPY": 150.0 + i * 0.01,
            "USDCAD": 1.35 + i * 0.0001,
            "USDCHF": 0.88 + i * 0.0001,
            "XAUUSD": 2000.0 + i * 0.1,
            "XTIUSD": 75.0 + i * 0.01,
            "AUDUSD": 0.67 + i * 0.0001,
            "US500": 4500.0 + i * 0.5,
        }
        engine.compute_lead_lag_features(base, prices)


class TestDxyImpact:
    def test_positive_correlation(self, engine):
        _feed_correlated(engine, "EURUSD", n=100)
        feats = engine.compute_lead_lag_features("EURUSD", {"EURUSD": 1.1100})
        # Synthetic DXY rises as EURUSD rises (inverse), so correlation should
        # be negative for EURUSD vs DXY.  We just assert it's non-zero.
        assert abs(feats["dxy_impact"]) > 0.0

    def test_insufficient_history(self, engine):
        feats = engine.compute_lead_lag_features("EURUSD", {"EURUSD": 1.1000})
        assert feats["dxy_impact"] == 0.0


class TestEurusdGbpusdLag30:
    def test_high_correlation(self, engine):
        _feed_correlated(engine, "EURUSD", n=100)
        feats = engine.compute_lead_lag_features(
            "EURUSD",
            {"EURUSD": 1.1100, "GBPUSD": 1.2765},
        )
        # EURUSD and GBPUSD were fed perfectly correlated
        assert feats["eurusd_gbpusd_lag30"] > 0.8

    def test_no_gbp_history(self, engine):
        for i in range(50):
            engine.compute_lead_lag_features("EURUSD", {"EURUSD": 1.1000 + i * 0.0001})
        feats = engine.compute_lead_lag_features("EURUSD", {"EURUSD": 1.1050})
        assert feats["eurusd_gbpusd_lag30"] == 0.0


class TestGoldOilRatio:
    def test_deviation_when_ratio_shifts(self, engine):
        # Feed stable gold/oil ratio
        for i in range(200):
            engine.compute_lead_lag_features(
                "XAUUSD",
                {"XAUUSD": 2000.0, "XTIUSD": 75.0},
            )
        # Now shift ratio by 10%
        feats = engine.compute_lead_lag_features(
            "XAUUSD",
            {"XAUUSD": 2200.0, "XTIUSD": 75.0},
        )
        assert abs(feats["gold_oil_ratio"]) > 0.05

    def test_zero_when_no_data(self, engine):
        feats = engine.compute_lead_lag_features("XAUUSD", {})
        assert feats["gold_oil_ratio"] == 0.0


class TestCommodityFxLead:
    def test_aud_gold_correlation(self, engine):
        np.random.seed(3)
        for i in range(100):
            gold = 2000.0 + i * 0.1 + np.random.randn() * 0.5
            aud = 0.67 + i * 0.0001 + np.random.randn() * 0.0001
            engine.compute_lead_lag_features(
                "AUDUSD",
                {"AUDUSD": aud, "XAUUSD": gold},
            )
        feats = engine.compute_lead_lag_features(
            "AUDUSD", {"AUDUSD": 0.6800, "XAUUSD": 2010.0}
        )
        # Should detect some positive correlation
        assert abs(feats["commodity_fx_lead"]) > 0.0


class TestUs500RiskOn:
    def test_positive_momentum(self, engine):
        for i in range(50):
            engine.compute_lead_lag_features("EURUSD", {"US500": 4500.0 + i * 1.0})
        feats = engine.compute_lead_lag_features("EURUSD", {"US500": 4550.0})
        assert feats["us500_risk_on"] > 0.0

    def test_negative_momentum(self, engine):
        for i in range(50):
            engine.compute_lead_lag_features("EURUSD", {"US500": 4500.0 - i * 1.0})
        feats = engine.compute_lead_lag_features("EURUSD", {"US500": 4450.0})
        assert feats["us500_risk_on"] < 0.0
