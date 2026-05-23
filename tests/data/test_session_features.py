"""Tests for SessionBehaviorEngine."""

import numpy as np
import pandas as pd
import pytest

from src.data.session_features import SessionBehaviorEngine


def _make_df(n: int = 100, base_price: float = 1.1000) -> pd.DataFrame:
    """Create a synthetic 1-minute OHLCV DataFrame."""
    timestamps = pd.date_range("2024-01-01 00:00", periods=n, freq="min")
    np.random.seed(42)
    opens = base_price + np.cumsum(np.random.randn(n) * 0.0001)
    highs = opens + np.abs(np.random.randn(n) * 0.0002)
    lows = opens - np.abs(np.random.randn(n) * 0.0002)
    closes = opens + np.random.randn(n) * 0.0001
    volumes = np.random.randint(100, 1000, size=n).astype(float)
    return pd.DataFrame(
        {
            "timestamp": timestamps.astype("int64") // 10**9,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )


@pytest.fixture
def engine() -> SessionBehaviorEngine:
    return SessionBehaviorEngine("EURUSD")


class TestSessionType:
    def test_london_open(self, engine):
        ts = pd.Timestamp("2024-01-02 08:30", tz="UTC").timestamp()
        feats = engine.compute_session_features(_make_df(), ts)
        assert feats["session_type"] == 2.0

    def test_ny_overlap(self, engine):
        ts = pd.Timestamp("2024-01-02 14:00", tz="UTC").timestamp()
        feats = engine.compute_session_features(_make_df(), ts)
        assert feats["session_type"] == 1.0

    def test_asian_session(self, engine):
        ts = pd.Timestamp("2024-01-02 23:00", tz="UTC").timestamp()
        feats = engine.compute_session_features(_make_df(), ts)
        assert feats["session_type"] == 3.0

    def test_overnight_session(self, engine):
        ts = pd.Timestamp("2024-01-02 18:00", tz="UTC").timestamp()
        feats = engine.compute_session_features(_make_df(), ts)
        assert feats["session_type"] == 0.0


class TestSessionMomentumBias:
    def test_london_bias_positive(self, engine):
        ts = pd.Timestamp("2024-01-02 09:00", tz="UTC").timestamp()
        feats = engine.compute_session_features(_make_df(), ts)
        assert feats["session_momentum_bias"] == 1.0

    def test_asian_bias_negative(self, engine):
        ts = pd.Timestamp("2024-01-02 02:00", tz="UTC").timestamp()
        feats = engine.compute_session_features(_make_df(), ts)
        assert feats["session_momentum_bias"] == -1.0


class TestWeekendGapRisk:
    def test_friday_evening(self, engine):
        ts = pd.Timestamp("2024-01-05 21:00", tz="UTC").timestamp()
        feats = engine.compute_session_features(_make_df(), ts)
        assert feats["weekend_gap_risk"] == 1.0

    def test_sunday_pre_open(self, engine):
        ts = pd.Timestamp("2024-01-07 20:00", tz="UTC").timestamp()
        feats = engine.compute_session_features(_make_df(), ts)
        assert feats["weekend_gap_risk"] == 1.0

    def test_midweek_no_risk(self, engine):
        ts = pd.Timestamp("2024-01-03 12:00", tz="UTC").timestamp()
        feats = engine.compute_session_features(_make_df(), ts)
        assert feats["weekend_gap_risk"] == 0.0


class TestVolExpansionAndSurge:
    def test_london_vol_expansion_high(self, engine):
        """London ATR should exceed Asian ATR when London bars are wilder."""
        df = _make_df(n=1440)  # 24 h of data
        # Inflate London hours (07:00-11:59) volatility
        london_mask = (df["timestamp"] % 86400 // 3600).between(7, 11)
        df.loc[london_mask, "high"] += 0.005
        df.loc[london_mask, "low"] -= 0.005
        ts = pd.Timestamp("2024-01-02 10:00", tz="UTC").timestamp()
        feats = engine.compute_session_features(df, ts)
        assert feats["london_vol_expansion"] > 1.5

    def test_ny_overlap_volume_surge(self, engine):
        """NY volume should exceed 24 h average when NY bars are heavy."""
        df = _make_df(n=1440)
        ny_mask = (df["timestamp"] % 86400 // 3600).between(12, 15)
        df.loc[ny_mask, "volume"] *= 5.0
        ts = pd.Timestamp("2024-01-02 14:00", tz="UTC").timestamp()
        feats = engine.compute_session_features(df, ts)
        assert feats["ny_overlap_volume_surge"] > 2.0

    def test_asian_range_contraction(self, engine):
        """Asian range should be a smaller % of full-day range."""
        df = _make_df(n=1440)
        asian_mask = (df["timestamp"] % 86400 // 3600).isin(
            list(range(21, 24)) + list(range(0, 8))
        )
        df.loc[asian_mask, "high"] = df.loc[asian_mask, "open"] + 0.0001
        df.loc[asian_mask, "low"] = df.loc[asian_mask, "open"] - 0.0001
        ts = pd.Timestamp("2024-01-02 02:00", tz="UTC").timestamp()
        feats = engine.compute_session_features(df, ts)
        assert feats["asian_range_contraction"] < 1.0

    def test_empty_df_fallback(self, engine):
        feats = engine.compute_session_features(None, 0.0)
        assert feats["london_vol_expansion"] == 1.0
        assert feats["ny_overlap_volume_surge"] == 1.0
        assert feats["asian_range_contraction"] == 1.0
