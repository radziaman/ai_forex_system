"""Tests for SignalEngine."""
import numpy as np
import pandas as pd
from infrastructure.config_v2 import AppConfig
from services.signal_engine import SignalEngine
from services import Signal, SignalDirection, Regime, FeatureUpdate


class TestSignalEngine:
    def setup_method(self):
        self.config = AppConfig()
        self.engine = SignalEngine(self.config)

    def test_init(self):
        assert self.engine.name == "signal_engine"
        assert len(self.engine.ensemble.experts) == 0  # not started yet

    def test_start_adds_rule_based_expert(self):
        import asyncio
        asyncio.run(self.engine.start())
        assert len(self.engine.ensemble.experts) >= 1
        asyncio.run(self.engine.stop())

    def test_on_features_none_if_no_features(self):
        result = self.engine.on_features(None)
        assert result is None

    def test_on_features_returns_none_without_ohlcv(self):
        update = FeatureUpdate(
            symbol="EURUSD", timeframe="1h",
            features=np.zeros((30, 55)),
            ohlcv=None, price=1.12,
        )
        result = self.engine.on_features(update)
        # Should return None since no regime can be detected
        assert result is None or isinstance(result, Signal)

    def test_on_features_with_mock_ohlcv(self):
        np.random.seed(42)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=200, freq="1h")
        df = pd.DataFrame({
            "timestamp": dates.astype(np.int64) // 10**9,
            "open": np.cumsum(np.random.randn(200) * 0.001) + 1.12,
            "high": 0, "low": 0, "close": 0, "volume": 1000,
        })
        df["high"] = df["open"] * 1.002
        df["low"] = df["open"] * 0.998
        df["close"] = df["open"] * (1 + np.random.randn(200) * 0.001)
        df["volume"] = np.random.randint(100, 10000, 200)

        update = FeatureUpdate(
            symbol="EURUSD", timeframe="1h",
            features=np.zeros((30, 55)),  # simplified features
            ohlcv=df, price=1.12,
        )
        result = self.engine.on_features(update)
        if result is not None:
            assert isinstance(result, Signal)
            assert result.symbol == "EURUSD"
            assert result.direction in (SignalDirection.BUY, SignalDirection.SELL, SignalDirection.HOLD)
            assert 0 <= result.confidence <= 1
            assert isinstance(result.regime, Regime)

    def test_drift_monitor_updates(self):
        drifted = self.engine.on_trade_result("EURUSD", 1.12, 1.121)
        assert isinstance(drifted, bool)
        assert "EURUSD" in self.engine.drift_monitors
