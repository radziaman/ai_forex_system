"""Tests for foundation model adapters."""

import numpy as np
from ai.foundation_models import (
    SimpleLinearAdapter,
    FoundationModelRegistry,
    FoundationPrediction,
    TimesFMAdapter,
    MOIRAIAdapter,
)


class TestSimpleLinearAdapter:
    def test_predict_returns_float(self):
        adapter = SimpleLinearAdapter(lookback=30)
        prices = np.cumsum(np.random.randn(50)) + 100.0
        pred = adapter.predict(prices)
        assert isinstance(pred, float)

    def test_predict_short_series(self):
        adapter = SimpleLinearAdapter(lookback=30)
        pred = adapter.predict(np.array([1.0, 2.0]))
        assert pred == 0.0

    def test_confidence(self):
        adapter = SimpleLinearAdapter()
        assert adapter.get_confidence() == 0.5


class TestTimesFMAdapter:
    def test_predict_returns_float(self):
        adapter = TimesFMAdapter(lookback=30)
        prices = np.cumsum(np.random.randn(50)) + 100.0
        pred = adapter.predict(prices)
        assert isinstance(pred, float)

    def test_confidence_fallback(self):
        adapter = TimesFMAdapter()
        # Without transformers installed, confidence is 0.5 (fallback)
        assert adapter.get_confidence() in (0.5, 0.65)


class TestMOIRAIAdapter:
    def test_predict_returns_float(self):
        adapter = MOIRAIAdapter(lookback=30)
        prices = np.cumsum(np.random.randn(50)) + 100.0
        pred = adapter.predict(prices)
        assert isinstance(pred, float)

    def test_confidence_fallback(self):
        adapter = MOIRAIAdapter()
        # Without transformers installed, confidence is 0.5 (fallback)
        assert adapter.get_confidence() in (0.5, 0.62)


class TestFoundationModelRegistry:
    def test_get_adapter(self):
        registry = FoundationModelRegistry()
        adapter = registry.get_adapter("EURUSD")
        assert isinstance(adapter, SimpleLinearAdapter)

    def test_predict(self):
        registry = FoundationModelRegistry()
        prices = np.cumsum(np.random.randn(50)) + 1.0
        result = registry.predict("EURUSD", prices)
        assert isinstance(result, FoundationPrediction)
        assert result.symbol == "EURUSD"
        assert isinstance(result.prediction, float)

    def test_predict_all(self):
        registry = FoundationModelRegistry()
        data = {
            "EURUSD": np.cumsum(np.random.randn(50)) + 1.0,
            "GBPUSD": np.cumsum(np.random.randn(50)) + 1.3,
        }
        results = registry.predict_all(data)
        assert len(results) == 2
        assert "EURUSD" in results
        assert "GBPUSD" in results
