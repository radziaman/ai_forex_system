#!/usr/bin/env python3
"""
Tests for AI models - LSTM-CNN Hybrid, Profitability Classifier, PPO Agent, and Ensemble.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rts_ai_fx.model import LSTMCNNHybrid, ProfitabilityClassifier
from rts_ai_fx.ensemble import MoEEnsemble, Expert, EnsemblePrediction


class TestLSTMCNNHybrid:
    """Test cases for LSTM-CNN Hybrid model."""

    @pytest.fixture
    def model(self):
        return LSTMCNNHybrid(lookback=30, n_features=51)

    @pytest.fixture
    def sample_data(self):
        X = np.random.randn(100, 30, 51).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        return X, y

    def test_build_model(self, model):
        built_model = model.build()
        assert built_model is not None
        assert model.model is not None

    def test_predict_without_build_raises(self, model, sample_data):
        X, _ = sample_data
        with pytest.raises(ValueError, match="Model not built"):
            model.predict(X[:5])

    def test_predict_shape(self, model, sample_data):
        X, y = sample_data
        model.build()
        predictions = model.predict(X[:10])
        assert predictions.shape == (10, 1)

    def test_train_model(self, model, sample_data):
        X, y = sample_data
        X_val, y_val = X[:20], y[:20]
        X_train, y_train = X[20:], y[20:]
        history = model.train(X_train, y_train, X_val, y_val, epochs=2, batch_size=32)
        assert history is not None
        assert "loss" in history.history

    def test_save_load(self, model, sample_data, tmp_path):
        X, y = sample_data
        model.build()
        model.train(X[:50], y[:50], X[:10], y[:10], epochs=1, batch_size=32)
        path = str(tmp_path / "lstm_cnn_test.keras")
        model.save(path)
        loaded = LSTMCNNHybrid.load(path)
        assert loaded.model is not None
        pred1 = model.predict(X[:5])
        pred2 = loaded.predict(X[:5])
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=5)


class TestProfitabilityClassifier:
    """Test cases for Profitability Classifier."""

    @pytest.fixture
    def model(self):
        return ProfitabilityClassifier(lookback=30, n_features=51)

    @pytest.fixture
    def sample_data(self):
        X = np.random.randn(100, 30, 51).astype(np.float32)
        prices = np.cumsum(np.random.randn(101)) + 100
        return X, prices

    def test_build_model(self, model):
        built = model.build()
        assert built is not None
        assert model.model is not None

    def test_make_labels(self, model):
        prices = np.array([1.0, 1.01, 0.99, 1.02, 1.02])
        labels = model._make_labels(prices)
        assert len(labels) == 4
        assert labels[0] == 1
        assert labels[1] == 0
        assert labels[2] == 1
        assert labels[3] == 0

    def test_predict_proba(self, model, sample_data):
        X, prices = sample_data
        model.build()
        proba = model.predict_proba(X[:10])
        assert proba.shape == (10, 1)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_train_model(self, model, sample_data):
        X, prices = sample_data
        X_val, prices_val = X[:20], prices[:21]
        X_train, prices_train = X[20:], prices[20:]
        history = model.train(
            X_train, prices_train, X_val, prices_val, epochs=2, batch_size=32
        )
        assert history is not None
        assert "loss" in history.history


class TestMoEEnsemble:
    """Test cases for Mixture-of-Experts Ensemble."""

    @pytest.fixture
    def ensemble(self):
        return MoEEnsemble()

    @pytest.fixture
    def sample_experts(self):
        def predict_up(X):
            return np.array([1.12])

        def predict_down(X):
            return np.array([1.10])

        def conf_high(X):
            return np.array([0.9])

        def conf_low(X):
            return np.array([0.6])

        return [
            Expert(
                name="bull_expert",
                predict=predict_up,
                confidence=conf_high,
                regime="trending",
                elo=1300.0,
            ),
            Expert(
                name="bear_expert",
                predict=predict_down,
                confidence=conf_low,
                regime="volatile",
                elo=1100.0,
            ),
        ]

    def test_empty_ensemble_predict(self, ensemble):
        X = np.random.randn(1, 30, 51)
        pred = ensemble.predict(X, regime="trending")
        assert pred.price == 0.0
        assert pred.confidence == 0.0
        assert pred.direction == "HOLD"

    def test_add_expert(self, ensemble, sample_experts):
        for expert in sample_experts:
            ensemble.add_expert(
                expert.name, expert.predict, expert.confidence, expert.regime
            )
        assert len(ensemble.experts) == 2

    def test_predict_with_experts(self, ensemble, sample_experts):
        for expert in sample_experts:
            ensemble.add_expert(
                expert.name, expert.predict, expert.confidence, expert.regime
            )
        X = np.random.randn(1, 30, 51)
        pred = ensemble.predict(X, regime="trending")
        assert pred.price > 0
        assert 0 <= pred.confidence <= 1
        assert len(pred.expert_outputs) == 2

    def test_regime_weighting(self, ensemble, sample_experts):
        for expert in sample_experts:
            ensemble.add_expert(
                expert.name, expert.predict, expert.confidence, expert.regime
            )
        X = np.random.randn(1, 30, 51)
        pred_trending = ensemble.predict(X, regime="trending")
        pred_volatile = ensemble.predict(X, regime="volatile")
        assert (
            pred_trending.expert_outputs["bull_expert"]["weight"]
            > pred_trending.expert_outputs["bear_expert"]["weight"]
        )
        assert (
            pred_volatile.expert_outputs["bear_expert"]["weight"]
            > pred_volatile.expert_outputs["bull_expert"]["weight"]
        )

    def test_should_trade(self, ensemble, sample_experts):
        for expert in sample_experts:
            ensemble.add_expert(
                expert.name, expert.predict, expert.confidence, expert.regime
            )
        X = np.random.randn(1, 30, 51)
        pred = ensemble.predict(X, regime="trending")
        current_price = 1.11
        should_trade, direction, confidence = ensemble.should_trade(pred, current_price)
        assert isinstance(should_trade, bool)
        assert direction in ["BUY", "SELL", "HOLD"]
        assert 0 <= confidence <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
