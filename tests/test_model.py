"""Tests for LSTM-CNN hybrid model"""

import numpy as np
import sys
from pathlib import Path

from rts_ai_fx.model import LSTMCNNHybrid, ProfitabilityClassifier

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestLSTMCNNHybrid:
    """Test LSTM-CNN hybrid model"""

    def test_model_build(self):
        """Test model architecture builds correctly"""
        model_wrapper = LSTMCNNHybrid(lookback=30, n_features=51)
        model = model_wrapper.build()

        assert model is not None
        assert model.input_shape == (None, 30, 51)
        assert len(model.layers) > 5

    def test_model_predict(self):
        """Test model prediction"""
        model_wrapper = LSTMCNNHybrid(lookback=30, n_features=51)
        model_wrapper.build()

        X_test = np.random.randn(10, 30, 51)
        predictions = model_wrapper.predict(X_test)

        assert predictions.shape == (10, 1)

    def test_model_save_load(self, tmp_path):
        """Test model save and load"""
        model_wrapper = LSTMCNNHybrid(lookback=30, n_features=51)
        model_wrapper.build()

        save_path = str(tmp_path / "test_model.keras")
        model_wrapper.save(save_path)

        loaded = LSTMCNNHybrid.load(save_path)
        assert loaded.model is not None


class TestProfitabilityClassifier:
    """Test profitability classifier"""

    def test_build(self):
        """Test classifier builds correctly"""
        classifier = ProfitabilityClassifier(lookback=30, n_features=51)
        model = classifier.build()

        assert model is not None
        assert model.output_shape == (None, 1)

    def test_predict_proba(self):
        """Test probability prediction"""
        classifier = ProfitabilityClassifier(lookback=30, n_features=51)
        classifier.build()

        X_test = np.random.randn(5, 30, 51)
        probas = classifier.predict_proba(X_test)

        assert probas.shape == (5, 1)
        assert (probas >= 0).all() and (probas <= 1).all()
