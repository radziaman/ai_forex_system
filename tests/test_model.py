"""Tests for AI models - LSTMCNNHybrid and ProfitabilityClassifier."""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rts_ai_fx.model import LSTMCNNHybrid, ProfitabilityClassifier


class TestLSTMCNNHybrid:
    """Test suite for LSTMCNNHybrid model."""

    @pytest.fixture
    def model(self):
        return LSTMCNNHybrid(lookback=30, n_features=51)

    def test_build_creates_model(self, model):
        """Model should build without errors."""
        result = model.build()
        assert result is not None
        assert model.model is not None

    def test_predict_requires_model(self, model):
        """Predict should raise error if model not built."""
        X = np.random.randn(10, 30, 51)
        with pytest.raises(ValueError, match="Model not built"):
            model.predict(X)

    def test_predict_returns_correct_shape(self, model):
        """Predict should return (n_samples, 1) shape."""
        model.build()
        X = np.random.randn(10, 30, 51)
        preds = model.predict(X)
        assert preds.shape == (10, 1)

    def test_save_and_load(self, model, tmp_path):
        """Model should save and load correctly."""
        model.build()
        save_path = tmp_path / "test_model.keras"
        model.save(str(save_path))

        loaded = LSTMCNNHybrid.load(str(save_path))
        assert loaded.model is not None

        X = np.random.randn(5, 30, 51)
        preds_orig = model.predict(X)
        preds_loaded = loaded.predict(X)
        np.testing.assert_array_almost_equal(preds_orig, preds_loaded)
