"""Tests for Mamba state-space model."""

import numpy as np
import pytest
from rts_ai_fx.mamba_model import (
    MambaBlock,
    MambaTimeSeriesModel,
    selective_scan,
    TORCH_AVAILABLE,
)


class TestSelectiveScan:
    def test_scan_runs(self):
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        import torch

        batch, seq, d_state = 2, 10, 8
        u = torch.randn(batch, seq, d_state)
        delta = torch.randn(batch, seq, d_state)
        A = torch.randn(d_state)  # diagonal A (d_state,)
        B = torch.randn(batch, seq, d_state)
        y = selective_scan(u, delta, A, B)
        assert y.shape == (batch, seq, d_state)


class TestMambaBlock:
    def test_forward_shape(self):
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        import torch

        block = MambaBlock(d_model=32, d_state=8)
        x = torch.randn(4, 20, 32)
        out = block(x)
        assert out.shape == (4, 20, 32)


class TestMambaTimeSeriesModel:
    def test_forward_shape(self):
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        import torch

        model = MambaTimeSeriesModel(
            n_features=10, d_model=32, n_layers=1, max_seq_len=50
        )
        x = torch.randn(4, 30, 10)
        out = model(x)
        assert out.shape == (4, 1)

    def test_predict_numpy(self):
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        model = MambaTimeSeriesModel(
            n_features=5, d_model=16, n_layers=1, max_seq_len=30
        )
        X = np.random.randn(2, 20, 5).astype(np.float32)
        preds = model.predict(X)
        assert preds.shape == (2,)

    def test_model_config(self):
        model = MambaTimeSeriesModel(
            n_features=49, d_model=64, d_state=16, n_layers=2, max_seq_len=256
        )
        assert model.n_features == 49
        assert model.d_model == 64
        assert model.max_seq_len == 256
