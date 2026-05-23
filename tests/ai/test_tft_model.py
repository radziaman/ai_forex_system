"""Tests for TemporalFusionTransformer model."""

import os
import tempfile

import numpy as np
import torch

from ai.tft_model import TemporalFusionTransformer


class TestTFTModelBuild:
    def test_build_and_forward(self):
        model = TemporalFusionTransformer(
            input_size=49,
            hidden_size=64,
            num_heads=4,
            num_layers=2,
            output_size=1,
            dropout=0.1,
            static_size=8,
            num_variables=7,
        )
        x = torch.randn(2, 30, 49)
        static = torch.randn(2, 8)
        out = model.forward(x, static)
        assert out.shape == (2, 1)
        assert torch.isfinite(out).all()

    def test_forward_without_static(self):
        model = TemporalFusionTransformer(
            input_size=14,
            hidden_size=32,
            num_heads=2,
            num_layers=1,
            output_size=1,
            dropout=0.0,
            static_size=8,
            num_variables=7,
        )
        x = torch.randn(1, 10, 14)
        out = model.forward(x, None)
        assert out.shape == (1, 1)

    def test_predict_numpy(self):
        model = TemporalFusionTransformer(
            input_size=21,
            hidden_size=32,
            num_heads=2,
            num_layers=1,
            output_size=1,
            dropout=0.0,
            static_size=4,
            num_variables=7,
        )
        x = np.random.randn(1, 15, 21).astype(np.float32)
        static = np.random.randn(1, 4).astype(np.float32)
        probs = model.predict(x, static)
        assert probs.shape == (1, 1)
        assert 0.0 <= probs[0, 0] <= 1.0

    def test_predict_2d_input(self):
        model = TemporalFusionTransformer(
            input_size=14,
            hidden_size=16,
            num_heads=2,
            num_layers=1,
            output_size=1,
            dropout=0.0,
            static_size=4,
            num_variables=7,
        )
        x = np.random.randn(20, 14).astype(np.float32)
        probs = model.predict(x)
        assert probs.shape == (1, 1)


class TestTFTSaveLoad:
    def test_save_and_load(self):
        model = TemporalFusionTransformer(
            input_size=28,
            hidden_size=32,
            num_heads=2,
            num_layers=1,
            output_size=1,
            dropout=0.0,
            static_size=4,
            num_variables=7,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tft_test.pt")
            model.save(path)
            assert os.path.exists(path)

            loaded = TemporalFusionTransformer.load(path)
            assert loaded.input_size == model.input_size
            assert loaded.hidden_size == model.hidden_size

            # Inference parity
            x = torch.randn(1, 10, 28)
            static = torch.randn(1, 4)
            with torch.no_grad():
                original_out = torch.sigmoid(model(x, static))
                loaded_out = torch.sigmoid(loaded(x, static))
            np.testing.assert_allclose(
                original_out.numpy(), loaded_out.numpy(), rtol=1e-5
            )

    def test_load_state_dict_keys(self):
        model = TemporalFusionTransformer(
            input_size=14,
            hidden_size=16,
            num_heads=2,
            num_layers=1,
            output_size=1,
            dropout=0.0,
            static_size=4,
            num_variables=7,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tft_keys.pt")
            model.save(path)
            loaded = TemporalFusionTransformer.load(path)
            for k in model.state_dict().keys():
                assert k in loaded.state_dict()


class TestTFTGating:
    def test_grn_with_context(self):
        from ai.tft_model import GatedResidualNetwork

        grn = GatedResidualNetwork(8, 16, 8, dropout=0.0, context_size=4)
        x = torch.randn(2, 8)
        ctx = torch.randn(2, 4)
        out = grn(x, ctx)
        assert out.shape == (2, 8)

    def test_variable_selection(self):
        from ai.tft_model import VariableSelectionNetwork

        vsn = VariableSelectionNetwork(
            num_inputs=7,
            input_size=7,  # 49 / 7
            hidden_size=32,
            dropout=0.0,
            context_size=8,
        )
        x = torch.randn(2, 10, 7, 7)
        ctx = torch.randn(2, 8)
        selected, weights = vsn(x, ctx)
        assert selected.shape == (2, 10, 32)
        assert weights.shape == (2, 10, 7)
        # Weights must sum to 1 across variables
        np.testing.assert_allclose(
            weights.sum(dim=-1).detach().numpy(), np.ones((2, 10)), atol=1e-6
        )
