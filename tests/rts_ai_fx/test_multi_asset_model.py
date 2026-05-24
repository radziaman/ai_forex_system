import numpy as np
import pytest
from rts_ai_fx.multi_asset_model import GlobalMultiAssetModel


class TestGlobalMultiAssetModel:
    def test_symbol_to_id_mapping(self):
        assert GlobalMultiAssetModel.symbol_to_id("EURUSD") == 0
        assert GlobalMultiAssetModel.symbol_to_id("GBPUSD") == 1
        assert GlobalMultiAssetModel.symbol_to_id("USDJPY") == 2

    def test_symbol_to_id_unknown_raises(self):
        with pytest.raises(ValueError):
            GlobalMultiAssetModel.symbol_to_id("XXXYYY")

    def test_build_creates_model(self):
        model = GlobalMultiAssetModel(n_symbols=5, lookback=30, n_features=49)
        model.build()
        assert model.model is not None
        assert len(model.model.inputs) == 2
        assert model.model.output_shape[-1] == 5

    def test_build_input_shapes(self):
        model = GlobalMultiAssetModel(n_symbols=3, lookback=20, n_features=10)
        model.build()
        # Check that input shapes are correct
        input_shapes = [inp.shape for inp in model.model.inputs]
        shapes = [(None, 20, 10), (None, 1)]
        for actual, expected in zip(input_shapes, shapes):
            assert actual[1:] == expected[1:]

    def test_predict_returns_correct_shape(self):
        model = GlobalMultiAssetModel(n_symbols=3, lookback=20, n_features=10)
        model.build()
        X = np.random.randn(5, 20, 10)
        preds = model.predict(X, symbol_id=0)
        assert preds.shape == (5,)

    def test_predict_different_symbols(self):
        model = GlobalMultiAssetModel(n_symbols=3, lookback=20, n_features=10)
        model.build()
        X = np.random.randn(3, 20, 10)
        pred_eurusd = model.predict(X, symbol_id=0)
        pred_gbpusd = model.predict(X, symbol_id=1)
        assert pred_eurusd.shape == pred_gbpusd.shape

    def test_get_embeddings(self):
        model = GlobalMultiAssetModel(n_symbols=3, lookback=20, n_features=10)
        model.build()
        embeddings = model.get_embeddings()
        assert len(embeddings) == 3
        for name in ["EURUSD", "GBPUSD", "USDJPY"]:
            assert name in embeddings
            assert embeddings[name].shape[0] == 16  # embedding_dim

    def test_save_load_roundtrip(self, tmp_path):
        model = GlobalMultiAssetModel(n_symbols=3, lookback=20, n_features=10)
        model.build()
        path = str(tmp_path / "test_model.keras")
        model.save(path)
        loaded = GlobalMultiAssetModel(n_symbols=3, lookback=20, n_features=10)
        loaded.load(path)
        assert loaded.model is not None
        X = np.random.randn(2, 20, 10)
        pred = loaded.predict(X, symbol_id=0)
        assert pred.shape == (2,)
