"""Tests for GNN cross-asset relationship model."""

import numpy as np
import pytest
from rts_ai_fx.gnn_cross_asset import (
    CrossAssetGNNAdapter,
    FXGNNModel,
    FXGraphBuilder,
    SimpleGATLayer,
    SYMBOL_NODES,
    FX_EDGES,
)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestFXGraphBuilder:
    def test_node_count(self):
        builder = FXGraphBuilder()
        assert builder.n_nodes == len(SYMBOL_NODES)

    def test_build_node_features_shape(self):
        builder = FXGraphBuilder()
        price_data = {
            sym: np.cumsum(np.random.randn(100)) + 1.0 for sym in SYMBOL_NODES[:5]
        }
        features = builder.build_node_features(price_data)
        assert features.shape[0] == len(SYMBOL_NODES)
        assert features.shape[1] == 8

    def test_build_node_features_missing_data(self):
        builder = FXGraphBuilder()
        price_data: dict = {}
        features = builder.build_node_features(price_data)
        assert features.shape == (len(SYMBOL_NODES), 8)
        assert np.all(features == 0)

    def test_build_edge_index(self):
        builder = FXGraphBuilder()
        edge_index, edge_attr = builder.build_edge_index()
        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] == len(FX_EDGES) * 2  # undirected
        assert len(edge_attr) == len(FX_EDGES) * 2

    def test_edges_within_bounds(self):
        builder = FXGraphBuilder()
        edge_index, _ = builder.build_edge_index()
        max_node = edge_index.max()
        assert max_node < len(SYMBOL_NODES)


class TestSimpleGATLayer:
    def test_forward_shape(self):
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        layer = SimpleGATLayer(in_dim=8, out_dim=16)
        x = torch.randn(11, 8)  # 11 nodes
        edge_index = torch.randint(0, 11, (2, 30))
        out = layer(x, edge_index)
        assert out.shape == (11, 16)


class TestFXGNNModel:
    def test_forward_shape(self):
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        model = FXGNNModel(node_dim=8, hidden_dim=32, output_dim=16)
        x = torch.randn(11, 8)
        edge_index = torch.randint(0, 11, (2, 30))
        out = model(x, edge_index)
        assert out.shape == (11, 16)

    def test_get_all_embeddings(self):
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        model = FXGNNModel(node_dim=8, hidden_dim=16, output_dim=8)
        x = torch.randn(5, 8)
        edge_index = torch.randint(0, 5, (2, 10))
        embeddings = model.get_all_embeddings(x, edge_index)
        assert len(embeddings) == 5


class TestCrossAssetGNNAdapter:
    def test_compute_embeddings_shape(self):
        adapter = CrossAssetGNNAdapter(embedding_dim=16)
        price_data = {
            sym: np.cumsum(np.random.randn(50)) + 1.0 for sym in SYMBOL_NODES[:5]
        }
        embeddings = adapter.compute_embeddings(price_data)
        assert len(embeddings) == len(SYMBOL_NODES)
        for sym, emb in embeddings.items():
            assert len(emb) == 16

    def test_compute_embeddings_fallback_without_torch(self):
        if TORCH_AVAILABLE:
            pytest.skip("Fallback path only applies when PyTorch is not available")
        adapter = CrossAssetGNNAdapter(embedding_dim=8)
        price_data: dict = {}
        embeddings = adapter.compute_embeddings(price_data)
        for emb in embeddings.values():
            assert np.all(emb == 0)
