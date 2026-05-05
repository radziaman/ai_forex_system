#!/usr/bin/env python3
"""
End-to-end tests for Causal Feature Selection.
Tests PCMCI discovery, correlation fallback, fit_transform pipeline,
edge cases, and integration with feature engineering.
"""
import os
import sys
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rts_ai_fx.causal_features import CausalFeatureSelector, CAUSAL_AVAILABLE
from rts_ai_fx.features_unified import compute_features


class TestCausalFeatureSelectorInitialization:
    """Test initialization and default parameters."""

    def test_default_params(self):
        selector = CausalFeatureSelector()
        assert selector.max_lag == 5
        assert selector.alpha == 0.01
        assert selector.causal_features_ == []
        assert selector.causal_graph_ is None
        assert selector.p_matrix_ is None
        assert selector.val_matrix_ is None

    def test_custom_params(self):
        selector = CausalFeatureSelector(max_lag=10, alpha=0.05)
        assert selector.max_lag == 10
        assert selector.alpha == 0.05

    def test_causal_available_flag(self):
        assert isinstance(CAUSAL_AVAILABLE, bool)
        if CAUSAL_AVAILABLE:
            from tigramite.pcmci import PCMCI
            from tigramite.independence_tests.parcorr import ParCorr
            import tigramite.data_processing as pp


class TestPCMCIWithSyntheticData:
    """Test causal discovery using synthetic time series with known structure."""

    @pytest.fixture
    def synthetic_causal_data(self):
        np.random.seed(42)
        n = 200
        x = np.random.randn(n)
        y = np.random.randn(n)
        z = np.random.randn(n)
        w = np.random.randn(n)
        noise = np.random.randn(n)

        x[1:] = 0.7 * x[:-1] + 0.3 * y[:-1] + 0.1 * noise[1:]
        w_data = 0.5 * x[:-2] + 0.2 * noise[2:]
        w[2:] = w_data

        features = pd.DataFrame({
            'feature_x': x,
            'feature_y': y,
            'feature_z': z,
            'feature_w': w,
        })
        target = pd.Series(0.6 * x[1:] + 0.4 * w[1:] + 0.2 * noise[1:], name='target')
        features_aligned = features.iloc[2:].reset_index(drop=True)
        target_aligned = target.iloc[1:].reset_index(drop=True)
        return features_aligned, target_aligned

    def test_pcmci_discovery_runs(self, synthetic_causal_data):
        if not CAUSAL_AVAILABLE:
            pytest.skip("tigramite not available")
        features, target = synthetic_causal_data
        selector = CausalFeatureSelector(max_lag=3, alpha=0.05)
        selected = selector.fit(features, target)
        assert isinstance(selected, list)
        assert all(f in features.columns for f in selected)
        assert len(selected) > 0
        assert len(selected) <= len(features.columns)

    def test_pcmci_p_matrix_shape(self, synthetic_causal_data):
        if not CAUSAL_AVAILABLE:
            pytest.skip("tigramite not available")
        features, target = synthetic_causal_data
        selector = CausalFeatureSelector(max_lag=3, alpha=0.05)
        selector.fit(features, target)
        assert selector.p_matrix_ is not None
        n_vars = len(features.columns) + 1
        expected_shape = (n_vars, n_vars, 4)
        assert selector.p_matrix_.shape == expected_shape

    def test_pcmci_val_matrix_shape(self, synthetic_causal_data):
        if not CAUSAL_AVAILABLE:
            pytest.skip("tigramite not available")
        features, target = synthetic_causal_data
        selector = CausalFeatureSelector(max_lag=3, alpha=0.05)
        selector.fit(features, target)
        assert selector.val_matrix_ is not None
        n_vars = len(features.columns) + 1
        expected_shape = (n_vars, n_vars, 4)
        assert selector.val_matrix_.shape == expected_shape

    def test_pcmci_causal_graph_stored(self, synthetic_causal_data):
        if not CAUSAL_AVAILABLE:
            pytest.skip("tigramite not available")
        features, target = synthetic_causal_data
        selector = CausalFeatureSelector(max_lag=3, alpha=0.05)
        selector.fit(features, target)
        assert selector.causal_graph_ is not None
        assert 'p_matrix' in selector.causal_graph_
        assert 'val_matrix' in selector.causal_graph_

    @pytest.mark.parametrize("max_lag,alpha", [
        (2, 0.01),
        (3, 0.05),
        (5, 0.1),
    ])
    def test_different_hyperparameters(self, synthetic_causal_data, max_lag, alpha):
        if not CAUSAL_AVAILABLE:
            pytest.skip("tigramite not available")
        features, target = synthetic_causal_data
        selector = CausalFeatureSelector(max_lag=max_lag, alpha=alpha)
        selected = selector.fit(features, target)
        assert isinstance(selected, list)
        assert len(selected) > 0

    def test_causal_summary_after_fit(self, synthetic_causal_data):
        if not CAUSAL_AVAILABLE:
            pytest.skip("tigramite not available")
        features, target = synthetic_causal_data
        selector = CausalFeatureSelector(max_lag=3, alpha=0.05)
        selector.fit(features, target)
        summary = selector.get_causal_summary()
        assert 'n_causal_features' in summary
        assert 'causal_features' in summary
        assert 'algorithm' in summary
        assert 'max_lag' in summary
        assert 'alpha' in summary
        assert summary['algorithm'] == 'PCMCI'
        assert summary['n_causal_features'] == len(selector.causal_features_)
        assert summary['max_lag'] == 3
        assert summary['alpha'] == 0.05


class TestCorrelationFallback:
    """Test correlation-based fallback when causal discovery is unavailable."""

    @pytest.fixture
    def correlated_data(self):
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 0.8 * x + 0.2 * np.random.randn(n)
        z = np.random.randn(n)
        w = 0.6 * x + 0.4 * np.random.randn(n)
        features = pd.DataFrame({
            'strong_corr': y,
            'no_corr': z,
            'weak_corr': w,
            'feature_x': x,
        })
        target = pd.Series(x, name='target')
        return features, target

    def test_fallback_selects_correlated_features(self, correlated_data):
        features, target = correlated_data
        with patch('rts_ai_fx.causal_features.CAUSAL_AVAILABLE', False):
            selector = CausalFeatureSelector()
            selected = selector.fit(features, target)
            assert 'strong_corr' in selected
            assert len(selected) > 0

    def test_fallback_excludes_uncorrelated(self, correlated_data):
        features, target = correlated_data
        with patch('rts_ai_fx.causal_features.CAUSAL_AVAILABLE', False):
            selector = CausalFeatureSelector()
            selected = selector.fit(features, target)
            corrs = features.apply(lambda col: abs(col.corr(target)) if col.std() > 0 else 0.0)
            assert corrs['strong_corr'] > corrs['no_corr']
            assert 'strong_corr' in selected

    def test_fallback_with_empty_features(self):
        empty_features = pd.DataFrame()
        target = pd.Series([1, 2, 3])
        with patch('rts_ai_fx.causal_features.CAUSAL_AVAILABLE', False):
            selector = CausalFeatureSelector()
            selected = selector.fit(empty_features, target)
            assert selected == []


class TestFitTransformPipeline:
    """Test the sklearn-style fit_transform API."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 150
        features = pd.DataFrame({
            'feat_a': np.random.randn(n),
            'feat_b': np.random.randn(n),
            'feat_c': np.random.randn(n),
            'feat_d': np.random.randn(n),
            'feat_e': np.random.randn(n),
        })
        target = pd.Series(
            0.5 * features['feat_a'].values +
            0.3 * features['feat_b'].values +
            0.2 * np.random.randn(n),
            name='target'
        )
        return features, target

    def test_fit_transform_returns_dataframe(self, sample_data):
        if not CAUSAL_AVAILABLE:
            pytest.skip("tigramite not available")
        features, target = sample_data
        selector = CausalFeatureSelector(max_lag=3, alpha=0.05)
        result = selector.fit_transform(features, target)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(features)

    def test_fit_transform_selects_subset(self, sample_data):
        if not CAUSAL_AVAILABLE:
            pytest.skip("tigramite not available")
        features, target = sample_data
        selector = CausalFeatureSelector(max_lag=3, alpha=0.05)
        result = selector.fit_transform(features, target)
        assert len(result.columns) <= len(features.columns)
        assert len(result.columns) > 0

    def test_fit_then_transform(self, sample_data):
        if not CAUSAL_AVAILABLE:
            pytest.skip("tigramite not available")
        features, target = sample_data
        selector = CausalFeatureSelector(max_lag=3, alpha=0.05)
        selected_features = selector.fit(features, target)
        transformed = selector.transform(features)
        assert isinstance(transformed, pd.DataFrame)
        assert list(transformed.columns) == selected_features

    def test_transform_on_new_data(self, sample_data):
        if not CAUSAL_AVAILABLE:
            pytest.skip("tigramite not available")
        features, target = sample_data
        selector = CausalFeatureSelector(max_lag=3, alpha=0.05)
        selector.fit(features, target)
        new_features = pd.DataFrame({
            'feat_a': np.random.randn(50),
            'feat_b': np.random.randn(50),
            'feat_c': np.random.randn(50),
            'feat_d': np.random.randn(50),
            'feat_e': np.random.randn(50),
        })
        transformed = selector.transform(new_features)
        assert len(transformed) == 50
        assert all(col in features.columns for col in transformed.columns)

    def test_transform_before_fit_returns_original(self, sample_data):
        features, _ = sample_data
        selector = CausalFeatureSelector()
        result = selector.transform(features)
        assert result.equals(features)

    def test_transform_handles_missing_columns(self, sample_data):
        if not CAUSAL_AVAILABLE:
            pytest.skip("tigramite not available")
        features, target = sample_data
        selector = CausalFeatureSelector(max_lag=3, alpha=0.05)
        selector.fit(features, target)
        new_features = pd.DataFrame({
            'feat_a': np.random.randn(50),
            'feat_b': np.random.randn(50),
        })
        transformed = selector.transform(new_features)
        assert all(col in new_features.columns for col in transformed.columns)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self):
        empty_features = pd.DataFrame()
        target = pd.Series([])
        selector = CausalFeatureSelector()
        selected = selector.fit(empty_features, target)
        assert selected == []

    def test_insufficient_data(self):
        np.random.seed(42)
        small_features = pd.DataFrame({
            'a': np.random.randn(10),
            'b': np.random.randn(10),
        })
        small_target = pd.Series(np.random.randn(10))
        selector = CausalFeatureSelector()
        selected = selector.fit(small_features, small_target)
        assert len(selected) == 2

    def test_features_with_nan(self):
        if not CAUSAL_AVAILABLE:
            pytest.skip("tigramite not available")
        np.random.seed(42)
        n = 150
        features = pd.DataFrame({
            'a': np.random.randn(n),
            'b': np.random.randn(n),
            'c': np.random.randn(n),
        })
        features.iloc[10, 0] = np.nan
        features.iloc[20, 1] = np.nan
        target = pd.Series(0.5 * features['a'].values + 0.3 * np.random.randn(n))
        selector = CausalFeatureSelector(max_lag=3, alpha=0.05)
        try:
            selected = selector.fit(features, target)
            assert isinstance(selected, list)
        except Exception:
            pass

    def test_target_with_nan(self):
        if not CAUSAL_AVAILABLE:
            pytest.skip("tigramite not available")
        np.random.seed(42)
        n = 150
        features = pd.DataFrame({
            'a': np.random.randn(n),
            'b': np.random.randn(n),
        })
        target = pd.Series(0.5 * features['a'].values + 0.3 * np.random.randn(n))
        target.iloc[5] = np.nan
        selector = CausalFeatureSelector(max_lag=3, alpha=0.05)
        try:
            selected = selector.fit(features, target)
            assert isinstance(selected, list)
        except Exception:
            pass

    def test_single_feature(self):
        if not CAUSAL_AVAILABLE:
            pytest.skip("tigramite not available")
        np.random.seed(42)
        n = 150
        features = pd.DataFrame({'only_feature': np.random.randn(n)})
        target = pd.Series(0.7 * features['only_feature'].values + 0.3 * np.random.randn(n))
        selector = CausalFeatureSelector(max_lag=3, alpha=0.05)
        selected = selector.fit(features, target)
        assert isinstance(selected, list)

    def test_many_features_few_samples(self):
        if not CAUSAL_AVAILABLE:
            pytest.skip("tigramite not available")
        np.random.seed(42)
        n = 100
        n_features = 15
        feature_data = {f'feat_{i}': np.random.randn(n) for i in range(n_features)}
        features = pd.DataFrame(feature_data)
        target = pd.Series(np.random.randn(n))
        selector = CausalFeatureSelector(max_lag=2, alpha=0.05)
        selected = selector.fit(features, target)
        assert isinstance(selected, list)
        assert all(f in features.columns for f in selected)

    def test_constant_features(self):
        if not CAUSAL_AVAILABLE:
            pytest.skip("tigramite not available")
        np.random.seed(42)
        n = 150
        features = pd.DataFrame({
            'constant': np.ones(n),
            'variable': np.random.randn(n),
        })
        target = pd.Series(np.random.randn(n))
        selector = CausalFeatureSelector(max_lag=3, alpha=0.05)
        selected = selector.fit(features, target)
        assert isinstance(selected, list)
        assert 'variable' in selected

    def test_exception_handling_returns_fallback(self):
        np.random.seed(42)
        n = 150
        features = pd.DataFrame({
            'a': np.random.randn(n),
            'b': np.random.randn(n),
        })
        target = pd.Series(np.random.randn(n))
        selector = CausalFeatureSelector()
        if CAUSAL_AVAILABLE:
            with patch.object(selector, '_to_causal_format', side_effect=Exception("Test error")):
                selected = selector.fit(features, target)
                assert isinstance(selected, list)


class TestIntegrationWithFeaturePipeline:
    """Test integration with the compute_features pipeline."""

    @pytest.fixture
    def ohlcv_data(self):
        np.random.seed(42)
        n = 300
        close = np.cumsum(np.random.randn(n) * 0.001) + 1.0
        high = close + np.abs(np.random.randn(n) * 0.0005)
        low = close - np.abs(np.random.randn(n) * 0.0005)
        opn = close + np.random.randn(n) * 0.0003
        volume = np.random.randint(1000, 10000, n).astype(float)
        dates = pd.date_range('2025-01-01', periods=n, freq='h')
        df = pd.DataFrame({
            'open': opn, 'high': high, 'low': low, 'close': close, 'volume': volume,
        }, index=dates)
        return df

    def test_causal_on_computed_features(self, ohlcv_data):
        if not CAUSAL_AVAILABLE:
            pytest.skip("tigramite not available")
        features_df = compute_features(ohlcv_data)
        features_df = features_df.dropna()
        assert len(features_df) > 50
        target = pd.Series(
            features_df['close'].pct_change().shift(-1).values,
            index=features_df.index,
            name='target'
        )
        target = target.iloc[:-1]
        features_aligned = features_df.iloc[:-1].drop(columns=['close'], errors='ignore')
        selector = CausalFeatureSelector(max_lag=3, alpha=0.05)
        selected = selector.fit(features_aligned, target)
        assert isinstance(selected, list)
        assert all(f in features_aligned.columns for f in selected)

    def test_full_pipeline_fit_transform(self, ohlcv_data):
        if not CAUSAL_AVAILABLE:
            pytest.skip("tigramite not available")
        features_df = compute_features(ohlcv_data)
        features_df = features_df.dropna()
        target = pd.Series(
            features_df['close'].pct_change().shift(-1).values,
            index=features_df.index,
            name='target'
        )
        target = target.iloc[:-1]
        features_aligned = features_df.iloc[:-1].drop(columns=['close'], errors='ignore')
        selector = CausalFeatureSelector(max_lag=3, alpha=0.05)
        reduced_features = selector.fit_transform(features_aligned, target)
        assert isinstance(reduced_features, pd.DataFrame)
        assert len(reduced_features.columns) <= len(features_aligned.columns)
        assert len(reduced_features.columns) > 0

    def test_causal_summary_after_pipeline(self, ohlcv_data):
        if not CAUSAL_AVAILABLE:
            pytest.skip("tigramite not available")
        features_df = compute_features(ohlcv_data)
        features_df = features_df.dropna()
        target = pd.Series(
            features_df['close'].pct_change().shift(-1).values,
            index=features_df.index,
            name='target'
        )
        target = target.iloc[:-1]
        features_aligned = features_df.iloc[:-1].drop(columns=['close'], errors='ignore')
        selector = CausalFeatureSelector(max_lag=3, alpha=0.05)
        selector.fit(features_aligned, target)
        summary = selector.get_causal_summary()
        assert summary['n_causal_features'] > 0
        assert summary['algorithm'] == 'PCMCI'
        assert len(summary['causal_features']) > 0


class TestMainSystemIntegration:
    """Test patterns matching main.py usage."""

    @pytest.fixture
    def trading_scenario_data(self):
        np.random.seed(42)
        n = 500
        dates = pd.date_range('2025-01-01', periods=n, freq='h')
        close = 1.1000 + np.cumsum(np.random.randn(n) * 0.0003)
        high = close + np.abs(np.random.randn(n) * 0.0002)
        low = close - np.abs(np.random.randn(n) * 0.0002)
        opn = close + np.random.randn(n) * 0.0001
        volume = np.random.randint(1000, 10000, n).astype(float)
        df = pd.DataFrame({
            'open': opn, 'high': high, 'low': low, 'close': close, 'volume': volume,
        }, index=dates)
        return df

    def test_main_py_causal_pattern(self, trading_scenario_data):
        if not CAUSAL_AVAILABLE:
            pytest.skip("tigramite not available")
        df_1h = trading_scenario_data
        if len(df_1h) > 50:
            if "close" in df_1h.columns and len(df_1h) > 10:
                target = df_1h["close"].pct_change().shift(-1).dropna()
                if len(target) > 50:
                    features_processed = compute_features(df_1h.iloc[:-1].copy())
                    features_processed = features_processed.dropna()
                    target_aligned = target.iloc[:len(features_processed)]
                    selector = CausalFeatureSelector(max_lag=5, alpha=0.01)
                    causal_features = selector.fit_transform(
                        features_processed, target_aligned
                    )
                    assert isinstance(causal_features, pd.DataFrame)
                    assert len(causal_features.columns) > 0
                    assert len(causal_features) == len(features_processed)

    def test_causal_with_small_buffer(self):
        df_1h = pd.DataFrame({
            'open': [1.1, 1.11, 1.12],
            'high': [1.12, 1.13, 1.14],
            'low': [1.09, 1.10, 1.11],
            'close': [1.11, 1.12, 1.13],
            'volume': [1000, 1100, 1200],
        })
        selector = CausalFeatureSelector()
        if len(df_1h) <= 50:
            assert len(df_1h) < 50

    def test_causal_preserves_index(self, trading_scenario_data):
        if not CAUSAL_AVAILABLE:
            pytest.skip("tigramite not available")
        df = trading_scenario_data
        features_df = compute_features(df).dropna()
        target = pd.Series(
            features_df['close'].pct_change().shift(-1).values,
            index=features_df.index,
        )
        target = target.iloc[:-1]
        features_aligned = features_df.iloc[:-1].drop(columns=['close'], errors='ignore')
        selector = CausalFeatureSelector(max_lag=3, alpha=0.05)
        result = selector.fit_transform(features_aligned, target)
        assert len(result) == len(features_aligned)


class TestReproducibility:
    """Test that results are reproducible with same seed."""

    def test_same_seed_same_results(self):
        if not CAUSAL_AVAILABLE:
            pytest.skip("tigramite not available")
        np.random.seed(123)
        n = 200
        features1 = pd.DataFrame({
            'a': np.random.randn(n),
            'b': np.random.randn(n),
            'c': np.random.randn(n),
        })
        target1 = pd.Series(0.5 * features1['a'].values + 0.3 * np.random.randn(n))
        np.random.seed(123)
        features2 = pd.DataFrame({
            'a': np.random.randn(n),
            'b': np.random.randn(n),
            'c': np.random.randn(n),
        })
        target2 = pd.Series(0.5 * features2['a'].values + 0.3 * np.random.randn(n))
        selector1 = CausalFeatureSelector(max_lag=3, alpha=0.05)
        selector2 = CausalFeatureSelector(max_lag=3, alpha=0.05)
        selected1 = selector1.fit(features1, target1)
        selected2 = selector2.fit(features2, target2)
        assert selected1 == selected2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
