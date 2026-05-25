"""Test dimensional invariants of the FeaturePipeline."""

import numpy as np
import pandas as pd
from rts_ai_fx.features_unified import FeaturePipeline, EXPECTED_FEATURE_DIM


def test_feature_dim_is_49():
    """EXPECTED_FEATURE_DIM must remain 49 for PPO dimensional compatibility."""
    assert EXPECTED_FEATURE_DIM == 49, (
        f"PPO regime agents are hardcoded to state_dim=49. "
        f"Changing EXPECTED_FEATURE_DIM to {EXPECTED_FEATURE_DIM} "
        f"requires retraining all models."
    )


def test_feature_pipeline_output_has_correct_dim():
    """FeaturePipeline.transform must output EXPECTED_FEATURE_DIM features."""
    pipeline = FeaturePipeline(lookback=30, timeframes=["1h"])

    # Create minimal OHLCV data
    n = 100
    df = pd.DataFrame(
        {
            "timestamp": np.arange(n),
            "open": np.random.uniform(1.0, 1.1, n),
            "high": np.random.uniform(1.01, 1.12, n),
            "low": np.random.uniform(0.98, 1.09, n),
            "close": np.random.uniform(1.0, 1.11, n),
            "volume": np.random.randint(1000, 10000, n),
        }
    )

    dfs = {"1h": df}
    result = pipeline.transform(dfs, symbol="EURUSD")

    assert result is not None, "Pipeline returned None"
    assert result.ndim == 2, f"Expected 2D output, got {result.ndim}D"
    assert result.shape[1] == EXPECTED_FEATURE_DIM, (
        f"Feature dimension mismatch: expected {EXPECTED_FEATURE_DIM}, "
        f"got {result.shape[1]}"
    )


def test_feature_dim_stable_after_normalization():
    """Feature dimension must stay consistent after fit/transform cycle."""
    pipeline = FeaturePipeline(lookback=30, timeframes=["1h"])

    n = 100
    df = pd.DataFrame(
        {
            "timestamp": np.arange(n),
            "open": np.random.uniform(1.0, 1.1, n),
            "high": np.random.uniform(1.01, 1.12, n),
            "low": np.random.uniform(0.98, 1.09, n),
            "close": np.random.uniform(1.0, 1.11, n),
            "volume": np.random.randint(1000, 10000, n),
        }
    )

    dfs = {"1h": df}
    pipeline.fit(dfs, symbol="EURUSD")
    result = pipeline.transform(dfs, symbol="EURUSD")

    assert result is not None
    assert result.shape[1] == EXPECTED_FEATURE_DIM
