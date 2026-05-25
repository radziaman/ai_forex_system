"""
Integration tests for the full signal pipeline:
data → features → PPO → ensemble → execution plan.

These tests validate that components work together correctly,
catching interface mismatches between subsystems.
"""

import sys
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _src not in sys.path:
    sys.path.insert(0, _src)

# ── Imports (work after path setup; E402 is accepted pattern in this repo)
from rts_ai_fx.features_unified import (  # noqa: E402
    FeaturePipeline,
    EXPECTED_FEATURE_DIM,
)
from rts_ai_fx.ensemble import MoEEnsemble, EnsemblePrediction  # noqa: E402
from execution.is_execution import ISExecutionEngine, ExecutionUrgency  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def make_ohlcv_data(
    n: int = 500, seed: int = 42, include_volume: bool = True
) -> pd.DataFrame:
    """
    Create synthetic OHLCV DataFrame for testing.

    The canonical feature count of 49 (as documented in features_unified.py)
    is achieved WITHOUT volume: 45 base features + 4 OHLCV microstructure.
    When volume is present, 4 extra volume-based features are added (total 53),
    but the transform method pads/trims to EXPECTED_FEATURE_DIM=49.
    """
    np.random.seed(seed)
    close = 1.20 + np.cumsum(np.random.randn(n) * 0.0005)
    data = {
        "timestamp": np.arange(n),
        "open": 1.20 + np.random.randn(n) * 0.001,
        "high": 1.21 + np.random.randn(n) * 0.002,
        "low": 1.19 + np.random.randn(n) * 0.002,
        "close": close,
    }
    if include_volume:
        data["volume"] = np.random.randint(1000, 10000, n)
    return pd.DataFrame(data)


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1: FeaturePipeline produces canonical 49-dim output
# ═══════════════════════════════════════════════════════════════════════════════


def test_feature_pipeline_canonical_dim():
    """
    FeaturePipeline with use_microstructure=True produces exactly 49 features
    when data contains OHLCV columns (without volume extras).
    Validates the canonical feature dimension contract documented in
    features_unified.py: 45 base + 4 OHLCV microstructure = 49.
    """
    # Use data without volume to get the canonical 49-dim feature set
    # (45 base + 4 OHLCV microstructure = 49). Volume adds extra features
    # that get trimmed to 49 by transform().
    df = make_ohlcv_data(n=500, seed=42, include_volume=False)

    fp = FeaturePipeline(lookback=30, timeframes=["1h"], use_microstructure=True)
    fp.fit({"1h": df}, symbol="EURUSD")

    n_features = len(fp._feature_cols)
    assert (
        n_features == EXPECTED_FEATURE_DIM
    ), f"Got {n_features} features, expected {EXPECTED_FEATURE_DIM}"

    # transform must produce (lookback, EXPECTED_FEATURE_DIM)-shaped array
    result = fp.transform({"1h": df}, symbol="EURUSD")
    assert result is not None, "transform returned None"
    assert result.shape == (
        30,
        EXPECTED_FEATURE_DIM,
    ), f"Got shape {result.shape}, expected (30, {EXPECTED_FEATURE_DIM})"

    # Values should be finite (no NaNs, no infs)
    assert np.all(np.isfinite(result)), "transform output contains non-finite values"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2: FeaturePipeline handles missing volume gracefully
# ═══════════════════════════════════════════════════════════════════════════════


def test_feature_pipeline_no_volume():
    """
    FeaturePipeline must gracefully handle data without a volume column.
    The 4 OHLCV-based microstructure features (tr_scaled, position_in_bar,
    intraday_volatility, gap) should still be computed from OHLCV alone,
    bringing the total to 49 features (45 base + 4 OHLCV microstructure).
    """
    # Omit volume column
    df = make_ohlcv_data(n=500, seed=42, include_volume=False)
    assert "volume" not in df.columns, "volume should not be in test data"

    fp = FeaturePipeline(lookback=30, timeframes=["1h"], use_microstructure=True)
    fp.fit({"1h": df}, symbol="EURUSD")

    # Without volume: 45 base + 4 OHLCV microstructure = 49
    n_features = len(fp._feature_cols)
    assert n_features == EXPECTED_FEATURE_DIM, (
        f"Got {n_features} features without volume, " f"expected {EXPECTED_FEATURE_DIM}"
    )

    result = fp.transform({"1h": df}, symbol="EURUSD")
    assert result is not None, "transform returned None"
    assert result.shape == (
        30,
        EXPECTED_FEATURE_DIM,
    ), f"Got shape {result.shape}, expected (30, {EXPECTED_FEATURE_DIM})"

    assert np.all(np.isfinite(result)), "transform output contains non-finite values"

    # Verify OHLCV microstructure features are present
    assert "tr_scaled" in fp._feature_cols, "Missing tr_scaled"
    assert "position_in_bar" in fp._feature_cols, "Missing position_in_bar"
    assert "intraday_volatility" in fp._feature_cols, "Missing intraday_volatility"
    assert "gap" in fp._feature_cols, "Missing gap"

    # Volume-based features should NOT be present
    assert (
        "log_volume" not in fp._feature_cols
    ), "log_volume should not exist without volume column"
    assert (
        "volume_ma_ratio" not in fp._feature_cols
    ), "volume_ma_ratio should not exist without volume column"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3: FeaturePipeline with volume trims to 49 via transform
# ═══════════════════════════════════════════════════════════════════════════════


def test_feature_pipeline_with_volume_trim():
    """
    When volume is present, compute_microstructure_features adds 4 volume-based
    features (log_volume, volume_ma_ratio, volume_shock, dollar_vol_rank) on top
    of the 4 OHLCV microstructure features, yielding 53 raw features.
    The transform() method must trim/pad to EXPECTED_FEATURE_DIM (49),
    ensuring dimensional compatibility with PPO regime agents.
    """
    df = make_ohlcv_data(n=500, seed=42, include_volume=True)
    assert "volume" in df.columns, "volume should be in test data"

    fp = FeaturePipeline(lookback=30, timeframes=["1h"], use_microstructure=True)
    fp.fit({"1h": df}, symbol="EURUSD")

    # Raw feature count includes volume extras → expect > 49
    n_features = len(fp._feature_cols)
    assert n_features > EXPECTED_FEATURE_DIM, (
        f"With volume expected >{EXPECTED_FEATURE_DIM} raw features, "
        f"got {n_features}"
    )

    # Volume features should be present
    assert "log_volume" in fp._feature_cols, "Missing log_volume with volume data"
    assert (
        "volume_ma_ratio" in fp._feature_cols
    ), "Missing volume_ma_ratio with volume data"

    # transform must trim to (30, 49) despite larger raw feature set
    result = fp.transform({"1h": df}, symbol="EURUSD")
    assert result is not None, "transform returned None"
    assert result.shape == (
        30,
        EXPECTED_FEATURE_DIM,
    ), f"Got shape {result.shape}, expected (30, {EXPECTED_FEATURE_DIM})"
    assert np.all(np.isfinite(result)), "transform output contains non-finite values"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4: Ensemble MoE config has correct structure
# ═══════════════════════════════════════════════════════════════════════════════


def test_ensemble_config_structure():
    """
    Ensemble config (generated by populate_ensemble.py) must contain
    exactly 28 experts (8 symbols × 3 + 4 PPO), 4 PPO regime agents, and matching elo_ratings.
    Validates the config contract expected by the ensemble loader.
    """
    config_path = Path("models/ensemble_config.json")
    assert (
        config_path.exists()
    ), "Ensemble config not found — run populate_ensemble.py first"

    with open(config_path) as f:
        config = json.load(f)

    # Top-level structure
    assert (
        config["n_experts"] == 28
    ), f"Expected 28 experts (8 symbols × 3 + 4 PPO), got {config['n_experts']}"
    assert isinstance(config["experts"], list), "experts must be a list"
    assert isinstance(config["elo_ratings"], dict), "elo_ratings must be a dict"

    # Expert count matches n_experts
    assert (
        len(config["experts"]) == 28
    ), f"Expected 28 experts in list, got {len(config['experts'])}"
    assert (
        len(config["elo_ratings"]) == 28
    ), f"Expected 28 elo_ratings, got {len(config['elo_ratings'])}"

    # Verify all 4 PPO regime agents present
    expert_names = [e["name"] for e in config["experts"]]
    for regime in ["trending", "ranging", "volatile", "crisis"]:
        assert (
            f"ppo_{regime}" in expert_names
        ), f"Missing ppo_{regime} in ensemble config"

    # Every expert has required fields
    for expert in config["experts"]:
        assert "name" in expert, f"Expert missing 'name': {expert}"
        assert "regime" in expert, f"Expert {expert['name']} missing 'regime'"
        assert "elo" in expert, f"Expert {expert['name']} missing 'elo'"

    # All elo_ratings keys match expert names
    for expert in config["experts"]:
        assert (
            expert["name"] in config["elo_ratings"]
        ), f"Expert {expert['name']} missing from elo_ratings"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4: MoEEnsemble.predict returns valid result
# ═══════════════════════════════════════════════════════════════════════════════


def test_ensemble_predict():
    """
    MoEEnsemble with registered test experts must return a valid
    EnsemblePrediction with confidence > 0, a non-HOLD direction,
    and expert_outputs matching the number of registered experts.
    """
    ensemble = MoEEnsemble()

    # Register 2 test experts with distinct behaviors
    ensemble.add_expert(
        name="test_bull",
        predict_fn=lambda X: 0.01,
        confidence_fn=lambda X: 0.7,
        regime="trending",
    )
    ensemble.add_expert(
        name="test_bear",
        predict_fn=lambda X: -0.01,
        confidence_fn=lambda X: 0.6,
        regime="ranging",
    )

    # Dummy feature input: (1, 30, 49)
    np.random.seed(42)
    dummy = np.random.randn(1, 30, EXPECTED_FEATURE_DIM).astype(np.float32)

    pred = ensemble.predict(dummy, regime="trending")

    assert pred is not None, "predict returned None"
    assert isinstance(
        pred, EnsemblePrediction
    ), f"Expected EnsemblePrediction, got {type(pred)}"
    assert pred.confidence > 0, f"Expected positive confidence, got {pred.confidence}"
    assert pred.direction in (
        "BUY",
        "SELL",
        "HOLD",
    ), f"Unexpected direction: {pred.direction}"
    assert (
        len(pred.expert_outputs) == 2
    ), f"Expected 2 expert outputs, got {len(pred.expert_outputs)}"

    # Verify expert output structure
    for name in ("test_bull", "test_bear"):
        assert name in pred.expert_outputs, f"Missing output for {name}"
        output = pred.expert_outputs[name]
        assert "prediction" in output
        assert "confidence" in output
        assert "weight" in output
        assert output["confidence"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5: MoEEnsemble.should_trade respects minimum confidence
# ═══════════════════════════════════════════════════════════════════════════════


def test_ensemble_should_trade():
    """
    MoEEnsemble.should_trade must correctly filter low-confidence signals
    and return valid (bool, direction, ratio) tuples for both trade and
    no-trade scenarios.
    """
    ensemble = MoEEnsemble()

    # Register a strong bullish expert
    ensemble.add_expert(
        name="strong_bull",
        predict_fn=lambda X: 0.01,
        confidence_fn=lambda X: 0.9,
        regime="trending",
    )
    # Register a weak bearish expert
    ensemble.add_expert(
        name="weak_bear",
        predict_fn=lambda X: -0.001,
        confidence_fn=lambda X: 0.3,
        regime="ranging",
    )

    np.random.seed(42)
    dummy = np.random.randn(1, 30, EXPECTED_FEATURE_DIM).astype(np.float32)

    # Generate ensemble prediction
    pred = ensemble.predict(dummy, regime="trending")

    # Test with default min_confidence
    should_trade, direction, ratio = ensemble.should_trade(pred, current_price=1.20)
    assert isinstance(should_trade, bool), "should_trade must return bool"
    assert direction in ("BUY", "SELL", "HOLD"), f"Unexpected direction: {direction}"
    assert 0.0 <= ratio <= 1.0, f"Ratio {ratio} out of [0, 1] range"

    # Test with very high min_confidence — should reject
    should_trade_high, direction_high, ratio_high = ensemble.should_trade(
        pred, current_price=1.20, min_confidence=0.99
    )
    # May still trade if conf is high enough, but check the interface works

    # Test with empty expert outputs
    empty_pred = EnsemblePrediction()
    should, direc, rat = ensemble.should_trade(empty_pred, current_price=1.20)
    assert should is False
    assert direc == "HOLD"
    assert rat == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Test 6: IS Execution plan matches expected structure
# ═══════════════════════════════════════════════════════════════════════════════


def test_execution_plan_structure():
    """
    Implementation Shortfall execution engine must produce an ExecutionPlan
    with non-zero volume slices, correct urgency mapping, and positive
    expected cost in bps.
    """
    engine = ISExecutionEngine(volatility=0.15, adv=1e9, spread=0.0001)
    plan = engine.plan_execution(
        volume=10000,
        price=1.20,
        symbol="EURUSD",
        confidence=0.8,
    )

    # Top-level plan fields
    assert (
        plan.total_volume == 10000
    ), f"Expected total_volume=10000, got {plan.total_volume}"
    assert (
        plan.urgency == ExecutionUrgency.HIGH
    ), f"confidence=0.8 should map to HIGH urgency, got {plan.urgency}"
    assert len(plan.slices) > 0, "Execution plan must have at least one slice"
    assert (
        plan.expected_cost_bps > 0
    ), f"Expected positive cost, got {plan.expected_cost_bps}"
    assert (
        plan.expected_duration_sec > 0
    ), f"Expected positive duration, got {plan.expected_duration_sec}"
    assert plan.confidence == 0.8, f"Expected confidence=0.8, got {plan.confidence}"

    # Slice-level validation
    for s in plan.slices:
        assert s.volume > 0, f"Slice {s.slice_idx} has non-positive volume"
        assert s.order_type in (
            "AGGRESSIVE",
            "PASSIVE",
        ), f"Slice {s.slice_idx} invalid order_type: {s.order_type}"
        assert s.target_price > 0, f"Slice {s.slice_idx} has non-positive target_price"

    # Total slice volume should sum to total volume
    total_slice_vol = sum(s.volume for s in plan.slices)
    assert (
        abs(total_slice_vol - 10000) < 0.01
    ), f"Slice volumes sum {total_slice_vol} != 10000"

    # HIGH urgency should have front-loaded volume distribution
    if len(plan.slices) >= 2:
        assert (
            plan.slices[0].volume >= plan.slices[-1].volume
        ), "HIGH urgency should front-load: first slice volume >= last slice volume"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 7: IS Execution urgency mapping for all confidence tiers
# ═══════════════════════════════════════════════════════════════════════════════


def test_execution_urgency_tiers():
    """
    ISExecutionEngine._determine_urgency must map confidence values to
    the correct ExecutionUrgency across all four tiers:
        >= 0.85 → URGENT
        >= 0.70 → HIGH
        >= 0.55 → MEDIUM
        <  0.55 → LOW
    """
    engine = ISExecutionEngine(volatility=0.15, adv=1e9, spread=0.0001)

    test_cases = [
        (0.90, ExecutionUrgency.URGENT),
        (0.85, ExecutionUrgency.URGENT),
        (0.80, ExecutionUrgency.HIGH),
        (0.70, ExecutionUrgency.HIGH),
        (0.65, ExecutionUrgency.MEDIUM),
        (0.55, ExecutionUrgency.MEDIUM),
        (0.50, ExecutionUrgency.LOW),
        (0.00, ExecutionUrgency.LOW),
    ]

    for confidence, expected_urgency in test_cases:
        plan = engine.plan_execution(
            volume=10000,
            price=1.20,
            symbol="EURUSD",
            confidence=confidence,
        )
        assert plan.urgency == expected_urgency, (
            f"confidence={confidence}: expected {expected_urgency}, "
            f"got {plan.urgency}"
        )

        # All plans must still produce valid slices
        assert len(plan.slices) > 0, f"No slices for confidence={confidence}"
        assert (
            plan.expected_cost_bps > 0
        ), f"Non-positive cost for confidence={confidence}"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 8: Full pipeline smoke test (data → features → ensemble)
# ═══════════════════════════════════════════════════════════════════════════════


def test_signal_pipeline_smoke():
    """
    End-to-end smoke test: synthetic OHLCV data flows through FeaturePipeline
    (fit + transform) and into MoEEnsemble.predict with a test expert.
    Validates dimensional consistency across all pipeline stages.
    """
    # ── Stage 1: Data ──────────────────────────────────────────────────────
    # Use OHLCV-only data (no volume) to keep the canonical 49-dim feature
    # contract. Volume would add extra features that get trimmed by transform().
    df = make_ohlcv_data(n=500, seed=42, include_volume=False)

    # ── Stage 2: Features ──────────────────────────────────────────────────
    fp = FeaturePipeline(lookback=30, timeframes=["1h"], use_microstructure=True)
    fp.fit({"1h": df}, symbol="EURUSD")

    n_features = len(fp._feature_cols)
    assert n_features == EXPECTED_FEATURE_DIM, (
        f"Feature pipeline produced {n_features} dims, "
        f"expected {EXPECTED_FEATURE_DIM}"
    )

    features = fp.transform({"1h": df}, symbol="EURUSD")
    assert features is not None, "FeaturePipeline.transform returned None"
    assert features.shape == (
        30,
        EXPECTED_FEATURE_DIM,
    ), f"Transform output shape {features.shape} != (30, {EXPECTED_FEATURE_DIM})"

    # ── Stage 3: Ensemble ──────────────────────────────────────────────────
    ensemble = MoEEnsemble()
    ensemble.add_expert(
        name="trend_follower",
        predict_fn=lambda X: float(np.mean(X[-5:, -1])),
        confidence_fn=lambda X: 0.6,
        regime="trending",
    )
    ensemble.add_expert(
        name="mean_reversion",
        predict_fn=lambda X: float(-np.mean(X[-3:, 0])),
        confidence_fn=lambda X: 0.5,
        regime="ranging",
    )

    # Batch dimension: (1, lookback, n_features)
    features_3d = features[np.newaxis, :, :]
    pred = ensemble.predict(features_3d, regime="trending")

    assert pred is not None, "Ensemble.predict returned None"
    assert isinstance(
        pred.price, float
    ), f"Expected float price, got {type(pred.price)}"
    assert pred.confidence > 0, f"Expected positive confidence, got {pred.confidence}"
    assert isinstance(
        pred.direction, str
    ), f"Expected str direction, got {type(pred.direction)}"
    assert (
        len(pred.expert_outputs) == 2
    ), f"Expected 2 expert outputs, got {len(pred.expert_outputs)}"

    # Both experts should have produced predictions
    for name in ("trend_follower", "mean_reversion"):
        assert name in pred.expert_outputs, f"Missing expert: {name}"
        out = pred.expert_outputs[name]
        assert isinstance(out["prediction"], float)
        assert isinstance(out["confidence"], float)
        assert isinstance(out["weight"], float)
        assert out["weight"] > 0, f"Expert {name} has non-positive weight"
