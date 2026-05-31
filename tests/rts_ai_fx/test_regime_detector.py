"""
Comprehensive tests for HMMRegimeDetector and SimpleRegimeDetector.
"""

import math

import numpy as np
import pandas as pd
import pytest

from rts_ai_fx.regime_detector import (
    HMMRegimeDetector,
    HMM_AVAILABLE,
    SimpleRegimeDetector,
)


# ── Helper ────────────────────────────────────────────────────────────


def _make_df(adx=None, atr=None, close=1.12):
    data = {"close": [close] * 100}
    if adx is not None:
        data["adx_14"] = [adx] * 100
    if atr is not None:
        data["atr_14"] = [atr] * 100
    return pd.DataFrame(data)


# ══════════════════════════════════════════════════════════════════════
# SimpleRegimeDetector
# ══════════════════════════════════════════════════════════════════════


class TestSimpleDetect:
    """SimpleRegimeDetector.detect() — ADX/ATR based, no ML."""

    def test_simple_detect_trending(self):
        """ADX >= 20 → labels as trending."""
        df = _make_df(adx=25)
        detector = SimpleRegimeDetector(adx_threshold=20.0)
        labels, names = detector.detect(df)

        assert len(labels) == 100
        assert len(names) == 100
        assert np.all(labels == 3), f"Expected all trending(3), got {np.unique(labels)}"
        assert np.all(names == "trending")

    def test_simple_detect_ranging(self):
        """ADX < 20 → labels as ranging."""
        df = _make_df(adx=15)
        detector = SimpleRegimeDetector(adx_threshold=20.0)
        labels, names = detector.detect(df)

        assert np.all(labels == 1), f"Expected all ranging(1), got {np.unique(labels)}"
        assert np.all(names == "ranging")

    def test_simple_detect_crisis(self):
        """ATR/price > 0.02 → labels as crisis (takes priority over ADX)."""
        # ATR=0.025 / close=1.0 → 0.025 > 0.02 → crisis
        df = _make_df(adx=30, atr=0.025, close=1.0)
        detector = SimpleRegimeDetector(crisis_atr_pct=0.02)
        labels, names = detector.detect(df)

        assert np.all(labels == 0), f"Expected all crisis(0), got {np.unique(labels)}"
        assert np.all(names == "crisis")

    def test_simple_detect_volatile(self):
        """ATR/price between 0.01 and 0.02 → labels as volatile."""
        # ATR=0.015 / close=1.0 → 0.015 > 0.01 and 0.015 <= 0.02 → volatile
        df = _make_df(atr=0.015, close=1.0)
        detector = SimpleRegimeDetector(crisis_atr_pct=0.02)
        labels, names = detector.detect(df)

        assert np.all(labels == 2), f"Expected all volatile(2), got {np.unique(labels)}"
        assert np.all(names == "volatile")

    def test_simple_detect_fallback(self):
        """Missing ADX/ATR columns → defaults to ranging."""
        # DataFrame with only close (no adx_14, no atr_14)
        df = _make_df()
        detector = SimpleRegimeDetector()
        labels, names = detector.detect(df)

        assert np.all(
            labels == 1
        ), f"Expected default ranging(1), got {np.unique(labels)}"
        assert np.all(names == "ranging")

    def test_simple_detect_fallback_no_close(self):
        """No close either → still defaults to ranging."""
        df = pd.DataFrame({"some_other_col": [1.0] * 100})
        detector = SimpleRegimeDetector()
        labels, names = detector.detect(df)

        assert np.all(labels == 1)
        assert np.all(names == "ranging")

    def test_simple_returns_names_array(self):
        """Returns both labels (int) and names (str) arrays of same length."""
        df = _make_df(adx=25, atr=0.01, close=1.12)
        detector = SimpleRegimeDetector()
        labels, names = detector.detect(df)

        assert isinstance(labels, np.ndarray)
        assert isinstance(names, np.ndarray)
        assert labels.dtype == int or labels.dtype == np.int64
        assert names.dtype == object or names.dtype.kind == "U"
        assert len(labels) == len(names) == 100

    def test_simple_volatile_bounds_edge(self):
        """ATR/price exactly at threshold boundaries."""
        # Exactly at crisis_atr_pct / 2 → should NOT be volatile (must be >)
        df = _make_df(atr=0.01, close=1.0)  # atr_ratio = 0.01 exactly
        detector = SimpleRegimeDetector(crisis_atr_pct=0.02)
        labels, names = detector.detect(df)

        # atr_ratio=0.01 is NOT > 0.01, so it's not volatile
        # No ADX column, so falls to default ranging
        assert np.all(
            labels == 1
        ), f"Expected ranging(1) at boundary, got {np.unique(labels)}"

    def test_simple_nan_atr(self):
        """NaN ATR values → fallback to ranging, no crash."""
        df = _make_df(adx=15, atr=float("nan"), close=1.12)
        detector = SimpleRegimeDetector()
        labels, names = detector.detect(df)

        # NaN atr → pd.isna is True → label=1, name="ranging", continue
        # So adx check never reached
        assert np.all(labels == 1)
        assert np.all(names == "ranging")

    def test_simple_nan_adx(self):
        """NaN ADX → not trending, defaults to ranging (no ATR crisis)."""
        df = _make_df(adx=float("nan"), atr=0.005, close=1.12)
        detector = SimpleRegimeDetector()
        labels, names = detector.detect(df)

        # atr_ratio = 0.005/1.12 ≈ 0.00446 < 0.01 → not volatile, not crisis
        # adx is NaN → not pd.isna check fails → not trending
        # So defaults to ranging
        assert np.all(labels == 1)
        assert np.all(names == "ranging")

    def test_simple_zero_close(self):
        """Close == 0 → skip crisis/volatile, fall through."""
        df = _make_df(adx=30, atr=0.03, close=0.0)
        detector = SimpleRegimeDetector()
        labels, names = detector.detect(df)

        # close_val <= 0 → label=1, name="ranging", continue
        # So adx check is skipped (continue)
        assert np.all(labels == 1)
        assert np.all(names == "ranging")

    def test_simple_custom_thresholds(self):
        """Custom adx_threshold and crisis_atr_pct."""
        df = _make_df(adx=18, atr=0.03, close=1.0)
        # Low threshold: ADX 18 >= 15 → trending
        # But ATR/close = 0.03 > 0.025 → crisis takes priority
        detector = SimpleRegimeDetector(adx_threshold=15.0, crisis_atr_pct=0.025)
        labels, names = detector.detect(df)

        assert np.all(labels == 0), "Crisis should take priority with custom threshold"
        assert np.all(names == "crisis")


# ══════════════════════════════════════════════════════════════════════
# HMMRegimeDetector
# ══════════════════════════════════════════════════════════════════════

hmm_test = pytest.mark.skipif(not HMM_AVAILABLE, reason="hmmlearn not installed")


class TestHMMInit:
    """HMMRegimeDetector construction and defaults."""

    def test_hmm_init(self):
        """Default parameters on construction."""
        detector = HMMRegimeDetector()
        assert detector.n_regimes == 4
        assert detector.lookback == 60
        assert detector.model is None
        assert detector.current_regime is None
        assert detector.regime_history == []

    def test_hmm_init_custom(self):
        """Custom parameters."""
        detector = HMMRegimeDetector(n_regimes=3, lookback=30)
        assert detector.n_regimes == 3
        assert detector.lookback == 30

    def test_hmm_regime_names(self):
        """REGIME_NAMES is a class constant with 4 names."""
        assert HMMRegimeDetector.REGIME_NAMES == [
            "crisis",
            "ranging",
            "volatile",
            "trending",
        ]


class TestHMMFeatures:
    """_extract_features shape and content."""

    def test_hmm_extract_features(self):
        """_extract_features returns correct shape from standard df."""
        df = _make_df(adx=25, atr=0.01, close=1.12)
        detector = HMMRegimeDetector()
        features = detector._extract_features(df)

        # diff reduces by 1, 8 feature columns
        assert features.shape == (99, 8), f"Expected (99, 8), got {features.shape}"
        assert not np.any(np.isnan(features)), "NaN in extracted features"
        assert not np.any(np.isinf(features)), "Inf in extracted features"

    def test_hmm_extract_features_no_indicators(self):
        """_extract_features works with only close price."""
        df = pd.DataFrame({"close": np.linspace(1.10, 1.15, 50)})
        detector = HMMRegimeDetector()
        features = detector._extract_features(df)

        assert features.shape == (49, 8), f"Expected (49, 8), got {features.shape}"
        assert np.all(np.isfinite(features))

    def test_hmm_extract_features_minimal_rows(self):
        """Very short df still produces valid (small) features."""
        df = pd.DataFrame({"close": [1.10, 1.11, 1.12]})
        detector = HMMRegimeDetector()
        features = detector._extract_features(df)

        # 3 rows → 2 returns → (2, 8)
        assert features.shape == (2, 8)

    def test_hmm_extract_features_none(self):
        """None df returns empty array."""
        detector = HMMRegimeDetector()
        features = detector._extract_features(None)
        assert isinstance(features, np.ndarray)
        assert features.size == 0

    def test_hmm_extract_features_with_rsi_bb(self):
        """Extra columns (rsi_14, bb_width) are consumed without error."""
        df = _make_df(adx=25, atr=0.01, close=1.12)
        df["rsi_14"] = 55.0
        df["bb_width"] = 0.025
        detector = HMMRegimeDetector()
        features = detector._extract_features(df)

        assert features.shape == (99, 8)
        assert np.all(np.isfinite(features))


@hmm_test
class TestHMMDetect:
    """HMM-based detect and fallback behaviour."""

    def test_hmm_detect_regime_unfit(self):
        """Not fitted → falls back to rule-based regime detection."""
        df = _make_df(adx=25, atr=0.005, close=1.12)
        detector = HMMRegimeDetector()
        regime = detector.detect_regime(df)

        # _fallback_regime with adx=25, default bb=0.02, default rvol=1.0
        # adx=25 not > 30, bb=0.02 not > 0.03, adx=25 not < 20, rvol=1.0 not > 1.5
        # → "ranging"
        assert regime == "ranging"
        assert detector.current_regime == "ranging"
        assert detector.regime_history == ["ranging"]

    def test_hmm_detect_regime_short_df(self):
        """Very short DataFrame (< 10 rows) → fallback returns ranging."""
        df = pd.DataFrame({"close": [1.10, 1.11, 1.12]})
        detector = HMMRegimeDetector()
        regime = detector.detect_regime(df)

        # len(recent) = 3 < 10 → "ranging"
        assert regime == "ranging"

    def test_hmm_fallback_crisis(self):
        """Fallback detects crisis when rvol > 2."""
        df = _make_df(adx=15, atr=0.005, close=1.12)
        df["vol_ratio"] = 3.0
        detector = HMMRegimeDetector()
        regime = detector.detect_regime(df)

        assert regime == "crisis"

    def test_hmm_fallback_trending(self):
        """Fallback detects trending when adx > 30 and bb_width > 0.03."""
        df = _make_df(adx=35, atr=0.005, close=1.12)
        df["bb_width"] = 0.04
        detector = HMMRegimeDetector()
        regime = detector.detect_regime(df)

        assert regime == "trending"

    def test_hmm_fallback_ranging(self):
        """Fallback detects ranging when adx < 20 and bb_width < 0.02."""
        df = _make_df(adx=15, atr=0.005, close=1.12)
        df["bb_width"] = 0.015
        detector = HMMRegimeDetector()
        regime = detector.detect_regime(df)

        assert regime == "ranging"


class TestHMMRegimeParams:
    """get_regime_params returns correct parameters."""

    def test_hmm_get_regime_params_trending(self):
        detector = HMMRegimeDetector()
        params = detector.get_regime_params("trending")
        assert params == {
            "sl_atr": 2.0,
            "tp_atr": 4.0,
            "pos_mult": 1.0,
            "min_conf": 0.60,
        }

    def test_hmm_get_regime_params_ranging(self):
        detector = HMMRegimeDetector()
        params = detector.get_regime_params("ranging")
        assert params == {
            "sl_atr": 1.5,
            "tp_atr": 3.0,
            "pos_mult": 1.0,
            "min_conf": 0.65,
        }

    def test_hmm_get_regime_params_volatile(self):
        detector = HMMRegimeDetector()
        params = detector.get_regime_params("volatile")
        assert params == {
            "sl_atr": 2.5,
            "tp_atr": 5.0,
            "pos_mult": 0.5,
            "min_conf": 0.75,
        }

    def test_hmm_get_regime_params_crisis(self):
        detector = HMMRegimeDetector()
        params = detector.get_regime_params("crisis")
        assert params == {
            "sl_atr": 1.0,
            "tp_atr": 2.0,
            "pos_mult": 0.0,
            "min_conf": 0.95,
        }

    def test_hmm_get_regime_params_unknown(self):
        """Unknown regime defaults to ranging."""
        detector = HMMRegimeDetector()
        params = detector.get_regime_params("unknown_regime")
        assert params == detector.get_regime_params("ranging")


class TestHMMShouldTrade:
    """should_trade decision per regime."""

    def test_hmm_should_trade_crisis(self):
        detector = HMMRegimeDetector()
        assert detector.should_trade("crisis") is False

    def test_hmm_should_trade_trending(self):
        detector = HMMRegimeDetector()
        assert detector.should_trade("trending") is True

    def test_hmm_should_trade_ranging(self):
        detector = HMMRegimeDetector()
        assert detector.should_trade("ranging") is True

    def test_hmm_should_trade_volatile(self):
        detector = HMMRegimeDetector()
        assert detector.should_trade("volatile") is True


class TestHMMPositionSize:
    """get_position_size_multiplier with sentiment adjustment."""

    def test_hmm_position_size_multiplier_neutral(self):
        """Neutral sentiment → 0.5x base for trending (1.0 * 0.5 = 0.5)."""
        detector = HMMRegimeDetector()
        mult = detector.get_position_size_multiplier("trending", sentiment_score=0.0)
        assert math.isclose(mult, 0.5, rel_tol=1e-6)

    def test_hmm_position_size_multiplier_positive(self):
        """Full positive sentiment → 1.0x multiplier."""
        detector = HMMRegimeDetector()
        mult = detector.get_position_size_multiplier("trending", sentiment_score=1.0)
        assert math.isclose(mult, 1.0, rel_tol=1e-6)

    def test_hmm_position_size_multiplier_negative(self):
        """Full negative sentiment → 0.0x multiplier."""
        detector = HMMRegimeDetector()
        mult = detector.get_position_size_multiplier("trending", sentiment_score=-1.0)
        assert math.isclose(mult, 0.0, rel_tol=1e-6)

    def test_hmm_position_size_multiplier_volatile(self):
        """Volatile base (0.5) with neutral sentiment → 0.25."""
        detector = HMMRegimeDetector()
        mult = detector.get_position_size_multiplier("volatile", sentiment_score=0.0)
        assert math.isclose(mult, 0.25, rel_tol=1e-6)

    def test_hmm_position_size_multiplier_crisis(self):
        """Crisis base (0.0) → always 0.0 regardless of sentiment."""
        detector = HMMRegimeDetector()
        mult = detector.get_position_size_multiplier("crisis", sentiment_score=0.8)
        assert math.isclose(mult, 0.0, abs_tol=1e-6)

    def test_hmm_position_size_sentiment_clipped(self):
        """Sentiment scores outside [-1, 1] are clipped."""
        detector = HMMRegimeDetector()
        m1 = detector.get_position_size_multiplier("trending", sentiment_score=2.0)
        m2 = detector.get_position_size_multiplier("trending", sentiment_score=1.0)
        assert math.isclose(m1, m2, rel_tol=1e-6)

        m3 = detector.get_position_size_multiplier("trending", sentiment_score=-2.0)
        m4 = detector.get_position_size_multiplier("trending", sentiment_score=-1.0)
        assert math.isclose(m3, m4, rel_tol=1e-6)

    def test_hmm_position_size_partial_sentiment(self):
        """Sentiment score of 0.5 → 0.5 + 0.5 * 0.5 = 0.75."""
        detector = HMMRegimeDetector()
        mult = detector.get_position_size_multiplier("trending", sentiment_score=0.5)
        assert math.isclose(mult, 0.75, rel_tol=1e-6)


class TestHMMTransition:
    """Enhancement #12: Regime transition detection."""

    def test_hmm_transition_detection_no_history(self):
        """No history → no transition."""
        detector = HMMRegimeDetector()
        result = detector.detect_transition()
        assert result["in_transition"] is False
        assert result["from_regime"] is None
        assert result["to_regime"] is None

    def test_hmm_transition_detection_single(self):
        """Only one entry → no transition (need 2+)."""
        detector = HMMRegimeDetector()
        detector.regime_history = ["ranging"]
        result = detector.detect_transition()
        assert result["in_transition"] is False

    def test_hmm_transition_detection_no_change(self):
        """Same regime twice → not a transition."""
        detector = HMMRegimeDetector()
        detector.regime_history = ["ranging", "ranging"]
        result = detector.detect_transition()
        assert result["in_transition"] is False
        assert result["from_regime"] == "ranging"
        assert result["to_regime"] == "ranging"

    def test_hmm_transition_detection(self):
        """Different regimes → detects transition."""
        detector = HMMRegimeDetector()
        detector.regime_history = ["ranging", "trending"]
        result = detector.detect_transition()

        assert result["in_transition"] is True
        assert result["from_regime"] == "ranging"
        assert result["to_regime"] == "trending"
        assert result["transition_key"] == "ranging_to_trending"
        assert result["should_trade"] is True

    def test_hmm_transition_volatile_to_crisis_no_trade(self):
        """Volatile → Crisis: should_trade is False (too dangerous)."""
        detector = HMMRegimeDetector()
        detector.regime_history = ["volatile", "crisis"]
        result = detector.detect_transition()

        assert result["in_transition"] is True
        assert result["should_trade"] is False

    def test_hmm_transition_crisis_to_ranging_no_trade(self):
        """Crisis → Ranging: should_trade is False (wait for stability)."""
        detector = HMMRegimeDetector()
        detector.regime_history = ["crisis", "ranging"]
        result = detector.detect_transition()

        assert result["in_transition"] is True
        assert result["should_trade"] is False

    def test_hmm_transition_sl_tp_adjustment_volatile(self):
        """Transitions involving volatile → sl_tp_adj = 1.5."""
        detector = HMMRegimeDetector()
        detector.regime_history = ["trending", "volatile"]
        result = detector.detect_transition()
        assert math.isclose(result["sl_tp_adjustment"], 1.5, rel_tol=1e-6)

    def test_hmm_transition_sl_tp_adjustment_trend_range(self):
        """Trending↔Ranging transitions → sl_tp_adj = 1.2."""
        detector = HMMRegimeDetector()

        # trending → ranging
        detector.regime_history = ["trending", "ranging"]
        result = detector.detect_transition()
        assert math.isclose(result["sl_tp_adjustment"], 1.2, rel_tol=1e-6)

        # ranging → trending
        detector.regime_history = ["ranging", "trending"]
        result = detector.detect_transition()
        assert math.isclose(result["sl_tp_adjustment"], 1.2, rel_tol=1e-6)

    def test_hmm_transition_sl_tp_adjustment_other(self):
        """Other transitions → sl_tp_adj = 1.3."""
        detector = HMMRegimeDetector()
        detector.regime_history = ["crisis", "trending"]
        result = detector.detect_transition()
        assert math.isclose(result["sl_tp_adjustment"], 1.3, rel_tol=1e-6)

    def test_hmm_transition_sl_tp_adjustment_no_transition(self):
        """No transition → sl_tp_adj = 1.0."""
        detector = HMMRegimeDetector()
        detector.regime_history = ["trending", "trending"]
        result = detector.detect_transition()
        assert math.isclose(result["sl_tp_adjustment"], 1.0, rel_tol=1e-6)

    def test_hmm_transition_confidence(self):
        """Confidence > 0 when in transition, based on transition frequency."""
        detector = HMMRegimeDetector()
        # Many same-regime entries → low transition frequency → high confidence
        detector.regime_history = ["ranging"] * 9 + ["trending"]
        result = detector.detect_transition()

        assert result["in_transition"] is True
        assert result["confidence"] > 0.0
        assert result["confidence"] <= 1.0


class TestHMMPause:
    """get_pause_conditions — high uncertainty pause logic."""

    def test_hmm_pause_conditions_normal(self):
        """Normal conditions (stable regime) → no pause."""
        detector = HMMRegimeDetector()
        detector.regime_history = ["ranging"] * 5
        result = detector.get_pause_conditions()

        assert result["should_pause"] is False
        assert result["reason"] == "normal_conditions"

    def test_hmm_pause_crisis(self):
        """Crisis regime → pause with reason 'crisis_regime_detected'."""
        detector = HMMRegimeDetector()
        detector.regime_history = ["ranging", "ranging", "ranging", "ranging", "crisis"]
        result = detector.get_pause_conditions()

        assert result["should_pause"] is True
        assert "crisis_regime" in result["reason"]
        assert result["regime"] == "crisis"

    def test_hmm_pause_frequent_transitions(self):
        """3+ transitions in 5 periods → pause."""
        detector = HMMRegimeDetector()
        # 4 transitions: ranging→trending→ranging→trending→ranging
        detector.regime_history = [
            "ranging",
            "trending",
            "ranging",
            "trending",
            "ranging",
        ]
        result = detector.get_pause_conditions()

        assert result["should_pause"] is True
        assert "high_transition_frequency" in result["reason"]
        assert result["transitions"] >= 3

    def test_hmm_pause_insufficient_history(self):
        """Less than 5 entries → insufficient_history, no pause."""
        detector = HMMRegimeDetector()
        detector.regime_history = ["ranging", "ranging", "ranging"]
        result = detector.get_pause_conditions()

        assert result["should_pause"] is False
        assert result["reason"] == "insufficient_history"

    def test_hmm_pause_empty_history(self):
        """Empty history → insufficient_history."""
        detector = HMMRegimeDetector()
        result = detector.get_pause_conditions()
        assert result["should_pause"] is False
        assert result["reason"] == "insufficient_history"

    def test_hmm_pause_two_transitions_allowed(self):
        """2 transitions in 5 periods → no pause (< 3)."""
        detector = HMMRegimeDetector()
        # 2 transitions: ranging→trending→ranging
        detector.regime_history = [
            "ranging",
            "trending",
            "ranging",
            "ranging",
            "ranging",
        ]
        result = detector.get_pause_conditions()

        assert result["should_pause"] is False
        assert result["reason"] == "normal_conditions"


class TestHMMTransitionSignal:
    """get_transition_trading_signal — actions per transition type."""

    def test_hmm_signal_no_transition(self):
        detector = HMMRegimeDetector()
        detector.regime_history = ["ranging", "ranging"]
        signal = detector.get_transition_trading_signal()
        assert signal["action"] == "HOLD"
        assert signal["reason"] == "no_transition"

    def test_hmm_signal_no_history(self):
        detector = HMMRegimeDetector()
        signal = detector.get_transition_trading_signal()
        assert signal["action"] == "HOLD"

    def test_hmm_signal_to_crisis(self):
        """Any transition TO crisis → CLOSE_ALL."""
        detector = HMMRegimeDetector()
        detector.regime_history = ["volatile", "crisis"]
        signal = detector.get_transition_trading_signal()
        assert signal["action"] == "CLOSE_ALL"
        assert signal["reason"] == "entering_crisis_regime"

    def test_hmm_signal_ranging_to_trending(self):
        """Ranging → Trending → breakout_from_range."""
        detector = HMMRegimeDetector()
        detector.regime_history = ["ranging", "trending"]
        signal = detector.get_transition_trading_signal()
        assert signal["action"] in ("BUY", "SELL")
        assert signal["reason"] == "breakout_from_range"

    def test_hmm_signal_volatile_to_ranging(self):
        """Volatile → Ranging → wait_for_range_confirmation."""
        detector = HMMRegimeDetector()
        detector.regime_history = ["volatile", "ranging"]
        signal = detector.get_transition_trading_signal()
        assert signal["action"] == "HOLD"
        assert signal["reason"] == "wait_for_range_confirmation"


class TestHMMFit:
    """HMM fitting — integration level."""

    @hmm_test
    def test_hmm_fit_sufficient_data(self):
        """Fitting with 200+ rows creates a model (need 100+ features)."""
        np.random.seed(42)
        n = 250
        # Use realistic random-walk prices + indicators so the HMM
        # covariance matrix is positive-definite (avoid zero-variance cols).
        close = 1.12 + np.cumsum(np.random.randn(n) * 0.002)
        close = np.clip(close, 1.0, 1.3)
        df = pd.DataFrame(
            {
                "close": close,
                "adx_14": np.random.uniform(15, 45, n),
                "atr_14": np.abs(np.random.randn(n) * 0.005 + 0.01),
                "rsi_14": np.random.uniform(20, 80, n),
                "bb_width": np.random.uniform(0.01, 0.06, n),
            }
        )
        detector = HMMRegimeDetector()
        detector.fit(df)

        assert detector.model is not None

    @hmm_test
    def test_hmm_fit_insufficient_data(self):
        """Fitting with < 100 rows does not create a model."""
        df = pd.DataFrame({"close": np.linspace(1.10, 1.15, 50)})
        detector = HMMRegimeDetector()
        detector.fit(df)

        assert detector.model is None

    def test_hmm_fit_no_hmm(self):
        """When HMM not available, fit is a no-op."""
        # This is always safe to call
        df = _make_df(adx=25, atr=0.01, close=1.12)
        detector = HMMRegimeDetector()
        detector.fit(df)


class TestHMMRegimeHistory:
    """Regime history tracking."""

    def test_hmm_history_appended_on_detect(self):
        """detect_regime appends to regime_history."""
        df = _make_df(adx=25, atr=0.005, close=1.12)
        detector = HMMRegimeDetector()

        detector.detect_regime(df)
        assert len(detector.regime_history) == 1

        detector.detect_regime(df)
        assert len(detector.regime_history) == 2

    def test_hmm_history_order(self):
        """Regime history maintains insertion order."""
        df1 = _make_df(adx=15, atr=0.005, close=1.12)
        df2 = _make_df(adx=35, atr=0.005, close=1.12)
        df2["bb_width"] = 0.04

        detector = HMMRegimeDetector()
        detector.detect_regime(df1)  # ranging
        detector.detect_regime(df2)  # trending (via fallback)

        assert detector.regime_history == ["ranging", "trending"]


class TestHMMGetTrendDirection:
    """_get_trend_direction helper."""

    def test_hmm_trend_direction_empty(self):
        detector = HMMRegimeDetector()
        assert detector._get_trend_direction() == 0

    def test_hmm_trend_direction_trending(self):
        detector = HMMRegimeDetector()
        detector.regime_history = ["trending"]
        # Provide upward-sloping prices so slope > 0 -> return 1
        detector._last_prices = np.linspace(1.10, 1.15, 10)
        assert detector._get_trend_direction() == 1

    def test_hmm_trend_direction_downtrend(self):
        detector = HMMRegimeDetector()
        detector.regime_history = ["trending"]
        # Provide downward-sloping prices so slope < 0 -> return -1
        detector._last_prices = np.linspace(1.15, 1.10, 10)
        assert detector._get_trend_direction() == -1

    def test_hmm_trend_direction_non_trending(self):
        detector = HMMRegimeDetector()
        detector.regime_history = ["ranging"]
        assert detector._get_trend_direction() == 0
