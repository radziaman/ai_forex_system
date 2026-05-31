"""Tests for ExpertRegistry — extracted strategy registry from SignalEngine."""

import numpy as np
import pandas as pd

from pipeline.expert_registry import ExpertRegistry


# ── Helpers ──────────────────────────────────────────────────────────


def _make_registry(n_rows: int = 200) -> ExpertRegistry:
    """Create ExpertRegistry with realistic mock data (200 rows by default)."""
    reg = ExpertRegistry()
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "open": np.linspace(1.12, 1.13, n_rows),
            "high": np.linspace(1.121, 1.131, n_rows),
            "low": np.linspace(1.119, 1.129, n_rows),
            "close": np.linspace(1.12, 1.13, n_rows),
            "volume": np.random.randint(100, 1000, n_rows),
            "rsi_14": np.random.uniform(30, 70, n_rows),
            "adx_14": np.random.uniform(15, 35, n_rows),
            "bb_width": np.random.uniform(0.01, 0.05, n_rows),
            "bb_mid": np.linspace(1.12, 1.13, n_rows),
            "mom_1": np.random.uniform(-0.005, 0.005, n_rows),
            "mom_5": np.random.uniform(-0.01, 0.01, n_rows),
            "mom_10": np.random.uniform(-0.02, 0.02, n_rows),
            "vol_ratio": np.random.uniform(0.5, 1.5, n_rows),
        }
    )
    reg.update_context(last_df=df, current_symbol="EURUSD", current_session="london")
    return reg


# ── 1. Initialization ───────────────────────────────────────────────


class TestInit:
    """ExpertRegistry construction."""

    def test_init_defaults(self):
        """Created with no args, fields should be empty."""
        reg = ExpertRegistry()
        assert reg._last_df is None
        assert reg._current_symbol == ""
        assert reg._current_session == "asia"
        assert reg._data_manager is None
        assert reg._feature_pipeline is None
        assert reg._alpha_strategies == {}
        assert reg._cross_sectional_alpha is None

    def test_init_with_data_manager(self):
        """Accepts optional data_manager."""
        dm = object()
        reg = ExpertRegistry(data_manager=dm)
        assert reg._data_manager is dm

    def test_init_with_feature_pipeline(self):
        """Accepts optional feature_pipeline."""
        fp = object()
        reg = ExpertRegistry(feature_pipeline=fp)
        assert reg._feature_pipeline is fp


# ── 2. Context updates ──────────────────────────────────────────────


class TestContext:
    """update_context behaviour."""

    def test_update_context(self):
        """update_context() sets _last_df, _current_symbol, _current_session."""
        reg = ExpertRegistry()
        df = pd.DataFrame({"close": [1.12, 1.13]})
        reg.update_context(
            last_df=df, current_symbol="GBPUSD", current_session="london"
        )
        assert reg._last_df is df
        assert reg._current_symbol == "GBPUSD"
        assert reg._current_session == "london"

    def test_update_context_partial(self):
        """update_context() only overrides non-None / non-empty args."""
        reg = ExpertRegistry()
        reg._current_symbol = "EURUSD"
        reg._current_session = "asia"
        reg.update_context(last_df=None, current_symbol="", current_session="")
        assert reg._current_symbol == "EURUSD"
        assert reg._current_session == "asia"


# ── 3. STRATEGY_SL_TP ───────────────────────────────────────────────


class TestStrategySlTp:
    """STRATEGY_SL_TP class constant."""

    def test_strategy_sl_tp_exists(self):
        """Class has STRATEGY_SL_TP dict with expected strategies."""
        assert isinstance(ExpertRegistry.STRATEGY_SL_TP, dict)
        assert len(ExpertRegistry.STRATEGY_SL_TP) >= 11

    def test_strategy_sl_tp_values(self):
        """Known strategies have expected (sl, tp) tuples."""
        st = ExpertRegistry.STRATEGY_SL_TP
        assert st["ppo_trending"] == (2.0, 4.0)
        assert st["rule_breakout"] == (2.0, 5.0)
        assert st["rule_mean_rev"] == (1.5, 2.0)
        assert st["bb_squeeze"] == (2.0, 3.0)
        assert st["orderflow"] == (1.5, 2.5)
        assert st["xgboost"] == (2.0, 4.0)


# ── 4. get_current_session ──────────────────────────────────────────


class TestGetCurrentSession:
    """get_current_session returns a valid session name."""

    def test_get_current_session(self):
        """Returns one of 'asia', 'london', 'overlap', 'newyork'."""
        reg = ExpertRegistry()
        session = reg.get_current_session()
        assert session in {"asia", "london", "overlap", "newyork"}


# ── 5. features_to_ppo_state ────────────────────────────────────────


class TestFeaturesToPpoState:
    """State vector extraction for PPO agents."""

    def test_features_to_ppo_state_2d(self):
        """2D input returns last row."""
        reg = ExpertRegistry()
        X = np.random.randn(10, 7).astype(np.float32)
        state = reg.features_to_ppo_state(X)
        assert state.shape == (49,)
        assert state.dtype == np.float32
        np.testing.assert_array_equal(state[:7], X[-1, :])

    def test_features_to_ppo_state_3d(self):
        """3D input extracts first batch then last row."""
        reg = ExpertRegistry()
        X = np.random.randn(1, 10, 7).astype(np.float32)
        state = reg.features_to_ppo_state(X)
        assert state.shape == (49,)
        np.testing.assert_array_equal(state[:7], X[0, -1, :])

    def test_features_to_ppo_state_padding(self):
        """Short input padded to 49."""
        reg = ExpertRegistry()
        X = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        state = reg.features_to_ppo_state(X)
        assert state.shape == (49,)
        np.testing.assert_array_equal(state[:3], [1.0, 2.0, 3.0])
        assert np.all(state[3:] == 0.0)

    def test_features_to_ppo_state_truncation(self):
        """Long input truncated to 49."""
        reg = ExpertRegistry()
        X = np.random.randn(100).astype(np.float32)
        state = reg.features_to_ppo_state(X)
        assert state.shape == (49,)
        np.testing.assert_array_equal(state, X[:49])


# ── 6. Strategy prediction methods ──────────────────────────────────


class TestComputeAtr:
    """ATR computation."""

    def test_compute_atr(self):
        """Returns non-negative float."""
        reg = _make_registry()
        atr = reg.compute_atr(14)
        assert isinstance(atr, float)
        assert atr >= 0.0

    def test_compute_atr_no_data(self):
        """Returns 0.0 when no data."""
        reg = ExpertRegistry()
        assert reg.compute_atr(14) == 0.0


class TestRuleBreakout:
    """Rule-based breakout prediction."""

    def test_rule_breakout_prediction(self):
        """Returns valid float (0.0 when no breakout)."""
        reg = _make_registry()
        X = np.array([1.0])
        pred = reg.rule_breakout_prediction(X)
        assert isinstance(pred, float)
        assert -0.01 <= pred <= 0.01

    def test_rule_breakout_asia_skips(self):
        """Asia session returns 0.0."""
        reg = _make_registry()
        reg.update_context(current_session="asia")
        X = np.array([1.0])
        assert reg.rule_breakout_prediction(X) == 0.0

    def test_rule_breakout_no_data(self):
        """No data returns 0.0."""
        reg = ExpertRegistry()
        X = np.array([1.0])
        assert reg.rule_breakout_prediction(X) == 0.0


class TestRuleMeanRev:
    """Rule-based mean reversion prediction."""

    def test_rule_mean_rev_prediction(self):
        """Returns valid float."""
        reg = _make_registry()
        X = np.array([1.0])
        pred = reg.rule_mean_rev_prediction(X)
        assert isinstance(pred, float)
        assert -0.01 <= pred <= 0.01

    def test_rule_mean_rev_no_data(self):
        """No data returns 0.0."""
        reg = ExpertRegistry()
        X = np.array([1.0])
        assert reg.rule_mean_rev_prediction(X) == 0.0


class TestBbSqueeze:
    """Bollinger Band Squeeze prediction."""

    def test_bb_squeeze_prediction(self):
        """Returns valid float."""
        reg = _make_registry(n_rows=200)
        X = np.array([1.0])
        pred = reg.bb_squeeze_prediction(X)
        assert isinstance(pred, float)
        assert -0.01 <= pred <= 0.01

    def test_bb_squeeze_no_data(self):
        """No data returns 0.0."""
        reg = ExpertRegistry()
        X = np.array([1.0])
        assert reg.bb_squeeze_prediction(X) == 0.0


class TestTsMomentum:
    """Time Series Momentum prediction."""

    def test_ts_momentum_prediction(self):
        """Returns valid float."""
        reg = _make_registry()
        X = np.array([1.0])
        pred = reg.ts_momentum_prediction(X)
        assert isinstance(pred, float)
        assert -0.01 <= pred <= 0.01

    def test_ts_momentum_no_data(self):
        """No data returns 0.0."""
        reg = ExpertRegistry()
        X = np.array([1.0])
        assert reg.ts_momentum_prediction(X) == 0.0


class TestVolMeanRev:
    """Volatility Mean Reversion prediction."""

    def test_vol_mean_rev_prediction(self):
        """Returns valid float."""
        reg = _make_registry()
        X = np.array([1.0])
        pred = reg.vol_mean_rev_prediction(X)
        assert isinstance(pred, float)
        assert -0.01 <= pred <= 0.01

    def test_vol_mean_rev_no_data(self):
        """No data returns 0.0."""
        reg = ExpertRegistry()
        X = np.array([1.0])
        assert reg.vol_mean_rev_prediction(X) == 0.0


class TestOrderflow:
    """Order flow prediction."""

    def test_orderflow_prediction_no_data(self):
        """No data_manager returns 0.0."""
        reg = ExpertRegistry()
        reg.update_context(current_symbol="EURUSD")
        X = np.array([1.0])
        assert reg.orderflow_prediction(X) == 0.0

    def test_orderflow_prediction_with_data_manager(self):
        """With data_manager returns 0.0 if no orderflow data."""
        reg = ExpertRegistry(data_manager=object())
        reg.update_context(current_symbol="EURUSD")
        X = np.array([1.0])
        pred = reg.orderflow_prediction(X)
        assert isinstance(pred, float)
        assert -0.01 <= pred <= 0.01

    def test_orderflow_confidence_no_data(self):
        """orderflow_confidence returns base value when no data_manager."""
        reg = ExpertRegistry()
        X = np.array([1.0])
        conf = reg.orderflow_confidence(X)
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_orderflow_no_symbol(self):
        """No symbol returns 0.0."""
        reg = ExpertRegistry()
        X = np.array([1.0])
        assert reg.orderflow_prediction(X) == 0.0


class TestMacroSentiment:
    """Macro sentiment prediction."""

    def test_macro_sentiment_prediction_no_data(self):
        """No data_manager returns 0.0."""
        reg = ExpertRegistry()
        X = np.array([1.0])
        assert reg.macro_sentiment_prediction(X) == 0.0

    def test_macro_sentiment_prediction_with_dm(self):
        """With data_manager returns valid float."""
        reg = ExpertRegistry(data_manager=object())
        X = np.array([1.0])
        pred = reg.macro_sentiment_prediction(X)
        assert isinstance(pred, float)
        assert -0.01 <= pred <= 0.01


class TestSocialSentiment:
    """Social sentiment prediction."""

    def test_social_sentiment_prediction_no_data(self):
        """No data_manager returns 0.0."""
        reg = ExpertRegistry()
        X = np.array([1.0])
        assert reg.social_sentiment_prediction(X) == 0.0

    def test_social_sentiment_prediction_with_dm(self):
        """With data_manager returns valid float."""
        reg = ExpertRegistry(data_manager=object())
        X = np.array([1.0])
        pred = reg.social_sentiment_prediction(X)
        assert isinstance(pred, float)
        assert -0.01 <= pred <= 0.01


class TestXgboost:
    """XGBoost prediction."""

    def test_xgboost_prediction_no_model(self):
        """No model file returns 0.0."""
        reg = ExpertRegistry()
        reg.update_context(current_symbol="EURUSD")
        X = np.array([1.0])
        assert reg.xgboost_prediction(X) == 0.0

    def test_xgboost_no_symbol(self):
        """No symbol returns 0.0."""
        reg = ExpertRegistry()
        X = np.array([1.0])
        assert reg.xgboost_prediction(X) == 0.0


# ── 7. Alpha strategy methods ───────────────────────────────────────


class TestAlphaRegime:
    """alpha_regime_for mapping."""

    def test_alpha_regime_for_known(self):
        """Known alpha strategies return correct regime."""
        reg = ExpertRegistry()
        assert reg.alpha_regime_for("stat_arb") == "ranging"
        assert reg.alpha_regime_for("carry_trade") == "trending"
        assert reg.alpha_regime_for("event_driven") == "volatile"
        assert reg.alpha_regime_for("vol_expansion") == "volatile"
        assert reg.alpha_regime_for("order_flow_momentum") == "ranging"

    def test_alpha_regime_for_unknown(self):
        """Unknown returns 'ranging'."""
        reg = ExpertRegistry()
        assert reg.alpha_regime_for("nonexistent") == "ranging"
        assert reg.alpha_regime_for("") == "ranging"


class TestAlphaHelpers:
    """Alpha strategy helper closures."""

    def test_make_alpha_predict_fn(self):
        """Returns a callable."""
        reg = ExpertRegistry()
        fn = reg.make_alpha_predict_fn("stat_arb")
        assert callable(fn)
        result = fn(np.array([1.0]))
        assert isinstance(result, float)

    def test_make_alpha_confidence_fn(self):
        """Returns a callable."""
        reg = ExpertRegistry()
        fn = reg.make_alpha_confidence_fn("stat_arb")
        assert callable(fn)
        result = fn(np.array([1.0]))
        assert isinstance(result, float)


# ── 8. get_strategy_specs ───────────────────────────────────────────


class TestGetStrategySpecs:
    """Strategy spec building."""

    def test_get_strategy_specs_returns_list(self):
        """Returns a list."""
        reg = ExpertRegistry()
        specs = reg.get_strategy_specs()
        assert isinstance(specs, list)

    def test_get_strategy_specs_has_expected_strategies(self):
        """List contains expected rule-based strategy names."""
        reg = ExpertRegistry()
        specs = reg.get_strategy_specs()
        names = [s["name"] for s in specs]
        for expected in [
            "rule_breakout",
            "rule_mean_rev",
            "bb_squeeze",
            "ts_momentum",
            "vol_mean_rev",
            "orderflow",
            "macro_sentiment",
            "social_sentiment",
            "xgboost",
        ]:
            assert expected in names, f"Missing strategy: {expected}"

    def test_get_strategy_specs_spec_format(self):
        """Each spec has correct keys: name, predict_fn, confidence_fn, regime."""
        reg = ExpertRegistry()
        specs = reg.get_strategy_specs()
        for spec in specs:
            assert "name" in spec
            assert "predict_fn" in spec
            assert "confidence_fn" in spec
            assert "regime" in spec


# ── 9. load_alpha_strategies ────────────────────────────────────────


class TestLoadAlpha:
    """Alpha strategy loading."""

    def test_load_alpha_strategies(self):
        """Doesn't raise; alpha strategies are populated if module exists."""
        reg = ExpertRegistry()
        reg.update_context(current_symbol="EURUSD")
        reg.load_alpha_strategies()
        # Strategies may or may not be importable; either way no exception
        assert isinstance(reg._alpha_strategies, dict)
