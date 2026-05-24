import numpy as np
import pandas as pd
import pytest
from ai.alpha_research import AlphaResearchPipeline, AlphaPipelineConfig, SignalStatus


class TestAlphaResearchPipeline:
    def test_register_and_evaluate_strong_signal(self):
        """A deliberately correlated signal should pass validation."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=500, freq="h")
        # Create price with predictable pattern
        trend = np.cumsum(np.random.randn(500) * 0.001 + 0.0001)
        noise = np.random.randn(500) * 0.0005
        prices = pd.DataFrame({"EURUSD": 1.10 + trend + noise}, index=dates)
        pipeline = AlphaResearchPipeline(
            config=AlphaPipelineConfig(min_ic=0.01, min_ir=0.2, min_t_stat=0.5)
        )

        # Register a simple momentum signal
        def mom5(df):
            return df.iloc[:, 0].pct_change(5)

        pipeline.register_signal("momentum_5", mom5)
        result = pipeline.evaluate("momentum_5", prices)

        assert result.status == SignalStatus.ACTIVE
        assert result.is_significant

    def test_register_and_evaluate_weak_signal(self):
        """Random noise should fail validation."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=500, freq="h")
        prices = pd.DataFrame(
            {"EURUSD": np.cumsum(np.random.randn(500)) + 1.1}, index=dates
        )

        pipeline = AlphaResearchPipeline(
            config=AlphaPipelineConfig(min_ic=0.05, min_ir=1.0, min_t_stat=1.0)
        )

        def random_signal(df):
            return pd.Series(np.random.randn(len(df)), index=df.index)

        pipeline.register_signal("random", random_signal)
        result = pipeline.evaluate("random", prices)

        assert result.status == SignalStatus.REJECTED
        assert not result.is_significant

    def test_ic_by_horizon(self):
        """IC should be computable at multiple horizons."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=500, freq="h")
        prices = pd.DataFrame(
            {"EURUSD": np.cumsum(np.random.randn(500) * 0.001 + 0.0001) + 1.1},
            index=dates,
        )

        pipeline = AlphaResearchPipeline(
            config=AlphaPipelineConfig(min_ic=0.0, min_ir=0.0, min_t_stat=0.0)
        )

        def signal(df):
            return df.iloc[:, 0].pct_change(5)

        pipeline.register_signal("test", signal)
        result = pipeline.evaluate("test", prices)

        assert len(result.ic_by_horizon) > 0
        for h in [1, 4, 24]:
            if h in result.ic_by_horizon:
                assert isinstance(result.ic_by_horizon[h], float)

    def test_get_active_signals(self):
        pipeline = AlphaResearchPipeline()
        assert pipeline.get_active_signals() == []

    def test_generate_signal_portfolio_empty(self):
        pipeline = AlphaResearchPipeline()
        result = pipeline.generate_signal_portfolio(pd.DataFrame())
        assert result == {}

    def test_invalid_signal_name_raises(self):
        pipeline = AlphaResearchPipeline()
        with pytest.raises(ValueError):
            pipeline.evaluate("nonexistent", pd.DataFrame())

    def test_regime_specific_ic(self):
        """Regime-specific IC should be calculated when regimes are provided."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=500, freq="h")
        prices = pd.DataFrame(
            {"EURUSD": np.cumsum(np.random.randn(500) * 0.001) + 1.1}, index=dates
        )
        regimes = pd.Series(np.random.choice(["trending", "ranging"], 500), index=dates)

        pipeline = AlphaResearchPipeline(
            config=AlphaPipelineConfig(min_ic=0.0, min_ir=0.0, min_t_stat=0.0)
        )

        def signal(df):
            return df.iloc[:, 0].pct_change(5)

        pipeline.register_signal("test", signal)
        result = pipeline.evaluate("test", prices, regimes=regimes)

        assert isinstance(result.regime_ics, dict)
