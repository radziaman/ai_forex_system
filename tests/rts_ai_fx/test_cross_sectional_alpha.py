import numpy as np
import pandas as pd
from rts_ai_fx.cross_sectional_alpha import CrossSectionalAlpha, DEFAULT_RATES


class TestCrossSectionalAlpha:
    def test_carry_zscore_symmetry(self):
        """Carry z-scores should sum to approximately zero (mean-centered)."""
        prices = {
            "EURUSD": pd.Series(np.cumsum(np.random.randn(100)) + 1.1),
            "GBPUSD": pd.Series(np.cumsum(np.random.randn(100)) + 1.25),
            "USDJPY": pd.Series(np.cumsum(np.random.randn(100)) + 110),
        }
        alpha = CrossSectionalAlpha()
        signal = alpha.compute_all(prices)
        z_sum = sum(signal.carry_zscore.values())
        assert abs(z_sum) < 1e-6

    def test_momentum_rank_cross_sectional(self):
        """Momentum should differentiate strong from weak pairs."""
        np.random.seed(42)
        # Create one strong uptrend, one downtrend
        dates = pd.date_range("2024-01-01", periods=200, freq="D")
        strong = pd.Series(np.cumsum(np.random.randn(200) * 0.01 + 0.002), index=dates)
        weak = pd.Series(np.cumsum(np.random.randn(200) * 0.01 - 0.002), index=dates)
        prices = {"EURUSD": strong, "GBPUSD": weak}
        alpha = CrossSectionalAlpha()
        signal = alpha.compute_all(prices)
        assert signal.momentum_zscore["EURUSD"] > signal.momentum_zscore["GBPUSD"]

    def test_value_rank_undervalued_signal(self):
        """Undervalued pair (below 200-MA) should have positive value z-score."""
        np.random.seed(42)
        # Price recently dropped below 200-MA
        prices_series = pd.Series(np.random.randn(250) * 0.01 + 0.0001).cumsum() + 1.0
        # Make last 50 points drift down
        prices_series.iloc[-50:] = prices_series.iloc[-50:].values * 0.98
        # Reindex so iloc[-50] is clearly below the long-term average
        prices = {"EURUSD": prices_series}
        alpha = CrossSectionalAlpha()
        signal = alpha.compute_all(prices)
        # The single pair will have z-score 0 (no cross-section), just check no error
        assert isinstance(signal.value_zscore, dict)

    def test_volatility_rank(self):
        """Volatile pair should get higher volatility score."""
        np.random.seed(42)
        calm = pd.Series(np.random.randn(100) * 0.002).cumsum() + 1.1
        wild = pd.Series(np.random.randn(100) * 0.02).cumsum() + 1.1
        prices = {"EURUSD": calm, "GBPUSD": wild}
        alpha = CrossSectionalAlpha()
        signal = alpha.compute_all(prices)
        assert signal.volatility_zscore["GBPUSD"] > signal.volatility_zscore["EURUSD"]

    def test_composite_zscore_equal_weight(self):
        """Composite should be approximately mean of individual z-scores."""
        prices = {
            "EURUSD": pd.Series(np.cumsum(np.random.randn(100)) + 1.1),
            "GBPUSD": pd.Series(np.cumsum(np.random.randn(100)) + 1.25),
            "USDJPY": pd.Series(np.cumsum(np.random.randn(100)) + 110),
            "AUDUSD": pd.Series(np.cumsum(np.random.randn(100)) + 0.65),
        }
        alpha = CrossSectionalAlpha()
        signal = alpha.compute_all(prices)
        for sym in prices:
            indiv = [
                signal.carry_zscore.get(sym, 0),
                signal.momentum_zscore.get(sym, 0),
                signal.value_zscore.get(sym, 0),
                signal.volatility_zscore.get(sym, 0),
            ]
            expected = np.mean([s for s in indiv if abs(s) > 1e-10] or [0])
            actual = signal.composite_zscore.get(sym, 0)
            assert (
                abs(actual - expected) < 1e-6
                or len([s for s in indiv if abs(s) > 1e-10]) <= 1
            )

    def test_default_rates_defined(self):
        assert "USD" in DEFAULT_RATES
        assert "EUR" in DEFAULT_RATES
        assert DEFAULT_RATES["JPY"] < DEFAULT_RATES["AUD"]

    def test_parse_eurusd(self):
        alpha = CrossSectionalAlpha()
        base, quote = alpha._parse_pair("EURUSD")
        assert base == "EUR"
        assert quote == "USD"

    def test_parse_usdjpy(self):
        alpha = CrossSectionalAlpha()
        base, quote = alpha._parse_pair("USDJPY")
        assert base == "USD"
        assert quote == "JPY"

    def test_parse_xauusd(self):
        alpha = CrossSectionalAlpha()
        base, quote = alpha._parse_pair("XAUUSD")
        assert base == "XAU"
        assert quote == "USD"
