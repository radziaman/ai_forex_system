"""Macroeconomic divergence features for RTS AI Forex Trading System.

Hard-coded central bank policy rates and derived divergence metrics.
"""

from typing import Dict


class MacroEconomicEngine:
    """Computes central bank divergence and carry-trade features."""

    # Approximate current policy rates (%)
    RATES = {
        "USD": 5.50,
        "EUR": 4.00,
        "GBP": 5.25,
        "JPY": 0.10,
        "AUD": 4.35,
        "CAD": 4.75,
        "CHF": 1.25,
        "NZD": 5.50,
    }

    # Hard-coded 1-month rate slope direction (% per month)
    RATE_SLOPE = {
        "USD": -0.05,
        "EUR": -0.05,
        "GBP": -0.05,
        "JPY": 0.10,
        "AUD": -0.05,
        "CAD": -0.05,
        "CHF": -0.05,
        "NZD": -0.05,
    }

    def __init__(self):
        pass

    def compute_divergence_features(self, symbol: str) -> Dict[str, float]:
        """Return macroeconomic divergence features for a symbol.

        Args:
            symbol: Six-character FX pair (e.g. ``EURUSD``).

        Returns:
            Dict of float-valued macro features.
        """
        if len(symbol) != 6:
            return {
                "rate_differential": 0.0,
                "rate_slope_1m": 0.0,
                "forward_divergence": 0.0,
                "hawkish_dovish_score": 0.0,
                "yield_carry_annualized": 0.0,
            }

        base = symbol[:3]
        quote = symbol[3:]

        base_rate = self.RATES.get(base, 0.0)
        quote_rate = self.RATES.get(quote, 0.0)
        rate_differential = base_rate - quote_rate

        base_slope = self.RATE_SLOPE.get(base, 0.0)
        quote_slope = self.RATE_SLOPE.get(quote, 0.0)
        rate_slope_1m = base_slope - quote_slope

        # Forward divergence: 1-year forward implied vs spot differential
        # Hard-coded: spot differential + annualised slope drift
        forward_divergence = rate_differential + rate_slope_1m * 12.0

        # Hawkish (+1) / dovish (-1) score for base currency
        if base_rate >= 5.0:
            hawkish_dovish_score = 1.0
        elif base_rate <= 0.5:
            hawkish_dovish_score = -1.0
        else:
            hawkish_dovish_score = 0.0

        # Annualised yield carry in pips
        yield_carry_annualized = rate_differential * 10000.0

        return {
            "rate_differential": rate_differential,
            "rate_slope_1m": rate_slope_1m,
            "forward_divergence": forward_divergence,
            "hawkish_dovish_score": hawkish_dovish_score,
            "yield_carry_annualized": yield_carry_annualized,
        }
