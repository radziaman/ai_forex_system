"""
Cross-Sectional Alpha Signals for FX.
Generates relative-value signals across all currency pairs:
  - Carry rank: long high-yield, short low-yield
  - Momentum rank: relative strength across pairs (Moskowitz 2012)
  - Value rank: PPP deviation / real exchange rate valuation
  - Volatility rank: short calm pairs, long volatile pairs

Each signal produces a z-score relative to the cross-sectional universe.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class CrossSectionalSignal:
    carry_zscore: Dict[str, float] = field(default_factory=dict)
    momentum_zscore: Dict[str, float] = field(default_factory=dict)
    value_zscore: Dict[str, float] = field(default_factory=dict)
    volatility_zscore: Dict[str, float] = field(default_factory=dict)
    composite_zscore: Dict[str, float] = field(default_factory=dict)
    timestamp: float = 0.0


# Risk-free rates by currency (approximate annualized, can be updated)
DEFAULT_RATES: Dict[str, float] = {
    "USD": 0.0525,  # US Federal Funds Rate
    "EUR": 0.0400,  # ECB Rate
    "GBP": 0.0525,  # BOE Rate
    "JPY": 0.0010,  # BOJ Rate
    "AUD": 0.0435,  # RBA Rate
    "NZD": 0.0550,  # RBNZ Rate
    "CAD": 0.0500,  # BOC Rate
    "CHF": 0.0175,  # SNB Rate
    "XAU": 0.0000,  # Gold (no yield)
    "BTC": 0.0000,  # Bitcoin (no yield)
    "XTI": 0.0000,  # Oil (no yield)
}


class CrossSectionalAlpha:
    """Generates cross-sectional alpha signals for multi-pair trading.

    For N pairs, each signal z-scores the pair's value relative to the universe.
    Composite = equal-weighted average of available signals.
    """

    def __init__(self, interest_rates: Optional[Dict[str, float]] = None):
        self.rates = interest_rates or DEFAULT_RATES
        self._lookback = 60  # for momentum calculation

    def compute_all(
        self,
        prices: Dict[str, pd.Series],
        volumes: Optional[Dict[str, pd.Series]] = None,
    ) -> CrossSectionalSignal:
        """Compute all cross-sectional signals from price data."""
        carry = self._compute_carry(prices)
        momentum = self._compute_momentum_rank(prices)
        value = self._compute_value_rank(prices)
        volatility = self._compute_volatility_rank(prices, volumes)

        # Composite: average of z-scores that exist
        symbols = set(carry.keys()) | set(momentum.keys())
        composite = {}
        for sym in symbols:
            scores = []
            for signal in [carry, momentum, value, volatility]:
                if sym in signal:
                    scores.append(signal[sym])
            composite[sym] = float(np.mean(scores)) if scores else 0.0

        import time

        return CrossSectionalSignal(
            carry_zscore=carry,
            momentum_zscore=momentum,
            value_zscore=value,
            volatility_zscore=volatility,
            composite_zscore=composite,
            timestamp=time.time(),
        )

    def _compute_carry(self, prices: Dict[str, pd.Series]) -> Dict[str, float]:
        """Carry: long pairs where base currency has higher rate than quote.

        For each pair (e.g. AUDUSD), carry = rate_base - rate_quote.
        Z-score across all pairs.
        """
        carries = {}
        for sym, series in prices.items():
            if len(series) < 2:
                continue
            base, quote = self._parse_pair(sym)
            base_rate = self.rates.get(base, 0.0)
            quote_rate = self.rates.get(quote, 0.0)
            # For XXX/USD pairs: base_rate - usd_rate
            # For USD/XXX pairs: usd_rate - quote_rate
            if quote == "USD":
                carry = base_rate - self.rates.get("USD", 0.05)
            elif base == "USD":
                carry = self.rates.get("USD", 0.05) - quote_rate
            else:
                # Cross pair: approximate
                carry = base_rate - quote_rate
            carries[sym] = carry
        return self._zscore_dict(carries)

    def _compute_momentum_rank(self, prices: Dict[str, pd.Series]) -> Dict[str, float]:
        """Time-series momentum rank (Moskowitz et al. 2012).

        For each pair: sign(return_12m_1m) * (return_12m_1m / vol_12m)
        Then z-score cross-sectionally.
        """
        mom_signals = {}
        for sym, series in prices.items():
            if len(series) < self._lookback + 20:
                continue
            ret_12m = (series.iloc[-1] / series.iloc[-self._lookback]) - 1.0
            vol_12m = series.pct_change().dropna().tail(self._lookback).std()
            if vol_12m > 0:
                mom = np.sign(ret_12m) * (ret_12m / vol_12m)
                mom_signals[sym] = float(mom)
        return self._zscore_dict(mom_signals)

    def _compute_value_rank(self, prices: Dict[str, pd.Series]) -> Dict[str, float]:
        """Value: z-score of real exchange rate deviation.

        Uses 200-day moving average as proxy for purchasing power parity.
        Positive = overvalued (expect mean reversion), negative = undervalued.
        """
        value_signals = {}
        for sym, series in prices.items():
            if len(series) < 200:
                continue
            current = series.iloc[-1]
            ma_200 = series.tail(200).mean()
            if ma_200 > 0:
                deviation = (current - ma_200) / ma_200
                value_signals[sym] = float(
                    -deviation
                )  # Negative = signal to buy undervalued
        return self._zscore_dict(value_signals)

    def _compute_volatility_rank(
        self,
        prices: Dict[str, pd.Series],
        volumes: Optional[Dict[str, pd.Series]] = None,
    ) -> Dict[str, float]:
        """Volatility rank: short the calmest pairs, long the most volatile.

        Uses 20-day realized vol, z-scored cross-sectionally.
        """
        vol_signals = {}
        for sym, series in prices.items():
            if len(series) < 20:
                continue
            returns = series.pct_change().dropna().tail(20)
            vol = returns.std() * np.sqrt(252)  # Annualized
            if vol > 0:
                vol_signals[sym] = float(vol)
        return self._zscore_dict(vol_signals)

    def _parse_pair(self, symbol: str) -> Tuple[str, str]:
        """Parse forex pair into base and quote currencies."""
        sym = symbol.upper()
        known = {
            "XAUUSD": ("XAU", "USD"),
            "XTIUSD": ("XTI", "USD"),
            "BTCUSD": ("BTC", "USD"),
            "US500": ("US500", "USD"),
        }
        if sym in known:
            return known[sym]
        majors_3 = ["EUR", "GBP", "AUD", "NZD"]
        for base in majors_3:
            if sym.startswith(base):
                quote = sym[len(base) :]
                return base, quote
        if sym.startswith("USD"):
            quote = sym[3:]
            return "USD", quote if quote else "USD"
        return sym[:3], sym[3:] if len(sym) > 3 else "USD"

    @staticmethod
    def _zscore_dict(data: Dict[str, float]) -> Dict[str, float]:
        """Convert dict values to z-scores."""
        if not data:
            return {}
        values = np.array(list(data.values()))
        if np.std(values) == 0:
            return {k: 0.0 for k in data}
        z = (values - np.mean(values)) / np.std(values)
        return {k: float(z[i]) for i, k in enumerate(data.keys())}
