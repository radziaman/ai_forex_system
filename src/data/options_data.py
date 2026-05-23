"""Options Market Data Framework — FX options implied-volatility analytics.

Provides Greeks-driven features for the AI model:
  * 25-delta risk reversal (sentiment / directional skew)
  * butterfly spread (smile convexity)
  * skew index (downside protection demand)
  * term structure (event expectations)

All methods return **mock data** for now.  When a Bloomberg terminal,
OTC broker feed, or Deribit crypto-options API is wired in,
replace the mock generators with real calls.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from loguru import logger


@dataclass
class OptionsQuote:
    """Single-strike options quote."""

    strike: float
    expiry: str  # ISO date, e.g. "2025-06-20"
    call_iv: float
    put_iv: float
    delta: float
    gamma: float
    theta: float
    vega: float


class OptionsDataProvider:
    """Framework for FX options data.

    Mock implementation returns deterministic synthetic quotes so the
    feature pipeline can be exercised immediately.  Real integrations
    (Bloomberg BPIPE, OTC broker, CME Globex) can be added later by
    overriding ``_fetch_chain()``.
    """

    def __init__(self, source: Optional[str] = None):
        self.source = source  # e.g. "bloomberg", "otc", "mock"
        self._mock_cache: Dict[str, List[OptionsQuote]] = {}
        logger.info(f"OptionsDataProvider initialized (source={source or 'mock'})")

    def is_available(self) -> bool:
        """Return ``True`` if a non-mock data source is configured."""
        return self.source is not None and self.source != "mock"

    # -- public analytics ----------------------------------------------------

    def get_25d_risk_reversal(self, symbol: str) -> float:
        """25-delta call IV minus 25-delta put IV.

        Positive = market is bidding up calls (bullish skew).
        Negative = put premium elevated (bearish / hedging demand).
        """
        chain = self._get_chain(symbol)
        call25 = self._find_nearest_delta(chain, "call", target_delta=0.25)
        put25 = self._find_nearest_delta(chain, "put", target_delta=-0.25)
        if call25 is None or put25 is None:
            return 0.0
        return call25.call_iv - put25.put_iv

    def get_butterfly_spread(self, symbol: str) -> float:
        """ATM vol minus average of 25-delta call + put vols.

        Measures smile convexity.  High = wings are expensive
        (crash protection priced in).
        """
        chain = self._get_chain(symbol)
        atm = self._find_atm(chain)
        call25 = self._find_nearest_delta(chain, "call", target_delta=0.25)
        put25 = self._find_nearest_delta(chain, "put", target_delta=-0.25)
        if atm is None or call25 is None or put25 is None:
            return 0.0
        wing_avg = (call25.call_iv + put25.put_iv) / 2.0
        return wing_avg - atm.call_iv

    def get_skew_index(self, symbol: str) -> float:
        """Normalised measure of downside protection demand.

        Returns a value in ``[0, 1]`` where ``1`` means extreme
        demand for OTM puts (risk-off).
        """
        rr = self.get_25d_risk_reversal(symbol)
        # Typical FX 25-delta RR ranges +/- 0.02 (2 vol points)
        return float(np.clip(-rr / 0.04 + 0.5, 0.0, 1.0))

    def get_term_structure(self, symbol: str) -> Dict[str, float]:
        """ATM implied vol across tenors.

        Steep upward slope = event risk expected.
        Inverted = immediate stress, calm thereafter.
        """
        chain = self._get_chain(symbol)
        atm_by_expiry: Dict[str, float] = {}
        for q in chain:
            if abs(q.delta) < 0.55:  # roughly ATM
                atm_by_expiry[q.expiry] = q.call_iv
        # Sort by expiry date string for consistency
        return dict(sorted(atm_by_expiry.items()))

    # -- internal helpers ----------------------------------------------------

    def _get_chain(self, symbol: str) -> List[OptionsQuote]:
        """Return the full options chain for *symbol* (cached)."""
        if symbol not in self._mock_cache:
            self._mock_cache[symbol] = self._generate_mock_chain(symbol)
        return self._mock_cache[symbol]

    @staticmethod
    def _find_nearest_delta(
        chain: List[OptionsQuote], side: str, target_delta: float
    ) -> Optional[OptionsQuote]:
        """Return the quote whose delta is closest to *target_delta*."""
        best: Optional[OptionsQuote] = None
        best_err = float("inf")
        for q in chain:
            d = q.delta if side == "call" else -q.delta
            err = abs(d - abs(target_delta))
            if err < best_err:
                best_err = err
                best = q
        return best

    @staticmethod
    def _find_atm(chain: List[OptionsQuote]) -> Optional[OptionsQuote]:
        """Return the quote whose delta is closest to zero."""
        best: Optional[OptionsQuote] = None
        best_err = float("inf")
        for q in chain:
            err = abs(q.delta)
            if err < best_err:
                best_err = err
                best = q
        return best

    # -- mock generator ------------------------------------------------------

    def _generate_mock_chain(self, symbol: str) -> List[OptionsQuote]:
        """Deterministic synthetic options chain for testing."""
        # Seed RNG from symbol for reproducibility
        rng = np.random.default_rng(seed=abs(hash(symbol)) % (2**31))
        base_price = self._base_price(symbol)
        tenors = ["1W", "1M", "3M", "6M", "1Y"]
        strikes = [
            base_price * (0.90 + i * 0.025)
            for i in range(9)  # 90 % … 110 % in 2.5 % steps
        ]
        chain: List[OptionsQuote] = []
        for tenor in tenors:
            atm_vol = 0.08 + rng.random() * 0.04  # 8–12 % ATM vol
            for k in strikes:
                moneyness = k / base_price - 1.0
                # Smile: wings are ~2 vol points higher
                call_iv = atm_vol + abs(moneyness) * 0.10 + rng.random() * 0.005
                put_iv = atm_vol + abs(moneyness) * 0.12 + rng.random() * 0.005
                # Simplified Greeks approximations
                delta = 0.5 + moneyness * 3.0 + rng.random() * 0.02
                delta = float(np.clip(delta, 0.01, 0.99))
                if moneyness < 0:
                    delta = -delta  # puts have negative delta in this convention
                gamma = 0.05 + rng.random() * 0.02
                theta = -(0.001 + rng.random() * 0.001)
                vega = 0.10 + rng.random() * 0.05
                chain.append(
                    OptionsQuote(
                        strike=k,
                        expiry=tenor,
                        call_iv=round(call_iv, 4),
                        put_iv=round(put_iv, 4),
                        delta=round(delta, 4),
                        gamma=round(gamma, 4),
                        theta=round(theta, 6),
                        vega=round(vega, 4),
                    )
                )
        return chain

    @staticmethod
    def _base_price(symbol: str) -> float:
        defaults = {
            "EURUSD": 1.12,
            "GBPUSD": 1.28,
            "USDJPY": 150.0,
            "AUDUSD": 0.67,
            "USDCAD": 1.35,
            "USDCHF": 0.88,
            "NZDUSD": 0.61,
            "XAUUSD": 2000.0,
            "XTIUSD": 75.0,
            "US500": 4500.0,
            "BTCUSD": 45000.0,
        }
        return defaults.get(symbol, 1.0)
