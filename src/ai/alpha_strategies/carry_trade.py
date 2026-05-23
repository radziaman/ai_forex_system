"""Carry trade strategy — long high-yield, short funding currency."""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ai.alpha_strategies.base_strategy import BaseAlphaStrategy

# Hardcoded interest rate differentials for major currencies (central bank rates)
_CARRY_RATES: Dict[str, float] = {
    "AUD": 4.35,
    "NZD": 5.50,
    "USD": 5.25,
    "GBP": 5.25,
    "EUR": 4.00,
    "CAD": 5.00,
    "CHF": 1.75,
    "JPY": 0.10,
}


class CarryTradeStrategy(BaseAlphaStrategy):
    """FX carry trade: long high-yield, short low-yield.

    Only active when ATR is below a volatility threshold (low-vol filter).
    """

    def __init__(self, symbol: str, params: Dict[str, Any]):
        super().__init__(symbol, params)
        self.atr_threshold = params.get("atr_threshold", 0.005)
        self.lookback = params.get("lookback", 20)

    def get_name(self) -> str:
        return "carry_trade"

    def get_required_features(self) -> List[str]:
        return ["close", "high", "low"]

    def _extract_currencies(self) -> tuple:
        """Extract base/quote from 6-char symbol (e.g. EURUSD -> EUR, USD)."""
        sym = self.symbol.upper()
        if len(sym) >= 6:
            return sym[:3], sym[3:6]
        return sym, "USD"

    def _compute_atr(self, df: pd.DataFrame) -> float:
        if len(df) < self.lookback + 1:
            return float("inf")
        high_low = df["high"] - df["low"]
        atr = high_low.rolling(self.lookback).mean()
        val = atr.iloc[-1]
        return float(val) if not np.isnan(val) else float("inf")

    def generate_signal(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        if len(features_df) < self.lookback + 1:
            return {"direction": "HOLD", "confidence": 0.0, "meta": {}}

        try:
            base, quote = self._extract_currencies()
            base_rate = _CARRY_RATES.get(base, 0.0)
            quote_rate = _CARRY_RATES.get(quote, 0.0)
            rate_diff = base_rate - quote_rate  # positive = base yields more

            # Low-vol filter
            atr = self._compute_atr(features_df)
            close = float(features_df["close"].iloc[-1])
            atr_pct = atr / close if close > 0 else float("inf")
            if atr_pct > self.atr_threshold:
                return {
                    "direction": "HOLD",
                    "confidence": 0.0,
                    "meta": {"reason": "vol_too_high", "atr_pct": round(atr_pct, 5)},
                }

            if rate_diff > 1.0:
                direction = "BUY"
                confidence = min(0.8, rate_diff / 10.0)
            elif rate_diff < -1.0:
                direction = "SELL"
                confidence = min(0.8, abs(rate_diff) / 10.0)
            else:
                direction = "HOLD"
                confidence = 0.0

            return {
                "direction": direction,
                "confidence": confidence,
                "meta": {
                    "rate_diff": round(rate_diff, 2),
                    "atr_pct": round(atr_pct, 5),
                },
            }
        except Exception:
            return {"direction": "HOLD", "confidence": 0.0, "meta": {}}
