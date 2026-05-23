"""Statistical arbitrage (pairs trading) strategy using OLS cointegration."""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ai.alpha_strategies.base_strategy import BaseAlphaStrategy


class StatArbStrategy(BaseAlphaStrategy):
    """Pairs trading via cointegration / OLS hedge ratio.

    Tracks the spread between two cointegrated pairs (e.g. EURUSD vs GBPUSD)
    and signals when the spread deviates >2 standard deviations from its
    rolling mean.
    """

    def __init__(
        self,
        symbol: str,
        params: Dict[str, Any],
        pair_symbol: str = "GBPUSD",
        lookback: int = 60,
    ):
        super().__init__(symbol, params)
        self.pair_symbol = pair_symbol
        self.lookback = params.get("lookback", lookback)
        self.z_threshold = params.get("z_threshold", 2.0)
        self._hedge_ratio: float = 1.0
        self._spread_mean: float = 0.0
        self._spread_std: float = 0.0

    def get_name(self) -> str:
        return "stat_arb"

    def get_required_features(self) -> List[str]:
        return ["close", f"close_pair_{self.pair_symbol}"]

    def generate_signal(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        if len(features_df) < self.lookback:
            return {"direction": "HOLD", "confidence": 0.0, "meta": {}}

        try:
            # Primary series
            y = features_df["close"].values[-self.lookback :]
            # Pair series (expect caller to provide aligned pair close)
            pair_col = f"close_pair_{self.pair_symbol}"
            if pair_col not in features_df.columns:
                return {"direction": "HOLD", "confidence": 0.0, "meta": {}}
            x = features_df[pair_col].values[-self.lookback :]

            # OLS hedge ratio via normal equation (slope)
            x_mean, y_mean = np.mean(x), np.mean(y)
            cov = np.mean((x - x_mean) * (y - y_mean))
            var = np.mean((x - x_mean) ** 2)
            if var > 0:
                self._hedge_ratio = cov / var
            else:
                self._hedge_ratio = 1.0

            # Spread = y - hedge_ratio * x
            spread = y - self._hedge_ratio * x
            self._spread_mean = float(np.mean(spread))
            self._spread_std = float(np.std(spread))
            if self._spread_std <= 0:
                return {"direction": "HOLD", "confidence": 0.0, "meta": {}}

            # Current spread
            current_spread = spread[-1]
            z_score = (current_spread - self._spread_mean) / self._spread_std

            if z_score > self.z_threshold:
                direction = "SELL"  # Spread too high → short y, long x
                confidence = min(0.95, abs(z_score) / 4.0)
            elif z_score < -self.z_threshold:
                direction = "BUY"  # Spread too low → long y, short x
                confidence = min(0.95, abs(z_score) / 4.0)
            else:
                direction = "HOLD"
                confidence = 0.0

            return {
                "direction": direction,
                "confidence": confidence,
                "meta": {
                    "z_score": round(z_score, 4),
                    "hedge_ratio": round(self._hedge_ratio, 4),
                    "spread_mean": round(self._spread_mean, 6),
                    "spread_std": round(self._spread_std, 6),
                },
            }
        except Exception:
            return {"direction": "HOLD", "confidence": 0.0, "meta": {}}
