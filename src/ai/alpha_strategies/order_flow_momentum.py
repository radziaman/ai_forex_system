"""Order flow momentum strategy using microstructure features."""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ai.alpha_strategies.base_strategy import BaseAlphaStrategy
from data.microstructure_features import MicrostructureEngine, PriceTick


class OrderFlowMomentumStrategy(BaseAlphaStrategy):
    """Uses microstructure order-flow metrics (CVD, OFI, DOM imbalance).

    Key signal: CVD divergence — CVD rising but price flat → accumulation → BUY.
    Also fades extreme DOM imbalance (>0.6) as mean reversion.
    """

    def __init__(self, symbol: str, params: Dict[str, Any]):
        super().__init__(symbol, params)
        self.cvd_slope_threshold = params.get("cvd_slope_threshold", 0.1)
        self.price_flat_threshold = params.get("price_flat_threshold", 0.0003)
        self.dom_imbalance_threshold = params.get("dom_imbalance_threshold", 0.6)
        self._engine = MicrostructureEngine(maxlen=500)

    def get_name(self) -> str:
        return "order_flow_momentum"

    def get_required_features(self) -> List[str]:
        return ["close"]

    def _ingest(self, features_df: pd.DataFrame):
        """Push the latest row into the microstructure engine as a synthetic tick."""
        if len(features_df) < 1:
            return
        row = features_df.iloc[-1]
        close = float(row.get("close", 0))
        bid = float(row.get("bid", close * 0.9999))
        ask = float(row.get("ask", close * 1.0001))
        volume = float(row.get("volume", 0))
        tick = PriceTick(
            symbol=self.symbol,
            bid=bid,
            ask=ask,
            mid=(bid + ask) / 2,
            timestamp=float(row.get("timestamp", 0.0)),
            volume=volume,
            trade_volume=float(row.get("tick_volume", volume)),
            trade_side=row.get("trade_side"),
        )
        self._engine.ingest_tick(tick)

    def generate_signal(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        try:
            self._ingest(features_df)

            cvd = self._engine.get_cvd(self.symbol)
            cvd_slope = self._engine.get_cvd_slope(self.symbol, lookback=20)
            dom_imb = self._engine.get_dom_depth_imbalance(self.symbol)
            ofi = self._engine.get_ofi(self.symbol)

            close_series = features_df["close"].values
            if len(close_series) >= 10:
                price_change = (close_series[-1] - close_series[-10]) / close_series[
                    -10
                ]
                price_flat = abs(price_change) < self.price_flat_threshold
            else:
                price_flat = False

            signal = 0.0
            confidence = 0.0
            meta: Dict[str, Any] = {
                "cvd": round(cvd, 2),
                "cvd_slope": round(cvd_slope, 4),
                "dom_imbalance": round(dom_imb, 4),
                "ofi": round(ofi, 4),
            }

            # CVD divergence: rising CVD + flat price → accumulation
            if cvd_slope > self.cvd_slope_threshold and price_flat:
                signal += 0.005
                confidence = min(0.75, 0.5 + abs(cvd_slope) * 2.0)
                meta["divergence"] = "accumulation"

            # CVD divergence: falling CVD + flat price → distribution
            if cvd_slope < -self.cvd_slope_threshold and price_flat:
                signal -= 0.005
                confidence = min(0.75, 0.5 + abs(cvd_slope) * 2.0)
                meta["divergence"] = "distribution"

            # Fade extreme DOM imbalance
            if abs(dom_imb) > self.dom_imbalance_threshold:
                signal -= dom_imb * 0.003
                confidence = max(confidence, min(0.6, abs(dom_imb)))
                meta["dom_fade"] = True

            # OFI confirmation
            if abs(ofi) > 0.3:
                signal += ofi * 0.002
                meta["ofi_signal"] = True

            direction = "HOLD"
            if signal > 0.001:
                direction = "BUY"
            elif signal < -0.001:
                direction = "SELL"

            return {
                "direction": direction,
                "confidence": confidence,
                "meta": meta,
            }
        except Exception:
            return {"direction": "HOLD", "confidence": 0.0, "meta": {}}
