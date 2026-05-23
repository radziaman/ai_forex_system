"""Volatility expansion / breakout strategy."""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ai.alpha_strategies.base_strategy import BaseAlphaStrategy


class VolExpansionStrategy(BaseAlphaStrategy):
    """Volatility breakout: enter when Bollinger Band width percentile > 80
    AND volume spike > 2x average.
    """

    def __init__(self, symbol: str, params: Dict[str, Any]):
        super().__init__(symbol, params)
        self.bb_percentile_threshold = params.get("bb_percentile_threshold", 80)
        self.bb_lookback = params.get("bb_lookback", 100)
        self.volume_spike_threshold = params.get("volume_spike_threshold", 2.0)
        self.volume_lookback = params.get("volume_lookback", 20)

    def get_name(self) -> str:
        return "vol_expansion"

    def get_required_features(self) -> List[str]:
        return ["bb_width", "volume", "close"]

    def generate_signal(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        if len(features_df) < self.bb_lookback:
            return {"direction": "HOLD", "confidence": 0.0, "meta": {}}

        try:
            if "bb_width" not in features_df.columns:
                return {"direction": "HOLD", "confidence": 0.0, "meta": {}}

            bb_width = features_df["bb_width"].values[-self.bb_lookback :]
            current_bb = bb_width[-1]
            if np.isnan(current_bb):
                return {"direction": "HOLD", "confidence": 0.0, "meta": {}}

            bb_percentile = np.percentile(bb_width, self.bb_percentile_threshold)
            bb_condition = current_bb >= bb_percentile

            # Volume spike check
            volume_ok = True
            if "volume" in features_df.columns:
                vol = features_df["volume"].values[-self.volume_lookback :]
                current_vol = vol[-1]
                avg_vol = np.mean(vol)
                if avg_vol > 0:
                    volume_ok = current_vol >= avg_vol * self.volume_spike_threshold
                else:
                    volume_ok = False

            if bb_condition and volume_ok:
                close = float(features_df["close"].iloc[-1])
                # Direction: break in direction of recent momentum
                if len(features_df) >= 3:
                    mom = close - float(features_df["close"].iloc[-3])
                else:
                    mom = 0.0
                direction = "BUY" if mom >= 0 else "SELL"
                confidence = min(
                    0.9,
                    0.5
                    + (current_bb / max(bb_percentile, 1e-8) - 1) * 0.3
                    + (current_vol / max(avg_vol, 1e-8) - 1) * 0.1,
                )
                return {
                    "direction": direction,
                    "confidence": confidence,
                    "meta": {
                        "bb_width": round(current_bb, 6),
                        "bb_percentile": round(bb_percentile, 6),
                        "volume_ratio": (
                            round(current_vol / avg_vol, 2) if avg_vol > 0 else 0.0
                        ),
                    },
                }

            return {"direction": "HOLD", "confidence": 0.0, "meta": {}}
        except Exception:
            return {"direction": "HOLD", "confidence": 0.0, "meta": {}}
