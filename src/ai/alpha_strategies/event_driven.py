"""Event-driven strategy — straddle before major economic releases."""

import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ai.alpha_strategies.base_strategy import BaseAlphaStrategy
from data.economic_calendar import EconomicCalendar


class EventDrivenStrategy(BaseAlphaStrategy):
    """Enters a straddle (long vol) shortly before high-impact events
    and exits shortly after.

    Uses Bollinger Band width to confirm low volatility before the event
    (vol compression → expansion play).
    """

    def __init__(self, symbol: str, params: Dict[str, Any]):
        super().__init__(symbol, params)
        self.enter_minutes_before = params.get("enter_minutes_before", 5)
        self.exit_minutes_after = params.get("exit_minutes_after", 15)
        self.bb_width_lookback = params.get("bb_width_lookback", 20)
        self.bb_width_percentile_threshold = params.get(
            "bb_width_percentile_threshold", 40
        )
        self._calendar = EconomicCalendar()
        self._calendar.fetch(days_forward=7)
        self._call_count = 0

    def get_name(self) -> str:
        return "event_driven"

    def get_required_features(self) -> List[str]:
        return ["bb_width", "close"]

    def _bb_width_low(self, df: pd.DataFrame) -> bool:
        """Return True if current BB width is below its lookback percentile."""
        if "bb_width" not in df.columns or len(df) < self.bb_width_lookback:
            return False
        bw = df["bb_width"].values[-self.bb_width_lookback :]
        current = bw[-1]
        if np.isnan(current):
            return False
        percentile = np.percentile(bw, self.bb_width_percentile_threshold)
        return current <= percentile

    def generate_signal(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        try:
            now = time.time()
            sym = self.symbol.upper()
            base = sym[:3] if len(sym) >= 3 else sym

            # Refresh calendar occasionally (every 100 calls)
            self._call_count += 1
            if not self._calendar.events or self._call_count % 100 == 0:
                self._calendar.fetch(days_forward=7)

            upcoming = self._calendar.get_upcoming_events(hours=24)
            # Filter to events relevant to our symbol's currencies
            relevant = [
                e
                for e in upcoming
                if e.currency in (base, sym[3:6] if len(sym) >= 6 else "USD")
            ]
            if not relevant:
                return {"direction": "HOLD", "confidence": 0.0, "meta": {}}

            next_event = relevant[0]
            minutes_until = (next_event.timestamp - now) / 60.0

            # Confirm low vol before event (BB squeeze)
            low_vol = self._bb_width_low(features_df)

            # Entry window: 5 min before event
            if 0 < minutes_until <= self.enter_minutes_before and low_vol:
                return {
                    "direction": "BUY",  # straddle proxy: long volatility
                    "confidence": 0.65 if next_event.is_high_impact else 0.45,
                    "meta": {
                        "event": next_event.title,
                        "minutes_until": round(minutes_until, 1),
                        "low_vol_confirmed": low_vol,
                        "impact": next_event.impact,
                    },
                }

            # Exit window: 15 min after event
            if -self.exit_minutes_after <= minutes_until <= 0:
                return {
                    "direction": "SELL",  # Close straddle
                    "confidence": 0.5,
                    "meta": {
                        "event": next_event.title,
                        "minutes_since": round(-minutes_until, 1),
                        "action": "exit",
                    },
                }

            return {"direction": "HOLD", "confidence": 0.0, "meta": {}}
        except Exception:
            return {"direction": "HOLD", "confidence": 0.0, "meta": {}}
