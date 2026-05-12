"""
Toxic Flow Detection — Institutional edge via VPIN (Volume-Synchronized Probability of Informed Trading).
Detects when you're trading against smart money / informed participants.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from loguru import logger


@dataclass
class ToxicFlowSnapshot:
    vpin: float = 0.5
    is_toxic: bool = False
    toxicity_level: str = "normal"  # normal | elevated | toxic
    bucket_imbalance: float = 0.0
    informed_direction: str = "neutral"  # buy | sell | neutral


class ToxicFlowDetector:
    """
    Detect toxic order flow using VPIN and order flow imbalance.
    High VPIN = high probability that informed traders are on the other side.
    """

    def __init__(
        self, lookback: int = 100, bucket_size: int = 1000, toxic_threshold: float = 0.8
    ):
        self.lookback = lookback
        self.bucket_size = bucket_size
        self.toxic_threshold = toxic_threshold
        self._ticks: deque = deque(maxlen=lookback * bucket_size)
        self._vpin_history: List[float] = []
        self._imbalance_history: List[float] = []

    def update(self, tick: Dict) -> ToxicFlowSnapshot:
        """
        Update with new tick. Returns toxicity snapshot.
        tick keys: price, mid, volume, bid, ask, timestamp
        """
        self._ticks.append(tick)
        return self._calculate_snapshot()

    def _calculate_snapshot(self) -> ToxicFlowSnapshot:
        if len(self._ticks) < self.bucket_size * 2:
            return ToxicFlowSnapshot()

        ticks = list(self._ticks)
        vpin = self._calculate_vpin(ticks)
        imbalance = self._calculate_imbalance(ticks[-self.bucket_size :])

        self._vpin_history.append(vpin)
        self._imbalance_history.append(imbalance)
        if len(self._vpin_history) > self.lookback:
            self._vpin_history = self._vpin_history[-self.lookback :]
            self._imbalance_history = self._imbalance_history[-self.lookback :]

        is_toxic = vpin > self.toxic_threshold
        level = "toxic" if vpin > 0.8 else "elevated" if vpin > 0.6 else "normal"

        # Determine informed direction from persistent imbalance
        recent_imbalance = (
            np.mean(self._imbalance_history[-10:]) if self._imbalance_history else 0.0
        )
        direction = (
            "buy"
            if recent_imbalance > 0.3
            else "sell" if recent_imbalance < -0.3 else "neutral"
        )

        if is_toxic:
            logger.warning(
                f"Toxic flow detected: VPIN={vpin:.2f}, direction={direction}"
            )

        return ToxicFlowSnapshot(
            vpin=vpin,
            is_toxic=is_toxic,
            toxicity_level=level,
            bucket_imbalance=imbalance,
            informed_direction=direction,
        )

    def _calculate_vpin(self, ticks: List[Dict]) -> float:
        """Calculate Volume-Synchronized Probability of Informed Trading."""
        buckets = []
        current_bucket = []
        current_vol = 0.0

        for tick in ticks:
            current_bucket.append(tick)
            current_vol += tick.get("volume", 1)
            if current_vol >= self.bucket_size:
                buy_vol = sum(
                    t.get("volume", 1)
                    for t in current_bucket
                    if t.get("price", t.get("mid", 0))
                    > self._get_mid(current_bucket[0])
                )
                sell_vol = sum(
                    t.get("volume", 1)
                    for t in current_bucket
                    if t.get("price", t.get("mid", 0))
                    < self._get_mid(current_bucket[0])
                )
                total = buy_vol + sell_vol
                if total > 0:
                    imbalance = abs(buy_vol - sell_vol) / total
                    buckets.append(imbalance)
                current_bucket = []
                current_vol = 0.0

        return np.mean(buckets[-10:]) if buckets else 0.5

    def _calculate_imbalance(self, ticks: List[Dict]) -> float:
        """Calculate order flow imbalance for a set of ticks."""
        if not ticks:
            return 0.0
        buy_vol = sum(
            t.get("volume", 1)
            for t in ticks
            if t.get("price", t.get("mid", 0)) >= t.get("mid", t.get("price", 0))
        )
        sell_vol = sum(
            t.get("volume", 1)
            for t in ticks
            if t.get("price", t.get("mid", 0)) < t.get("mid", t.get("price", 0))
        )
        total = buy_vol + sell_vol
        return (buy_vol - sell_vol) / total if total > 0 else 0.0

    def _get_mid(self, tick: Dict) -> float:
        return tick.get("mid", (tick.get("bid", 0) + tick.get("ask", 0)) / 2.0)

    def should_fade(self, symbol: str = "") -> Tuple[bool, str]:
        """
        Returns (should_fade: bool, reason: str).
        Fade the market when toxicity is high — informed money is positioning against you.
        """
        if not self._vpin_history:
            return False, "insufficient_data"
        recent_vpin = (
            np.mean(self._vpin_history[-3:])
            if len(self._vpin_history) >= 3
            else self._vpin_history[-1]
        )
        if recent_vpin > 0.85:
            direction = (
                "sell"
                if self._imbalance_history and np.mean(self._imbalance_history[-3:]) > 0
                else "buy"
            )
            return True, f"toxic_flow_vpin_{recent_vpin:.2f}_direction_{direction}"
        return False, "normal"
