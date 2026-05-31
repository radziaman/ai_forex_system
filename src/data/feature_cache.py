"""
Hash-based feature cache with smart invalidation based on incoming data time ranges.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from loguru import logger

from .data_manager import TF_MINUTES


class FeatureCache:
    """Manages feature caching with hash-based invalidation."""

    def __init__(self, dm):
        self.dm = dm  # Reference to DataManager for shared state

    def invalidate(self, symbol: str, ts_min: float, ts_max: float):
        """Smart invalidation based on time range of incoming data."""
        cache = self.dm._feature_cache.get(symbol, {})
        stale_tfs = []
        for tf in cache:
            df = self.dm.ohlcv[symbol].get(tf)
            if df is not None and not df.empty:
                last_bar_ts = float(df["timestamp"].iloc[-1])
                if ts_max >= last_bar_ts - TF_MINUTES.get(tf, 60) * 60:
                    stale_tfs.append(tf)
        for tf in stale_tfs:
            if tf in cache:
                del cache[tf]

    def get(self, symbol: str, timeframe: str) -> Optional[np.ndarray]:
        """Retrieve cached features if cache key still matches current OHLCV state."""
        entry = self.dm._feature_cache.get(symbol, {}).get(timeframe)
        if entry is None:
            return None
        features, cache_key = entry
        df = self.dm.ohlcv.get(symbol, {}).get(timeframe)
        if df is not None and not df.empty:
            current_key = hash(
                (
                    float(df["timestamp"].iloc[-1]),
                    float(df["close"].iloc[-1]),
                    float(df["volume"].iloc[-1]),
                    len(df),
                )
            )
            if cache_key == current_key:
                return features
        return None

    def set(self, symbol: str, timeframe: str, features: np.ndarray):
        """Cache features with a hash of the current OHLCV state as key."""
        df = self.dm.ohlcv.get(symbol, {}).get(timeframe)
        cache_key = 0
        if df is not None and not df.empty:
            cache_key = hash(
                (
                    float(df["timestamp"].iloc[-1]),
                    float(df["close"].iloc[-1]),
                    float(df["volume"].iloc[-1]),
                    len(df),
                )
            )
        if symbol not in self.dm._feature_cache:
            self.dm._feature_cache[symbol] = {}
        self.dm._feature_cache[symbol][timeframe] = (features, cache_key)
