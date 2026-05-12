"""
Institutional Order Flow Analytics + Gamma Exposure Mapping

Features:
- CVD (Cumulative Volume Delta) enhancement
- Footprint chart data (bid/ask volume at each price level)
- Gamma exposure estimation (placeholder for options data)
- Dark pool detection (if data available)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class FootprintBar:
    """Footprint chart bar showing volume at each price level."""

    timestamp: float
    symbol: str
    open: float
    high: float
    low: float
    close: float
    bid_volume: Dict[float, float]  # price -> volume
    ask_volume: Dict[float, float]  # price -> volume
    delta: float = 0.0  # net delta for this bar


@dataclass
class GammaLevel:
    """Gamma exposure at a specific price level."""

    price: float
    gamma: float  # positive = dealers long gamma, negative = short gamma
    oi: float = 0.0  # open interest
    is_call: bool = True


class OrderFlowAnalyzer:
    """Institutional-grade order flow analysis."""

    def __init__(self, n_levels: int = 10):
        self.n_levels = n_levels
        self.footprint_bars: Dict[str, List[FootprintBar]] = {}
        self._cvd_history: Dict[str, List[float]] = {}
        self._price_volume_map: Dict[str, Dict[float, Dict[str, float]]] = {}

    def update_tick(
        self, symbol: str, bid: float, ask: float, volume: float, timestamp: float
    ):
        """Update order flow with new tick data."""
        mid = (bid + ask) / 2.0
        price_key = round(mid, 5)  # Round to 5 decimal places

        if symbol not in self._price_volume_map:
            self._price_volume_map[symbol] = {}
        if price_key not in self._price_volume_map[symbol]:
            self._price_volume_map[symbol][price_key] = {
                "bid": 0.0,
                "ask": 0.0,
                "total": 0.0,
            }

        # Classify as buying or selling volume based on whether it's closer to bid or ask
        if mid >= ask:
            self._price_volume_map[symbol][price_key]["ask"] += volume
        elif mid <= bid:
            self._price_volume_map[symbol][price_key]["bid"] += volume
        else:
            # Middle - split based on momentum
            self._price_volume_map[symbol][price_key]["total"] += volume

        self._price_volume_map[symbol][price_key]["total"] += volume

        # Update CVD
        if symbol not in self._cvd_history:
            self._cvd_history[symbol] = []
        cvd_delta = volume if mid >= ask else -volume if mid <= bid else 0
        last_cvd = self._cvd_history[symbol][-1] if self._cvd_history[symbol] else 0.0
        self._cvd_history[symbol].append(last_cvd + cvd_delta)

        # Keep limited history
        if len(self._cvd_history[symbol]) > 1000:
            self._cvd_history[symbol] = self._cvd_history[symbol][-500:]

    def get_cvd(self, symbol: str) -> float:
        """Get current CVD (Cumulative Volume Delta)."""
        history = self._cvd_history.get(symbol, [])
        return history[-1] if history else 0.0

    def get_cvd_slope(self, symbol: str, periods: int = 20) -> float:
        """Get CVD slope (momentum indicator)."""
        history = self._cvd_history.get(symbol, [])
        if len(history) < periods:
            return 0.0
        recent = history[-periods:]
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        return float(slope)

    def get_price_imbalance(self, symbol: str, levels: int = 5) -> float:
        """Get bid/ask volume imbalance at current price levels."""
        pvm = self._price_volume_map.get(symbol, {})
        if not pvm:
            return 0.0

        sorted_prices = sorted(pvm.keys(), reverse=True)[:levels]
        bid_vol = sum(pvm[p]["bid"] for p in sorted_prices)
        ask_vol = sum(pvm[p]["ask"] for p in sorted_prices)
        total = bid_vol + ask_vol
        return (bid_vol - ask_vol) / total if total > 0 else 0.0

    def build_footprint_bar(
        self, symbol: str, tf: str = "1m"
    ) -> Optional[FootprintBar]:
        """Build a footprint chart bar from aggregated tick data."""
        pvm = self._price_volume_map.get(symbol, {})
        if not pvm:
            return None

        prices = sorted(pvm.keys())
        if not prices:
            return None

        bar = FootprintBar(
            timestamp=pd.Timestamp.now().timestamp(),
            symbol=symbol,
            open=prices[0],
            high=max(prices),
            low=min(prices),
            close=prices[-1],
            bid_volume={p: pvm[p]["bid"] for p in prices},
            ask_volume={p: pvm[p]["ask"] for p in prices},
        )
        bar.delta = sum(pvm[p]["bid"] - pvm[p]["ask"] for p in prices)
        return bar

    def get_order_flow_features(self, symbol: str) -> Dict:
        """Get all order flow features for ML model."""
        return {
            "cvd": self.get_cvd(symbol),
            "cvd_slope": self.get_cvd_slope(symbol),
            "price_imbalance": self.get_price_imbalance(symbol),
            "cvd_hist": self._cvd_history.get(symbol, [])[-100:],
        }


class GammaExposureMapper:
    """
    Gamma Exposure Mapping for institutional trading.

    Note: Requires options flow data (not available via cTrader).
    This module provides the structure for when options data is sourced.
    """

    def __init__(self):
        self.gamma_levels: Dict[str, List[GammaLevel]] = {}
        self.spot_price: Dict[str, float] = {}
        self._gamma_flip_points: Dict[str, float] = {}

    def update_spot(self, symbol: str, price: float):
        """Update spot price for gamma calculations."""
        self.spot_price[symbol] = price

    def add_gamma_level(
        self,
        symbol: str,
        price: float,
        gamma: float,
        oi: float = 0.0,
        is_call: bool = True,
    ):
        """Add gamma exposure at a specific price level."""
        if symbol not in self.gamma_levels:
            self.gamma_levels[symbol] = []
        self.gamma_levels[symbol].append(
            GammaLevel(price=price, gamma=gamma, oi=oi, is_call=is_call)
        )
        self._recalculate_flip_point(symbol)

    def _recalculate_flip_point(self, symbol: str):
        """Find gamma flip point (where dealer positioning changes)."""
        levels = self.gamma_levels.get(symbol, [])
        if not levels:
            return
        sorted_levels = sorted(levels, key=lambda x: x.price)
        cumulative_gamma = 0.0
        for level in sorted_levels:
            cumulative_gamma += level.gamma
            if cumulative_gamma > 0:
                self._gamma_flip_points[symbol] = level.price
                return
        self._gamma_flip_points[symbol] = (
            sorted_levels[-1].price if sorted_levels else 0.0
        )

    def get_gamma_exposure(self, symbol: str) -> Dict:
        """Get gamma exposure metrics."""
        spot = self.spot_price.get(symbol, 0.0)
        flip = self._gamma_flip_points.get(symbol, 0.0)
        levels = self.gamma_levels.get(symbol, [])

        if not levels:
            return {
                "gamma_exposure": 0.0,
                "gamma_flip_point": 0.0,
                "dealer_position": "unknown",
                "total_gamma": 0.0,
            }

        total_gamma = sum(l.gamma for l in levels)
        return {
            "gamma_exposure": total_gamma,
            "gamma_flip_point": flip,
            "dealer_position": "long_gamma" if total_gamma > 0 else "short_gamma",
            "total_gamma": total_gamma,
            "spot_vs_flip": spot - flip if flip else 0.0,
        }

    def detect_gamma_pin(self, symbol: str, threshold: float = 0.001) -> bool:
        """Detect if price is pinned to a gamma level."""
        spot = self.spot_price.get(symbol, 0.0)
        levels = self.gamma_levels.get(symbol, [])
        for level in levels:
            if abs(spot - level.price) / spot < threshold:
                return True
        return False


class DarkPoolDetector:
    """
    Dark Pool Detection for institutional trading.

    Note: cTrader does not provide dark pool data.
    This is a placeholder for when alternative data sources are integrated.
    """

    def __init__(self):
        self.dark_pool_volume: Dict[str, float] = {}
        self.total_volume: Dict[str, float] = {}
        self._alert_threshold = 0.3  # 30% dark pool volume triggers alert

    def update_volume(self, symbol: str, lit_volume: float, dark_volume: float = 0.0):
        """Update volume data (dark pool data from alternative source)."""
        self.total_volume[symbol] = self.total_volume.get(symbol, 0.0) + lit_volume
        self.dark_pool_volume[symbol] = (
            self.dark_pool_volume.get(symbol, 0.0) + dark_volume
        )

    def get_dark_pool_ratio(self, symbol: str) -> float:
        """Get ratio of dark pool to total volume."""
        total = self.total_volume.get(symbol, 0.0)
        dark = self.dark_pool_volume.get(symbol, 0.0)
        return dark / total if total > 0 else 0.0

    def is_dark_pool_active(self, symbol: str) -> bool:
        """Check if dark pool activity is significant."""
        return self.get_dark_pool_ratio(symbol) > self._alert_threshold

    def get_dark_pool_metrics(self, symbol: str) -> Dict:
        """Get dark pool detection metrics."""
        return {
            "dark_pool_ratio": self.get_dark_pool_ratio(symbol),
            "dark_pool_active": self.is_dark_pool_active(symbol),
            "total_dark_volume": self.dark_pool_volume.get(symbol, 0.0),
            "total_lit_volume": self.total_volume.get(symbol, 0.0)
            - self.dark_pool_volume.get(symbol, 0.0),
        }
