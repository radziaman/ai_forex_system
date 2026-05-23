"""
Execution quality tracking and slice planning.
Monitors slippage, fill rates, and market impact.
"""

import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class FillRecord:
    symbol: str
    direction: str
    expected_price: float
    filled_price: float
    volume: float
    slippage_pips: float
    slippage_pct: float
    timestamp: float


@dataclass
class ExecutionQualityReport:
    avg_slippage_pips: float = 0.0
    p95_slippage_pips: float = 0.0
    fill_rate: float = 1.0
    market_impact_bps: float = 0.0


class SlicePlanner:
    """Plan order slices for TWAP and VWAP execution."""

    @staticmethod
    def plan_twap(
        total_volume: float, n_slices: int, duration_sec: float
    ) -> List[float]:
        """Generate TWAP slices with ±10% randomization."""
        if n_slices <= 0:
            return [total_volume]
        base = total_volume / n_slices
        slices: List[float] = []
        for _ in range(n_slices - 1):
            dev = random.uniform(-0.1, 0.1)
            slice_vol = base * (1 + dev)
            slice_vol = max(0.0, slice_vol)
            slices.append(slice_vol)
        last = total_volume - sum(slices)
        last = max(0.0, last)
        slices.append(last)
        return slices

    @staticmethod
    def plan_vwap(total_volume: float, volume_profile: List[float]) -> List[float]:
        """Generate VWAP slices proportional to a volume profile."""
        if not volume_profile:
            return [total_volume]
        total_weight = sum(volume_profile)
        if total_weight <= 0:
            return [total_volume]
        normalized = [w / total_weight for w in volume_profile]
        slices: List[float] = []
        for w in normalized[:-1]:
            slices.append(total_volume * w)
        last = total_volume - sum(slices)
        slices.append(last)
        return slices


class ExecutionQualityTracker:
    """Track execution quality metrics: slippage, fill rate, market impact."""

    def __init__(self):
        self._fills: Dict[str, List[FillRecord]] = defaultdict(list)
        self._attempts: Dict[str, int] = defaultdict(int)
        self._adv: Dict[str, float] = {}
        self._market_conditions: Dict[str, Dict[str, float]] = {}

    @staticmethod
    def _pip_size(symbol: str) -> float:
        sym = symbol.upper()
        if sym in ("XAUUSD", "XTIUSD"):
            return 0.01
        if sym in ("BTCUSD",):
            return 1.0
        if sym in ("US500",):
            return 1.0
        if "JPY" in sym:
            return 0.01
        return 0.0001

    def record_order_attempt(self, symbol: str) -> None:
        """Record an order placement attempt."""
        self._attempts[symbol.upper()] += 1

    def record_fill(
        self,
        symbol: str,
        expected_price: float,
        filled_price: float,
        direction: str,
        volume: float,
    ) -> None:
        """Record a fill and compute slippage metrics."""
        sym = symbol.upper()
        pip_size = self._pip_size(sym)
        slippage_price = abs(filled_price - expected_price)
        slippage_pips = slippage_price / pip_size if pip_size > 0 else 0.0
        slippage_pct = (
            (slippage_price / expected_price * 100) if expected_price > 0 else 0.0
        )
        record = FillRecord(
            symbol=sym,
            direction=direction,
            expected_price=expected_price,
            filled_price=filled_price,
            volume=volume,
            slippage_pips=slippage_pips,
            slippage_pct=slippage_pct,
            timestamp=time.time(),
        )
        self._fills[sym].append(record)

    def get_slippage_distribution(self, symbol: str) -> Dict[str, float]:
        """Return mean, p50, and p95 slippage in pips."""
        sym = symbol.upper()
        fills = self._fills.get(sym, [])
        if not fills:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
        slippages = [f.slippage_pips for f in fills]
        mean = float(np.mean(slippages))
        p50 = float(np.percentile(slippages, 50))
        p95 = float(np.percentile(slippages, 95))
        return {"mean": mean, "p50": p50, "p95": p95}

    def set_adv(self, symbol: str, adv: float) -> None:
        """Set average daily volume for a symbol."""
        self._adv[symbol.upper()] = adv

    def set_market_conditions(self, symbol: str, atr: float, price: float) -> None:
        """Store latest market conditions for market-impact reporting."""
        self._market_conditions[symbol.upper()] = {"atr": atr, "price": price}

    def calculate_market_impact(
        self, symbol: str, volume: float, atr: float, price: float
    ) -> float:
        """Estimate market impact in basis points using a square-root model."""
        sym = symbol.upper()
        adv = self._adv.get(sym, 0.0)
        if adv <= 0 or volume <= 0 or atr <= 0 or price <= 0:
            return 0.0
        impact_fraction = (volume / adv) ** 0.5 * (atr / price)
        return impact_fraction * 10_000

    def get_fill_quality_report(self, symbol: str) -> ExecutionQualityReport:
        """Generate an execution quality report for a symbol."""
        sym = symbol.upper()
        fills = self._fills.get(sym, [])
        if not fills:
            return ExecutionQualityReport()

        slippages = [f.slippage_pips for f in fills]
        avg_slippage = float(np.mean(slippages))
        p95_slippage = float(np.percentile(slippages, 95))

        attempts = self._attempts.get(sym, 0)
        fill_rate = len(fills) / attempts if attempts > 0 else 1.0

        market_impact_bps = 0.0
        conditions = self._market_conditions.get(sym)
        adv = self._adv.get(sym, 0.0)
        if conditions and adv > 0:
            avg_volume = float(np.mean([f.volume for f in fills]))
            market_impact_bps = self.calculate_market_impact(
                sym, avg_volume, conditions["atr"], conditions["price"]
            )

        return ExecutionQualityReport(
            avg_slippage_pips=avg_slippage,
            p95_slippage_pips=p95_slippage,
            fill_rate=fill_rate,
            market_impact_bps=market_impact_bps,
        )

    def should_slice(self, volume: float) -> bool:
        """Recommend slicing if order exceeds 0.05 lots (5,000 units)."""
        return volume > 5_000.0

    def plan_slices(self, volume: float, method: str, **kwargs) -> List[float]:
        """Plan order slices using the requested method."""
        if method == "twap":
            return SlicePlanner.plan_twap(
                volume,
                kwargs.get("n_slices", 5),
                kwargs.get("duration_sec", 300.0),
            )
        if method == "vwap":
            return SlicePlanner.plan_vwap(
                volume,
                kwargs.get("volume_profile", [1.0]),
            )
        raise ValueError(f"Unknown slice method: {method}")

    def estimate_slippage(
        self,
        symbol: str,
        direction: str,
        volume: float,
        depth: Optional[Dict],
    ) -> float:
        """Estimate slippage in pips from DOM depth levels."""
        if depth is None or volume <= 0:
            return 0.0
        sym = symbol.upper()
        pip_size = self._pip_size(sym)

        levels = depth.get("asks", []) if direction == "BUY" else depth.get("bids", [])

        if not levels:
            return 0.0

        best_price = levels[0][0]
        remaining = volume
        weighted_sum = 0.0
        filled = 0.0

        for price, lvl_vol in levels:
            if remaining <= 0:
                break
            take = min(remaining, lvl_vol)
            weighted_sum += take * price
            filled += take
            remaining -= take

        if remaining > 0 and levels:
            last_price = levels[-1][0]
            weighted_sum += remaining * last_price
            filled += remaining

        if filled <= 0:
            return 0.0

        avg_price = weighted_sum / filled
        if direction == "BUY":
            slippage_price = avg_price - best_price
        else:
            slippage_price = best_price - avg_price

        slippage_price = max(0.0, slippage_price)
        return slippage_price / pip_size if pip_size > 0 else 0.0
