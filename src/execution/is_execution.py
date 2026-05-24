"""
Implementation Shortfall (IS) Execution Algorithm.
Dynamically splits orders into aggressive (cross spread) and passive (limit order)
components based on urgency, market conditions, and signal confidence.

Reference: Almgren & Chriss (2001), Kissell & Glantz (2003).

Key features:
  - Urgency scheduling: high-conviction signals execute fast (at cost)
  - Aggressive/passive split: adaptive ratio based on market conditions
  - Opportunistic price improvement: monitors for favorable micro-prices
  - Real-time adaptivity: shifts strategy if market moves against/with us
"""

import time
import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class ExecutionUrgency(Enum):
    LOW = "low"  # Slow execution, minimize cost
    MEDIUM = "medium"  # Balanced cost vs. speed
    HIGH = "high"  # Fast execution, accept higher cost
    URGENT = "urgent"  # Execute immediately (market order)


@dataclass
class SliceExecution:
    slice_idx: int
    volume: float
    order_type: str  # "AGGRESSIVE" (market) or "PASSIVE" (limit)
    target_price: float
    urgency: str
    timestamp: float = 0.0
    filled: bool = False
    fill_price: float = 0.0
    slippage_bps: float = 0.0


@dataclass
class ExecutionPlan:
    total_volume: float
    slices: List[SliceExecution]
    expected_cost_bps: float
    expected_duration_sec: float
    urgency: ExecutionUrgency
    confidence: float


class ISExecutionEngine:
    """Implementation Shortfall Execution Engine.

    Given an order, produces an execution plan that minimizes the sum of:
      - Market impact cost (price depression from our trading)
      - Timing risk (price moves against us while executing)
      - Opportunity cost (missed favorable moves)

    The optimal trade-off depends on:
      - Signal confidence (high -> execute faster)
      - Market volatility (high -> break into smaller slices)
      - Liquidity (thin -> slower execution)
      - Trade size relative to ADV
    """

    # Urgency parameters: (aggressive_ratio, n_slices, duration_sec)
    URGENCY_PARAMS = {
        ExecutionUrgency.LOW: (0.1, 20, 600),
        ExecutionUrgency.MEDIUM: (0.3, 10, 300),
        ExecutionUrgency.HIGH: (0.6, 5, 120),
        ExecutionUrgency.URGENT: (1.0, 1, 10),
    }

    def __init__(
        self,
        volatility: float = 0.15,
        adv: float = 1e9,
        spread: float = 0.0001,
        max_participation: float = 0.1,
    ):
        self.volatility = volatility
        self.adv = adv
        self.spread = spread
        self.max_participation = max_participation
        self._execution_history: List[SliceExecution] = []

    def plan_execution(
        self,
        volume: float,
        price: float,
        symbol: str = "",
        confidence: float = 0.5,
        urgency: Optional[ExecutionUrgency] = None,
        n_slices: Optional[int] = None,
    ) -> ExecutionPlan:
        """Generate an optimal execution plan.

        Args:
            volume: total order volume
            price: current market price
            symbol: trading symbol (for logging)
            confidence: signal confidence (0-1), higher = more urgent
            urgency: override urgency (auto-computed from confidence if None)
            n_slices: override number of slices (auto-computed if None)

        Returns:
            ExecutionPlan with slice-by-slice breakdown
        """
        # Auto-determine urgency from confidence
        if urgency is None:
            urgency = self._determine_urgency(confidence)

        # Get urgency parameters
        params = self.URGENCY_PARAMS[urgency]
        aggressive_ratio, default_slices, duration_sec = params

        # Allow n_slices override for testing
        if n_slices is not None:
            effective_slices = n_slices
        else:
            effective_slices = default_slices

        # Adjust for trade size vs ADV
        participation = volume / max(self.adv, 1.0)
        if participation > self.max_participation:
            # Very large order: break into more slices, longer duration
            effective_slices = max(
                effective_slices,
                int(participation / self.max_participation * 10),
            )
            duration_sec = duration_sec * (participation / self.max_participation)
            logger.info(
                f"IS: large order {symbol} {volume:.0f} "
                f"({participation:.2%} of ADV) -> {effective_slices} slices, "
                f"{duration_sec:.0f}s"
            )

        # Generate slices
        slices = []

        for i in range(effective_slices):
            # Volume per slice: front-loaded for urgency, uniform for low urgency
            if urgency in (ExecutionUrgency.HIGH, ExecutionUrgency.URGENT):
                weight = self._front_load_weight(i, effective_slices)
            else:
                weight = 1.0 / effective_slices

            # Normalize weights
            total_weight = sum(
                (
                    self._front_load_weight(j, effective_slices)
                    if urgency in (ExecutionUrgency.HIGH, ExecutionUrgency.URGENT)
                    else 1.0 / effective_slices
                )
                for j in range(effective_slices)
            )
            weight /= max(total_weight, 1e-10)

            slice_vol = volume * weight

            # Determine order type
            is_aggressive = (i / effective_slices) < aggressive_ratio

            # Price: aggressive crosses spread, passive uses mid
            if is_aggressive:
                target_price = price * (1 + self.spread / 2)
                order_type = "AGGRESSIVE"
            else:
                target_price = price
                order_type = "PASSIVE"

            slice_exec = SliceExecution(
                slice_idx=i,
                volume=slice_vol,
                order_type=order_type,
                target_price=target_price,
                urgency=urgency.value,
            )
            slices.append(slice_exec)

        # Estimate costs
        expected_cost_bps = self._estimate_total_cost(
            volume, price, aggressive_ratio, effective_slices
        )

        return ExecutionPlan(
            total_volume=volume,
            slices=slices,
            expected_cost_bps=expected_cost_bps,
            expected_duration_sec=duration_sec,
            urgency=urgency,
            confidence=confidence,
        )

    def execute_plan(
        self,
        plan: ExecutionPlan,
        market_order_fn: callable,
        limit_order_fn: Optional[callable] = None,
        price_feed_fn: Optional[callable] = None,
    ) -> List[SliceExecution]:
        """Execute a planned order schedule.

        Args:
            plan: execution plan from plan_execution()
            market_order_fn: callable(volume) -> fill_price or None
            limit_order_fn: optional callable(volume, limit_price) -> fill_price or None
            price_feed_fn: optional callable() -> current_market_price

        Returns:
            list of executed slices with fill prices
        """
        executed = []
        for i, slice_exec in enumerate(plan.slices):
            # Update price if feed available
            current_price = (
                price_feed_fn() if price_feed_fn else slice_exec.target_price
            )

            try:
                if slice_exec.order_type == "AGGRESSIVE":
                    fill_price = market_order_fn(slice_exec.volume)
                elif limit_order_fn:
                    fill_price = limit_order_fn(slice_exec.volume, current_price)
                else:
                    fill_price = market_order_fn(slice_exec.volume)

                if fill_price is not None and fill_price > 0:
                    slice_exec.filled = True
                    slice_exec.fill_price = fill_price
                    slice_exec.timestamp = time.time()
                    slippage_bps = (
                        abs(fill_price - slice_exec.target_price)
                        / max(slice_exec.target_price, 1e-10)
                        * 10_000
                    )
                    slice_exec.slippage_bps = slippage_bps
                    logger.debug(
                        f"IS: slice {i} filled {slice_exec.volume:.0f} "
                        f"@ {fill_price:.5f} "
                        f"(slippage: {slippage_bps:.1f} bps)"
                    )
                else:
                    logger.warning(
                        f"IS: slice {i} NOT filled, " f"volume={slice_exec.volume:.0f}"
                    )

            except Exception as e:
                logger.error(f"IS: slice {i} error: {e}")

            executed.append(slice_exec)

        self._execution_history.extend(executed)

        # Calculate cost summary
        filled_vol = sum(s.volume for s in executed if s.filled)
        total_cost = sum(s.slippage_bps * s.volume for s in executed if s.filled)
        avg_cost_bps = total_cost / max(filled_vol, 1e-10) if filled_vol > 0 else 0.0
        logger.info(
            f"IS: executed {filled_vol:.0f}/{plan.total_volume:.0f} units, "
            f"avg cost {avg_cost_bps:.2f} bps"
        )

        return executed

    def _determine_urgency(self, confidence: float) -> ExecutionUrgency:
        """Map signal confidence to execution urgency."""
        if confidence >= 0.85:
            return ExecutionUrgency.URGENT
        elif confidence >= 0.70:
            return ExecutionUrgency.HIGH
        elif confidence >= 0.55:
            return ExecutionUrgency.MEDIUM
        return ExecutionUrgency.LOW

    def _front_load_weight(self, idx: int, n_slices: int) -> float:
        """Front-loaded weight distribution (concave)."""
        return 1.0 / (idx + 1) ** 0.5

    def _estimate_total_cost(
        self,
        volume: float,
        price: float,
        aggressive_ratio: float,
        n_slices: int,
    ) -> float:
        """Estimate total execution cost in bps."""
        participation = volume / max(self.adv, 1.0)
        # Market impact: square-root model
        impact = self.volatility * (participation**0.5) * 10_000
        # Spread cost for aggressive portion
        spread_cost = (
            aggressive_ratio * (self.spread / price) * 10_000 if price > 0 else 0
        )
        # Timing risk
        timing = self.volatility * 10_000 / np.sqrt(n_slices)
        return float(impact + spread_cost + timing)

    def get_execution_summary(self) -> Dict:
        """Get summary of recent execution quality."""
        if not self._execution_history:
            return {"total_slices": 0, "avg_slippage_bps": 0.0, "fill_rate": 0.0}
        filled = [s for s in self._execution_history if s.filled]
        fill_rate = len(filled) / max(len(self._execution_history), 1)
        avg_slip = np.mean([s.slippage_bps for s in filled]) if filled else 0.0
        return {
            "total_slices": len(self._execution_history),
            "filled_slices": len(filled),
            "fill_rate": fill_rate,
            "avg_slippage_bps": float(avg_slip),
        }
