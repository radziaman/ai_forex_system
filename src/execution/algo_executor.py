"""
Smart Order Execution Algorithms: Iceberg, TWAP, VWAP, Implementation Shortfall.
Reduces market impact and slippage for large orders.
Enhanced with smart order routing and volume profile-based execution (Enhancement #8).
"""
import asyncio
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from .engine import TradeOrder, TradeRecord


class RoutingStrategy(Enum):
    """Smart order routing strategies."""
    BEST_PRICE = "best_price"
    LOWEST_IMPACT = "lowest_impact"
    VOLUME_PROFILE = "volume_profile"
    ADAPTIVE = "adaptive"


@dataclass
class ExecutionAlgoConfig:
    """Configuration for execution algorithms."""
    display_size: int = 1000  # Visible order size
    slice_interval: float = 300.0  # 5 minutes between slices
    max_participation_rate: float = 0.1  # Max % of volume to participate
    urgency: str = "normal"  # low | normal | high | immediate
    routing_strategy: RoutingStrategy = RoutingStrategy.ADAPTIVE
    max_slippage_pips: float = 5.0
    enable_smart_routing: bool = True


class AlgoExecutor:
    """
    Smart order execution algorithms.
    Reduces slippage and market impact for large orders.
    """

    def __init__(self, client, data_manager, cost_model):
        self.client = client
        self.data = data_manager
        self.cost_model = cost_model
        self.active_orders: Dict[str, List] = {}  # order_id -> [slices]

    async def iceberg_order(
        self,
        symbol: str,
        side: str,
        total_volume: float,
        display_size: Optional[int] = None,
        price: Optional[float] = None,
    ) -> List[TradeRecord]:
        """
        Hide true order size by showing small pieces (icebergs).
        Refreshes display when slice is filled.
        """
        display = display_size or ExecutionAlgoConfig.display_size
        remaining = total_volume
        results = []
        order_id = f"iceberg_{symbol}_{int(time.time())}"

        logger.info(f"ICEBERG: {side} {total_volume} {symbol} in slices of {display}")

        self.active_orders[order_id] = []

        while remaining > 0:
            show = min(display, int(remaining))
            current_price = price or self.data.get_price(symbol, "1h") or 1.12

            order = TradeOrder(
                symbol=symbol,
                side=side,
                order_type="LIMIT",
                volume=show,
                price=current_price,
            )

            result = await self.client.place_order(order)

            if result and result.status == "FILLED":
                results.append(result)
                remaining -= show
                self.active_orders[order_id].append(result)
                logger.debug(f"ICEBERG: Filled {show}, remaining {remaining}")
            else:
                # Order not filled, wait and retry
                await asyncio.sleep(1.0)

            # Avoid hammering
            await asyncio.sleep(0.1)

        logger.success(f"ICEBERG COMPLETE: {len(results)} slices executed")
        return results

    async def twap_execution(
        self,
        symbol: str,
        side: str,
        volume: float,
        duration_minutes: int = 30,
        start_immediately: bool = True,
    ) -> List[TradeRecord]:
        """
        Time-Weighted Average Price execution.
        Enhanced with adaptive slicing based on volatility (Enhancement #8).
        """
        chunks = max(1, int(duration_minutes / 5))  # One chunk per 5 min
        chunk_size = volume / chunks
        results = []
        order_id = f"twap_{symbol}_{int(time.time())}"

        logger.info(f"TWAP: {side} {volume} {symbol} over {duration_minutes}min ({chunks} chunks)")

        self.active_orders[order_id] = []

        # Get volatility for adaptive sizing
        atr = self.data.get_atr(symbol, "1h", 14) if hasattr(self.data, 'get_atr') else 0.0
        current_price = self.data.get_price(symbol, "1h") or 1.12

        for i in range(chunks):
            # Adaptive chunk sizing based on volatility
            if atr > 0 and current_price > 0:
                vol_adjustment = min(2.0, max(0.5, 0.01 / (atr / current_price)))
                adjusted_chunk = chunk_size * vol_adjustment
            else:
                adjusted_chunk = chunk_size

            order = TradeOrder(
                symbol=symbol,
                side=side,
                order_type="MARKET",
                volume=int(adjusted_chunk),
            )

            result = await self.client.place_order(order)

            if result:
                results.append(result)
                self.active_orders[order_id].append(result)

            # Wait for next time slice (except last chunk)
            if i < chunks - 1:
                await asyncio.sleep(300)  # 5 minutes

        avg_price = np.mean([r.filled_price for r in results if r.filled_price > 0]) if results else 0.0
        logger.success(f"TWAP COMPLETE: {len(results)} chunks, avg price={avg_price:.5f}")
        return results

    async def vwap_execution(
        self,
        symbol: str,
        side: str,
        volume: float,
        duration_minutes: int = 30,
    ) -> List[TradeRecord]:
        """
        Volume-Weighted Average Price execution.
        Enhanced with historical volume profile (Enhancement #8).
        """
        chunks = max(1, int(duration_minutes / 5))
        results = []
        order_id = f"vwap_{symbol}_{int(time.time())}"

        # Get volume profile from historical data
        volume_profile = self._get_volume_profile(symbol, duration_minutes)

        logger.info(f"VWAP: {side} {volume} {symbol} over {duration_minutes}min")

        self.active_orders[order_id] = []

        for i in range(chunks):
            # Slice size proportional to volume profile
            profile_weight = volume_profile[i] if i < len(volume_profile) else 1.0 / chunks
            chunk_size = volume * profile_weight

            order = TradeOrder(
                symbol=symbol,
                side=side,
                order_type="MARKET",
                volume=int(chunk_size),
            )

            result = await self.client.place_order(order)
            if result:
                results.append(result)
                self.active_orders[order_id].append(result)

            if i < chunks - 1:
                await asyncio.sleep(300)

        avg_price = np.mean([r.filled_price for r in results if r.filled_price > 0]) if results else 0.0
        logger.success(f"VWAP COMPLETE: {len(results)} chunks, avg price={avg_price:.5f}")
        return results

    def _get_volume_profile(self, symbol: str, duration_minutes: int) -> List[float]:
        """Get historical volume profile for VWAP execution."""
        try:
            df = self.data.get_ohlcv(symbol, "1h")
            if df is not None and len(df) > 24:
                # Get last 24 hours of volume data
                recent_volumes = df["volume"].tail(24).values
                if len(recent_volumes) > 0:
                    # Normalize to get proportional weights
                    total_vol = recent_volumes.sum()
                    if total_vol > 0:
                        weights = recent_volumes / total_vol
                        # Repeat pattern for duration
                        chunks_needed = max(1, int(duration_minutes / 5))
                        profile = []
                        while len(profile) < chunks_needed:
                            profile.extend(weights.tolist())
                        return profile[:chunks_needed]
        except Exception:
            pass
        # Fallback to equal weights
        chunks = max(1, int(duration_minutes / 5))
        return [1.0 / chunks] * chunks

    async def smart_order_routing(
        self,
        symbol: str,
        side: str,
        volume: float,
        strategy: RoutingStrategy = RoutingStrategy.ADAPTIVE,
    ) -> List[TradeRecord]:
        """
        Smart order routing across multiple venues (Enhancement #8).
        Currently simplified - in production, route across cTrader, FIX, etc.
        """
        logger.info(f"Smart routing: {side} {volume} {symbol} via {strategy.value}")

        # For now, use TWAP as default routing strategy
        if strategy == RoutingStrategy.BEST_PRICE:
            return await self._route_best_price(symbol, side, volume)
        elif strategy == RoutingStrategy.LOWEST_IMPACT:
            return await self.twap_execution(symbol, side, volume, duration_minutes=60)
        elif strategy == RoutingStrategy.VOLUME_PROFILE:
            return await self.vwap_execution(symbol, side, volume)
        else:  # ADAPTIVE
            return await self._route_adaptive(symbol, side, volume)

    async def _route_best_price(self, symbol: str, side: str, volume: float) -> List[TradeRecord]:
        """Route to venue with best price."""
        # Simplified - just use market order
        order = TradeOrder(symbol=symbol, side=side, order_type="MARKET", volume=int(volume))
        result = await self.client.place_order(order)
        return [result] if result else []

    async def _route_adaptive(self, symbol: str, side: str, volume: float) -> List[TradeRecord]:
        """Adaptive routing based on market conditions."""
        # Check volatility to decide strategy
        atr = self.data.get_atr(symbol, "1h", 14) if hasattr(self.data, 'get_atr') else 0.0
        current_price = self.data.get_price(symbol, "1h") or 1.12

        if atr > 0 and current_price > 0:
            vol_pct = atr / current_price
            if vol_pct > 0.02:  # High volatility
                logger.info(f"High volatility detected ({vol_pct:.2%}), using TWAP")
                return await self.twap_execution(symbol, side, volume, duration_minutes=60)
            else:
                logger.info(f"Normal volatility ({vol_pct:.2%}), using VWAP")
                return await self.vwap_execution(symbol, side, volume)
        else:
            return await self.twap_execution(symbol, side, volume)

    async def implementation_shortfall(
        self,
        symbol: str,
        side: str,
        volume: float,
        target_price: float,
        urgency: str = "normal",
    ) -> List[TradeRecord]:
        """
        Implementation Shortfall: balance market impact vs. opportunity cost.
        More aggressive if price moves away from target.
        """
        results = []
        order_id = f"is_{symbol}_{int(time.time())}"
        remaining = volume
        participated = 0.0

        # Urgency affects participation rate
        urgency_rates = {"low": 0.05, "normal": 0.1, "high": 0.2, "immediate": 1.0}
        participation_rate = urgency_rates.get(urgency, 0.1)

        logger.info(f"IS: {side} {volume} {symbol} @ target={target_price:.5f}, urgency={urgency}")

        self.active_orders[order_id] = []

        while remaining > 0 and participated < volume:
            current_price = self.data.get_price(symbol, "1h")
            if current_price is None:
                break

            # Calculate shortfall
            if side == "BUY":
                shortfall = max(0, current_price - target_price) / target_price
            else:
                shortfall = max(0, target_price - current_price) / target_price

            # Adjust slice size based on shortfall
            if shortfall > 0.01:  # Price moved against us >1%
                slice_size = remaining  # Go aggressive
            else:
                slice_size = min(remaining, volume * participation_rate)

            order = TradeOrder(
                symbol=symbol,
                side=side,
                order_type="MARKET",
                volume=int(slice_size),
            )

            result = await self.client.place_order(order)
            if result:
                results.append(result)
                self.active_orders[order_id].append(result)
                remaining -= slice_size
                participated += slice_size

            await asyncio.sleep(1.0)

        avg_price = np.mean([r.filled_price for r in results if r.filled_price > 0]) if results else 0.0
        logger.success(f"IS COMPLETE: {len(results)} orders, avg price={avg_price:.5f}")
        return results

    def cancel_all_active(self, order_id: Optional[str] = None):
        """Cancel active algo orders."""
        if order_id and order_id in self.active_orders:
            # In production: send cancel requests for all child orders
            del self.active_orders[order_id]
            logger.info(f"Cancelled algo order {order_id}")
        elif not order_id:
            self.active_orders.clear()
            logger.info("Cancelled all algo orders")
