"""
Smart Order Execution Algorithms: Iceberg, TWAP, VWAP, Implementation Shortfall.
Reduces market impact and slippage for large orders.
"""
import asyncio
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

from .engine import TradeOrder, TradeRecord


@dataclass
class ExecutionAlgoConfig:
    """Configuration for execution algorithms."""
    display_size: int = 1000  # Visible order size
    slice_interval: float = 300.0  # 5 minutes between slices
    max_participation_rate: float = 0.1  # Max % of volume to participate
    urgency: str = "normal"  # low | normal | high | immediate


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
        Splits order into equal time slices.
        """
        chunks = max(1, int(duration_minutes / 5))  # One chunk per 5 min
        chunk_size = volume / chunks
        results = []
        order_id = f"twap_{symbol}_{int(time.time())}"

        logger.info(f"TWAP: {side} {volume} {symbol} over {duration_minutes}min ({chunks} chunks)")

        self.active_orders[order_id] = []

        for i in range(chunks):
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
        Slices proportional to historical volume profile.
        """
        # Get historical volume profile (simplified: equal weights)
        # In production, use actual volume profile (more volume in active hours)
        chunks = max(1, int(duration_minutes / 5))
        results = []
        order_id = f"vwap_{symbol}_{int(time.time())}"

        # Simple volume profile: assume equal distribution
        chunk_size = volume / chunks

        logger.info(f"VWAP: {side} {volume} {symbol} over {duration_minutes}min")

        self.active_orders[order_id] = []

        for i in range(chunks):
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
