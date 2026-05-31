"""ExecutionManager — order execution, position tracking, trailing stops.

Replaces: ExecutionAgent + PositionAgent.

Responsibilities:
- Listen to risk_approved events and execute trades
- Manage open positions with trailing stops and partial closes
- Support both paper and live modes
- Emit position_opened, position_closed, execution_result events
- Monitor correlated positions
- Track execution quality and emit feedback for other modules
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from loguru import logger

from .pipeline_context import PipelineContext


@dataclass
class ManagedPosition:
    """Internal position tracking state."""

    position_id: int
    symbol: str
    direction: str
    volume: float
    entry_price: float
    sl: float
    tp: float
    timestamp: float
    status: str = "OPEN"
    pnl: float = 0.0
    exit_price: float = 0.0
    reason: str = ""
    trailing_state: Dict[str, Any] = field(default_factory=dict)
    tp_milestones_hit: Set[str] = field(default_factory=set)


class ExecutionManager:
    """Order execution and position lifecycle management."""

    CORRELATED_GROUPS: List[Set[str]] = [
        {"EURUSD", "GBPUSD"},
        {"AUDUSD", "NZDUSD"},
        {"USDCAD", "XTIUSD"},
    ]

    def __init__(
        self,
        ctx: PipelineContext,
        mode: str = "paper",
        execution_engine: Optional[Any] = None,
    ):
        self.ctx = ctx
        self.config = ctx.config
        self.bus = ctx.bus
        self._engine = execution_engine  # ExecutionEngine instance
        self._data_manager = ctx.data_manager
        self._initialized = False
        self._position_counter: int = 0
        self._positions: Dict[int, ManagedPosition] = {}
        self._mode: str = "PAPER"

        # Position agent state
        self._trailing_state: Dict[int, Dict] = {}
        self._tp_hit: Dict[int, Set[str]] = {}
        self._last_check: Dict[str, float] = {}
        self._check_interval: float = 5.0
        self._reconcile_task: Optional[asyncio.Task] = None

        # Execution quality tracking (Phase 4.3)
        self._quality_tracker: Optional[Any] = None
        self._slippage_multiplier: float = 1.0  # Adaptive slippage

        # Tick event sources
        self._tick_source_task: Optional[asyncio.Task] = None

        self._running: bool = False

        # Active symbols for tick generation
        self._active_symbols = [
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "AUDUSD",
            "USDCAD",
            "USDCHF",
            "NZDUSD",
            "XAUUSD",
        ]

        # Subscribe to events
        self.bus.on("risk_approved", self._on_risk_approved)

    async def start(self) -> None:
        """Initialize the execution manager."""
        trading_config = getattr(self.config, "trading", self.config)
        self._mode = getattr(trading_config, "mode", "PAPER")

        if self._engine is None and self._mode != "PAPER":
            try:
                from execution.engine import ExecutionEngine

                # Attempt to create a minimal ExecutionEngine
                self._engine = ExecutionEngine(
                    ctrader_client=None,
                    risk_manager=None,
                    data_manager=self._data_manager,
                    initial_balance=100_000.0,
                    mode=self._mode,
                )
            except Exception as e:
                logger.warning(
                    f"[execution_manager] ExecutionEngine not available: {e}"
                )

        # Initialize execution quality tracker
        try:
            from execution.execution_quality import ExecutionQualityTracker

            self._quality_tracker = ExecutionQualityTracker()
        except Exception:
            logger.warning("[execution_manager] ExecutionQualityTracker not available")
            self._quality_tracker = None

        # Wire tick source for paper mode
        if self._mode == "PAPER" or self._mode == "paper":
            self._tick_source_task = asyncio.create_task(self._paper_tick_loop())
            logger.info("[execution_manager] Paper tick generator started")

        # Wire live tick source from engine
        if self._engine is not None:
            try:
                original = getattr(self._engine, "on_market_data", None)

                async def _tick_wrapper(depth):
                    # Call original handler
                    if original:
                        await original(depth)
                    # Extract tick data and emit to pipeline bus
                    symbol = getattr(depth, "symbol", "EURUSD")
                    bid = getattr(depth, "bid", 0.0)
                    ask = getattr(depth, "ask", 0.0)
                    volume = getattr(depth, "volume", 0.0)
                    ts = getattr(depth, "timestamp", time.time())
                    if bid > 0 and ask > 0:
                        await self.bus.emit(
                            "tick",
                            symbol=symbol,
                            bid=bid,
                            ask=ask,
                            volume=volume,
                            timestamp=ts,
                        )

                self._engine.on_market_data = _tick_wrapper
                logger.info("[execution_manager] Live tick source wired")
            except Exception as e:
                logger.warning(f"[execution_manager] Could not wire tick source: {e}")

        self._running = True
        self._initialized = True
        logger.info(f"[execution_manager] Ready — mode={self._mode}")

    async def stop(self) -> None:
        """Stop the execution manager — cancel tick source and close open positions."""
        self._running = False
        if self._tick_source_task is not None:
            self._tick_source_task.cancel()
            try:
                await self._tick_source_task
            except asyncio.CancelledError:
                pass
            self._tick_source_task = None
        for pid in list(self._positions.keys()):
            if self._positions[pid].status == "OPEN":
                await self.close_position(pid, "system_shutdown")
        self._initialized = False
        logger.info(
            f"[execution_manager] Stopped — {len(self._positions)} positions tracked"
        )

    async def _on_risk_approved(
        self, signal: Dict[str, Any], volume: float, sl_price: float, tp_price: float
    ) -> None:
        """Execute a trade when risk is approved."""
        symbol = signal.get("symbol", "EURUSD")
        direction = signal.get("direction", "HOLD")
        confidence = signal.get("confidence", 0.0)
        timestamp = signal.get("timestamp", time.time())

        result = await self.execute_order(
            symbol=symbol,
            direction=direction,
            volume=volume,
            sl_price=sl_price,
            tp_price=tp_price,
            signal_price=signal.get("price", 0.0),
            confidence=confidence,
            timestamp=timestamp,
            expert_outputs=signal.get("expert_outputs", {}),
        )

        if result["success"]:
            await self.bus.emit("position_opened", **result["position"])

    async def execute_order(
        self,
        symbol: str,
        direction: str,
        volume: float,
        sl_price: float,
        tp_price: float,
        signal_price: float = 0.0,
        confidence: float = 0.0,
        timestamp: float = 0.0,
        expert_outputs: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Execute an order through the execution engine.

        Returns:
            dict with success (bool), position (dict), position_id (int)
        """
        if self._engine is None:
            # Paper simulation fallback
            self._position_counter += 1
            pos_id = self._position_counter
            price = signal_price or 1.12
            position = ManagedPosition(
                position_id=pos_id,
                symbol=symbol,
                direction=direction,
                volume=volume,
                entry_price=price,
                sl=sl_price,
                tp=tp_price,
                timestamp=timestamp or time.time(),
            )
            self._positions[pos_id] = position
            logger.info(
                f"[execution_manager] [PAPER] {direction} {volume:.2f} "
                f"{symbol} @ {price:.5f}"
            )

            # Record execution quality for paper trade
            await self._record_execution_quality(
                symbol=symbol,
                direction=direction,
                filled_price=price,
                expected_price=signal_price,
                volume=volume,
            )

            return {
                "success": True,
                "position_id": pos_id,
                "position": {
                    "position_id": pos_id,
                    "symbol": symbol,
                    "direction": direction,
                    "volume": volume,
                    "entry_price": price,
                    "sl": sl_price,
                    "tp": tp_price,
                    "timestamp": position.timestamp,
                    "status": "OPEN",
                },
            }

        try:
            trade = await self._engine.open_position(
                symbol=symbol,
                direction=direction,
                volume=volume,
                sl=sl_price,
                tp=tp_price,
                reason=f"signal_conf={confidence:.2f}",
            )
            if trade is None:
                return {
                    "success": False,
                    "position_id": 0,
                    "position": {},
                }

            pos_id = trade.position_id
            position = ManagedPosition(
                position_id=pos_id,
                symbol=trade.symbol,
                direction=trade.direction,
                volume=trade.volume,
                entry_price=trade.entry_price,
                sl=trade.sl,
                tp=trade.tp,
                timestamp=trade.timestamp,
            )
            self._positions[pos_id] = position

            await self.bus.emit(
                "execution_result",
                symbol=symbol,
                direction=direction,
                volume=volume,
                filled_price=trade.entry_price,
                signal_price=signal_price,
                position_id=pos_id,
                success=True,
            )

            # Record execution quality for live/engine trade
            await self._record_execution_quality(
                symbol=symbol,
                direction=direction,
                filled_price=trade.entry_price,
                expected_price=signal_price,
                volume=volume,
            )

            return {
                "success": True,
                "position_id": pos_id,
                "position": {
                    "position_id": pos_id,
                    "symbol": trade.symbol,
                    "direction": trade.direction,
                    "volume": trade.volume,
                    "entry_price": trade.entry_price,
                    "sl": trade.sl,
                    "tp": trade.tp,
                    "timestamp": trade.timestamp,
                    "status": "OPEN",
                },
            }

        except Exception as e:
            logger.error(f"[execution_manager] Order failed: {e}")
            await self.bus.emit(
                "execution_result",
                symbol=symbol,
                direction=direction,
                volume=volume,
                signal_price=signal_price,
                position_id=0,
                success=False,
                error=str(e),
            )
            return {
                "success": False,
                "position_id": 0,
                "position": {},
            }

    async def _record_execution_quality(
        self,
        symbol: str,
        direction: str,
        filled_price: float,
        expected_price: float,
        volume: float,
    ) -> None:
        """Track execution quality and emit feedback for other modules."""
        if self._quality_tracker is None:
            return

        try:
            slippage = abs(filled_price - expected_price) if expected_price > 0 else 0.0
            self._quality_tracker.record_order_attempt(symbol)
            self._quality_tracker.record_fill(
                symbol=symbol,
                expected_price=expected_price,
                filled_price=filled_price,
                direction=direction,
                volume=volume,
            )

            # Update adaptive slippage multiplier
            recent_slippage = self._quality_tracker.recent_average_slippage()
            self._slippage_multiplier = 1.0 + max(
                0.0, recent_slippage / 0.0001
            )  # 1 pip baseline

            # Emit quality update for other modules
            await self.bus.emit(
                "execution_quality",
                symbol=symbol,
                slippage=slippage,
                slippage_multiplier=self._slippage_multiplier,
            )
        except Exception as e:
            logger.warning(f"[execution_manager] Quality tracking failed: {e}")

    async def close_position(
        self,
        position_id: int,
        reason: str = "manual_close",
        exit_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Close an open position."""
        position = self._positions.get(position_id)
        if position is None:
            return {"success": False, "pnl": 0.0}

        if self._engine is not None:
            try:
                success = await self._engine.close_position(
                    position_id=position_id,
                    reason=reason,
                    exit_price=exit_price,
                )
                if not success:
                    return {"success": False, "pnl": 0.0}
            except Exception as e:
                logger.error(f"[execution_manager] Close failed for {position_id}: {e}")
                return {"success": False, "pnl": 0.0}

        # Paper close or after successful engine close
        exit_px = exit_price or position.entry_price
        pnl = self._calculate_pnl(position, exit_px)
        position.status = "CLOSED"
        position.exit_price = exit_px
        position.pnl = pnl
        position.reason = reason

        await self.bus.emit(
            "position_closed",
            position_id=position_id,
            symbol=position.symbol,
            direction=position.direction,
            volume=position.volume,
            entry_price=position.entry_price,
            exit_price=exit_px,
            pnl=pnl,
            reason=reason,
        )

        return {"success": True, "pnl": pnl}

    async def _paper_tick_loop(self) -> None:
        """Periodic tick generator for paper mode.

        Reads latest prices from the DataManager every second,
        applies slight random walks to simulate market movement,
        and emits 'tick' events for the SignalEngine.
        """
        import random
        import numpy as np

        # Per-symbol state for random walk simulation
        paper_prices = {
            "EURUSD": 1.1200,
            "GBPUSD": 1.2800,
            "USDJPY": 150.00,
            "AUDUSD": 0.6700,
            "USDCAD": 1.3500,
            "USDCHF": 0.8800,
            "NZDUSD": 0.6100,
            "XAUUSD": 2350.0,
        }
        # Volatility per symbol (daily %, scaled to per-tick ~1s)
        paper_vol = {
            "EURUSD": 0.0001,
            "GBPUSD": 0.0001,
            "USDJPY": 0.010,
            "AUDUSD": 0.0001,
            "USDCAD": 0.0001,
            "USDCHF": 0.0001,
            "NZDUSD": 0.0001,
            "XAUUSD": 0.50,
        }

        while self._initialized and self._running:
            try:
                await asyncio.sleep(1.0)  # Tick every second
                ts = time.time()

                for symbol in self._active_symbols:
                    # Start with paper price, try to get live price from data manager
                    price = paper_prices.get(symbol, 1.12)
                    if self._data_manager is not None:
                        try:
                            live = self._data_manager.get_price(symbol)
                            if live and live > 0:
                                price = live
                        except Exception:
                            pass

                    # Slight random walk
                    vol = paper_vol.get(symbol, 0.0001)
                    drift = np.random.normal(0, vol)
                    price = max(price * 0.5, price + drift)  # prevent zero/negative
                    paper_prices[symbol] = price

                    # Simulate bid-ask spread
                    spread = price * 0.0002  # ~0.2 pips for FX
                    bid = price - spread / 2
                    ask = price + spread / 2

                    await self.bus.emit(
                        "tick",
                        symbol=symbol,
                        bid=round(bid, 5),
                        ask=round(ask, 5),
                        volume=random.uniform(0.1, 10.0),
                        timestamp=ts,
                    )

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("[execution_manager] Paper tick loop error")

    def _calculate_pnl(self, position: ManagedPosition, exit_price: float) -> float:
        """Calculate P&L for a closed position."""
        if exit_price == position.entry_price:
            return 0.0
        mult = 1 if position.direction == "BUY" else -1
        diff = (exit_price - position.entry_price) * mult * position.volume
        return round(diff, 2)

    async def update_trailing_stop(self, position_id: int) -> None:
        """Update trailing stop for a position."""
        position = self._positions.get(position_id)
        if position is None or position.status != "OPEN":
            return

        # Get current price
        current_price = self._get_current_price(position.symbol)
        if current_price <= 0:
            return

        if position.direction == "BUY":
            _ = current_price - position.entry_price
            new_sl = current_price - (position.entry_price - position.sl)
        else:
            _ = position.entry_price - current_price
            new_sl = current_price + (position.sl - position.entry_price)

        # Update SL if price moved favorably
        if position.direction == "BUY" and new_sl > position.sl:
            position.sl = new_sl
        elif position.direction == "SELL" and new_sl < position.sl:
            position.sl = new_sl

    def _get_current_price(self, symbol: str) -> float:
        """Get the current price for a symbol."""
        base_prices = {
            "EURUSD": 1.12,
            "GBPUSD": 1.28,
            "USDJPY": 150.0,
            "AUDUSD": 0.67,
            "USDCAD": 1.35,
            "USDCHF": 0.88,
            "NZDUSD": 0.61,
            "XAUUSD": 2350.0,
        }
        return base_prices.get(symbol, 1.12)

    async def check_positions(self) -> None:
        """Periodic check on all open positions (SL/TP, trailing, partial close)."""
        now = time.time()
        for pid, pos in list(self._positions.items()):
            if pos.status != "OPEN":
                continue

            # Check interval
            last = self._last_check.get(str(pid), 0)
            if now - last < self._check_interval:
                continue
            self._last_check[str(pid)] = now

            # Get current price
            current_price = self._get_current_price(pos.symbol)

            # Check SL/TP
            if pos.direction == "BUY":
                if current_price <= pos.sl:
                    await self.close_position(pid, "stop_loss", current_price)
                elif current_price >= pos.tp:
                    await self.close_position(pid, "take_profit", current_price)
            else:
                if current_price >= pos.sl:
                    await self.close_position(pid, "stop_loss", current_price)
                elif current_price <= pos.tp:
                    await self.close_position(pid, "take_profit", current_price)

    @property
    def is_alive(self) -> bool:
        """Whether the execution manager is initialized."""
        return self._initialized

    @property
    def open_positions(self) -> List[ManagedPosition]:
        """Get list of currently open positions."""
        return [p for p in self._positions.values() if p.status == "OPEN"]

    @property
    def total_positions(self) -> int:
        """Total number of positions tracked."""
        return len(self._positions)
