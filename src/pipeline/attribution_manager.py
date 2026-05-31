"""
AttributionManager — wires StrategyAttributionEngine into the EventBus pipeline.

Subscribes to position_closed and execution_result events, feeds trade data
through the attribution engine, and periodically emits attribution_report events.
"""

from typing import Any, Dict
from loguru import logger

from validation.attribution import StrategyAttributionEngine


class AttributionManager:
    """Wires StrategyAttributionEngine to EventBus for real-time attribution."""

    def __init__(
        self,
        event_bus,
        slippage_estimate: float = 0.0001,
        luck_window: int = 50,
    ):
        self._bus = event_bus
        self._engine = StrategyAttributionEngine(
            slippage_estimate=slippage_estimate,
            luck_window=luck_window,
        )
        self._subscriptions: list = []

    async def start(self):
        """Subscribe to pipeline events."""
        self._bus.on("position_closed", self._on_position_closed)
        self._bus.on("execution_result", self._on_execution_result)
        logger.info(
            "AttributionManager started — listening to position_closed "
            "and execution_result"
        )

    async def stop(self):
        """Unsubscribe from pipeline events."""
        self._bus.off("position_closed", self._on_position_closed)
        self._bus.off("execution_result", self._on_execution_result)
        logger.info("AttributionManager stopped")

    async def _on_position_closed(self, **data: Any) -> None:
        """Handle position_closed event: decompose trade PnL."""
        position = data.get("position", data)
        if not isinstance(position, dict):
            position = data  # already flat

        trade_record = {
            "pnl": float(position.get("pnl", position.get("profit", 0.0))),
            "expected_pnl": float(
                position.get("expected_pnl", position.get("predicted_pnl", 0.0))
            ),
            "fill_price": float(
                position.get("fill_price", position.get("entry_price", 0.0))
            ),
            "signal_price": float(position.get("signal_price", 0.0)),
            "slippage_bps": position.get("slippage_bps"),
            "strategy": str(
                position.get("strategy", position.get("expert", "unknown"))
            ),
            "market_return": float(position.get("market_return", 0.0)),
            "symbol": str(position.get("symbol", "unknown")),
            "direction": str(position.get("direction", "unknown")),
        }

        attribution = self._engine.attribute_trade(trade_record)
        logger.debug(
            f"Attributed trade: {trade_record['strategy']} "
            f"pnl={attribution.total_pnl:.2f} "
            f"alpha={attribution.alpha_signal:.4f} "
            f"exec={attribution.execution_quality:.4f}"
        )

        # Check if this strategy should be disabled
        if self._engine.should_disable(trade_record["strategy"]):
            logger.warning(
                f"Alpha decay detected for strategy "
                f"'{trade_record['strategy']}' — "
                f"emitting strategy_disable signal"
            )
            await self._bus.emit(
                "strategy_disable",
                strategy=trade_record["strategy"],
                reason="alpha_decay",
                report=self._engine.calculate_alpha_decay(trade_record["strategy"]),
            )

        # Emit attribution event
        await self._bus.emit(
            "trade_attributed",
            strategy=trade_record["strategy"],
            symbol=trade_record["symbol"],
            attribution={
                "alpha_signal": attribution.alpha_signal,
                "execution_quality": attribution.execution_quality,
                "slippage": attribution.slippage,
                "luck": attribution.luck,
                "total_pnl": attribution.total_pnl,
                "unexplained": attribution.unexplained,
            },
        )

    async def _on_execution_result(self, **data: Any) -> None:
        """Handle execution_result event: capture slippage data."""
        result = data.get("result", data)
        if isinstance(result, dict):
            slippage = result.get("slippage", result.get("slippage_bps"))
            if slippage is not None:
                logger.debug(f"Captured execution slippage: {slippage}")

    def get_report(self) -> Dict[str, Any]:
        """Return current attribution report for all strategies."""
        return self._engine.get_report()

    async def emit_report(self) -> None:
        """Emit current attribution report as event."""
        report = self.get_report()
        if report:
            await self._bus.emit("attribution_report", report=report)
