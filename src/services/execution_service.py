"""Execution Engine: executes TradeDecision. Makes no decisions."""

import asyncio
from typing import Optional, Dict, List
from loguru import logger

from infrastructure.service_base import TradingService
from infrastructure.config_v2 import AppConfig
from infrastructure.secrets import Secrets
from infrastructure.event_bus import get_event_bus, EventType
from services import TradeDecision, ExecutionResult
from execution.engine import ExecutionEngine
from api.provider_factory import create_execution_provider
from data.data_manager import SYMBOLS


class ExecutionService(TradingService):
    """Dumb executor: receives decisions, sends orders. No deciding."""

    def __init__(
        self,
        config: AppConfig,
        secrets: Secrets,
        data_pipeline,
        risk_manager=None,
        initial_balance: float = 100_000.0,
    ):
        super().__init__("execution_service")
        self.config = config
        self.secrets = secrets
        self.data_pipeline = data_pipeline
        self._risk_manager = risk_manager
        self.initial_balance = initial_balance
        self.ctrader = None
        self.data_provider = None
        self.engine = None
        self.event_bus = get_event_bus()

    async def start(self) -> None:
        self.ctrader, self.data_provider = create_execution_provider(self.secrets)

        # Determine mode: live broker or paper simulation
        engine_mode = "LIVE" if not self.secrets.is_demo else "PAPER"
        self.engine = ExecutionEngine(
            self.ctrader,
            self._risk_manager,
            self.data_pipeline.data_manager,
            initial_balance=self.initial_balance,
            mode=engine_mode,
        )

        # Wire live price stream from broker to DataPipeline
        self.ctrader.on_price = lambda tick: self.data_pipeline.ingest_tick(tick)

        # Attempt broker connection
        connected = await self.ctrader.start()
        if (
            connected
            and hasattr(self.ctrader, "is_connected")
            and self.ctrader.is_connected()
        ):
            logger.info(
                f"ExecutionService: cTrader LIVE — streaming {len(SYMBOLS)} symbols"
            )
            # Subscribe to live market data for all symbols
            raw = getattr(self.ctrader, "raw", None) or self.ctrader
            if hasattr(raw, "subscribe_depth"):
                from api.ctrader_client import SYMBOL_MAP

                subscribed = 0
                for sym in SYMBOLS:
                    sym_id = SYMBOL_MAP.get(sym)
                    if sym_id:
                        success = await raw.subscribe_depth(sym_id)
                        if success:
                            subscribed += 1
                logger.info(
                    f"ExecutionService: subscribed to {subscribed}/{len(SYMBOLS)} symbols for live ticks"
                )
        else:
            logger.warning(
                "ExecutionService: cTrader not connected (running on historical data)"
            )
        self._running = True

    async def stop(self) -> None:
        if self.ctrader and hasattr(self.ctrader, "disconnect"):
            await self.ctrader.disconnect()
        self._running = False

    async def execute(self, decision: TradeDecision) -> Optional[ExecutionResult]:
        """Execute a trade decision. Returns result."""
        if not self._running:
            return ExecutionResult(success=False, error="service_not_running")

        trade = await self.engine.open_position(
            symbol=decision.signal.symbol,
            direction=decision.signal.direction.value,
            volume=decision.volume,
            sl=decision.sl_price,
            tp=decision.tp_price,
            reason=f"signal_conf={decision.signal.confidence:.2f}_regime={decision.signal.regime.value}",
        )

        if trade:
            await self.event_bus.emit(
                EventType.POSITION_OPENED,
                {
                    "symbol": decision.signal.symbol,
                    "volume": decision.volume,
                    "entry": trade.entry_price,
                    "position_id": trade.position_id,
                },
                source="execution_service",
            )
            return ExecutionResult(
                success=True,
                position_id=str(trade.position_id),
                filled_price=trade.entry_price,
                filled_volume=decision.volume,
            )
        else:
            await self.event_bus.emit(
                EventType.ORDER_REJECTED,
                {"symbol": decision.signal.symbol},
                source="execution_service",
            )
            return ExecutionResult(success=False, error="order_failed")

    async def close_position(self, position_id: str, reason: str = "AI signal") -> bool:
        return await self.engine.close_position(int(position_id), reason)

    async def close_all(self, reason: str = "system") -> None:
        await self.engine.close_all_positions(reason)

    def get_open_positions(self) -> List[Dict]:
        return self.engine.get_open_positions()

    async def get_account_info(self) -> Dict:
        result = self.engine.get_account_info()
        if asyncio.iscoroutine(result):
            return await result
        return result

    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        return self.engine.get_trade_history(limit)
