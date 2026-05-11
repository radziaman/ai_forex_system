"""Monitoring: dashboard updates + Telegram notifications."""
import asyncio
import time
from typing import Optional, Dict
from loguru import logger

from infrastructure.service_base import TradingService
from infrastructure.config_v2 import AppConfig
from infrastructure.secrets import Secrets
from infrastructure.event_bus import get_event_bus, EventType
from notifications.telegram import TelegramNotifier
from services import Signal, TradeDecision, ExecutionResult


class MonitoringService(TradingService):
    """Dashboard + notifications. Pure side effects, no decisions."""

    def __init__(self, config: AppConfig, secrets: Secrets):
        super().__init__("monitoring")
        self.config = config
        self.notifier = TelegramNotifier(
            bot_token=secrets.telegram_bot_token,
            chat_id=secrets.telegram_chat_id,
            send_trade_alerts=True,
            send_daily_summary=True,
            send_risk_alerts=True,
        )
        self.event_bus = get_event_bus()
        self._last_heartbeat = 0.0

    async def start(self) -> None:
        self._running = True
        logger.info("MonitoringService: Telegram + dashboard")

    async def stop(self) -> None:
        self._running = False

    def on_signal(self, signal: Signal) -> None:
        if signal.confidence > 0.7:
            logger.info(f"[SIGNAL] {signal.direction.value} {signal.symbol} "
                        f"conf={signal.confidence:.2f} regime={signal.regime.value}")

    def on_execution(self, result: ExecutionResult, signal: Signal) -> None:
        if result.success:
            self.notifier.trade_opened(
                symbol=signal.symbol, direction=signal.direction.value,
                volume=0, price=signal.price, regime=signal.regime.value,
                confidence=signal.confidence, atr=0,
            )
        else:
            self.notifier.send(
                f"Order rejected: {signal.symbol} - {result.error}",
                level="warning",
            )

    def heartbeat(self, cycle: int, positions: int, regime: str) -> None:
        now = time.time()
        if now - self._last_heartbeat > 60:
            self._last_heartbeat = now
            self.notifier.heartbeat(cycle, positions, regime)
