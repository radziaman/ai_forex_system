"""
Monitoring Agent — G27: reliable Telegram with retry and health tracking.
"""

from __future__ import annotations
import time
import asyncio
from typing import Dict, List, Optional, Any, Set
from loguru import logger

from agentic.core.base_agent import BaseAgent
from agentic.core.agent_message import MessageType, MessagePriority, AgentIntention
from agentic.core.agent_consciousness import ConsciousnessLevel


class MonitoringAgent(BaseAgent):
    def __init__(self, secrets=None):
        super().__init__(
            name="monitoring_agent",
            role="System Monitor & Notifier",
            purpose="Watch the system, alert on issues, keep humans informed",
            domain="monitoring",
            capabilities={
                "health_monitoring", "telegram_notifications",
                "dashboard_updates", "alert_aggregation",
                "event_logging", "uptime_tracking",
            },
            tick_interval=15.0,
            consciousness_level=ConsciousnessLevel.REFLECTIVE,
        )
        self.secrets = secrets
        self.notifier = None
        self._event_log: List[Dict] = []
        self._max_event_log = 1000
        self._last_telegram_ok: float = 0.0  # G27
        self._telegram_health: str = "unknown"  # G27
        self._telegram_retries: int = 0

        self.subscribe(MessageType.AGENT_HEARTBEAT)
        self.subscribe(MessageType.RISK_ALERT)
        self.subscribe(MessageType.CIRCUIT_BREAKER)
        self.subscribe(MessageType.SYSTEM_STATE_CHANGE)
        self.subscribe(MessageType.SIGNAL_GENERATED)
        self.subscribe(MessageType.POSITION_OPENED)
        self.subscribe(MessageType.POSITION_CLOSED)
        self.subscribe(MessageType.DIAGNOSTIC_REQUEST)

    async def _on_start(self):
        self.consciousness.current_intention = "initializing monitoring system"
        if self.secrets:
            try:
                from notifications.telegram import TelegramNotifier
                self.notifier = TelegramNotifier(
                    bot_token=getattr(self.secrets, 'telegram_bot_token', ''),
                    chat_id=getattr(self.secrets, 'telegram_chat_id', ''),
                    send_trade_alerts=True,
                    send_daily_summary=True,
                    send_risk_alerts=True,
                )
                # G27: Verify Telegram connectivity on startup
                try:
                    test_result = self.notifier.send("Agentic system online", level="info")
                    self._last_telegram_ok = time.time()
                    self._telegram_health = "ok"
                    self.log_state("Telegram verified — connected")
                except Exception:
                    self._telegram_health = "unavailable"
                    self.log_state("Telegram configured but not reachable", "warning")
            except Exception as e:
                self.log_state(f"Telegram not available: {e}", "warning")
                self._telegram_health = "unconfigured"

        self.set_world("monitoring.status", "active")
        self.set_world("monitoring.telegram_health", self._telegram_health)

    async def perceive(self) -> Dict[str, Any]:
        health = self.get_world("master.system_health", 1.0)
        performance = self.get_world("performance.stats", {})
        return {"health": health, "performance": performance}

    async def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        should_heartbeat = self.consciousness.cycle_count % 4 == 0
        alerts = []
        if perception.get("health", 1.0) < 0.5:
            alerts.append("System health critically low")
        perf = perception.get("performance", {})
        if perf.get("sharpe", 0) < -1.0:
            alerts.append(f"Sharpe critically low: {perf.get('sharpe', 0):.2f}")
        return {"heartbeat": should_heartbeat, "alerts": alerts}

    async def act(self, decision: Dict[str, Any]):
        if decision.get("heartbeat"):
            status = self._build_status()
            self.set_world("monitoring.status_snapshot", status)
        for alert in decision.get("alerts", []):
            self.memory.remember(event_type="monitoring_alert",
                description=alert, importance=0.7, emotion="warning")

    async def reflect(self, outcome: Dict[str, Any]):
        pass

    async def on_message(self, message: AgentMessage):
        event = {"time": time.time(), "type": message.msg_type.name,
                 "source": message.source_agent}
        self._event_log.append(event)
        if len(self._event_log) > self._max_event_log:
            self._event_log = self._event_log[-self._max_event_log:]

        if message.msg_type in (MessageType.POSITION_OPENED, MessageType.POSITION_CLOSED):
            self._safe_telegram(
                f"{message.msg_type.name}: {message.payload.get('symbol', '?')}",
                level="info")

        elif message.msg_type == MessageType.RISK_ALERT:
            payload = message.payload if isinstance(message.payload, dict) else {}
            self._safe_telegram(
                f"RISK: {payload.get('type', '?')}: {payload.get('reason', '?')}",
                level="critical")

        elif message.msg_type == MessageType.DIAGNOSTIC_REQUEST:
            await self.send(MessageType.DIAGNOSTIC_RESULT, payload={
                "agent": self.name, "events_logged": len(self._event_log),
                "telegram_health": self._telegram_health,
                "uptime": time.time() - self.consciousness.started_at,
            }, target=message.source_agent)

    # G27: Reliable Telegram with health tracking and retry
    def _safe_telegram(self, text: str, level: str = "info", max_retries: int = 2):
        if not self.notifier:
            return
        for attempt in range(max_retries):
            try:
                self.notifier.send(text, level=level)
                self._last_telegram_ok = time.time()
                self._telegram_health = "ok"
                self._telegram_retries = 0
                return
            except Exception as e:
                self._telegram_retries += 1
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    self._telegram_health = "error"
                    logger.warning(f"[{self.name}] Telegram failed after {max_retries} retries: {e}")

        # Track Telegram health in world state
        self.set_world("monitoring.telegram_health", self._telegram_health)
        if self._telegram_health == "error":
            self.set_world("monitoring.telegram_last_failure", time.time())

    def _build_status(self) -> Dict:
        return {
            "timestamp": time.time(),
            "uptime": time.time() - self.consciousness.started_at,
            "health": self.get_world("master.system_health", 1.0),
            "performance": self.get_world("performance.stats", {}),
            "regime": self.get_world("regime.current", "unknown"),
            "telegram": self._telegram_health,
            "events_logged": len(self._event_log),
        }
