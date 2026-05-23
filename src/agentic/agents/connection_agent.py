"""
Connection Agent — autonomous broker connectivity management.

Identity: I am the lifeline. I keep the connection to the broker alive.
Purpose: Without me, no orders reach the market.
Autonomy: I independently monitor connection health, reconnect with backoff, and report outages.
"""

from __future__ import annotations
import time
import asyncio
from typing import Dict, List, Optional, Any, Set
from loguru import logger

from agentic.core.base_agent import BaseAgent
from agentic.core.agent_message import (
    AgentMessage,
    MessageType,
    MessagePriority,
    AgentIntention,
)
from agentic.core.agent_consciousness import ConsciousnessLevel


class ConnectionAgent(BaseAgent):
    """
    Autonomous connection manager with auto-reconnect.

    Responsibilities:
    - Monitor broker connection health via heartbeat
    - Auto-reconnect with exponential backoff on disconnection
    - Track API health metrics (success rate, latency)
    - Notify system on connection state changes
    - Manage subscription to market data after reconnect
    """

    def __init__(self, symbols: List[str]):
        super().__init__(
            name="connection_agent",
            role="Broker Connection Manager",
            purpose="Maintain persistent, healthy connection to the broker at all times",
            domain="connectivity",
            capabilities={
                "connection_monitoring",
                "auto_reconnect",
                "exponential_backoff",
                "api_health_tracking",
                "market_data_subscription",
            },
            tick_interval=5.0,
            consciousness_level=ConsciousnessLevel.REFLECTIVE,
        )
        self._symbols = symbols
        self._connected = False
        self._retries = 0
        self._max_retries = 10
        self._base_delay = 1.0
        self._max_delay = 60.0
        self._last_reconnect_ts = 0.0  # Non-blocking timing for reconnect backoff
        self._api_health: Dict[str, Dict] = {}

        self.subscribe(MessageType.AGENT_DIRECTIVE)
        self.subscribe(MessageType.DIAGNOSTIC_REQUEST)

    async def _on_start(self):
        self._connected = self.get_world("execution.connected", False)
        self.set_world("connection.status", "checking")
        self.set_world("connection.connected", self._connected)
        self.log_state(
            f"Connection status: {'CONNECTED' if self._connected else 'DISCONNECTED'}"
        )

    async def perceive(self) -> Dict[str, Any]:
        was_connected = self._connected
        self._connected = self.get_world("execution.connected", False)
        return {
            "was_connected": was_connected,
            "is_connected": self._connected,
            "disconnected": was_connected and not self._connected,
            "reconnected": not was_connected and self._connected,
        }

    async def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        actions = {}

        if perception.get("disconnected"):
            self._retries = 0
            actions["start_reconnect"] = True
            self.memory.remember(
                event_type="connection_lost",
                description="Broker connection lost",
                importance=0.9,
                emotion="warning",
            )

        if perception.get("reconnected"):
            self._retries = 0
            actions["resubscribe"] = True
            self.memory.remember(
                event_type="connection_restored",
                description="Broker connection restored",
                importance=0.7,
                emotion="success",
            )

        if not self._connected and self._retries < self._max_retries:
            actions["reconnect_attempt"] = True
        elif not self._connected and self._retries >= self._max_retries:
            actions["give_up"] = True

        return actions

    async def act(self, decision: Dict[str, Any]):
        now = time.time()

        if decision.get("start_reconnect"):
            self.log_state("Starting reconnect sequence...", "warning")
            await self.send(
                MessageType.AGENT_DIRECTIVE,
                payload={"action": "reconnect", "reason": "connection_lost"},
                target="execution_agent",
                priority=MessagePriority.HIGH,
            )
            self._retries = 1
            self._last_reconnect_ts = now

        if decision.get("reconnect_attempt"):
            delay = min(self._base_delay * (2 ** (self._retries - 1)), self._max_delay)
            if now - self._last_reconnect_ts >= delay:
                self._retries += 1
                self._last_reconnect_ts = now
                self.log_state(
                    f"Reconnect attempt {self._retries}/{self._max_retries} "
                    f"(delay={delay:.0f}s)"
                )
                # Send directive to execution_agent — don't sleep, use
                # non-blocking timing so the agent stays responsive
                await self.send(
                    MessageType.AGENT_DIRECTIVE,
                    payload={"action": "reconnect", "reason": "connection_lost"},
                    target="execution_agent",
                    priority=MessagePriority.HIGH,
                )

            # Check if connection was restored between ticks
            self._connected = self.get_world("execution.connected", False)
            if self._connected:
                self._retries = 0
                self.log_state("Reconnected successfully")
                self.set_world("connection.connected", True)
                await self.send(
                    MessageType.AGENT_HEARTBEAT,
                    payload={"type": "reconnected", "retries": self._retries},
                    priority=MessagePriority.HIGH,
                )

        if decision.get("give_up"):
            self.log_state(
                f"Max retries ({self._max_retries}) reached. Manual intervention needed.",
                "critical",
            )
            self.set_world("connection.gave_up", True)
            await self.send(
                MessageType.RISK_ALERT,
                payload={"type": "connection_lost", "reason": "max_retries_exceeded"},
                priority=MessagePriority.CRITICAL,
            )

        if decision.get("resubscribe"):
            self._resubscribe()

        self.set_world("connection.connected", self._connected)
        self.set_world("connection.retries", self._retries)

    async def reflect(self, outcome: Dict[str, Any]):
        self.memory.know("connection.connected", self._connected, ttl=60)
        self.memory.know("connection.retries", self._retries, ttl=300)

    async def on_message(self, message: AgentMessage):
        if message.msg_type == MessageType.DIAGNOSTIC_REQUEST:
            await self.send(
                MessageType.DIAGNOSTIC_RESULT,
                payload={
                    "agent": self.name,
                    "connected": self._connected,
                    "retries": self._retries,
                    "max_retries": self._max_retries,
                    "api_health": self._api_health,
                },
                target=message.source_agent,
            )

    def _resubscribe(self):
        self.log_state(f"Resubscribing to {len(self._symbols)} symbols")
        self.set_world("connection.resubscribed", True)
        asyncio.ensure_future(self._do_resubscribe())

    async def _do_resubscribe(self):
        from api.symbol_map import get_symbol_id

        connected = self.get_world("execution.connected", False)
        if not connected:
            return
        for sym in self._symbols:
            try:
                sid = get_symbol_id(sym)
                await self.send(
                    MessageType.AGENT_DIRECTIVE,
                    payload={
                        "target": "execution_agent",
                        "action": "subscribe_depth",
                        "symbol_id": sid,
                        "reason": "resubscribe_after_reconnect",
                    },
                )
            except Exception:
                pass
