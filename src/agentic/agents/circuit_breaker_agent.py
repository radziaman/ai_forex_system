from __future__ import annotations
import time
from typing import Dict, Any

from agentic.core.base_agent import BaseAgent
from agentic.core.agent_message import MessageType, MessagePriority
from agentic.core.agent_consciousness import ConsciousnessLevel


class CircuitBreakerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="circuit_breaker_agent",
            role="Market Circuit Breaker",
            purpose="Independently monitor market stress and halt trading",
            domain="risk",
            capabilities={
                "flash_crash_detection",
                "liquidity_monitoring",
                "volatility_spike_detection",
                "degradation_mode",
                "graceful_recovery",
            },
            tick_interval=2.0,
            consciousness_level=ConsciousnessLevel.REFLECTIVE,
        )
        self._breaker = None

    async def _on_start(self):
        from risk.circuit_breaker import CircuitBreaker

        self._breaker = CircuitBreaker()
        self.log_state("Circuit breaker active — monitoring all symbols")

    async def perceive(self) -> Dict[str, Any]:
        if not self._breaker:
            return {"skip": True}

        # Skip circuit breaker checks when market is closed
        from data.market_session import MarketSession

        if not MarketSession.is_market_open():
            return {"skip": True, "reason": "market_closed"}

        halted_symbols = []
        market_healthy = True
        from data.data_manager import SYMBOLS

        for sym in SYMBOLS:
            bid = self.get_world(f"data.bid.{sym}", 0)
            ask = self.get_world(f"data.ask.{sym}", 0)
            if bid > 0 and ask > 0:
                if bid <= 0 or ask <= 0:
                    continue
                tick = {"bid": bid, "ask": ask, "price": (bid + ask) / 2.0, "volume": 0}
                should_halt, reason, snapshot = self._breaker.check_market_health(
                    sym, tick
                )
                if should_halt:
                    halted_symbols.append(sym)
                    market_healthy = False
        return {
            "halted_symbols": halted_symbols,
            "market_healthy": market_healthy,
            "degradation": self._breaker.current_degradation_mode.value,
        }

    async def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        actions = {}
        if perception.get("halted_symbols"):
            actions["halted"] = perception["halted_symbols"]
        if not perception.get("market_healthy") and not perception.get(
            "halted_symbols"
        ):
            actions["recovery_check"] = True
        return actions

    async def act(self, decision: Dict[str, Any]):
        if decision.get("halted"):
            self.set_world("risk.halted_symbols", decision["halted"])
            self.set_world("risk.circuit_breaker_active", True)
            await self.send(
                MessageType.CIRCUIT_BREAKER,
                payload={
                    "halted_symbols": decision["halted"],
                    "timestamp": time.time(),
                    "degradation": self._breaker.current_degradation_mode.value,
                },
                priority=MessagePriority.CRITICAL,
            )
        elif decision.get("recovery_check"):
            self._breaker._attempt_recovery()
            if self._breaker.current_degradation_mode.value == "normal":
                self.set_world("risk.circuit_breaker_active", False)

    async def reflect(self, outcome: Dict[str, Any]):
        pass

    async def on_message(self, message):
        if message.msg_type == MessageType.DIAGNOSTIC_REQUEST:
            summary = self._breaker.get_stress_summary() if self._breaker else {}
            await self.send(
                MessageType.DIAGNOSTIC_RESULT,
                payload={
                    "agent": self.name,
                    "degradation": (
                        self._breaker.current_degradation_mode.value
                        if self._breaker
                        else "unknown"
                    ),
                    "halted": (
                        list(self._breaker.last_halt_time.keys())
                        if self._breaker
                        else []
                    ),
                    "summary": summary,
                },
                target=message.source_agent,
            )
