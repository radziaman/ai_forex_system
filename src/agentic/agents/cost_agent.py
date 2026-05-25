from __future__ import annotations
import time
from typing import Dict, Any, Optional

from agentic.core.base_agent import BaseAgent
from agentic.core.agent_message import MessageType, MessagePriority
from agentic.core.agent_consciousness import ConsciousnessLevel
from execution.cost_model import CostModel


# Only monitor symbols we actively trade — major FX + XAUUSD
ACTIVE_SYMBOLS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "USDCHF",
    "NZDUSD",
    "XAUUSD",
]


class CostAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="cost_agent",
            role="Transaction Cost Monitor",
            purpose="Track live spreads, optimal execution times, and cost efficiency",
            domain="execution",
            capabilities={
                "live_spread_tracking",
                "cost_estimation",
                "optimal_execution",
                "spread_alerting",
            },
            tick_interval=5.0,
            consciousness_level=ConsciousnessLevel.REFLECTIVE,
        )
        self._cost_model = None
        self._spread_warnings: Dict[str, float] = {}
        # Rate-limit wide-spread alerts: one per symbol per 300s
        self._alert_cooldowns: Dict[str, float] = {}

    async def _on_start(self):
        self._cost_model = CostModel()
        self.log_state("Cost monitor initialized")

    async def perceive(self) -> Dict[str, Any]:
        if not self._cost_model:
            return {"skip": True}

        warnings = {}
        for sym in ACTIVE_SYMBOLS:
            bid = self.get_world(f"data.bid.{sym}", 0)
            ask = self.get_world(f"data.ask.{sym}", 0)
            if bid > 0 and ask > 0:
                spread_pips = (ask - bid) / CostModel.pip_to_price(sym)
                level = self._cost_model.get_spread_warning_level(sym, spread_pips)
                if level != "normal":
                    warnings[sym] = spread_pips
                    self.set_world(f"data.spread.{sym}", spread_pips)
        return {"spread_warnings": warnings, "total_warnings": len(warnings)}

    async def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        actions = {}
        if perception.get("spread_warnings"):
            actions["alert_wides"] = perception["spread_warnings"]
        return actions

    async def act(self, decision: Dict[str, Any]):
        if decision.get("alert_wides"):
            now = time.time()
            for sym, spread in decision["alert_wides"].items():
                # Rate limit: one alert per symbol per 300s (5 min)
                last_alert = self._alert_cooldowns.get(sym, 0.0)
                if now - last_alert < 300:
                    continue

                # Only alert if spread changed significantly from last warning
                last_spread = self._spread_warnings.get(sym)
                if last_spread is not None and abs(last_spread - spread) <= 2.0:
                    continue

                self._spread_warnings[sym] = spread
                self._alert_cooldowns[sym] = now
                await self.send(
                    MessageType.RISK_ALERT,
                    payload={
                        "type": "wide_spread",
                        "symbol": sym,
                        "spread_pips": round(spread, 1),
                        "timestamp": now,
                        "reason": f"wide_spread_{sym}_{spread:.1f}pips",
                    },
                    priority=MessagePriority.LOW,
                )

    async def estimate_cost(
        self, symbol: str, direction: str, volume: float, price: float, atr: float = 0.0
    ) -> Optional[Any]:
        if not self._cost_model:
            return None
        bid = self.get_world(f"data.bid.{symbol}", 0)
        ask = self.get_world(f"data.ask.{symbol}", 0)
        actual_spread = (
            (ask - bid) / CostModel.pip_to_price(symbol)
            if bid > 0 and ask > 0
            else None
        )
        return self._cost_model.calculate(
            symbol,
            direction,
            volume,
            price,
            atr,
            actual_spread_pips=actual_spread,
        )

    async def reflect(self, outcome: Dict[str, Any]):
        pass

    async def on_message(self, message):
        if message.msg_type == MessageType.DIAGNOSTIC_REQUEST:
            await self.send(
                MessageType.DIAGNOSTIC_RESULT,
                payload={
                    "agent": self.name,
                    "spread_warnings": len(self._spread_warnings),
                    "symbols": list(self._spread_warnings.keys()),
                },
                target=message.source_agent,
            )
