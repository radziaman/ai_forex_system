"""
Adaptive Risk Agent — autonomous dynamic risk adjustment.

Identity: I am the thermostat for risk. I turn risk up or down based on market conditions.
Purpose: I protect the account by dynamically adjusting position sizing in real-time.
Autonomy: I independently monitor volatility, drawdown, and win rate trends to tune Kelly.
"""

from __future__ import annotations
import time
import numpy as np
from typing import Dict, List, Optional, Any, Set
from collections import deque
from loguru import logger

from agentic.core.base_agent import BaseAgent
from agentic.core.agent_message import MessageType, MessagePriority, AgentIntention
from agentic.core.agent_consciousness import ConsciousnessLevel


class AdaptiveRiskAgent(BaseAgent):
    """
    Autonomous dynamic risk parameter tuner.

    Responsibilities:
    - Monitor market volatility regime (normal/high/extreme)
    - Monitor drawdown regime (safe/warning/critical)
    - Adjust effective Kelly fraction based on conditions
    - Adjust effective risk per trade based on conditions
    - Auto-halt trading at critical drawdown levels
    - Recover when conditions normalize
    """

    def __init__(self, base_kelly: float = 0.25, base_risk: float = 0.02):
        super().__init__(
            name="adaptive_risk_agent",
            role="Dynamic Risk Adjuster",
            purpose="Protect capital by dynamically adjusting risk parameters to market conditions",
            domain="risk",
            capabilities={
                "dynamic_kelly_adjustment", "volatility_regime_detection",
                "drawdown_regime_detection", "auto_halt", "auto_recovery",
                "confidence_penalty",
            },
            tick_interval=3.0,
            consciousness_level=ConsciousnessLevel.REFLECTIVE,
        )
        self.base_kelly = base_kelly
        self.base_risk = base_risk
        self.effective_kelly = base_kelly
        self.effective_risk = base_risk
        self._equity_high_water = 0.0
        self._consecutive_losses = 0
        self._regime_history: deque = deque(maxlen=20)
        self._volatility_regime: str = "normal"
        self._drawdown_regime: str = "safe"
        self._trade_pnls: deque = deque(maxlen=50)
        self.halted = False

        self.subscribe(MessageType.REGIME_CHANGED)
        self.subscribe(MessageType.DIAGNOSTIC_REQUEST)

    async def _on_start(self):
        self.set_world("adaptive_risk.kelly", self.effective_kelly)
        self.set_world("adaptive_risk.risk_per_trade", self.effective_risk)
        self.set_world("adaptive_risk.status", "active")
        self.log_state(f"Active: base_kelly={self.base_kelly}, base_risk={self.base_risk}")

    async def perceive(self) -> Dict[str, Any]:
        balance = self.get_world("account.balance", 0)
        equity = self.get_world("account.equity", 0)
        regime = self.get_world("regime.current", "ranging")
        atr = self.get_world(f"data.atr.EURUSD", 0.001)
        price = self.get_world("data.price.EURUSD", 1.12)

        if self._equity_high_water == 0:
            self._equity_high_water = max(balance, equity, 100000)

        return {
            "balance": balance, "equity": equity,
            "regime": regime, "atr": atr, "price": price,
        }

    async def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        equity = perception.get("equity", 0)
        price = perception.get("price", 1.12)
        atr = perception.get("atr", 0.001)
        regime = perception.get("regime", "ranging")

        # Skip drawdown check until account state is initialized
        if equity <= 0:
            return {}

        if equity > self._equity_high_water:
            self._equity_high_water = equity
        dd = ((self._equity_high_water - equity) / max(self._equity_high_water, 1)
              if self._equity_high_water > 0 else 0.0)

        if dd > 0.10:
            self._drawdown_regime = "critical"
        elif dd > 0.06:
            self._drawdown_regime = "warning"
        else:
            self._drawdown_regime = "safe"

        if price > 0 and atr > 0:
            vol = atr / price
            if vol > 0.02:
                self._volatility_regime = "extreme"
            elif vol > 0.01:
                self._volatility_regime = "high"
            else:
                self._volatility_regime = "normal"

        self._regime_history.append(regime)

        if self._drawdown_regime == "critical" and not self.halted:
            self.halted = True
            return {"halt": True, "reason": "critical_drawdown"}

        return {
            "adjust": True,
            "dd": dd,
            "drawdown_regime": self._drawdown_regime,
            "volatility_regime": self._volatility_regime,
            "regime": regime,
        }

    async def act(self, decision: Dict[str, Any]):
        if decision.get("halt"):
            self.halted = True
            self.effective_kelly = 0.0
            self.effective_risk = 0.0
            self.set_world("adaptive_risk.halted", True)
            self.set_world("adaptive_risk.kelly", 0.0)
            self.set_world("adaptive_risk.risk_per_trade", 0.0)
            self.memory.remember(
                event_type="adaptive_halt",
                description=f"Trading halted: {decision.get('reason', 'unknown')}",
                importance=0.9,
                emotion="warning",
            )
            await self.send(
                MessageType.RISK_ALERT,
                payload={"type": "halt", "reason": decision["reason"],
                         "drawdown_regime": self._drawdown_regime},
                priority=MessagePriority.CRITICAL,
            )
            return

        if decision.get("adjust") and not self.halted:
            mult = self._compute_multiplier(
                decision["drawdown_regime"],
                decision["volatility_regime"],
                decision["regime"],
            )
            self.effective_kelly = self.base_kelly * mult
            self.effective_risk = self.base_risk * mult
            self.set_world("adaptive_risk.kelly", self.effective_kelly)
            self.set_world("adaptive_risk.risk_per_trade", self.effective_risk)
            self.set_world("adaptive_risk.halted", False)

    async def reflect(self, outcome: Dict[str, Any]):
        if self.consciousness.cycle_count % 20 == 0:
            self.memory.know("adaptive_risk.effective_kelly", self.effective_kelly, ttl=120)
            self.memory.know("adaptive_risk.drawdown_regime", self._drawdown_regime, ttl=120)

    async def on_message(self, message: AgentMessage):
        if message.msg_type == MessageType.REGIME_CHANGED:
            payload = message.payload if isinstance(message.payload, dict) else {}
            if payload.get("to") == "crisis" and not self.halted:
                self.halted = True
                self.effective_kelly = 0.0
                self.set_world("adaptive_risk.halted", True)
                self.log_state("Auto-halted on crisis regime", "warning")
        elif message.msg_type == MessageType.DIAGNOSTIC_REQUEST:
            await self.send(
                MessageType.DIAGNOSTIC_RESULT,
                payload={
                    "agent": self.name,
                    "effective_kelly": self.effective_kelly,
                    "effective_risk": self.effective_risk,
                    "drawdown_regime": self._drawdown_regime,
                    "volatility_regime": self._volatility_regime,
                    "halted": self.halted,
                },
                target=message.source_agent,
            )

    def _compute_multiplier(self, dd_regime: str, vol_regime: str,
                            regime: str) -> float:
        mult = 1.0
        if dd_regime == "warning":
            mult *= 0.6
        elif dd_regime == "critical":
            mult *= 0.2

        if vol_regime == "high":
            mult *= 0.7
        elif vol_regime == "extreme":
            mult *= 0.3

        if regime == "crisis":
            mult *= 0.3
        elif regime == "volatile":
            mult *= 0.5

        recent_wr = self._recent_win_rate()
        if recent_wr is not None:
            if recent_wr < 0.3:
                mult *= 0.4
            elif recent_wr > 0.7:
                mult *= 1.1

        return max(mult, 0.05)

    def _recent_win_rate(self) -> Optional[float]:
        recent = list(self._trade_pnls)[-20:]
        if len(recent) < 5:
            return None
        wins = sum(1 for p in recent if p > 0)
        return wins / len(recent)
