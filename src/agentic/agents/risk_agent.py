"""
Risk Agent — G2: reads adaptive risk values, G7: acts on halted symbols, G9: human-in-loop approval.
"""

from __future__ import annotations
import time
import numpy as np
from typing import Dict, List, Optional, Any, Set
from loguru import logger

from agentic.core.base_agent import BaseAgent
from agentic.core.agent_message import (
    MessageType,
    MessagePriority,
    AgentIntention,
    AgentMessage,
)
from agentic.core.agent_consciousness import ConsciousnessLevel


class RiskAgent(BaseAgent):
    def __init__(self, config, initial_balance: float = 100_000.0):
        super().__init__(
            name="risk_agent",
            role="Risk Gatekeeper",
            purpose="Protect capital through rigorous pre-trade risk assessment",
            domain="risk",
            capabilities={
                "kelly_sizing",
                "var_cvar",
                "drawdown_tracking",
                "circuit_breaker",
                "cost_verification",
                "correlation_filtering",
                "dynamic_confidence",
            },
            tick_interval=1.0,
            consciousness_level=ConsciousnessLevel.REFLECTIVE,
        )
        self.config = config
        self.initial_balance = initial_balance
        self.risk_manager = None
        self.circuit_breaker = None
        self.cost_model = None
        self._approved = 0
        self._rejected = 0
        self._rejection_reasons: Dict[str, int] = {}

        self.subscribe(MessageType.SIGNAL_GENERATED)
        self.subscribe(MessageType.AGENT_DIRECTIVE)
        self.subscribe(MessageType.DIAGNOSTIC_REQUEST)

    async def _on_start(self):
        self.consciousness.current_intention = "initializing risk systems"
        from risk.manager import RiskManager, RiskParameters
        from risk.circuit_breaker import CircuitBreaker
        from execution.cost_model import CostModel

        params = RiskParameters(
            max_risk_per_trade=self.config.trading.max_risk_per_trade,
            max_drawdown=self.config.trading.max_drawdown,
            max_margin_usage=self.config.trading.max_margin_usage,
        )
        self.risk_manager = RiskManager(params, self.initial_balance)
        self.circuit_breaker = CircuitBreaker(
            price_velocity_threshold=0.005,
            spread_multiplier_threshold=5.0,
            volume_spike_multiplier=10.0,
        )
        # G3: Wire price_provider so cost_model uses live prices for cross-rate conversions
        self.cost_model = CostModel(
            commission_per_lot=self.config.trading.commission_per_lot,
            price_provider=lambda sym: self.get_world(f"data.price.{sym}", None) or 1.0,
        )
        self.set_world("risk.initial_balance", self.initial_balance)
        self.set_world("risk.status", "ready")
        self.log_state("Risk systems active")

    async def perceive(self) -> Dict[str, Any]:
        # G7: Actually check and act on halted symbols
        cb = self.circuit_breaker
        halted_symbols: List[str] = []
        if cb is not None:
            for sym in list(cb.last_halt_time.keys()):
                elapsed = time.time() - cb.last_halt_time[sym]
                if elapsed < cb.cooldown_seconds:
                    halted_symbols.append(sym)
        balance = self.get_world("account.balance", self.initial_balance)
        equity = self.get_world("account.equity", self.initial_balance)

        # G2: Read dynamically adjusted values from adaptive_risk_agent
        self._effective_kelly = self.get_world(
            "adaptive_risk.kelly", self.config.trading.kelly_fraction
        )
        self._effective_risk = self.get_world(
            "adaptive_risk.risk_per_trade", self.config.trading.max_risk_per_trade
        )
        self._adaptive_halted = self.get_world("adaptive_risk.halted", False)

        return {"halted_symbols": halted_symbols, "balance": balance, "equity": equity}

    async def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        actions = []

        # G7: Actually act on halted symbols — send risk alerts
        for sym in perception.get("halted_symbols", []):
            actions.append({"type": "alert_halt", "symbol": sym})

        # G9: If adaptive risk says halt, escalate for human confirmation
        if self._adaptive_halted and self._approved > 0:
            actions.append({"type": "human_confirm_halt"})

        return {"actions": actions, "halted": perception.get("halted_symbols", [])}

    async def act(self, decision: Dict[str, Any]):
        # G7: Act on halted symbols — not just detect them
        for action in decision.get("actions", []):
            if action.get("type") == "alert_halt":
                sym = action.get("symbol", "")
                self.memory.remember(
                    event_type="circuit_breaker_halt",
                    description=f"Market halt active for {sym}",
                    importance=0.8,
                    emotion="warning",
                )
                await self.send(
                    MessageType.RISK_ALERT,
                    payload={
                        "type": "circuit_breaker",
                        "symbol": sym,
                        "reason": f"Circuit breaker active for {sym}",
                    },
                    priority=MessagePriority.HIGH,
                )

            # G9: Request human confirmation for system halt
            elif action.get("type") == "human_confirm_halt":
                self.memory.remember(
                    event_type="human_approval_requested",
                    description="System halt recommended — awaiting human confirmation",
                    importance=0.9,
                    emotion="warning",
                )
                await self.send(
                    MessageType.RISK_ALERT,
                    payload={
                        "type": "halt_request",
                        "reason": "Adaptive risk agent recommends halt — requires confirmation",
                        "requires_confirmation": True,
                    },
                    target="monitoring_agent",
                    priority=MessagePriority.CRITICAL,
                )

    async def reflect(self, outcome: Dict[str, Any]):
        if self.consciousness.cycle_count % 50 == 0:
            self.memory.know("risk.approved", self._approved, ttl=3600)
            self.memory.know("risk.rejected", self._rejected, ttl=3600)
            self.set_world(
                "risk.stats",
                {
                    "approved": self._approved,
                    "rejected": self._rejected,
                    "rejection_breakdown": dict(self._rejection_reasons),
                },
            )

    async def on_message(self, message: AgentMessage):
        if message.msg_type == MessageType.SIGNAL_GENERATED:
            await self._evaluate_signal(message)
        elif message.msg_type == MessageType.DIAGNOSTIC_REQUEST:
            await self.send(
                MessageType.DIAGNOSTIC_RESULT,
                payload={
                    "agent": self.name,
                    "approved": self._approved,
                    "rejected": self._rejected,
                    "kill_switch": (
                        self.risk_manager.kill_switch_triggered
                        if self.risk_manager
                        else False
                    ),
                    "effective_kelly": (
                        self._effective_kelly
                        if hasattr(self, "_effective_kelly")
                        else "N/A"
                    ),
                    "adaptive_halted": (
                        self._adaptive_halted
                        if hasattr(self, "_adaptive_halted")
                        else False
                    ),
                },
                target=message.source_agent,
            )

    async def _evaluate_signal(self, message: AgentMessage):
        payload = message.payload if isinstance(message.payload, dict) else {}
        symbol = payload.get("symbol", "")
        direction = payload.get("direction", "")
        confidence = payload.get("confidence", 0.5)
        price = payload.get("price", 0)

        if self.risk_manager is None:
            await self._reject(message, "Risk manager not initialized")
            return
        if self.cost_model is None:
            await self._reject(message, "Cost model not initialized")
            return

        if self.risk_manager.kill_switch_triggered:
            await self._reject(message, "Kill switch active")
            return
        if self._adaptive_halted if hasattr(self, "_adaptive_halted") else False:
            await self._reject(message, "Adaptive risk halt active")
            return

        atr = self.get_world(f"data.atr.{symbol}", price * 0.001)
        balance = self.get_world("account.balance", self.initial_balance)
        equity = self.get_world("account.equity", self.initial_balance)
        margin = self.get_world("account.margin", 0)
        open_pos_count = self.get_world("account.open_positions", 0)
        max_positions = self.get_world("config.max_positions", 10)

        if open_pos_count >= max_positions:
            await self._reject(message, f"Max positions ({max_positions}) reached")
            return

        approved, reason = self.risk_manager.pre_trade_checks(
            balance, equity, margin, equity - balance
        )
        if not approved:
            await self._reject(message, reason)
            return

        # G2: Use adaptive risk values instead of raw config
        kelly_fraction = self.get_world(
            "adaptive_risk.kelly", self.config.trading.kelly_fraction
        )
        risk_per_trade = self.get_world(
            "adaptive_risk.risk_per_trade", self.config.trading.max_risk_per_trade
        )
        self.risk_manager.params.kelly_fraction = kelly_fraction
        self.risk_manager.params.max_risk_per_trade = risk_per_trade

        volume = self.risk_manager.calculate_kelly_size(
            balance,
            price,
            atr,
            confidence,
            symbol=symbol,
        )
        lot_min = 0.01
        volume = max(round(volume / lot_min) * lot_min, lot_min)
        volume = min(volume, balance * 0.5 / max(price, 0.0001))

        # Live spread from broker depth data (G1) — pips, or None if broker not streaming
        live_spread = self.get_world(f"data.spread.{symbol}", None)
        cost = self.cost_model.calculate(
            symbol=symbol,
            direction=direction,
            volume=volume,
            price=price,
            atr=atr,
            actual_spread_pips=live_spread,
        )
        if not cost.is_acceptable:
            await self._reject(message, cost.rejection_reason)
            return

        # G2: Read dynamically adjusted SL/TP from regime params
        sl_mult = self.get_world(
            "regime.params.sl_atr", self.config.trading.sl_atr_multiplier
        )
        tp_mult = self.get_world(
            "regime.params.tp_atr", self.config.trading.tp_atr_multiplier
        )

        if direction == "BUY":
            sl_price = price - atr * sl_mult
            tp_price = price + atr * tp_mult
        else:
            sl_price = price + atr * sl_mult
            tp_price = price - atr * tp_mult

        self._approved += 1
        self.memory.remember(
            event_type="trade_approved",
            description=f"{direction} {symbol} vol={volume:.2f} (kelly={kelly_fraction:.2f})",
            importance=0.7,
            emotion="success",
        )

        await self.send(
            MessageType.RISK_APPROVED,
            payload={
                "signal": payload,
                "volume": volume,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "cost_total": cost.total,
                "timestamp": time.time(),
            },
            priority=MessagePriority.HIGH,
            requires_ack=True,
            intention=AgentIntention(
                primary_goal=f"approve trade {direction} {symbol}",
                reasoning=f"kelly={kelly_fraction:.2f}, vol={volume:.2f}, cost=${cost.total:.2f}",
                expected_outcome="execution agent receives and sends order",
                confidence=float(confidence),
            ),
        )

    async def _reject(self, message: AgentMessage, reason: str):
        self._rejected += 1
        self._rejection_reasons[reason] = self._rejection_reasons.get(reason, 0) + 1
        self.memory.remember(
            event_type="trade_rejected",
            description=f"Rejected {message.payload.get('symbol', '?')}: {reason}",
            importance=0.5,
            emotion="warning",
        )
        await self.send(
            MessageType.RISK_REJECTED,
            payload={
                "signal": message.payload,
                "reason": reason,
                "timestamp": time.time(),
            },
        )
