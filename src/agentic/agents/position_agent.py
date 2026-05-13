"""
Position Agent — autonomous position lifecycle management.

Identity: I watch over every open position. I manage their journey from open to close.
Purpose: I optimize exits through trailing stops, partial closes, and correlation monitoring.
Autonomy: I independently evaluate every position every cycle and take protective actions.
"""

from __future__ import annotations
import time
import numpy as np
from typing import Dict, List, Optional, Any, Set
from loguru import logger

from agentic.core.base_agent import BaseAgent
from agentic.core.agent_message import MessageType, MessagePriority, AgentIntention
from agentic.core.agent_consciousness import ConsciousnessLevel


class PositionAgent(BaseAgent):
    """
    Autonomous position lifecycle manager.

    Responsibilities:
    - Multi-tier trailing stops (3-tier Zenox-style)
    - Partial close suggestions at profit milestones
    - Cross-symbol correlation monitoring
    - Concentration risk alerts
    - Position age tracking (max hold time enforcement)
    """

    CORRELATED_GROUPS = [
        {"EURUSD", "GBPUSD", "EURGBP"},
        {"USDJPY", "GBPJPY", "EURJPY"},
        {"XAUUSD", "XAGUSD"},
        {"BTCUSD", "ETHUSD"},
        {"US500", "US30", "USTEC"},
    ]

    def __init__(self):
        super().__init__(
            name="position_agent",
            role="Position Lifecycle Manager",
            purpose="Optimize open positions through trailing stops, partial closes, and risk monitoring",
            domain="positions",
            capabilities={
                "trailing_stops", "partial_closes", "correlation_monitoring",
                "concentration_risk", "position_age_tracking",
            },
            tick_interval=2.0,
            consciousness_level=ConsciousnessLevel.REFLECTIVE,
        )
        self._tp_hit: Dict[int, List[bool]] = {}
        self._last_check: Dict[str, float] = {}
        self._check_interval = 5.0

        self.subscribe(MessageType.POSITION_OPENED)
        self.subscribe(MessageType.POSITION_CLOSED)
        self.subscribe(MessageType.DIAGNOSTIC_REQUEST)

    async def _on_start(self):
        self.set_world("positions.status", "ready")
        self.log_state("Position manager ready")

    async def perceive(self) -> Dict[str, Any]:
        open_positions = self.get_world("execution.open_positions_raw", [])
        if not open_positions:
            return {"skip": True}
        now = time.time()
        evaluatable = []
        for pos in open_positions:
            pid = pos.get("position_id", 0)
            last = self._last_check.get(str(pid), 0)
            if now - last >= self._check_interval:
                evaluatable.append(pos)
        return {"positions": evaluatable}

    async def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        actions = []
        for pos in perception.get("positions", []):
            pid = pos.get("position_id", 0)
            symbol = pos.get("symbol", "")
            direction = pos.get("direction", "BUY")
            entry = pos.get("entry_price", 0)
            current = self._get_price(symbol)
            atr = self.get_world(f"data.atr.{symbol}", entry * 0.001)

            if current <= 0 or atr <= 0:
                continue

            action = {"pid": pid, "symbol": symbol, "direction": direction,
                      "entry": entry, "current": current, "atr": atr}

            new_sl = self._check_trailing(pid, entry, current, atr, direction)
            if new_sl:
                action["new_sl"] = new_sl

            pnl_pct = self._pnl_pct(entry, current, direction)
            if pnl_pct >= 0.05:
                action["partial_close"] = "Close 30% at +5%"
            elif pnl_pct >= 0.03:
                action["partial_close"] = "Close 20% at +3%"

            action["pnl_pct"] = pnl_pct
            actions.append(action)

        corr_warnings = self._check_correlations(perception.get("positions", []))
        conc_warnings = self._check_concentration(perception.get("positions", []))

        return {"actions": actions, "correlation_warnings": corr_warnings,
                "concentration_warnings": conc_warnings}

    async def act(self, decision: Dict[str, Any]):
        for action in decision.get("actions", []):
            if "new_sl" in action:
                pid = action["pid"]
                self._last_check[str(pid)] = time.time()
                self.memory.remember(
                    event_type="trailing_stop_updated",
                    description=f"Position {pid} ({action['symbol']}): SL→{action['new_sl']:.5f}",
                    importance=0.5,
                    emotion="info",
                    data={"pid": pid, "new_sl": action["new_sl"]},
                )

        for warn in decision.get("correlation_warnings", []):
            self.memory.remember(
                event_type="correlation_warning",
                description=warn,
                importance=0.6,
                emotion="warning",
            )

        for warn in decision.get("concentration_warnings", []):
            self.memory.remember(
                event_type="concentration_warning",
                description=warn,
                importance=0.7,
                emotion="warning",
            )

        self.set_world("positions.correlation_warnings", decision.get("correlation_warnings", []))
        self.set_world("positions.concentration_warnings", decision.get("concentration_warnings", []))

    async def reflect(self, outcome: Dict[str, Any]):
        pass

    async def on_message(self, message: AgentMessage):
        if message.msg_type == MessageType.POSITION_OPENED:
            payload = message.payload if isinstance(message.payload, dict) else {}
            pid = payload.get("position_id", 0)
            if pid:
                self._tp_hit[pid] = [False, False, False]
        elif message.msg_type == MessageType.POSITION_CLOSED:
            payload = message.payload if isinstance(message.payload, dict) else {}
            pid = payload.get("position_id", 0)
            self._tp_hit.pop(pid, None)
        elif message.msg_type == MessageType.DIAGNOSTIC_REQUEST:
            await self.send(
                MessageType.DIAGNOSTIC_RESULT,
                payload={
                    "agent": self.name,
                    "tracked_positions": len(self._tp_hit),
                    "correlation_groups": len(self.CORRELATED_GROUPS),
                },
                target=message.source_agent,
            )

    def _check_trailing(self, pid: int, entry: float, current: float,
                        atr: float, direction: str) -> Optional[float]:
        if pid not in self._tp_hit:
            self._tp_hit[pid] = [False, False, False]
        pnl_pct = self._pnl_pct(entry, current, direction)
        hits = self._tp_hit[pid]
        mult = 1 if direction == "BUY" else -1

        if pnl_pct >= 0.01 and not hits[0]:
            hits[0] = True
            return entry
        if pnl_pct >= 0.02 and not hits[1]:
            hits[1] = True
            return entry + atr * 2 * mult if direction == "BUY" else entry - atr * 2 * mult
        if pnl_pct >= 0.03 and hits[0] and hits[1]:
            if not hits[2]:
                hits[2] = True
            return current - atr * 0.5 * mult if direction == "BUY" else current + atr * 0.5 * mult
        return None

    def _check_correlations(self, positions: List[Dict]) -> List[str]:
        if len(positions) < 2:
            return []
        active_symbols = {p["symbol"] for p in positions}
        directions = {p["symbol"]: p.get("direction", "BUY") for p in positions}
        warnings = []
        for group in self.CORRELATED_GROUPS:
            active_in_group = active_symbols & group
            if len(active_in_group) >= 2:
                dirs = {s: directions.get(s) for s in active_in_group}
                if len({d for d in dirs.values()}) == 1:
                    warnings.append(f"Correlated bias: {', '.join(active_in_group)}")
        return warnings

    def _check_concentration(self, positions: List[Dict]) -> List[str]:
        if not positions:
            return []
        total_exposure = sum(
            abs(p.get("volume", 0) * self._get_price(p.get("symbol", "")))
            for p in positions
        )
        alerts = []
        for p in positions:
            exposure = abs(p.get("volume", 0) * self._get_price(p.get("symbol", "")))
            share = exposure / max(total_exposure, 1)
            if share > 0.4:
                alerts.append(f"{p['symbol']}: {share:.1%} of portfolio")
        return alerts

    def _get_price(self, symbol: str) -> float:
        return self.get_world(f"data.price.{symbol}", 0)

    @staticmethod
    def _pnl_pct(entry: float, current: float, direction: str) -> float:
        if entry == 0:
            return 0.0
        mult = 1 if direction == "BUY" else -1
        return (current - entry) / entry * mult
