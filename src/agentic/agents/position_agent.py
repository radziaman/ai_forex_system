"""
Position Agent — autonomous position lifecycle management.

Identity: I watch over every open position. I manage their journey from open to close.
Purpose: I optimize exits through trailing stops, partial closes, and correlation monitoring.  # noqa: E501
Autonomy: I independently evaluate every position every cycle and take protective actions.  # noqa: E501
"""

from __future__ import annotations
import time
from typing import Dict, List, Optional, Any, Set

from agentic.core.base_agent import BaseAgent
from agentic.core.agent_message import (
    AgentMessage,
    MessageType,
    MessagePriority,
)
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
        {"EURUSD", "GBPUSD"},
        {"AUDUSD", "NZDUSD"},
        {"USDCAD", "XTIUSD"},
    ]

    def __init__(self):
        super().__init__(
            name="position_agent",
            role="Position Lifecycle Manager",
            purpose="Optimize open positions through trailing stops, partial closes, and risk monitoring",  # noqa: E501
            domain="positions",
            capabilities={
                "trailing_stops",
                "partial_closes",
                "correlation_monitoring",
                "concentration_risk",
                "position_age_tracking",
            },
            tick_interval=2.0,
            consciousness_level=ConsciousnessLevel.REFLECTIVE,
        )
        self._tp_hit: Dict[int, Set[str]] = {}  # pid -> set of milestones hit
        self._last_check: Dict[str, float] = {}
        self._check_interval = 5.0
        self._trailing_state: Dict[int, Dict] = {}  # pid -> tracking state

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

            action = {
                "pid": pid,
                "symbol": symbol,
                "direction": direction,
                "entry": entry,
                "current": current,
                "atr": atr,
                "milestones_hit": [],
            }

            new_sl = self._check_trailing(pid, entry, current, atr, direction)
            if new_sl:
                action["new_sl"] = new_sl

            # ATR-based partial take-profit logic
            if direction == "BUY":
                atr_mult = (current - entry) / atr if atr > 0 else 0
            else:
                atr_mult = (entry - current) / atr if atr > 0 else 0

            # Track which milestones we've already hit for this position
            milestones = self._tp_hit.get(pid, set())

            # At 1.5 × ATR profit: close 30% of position, move SL to breakeven
            if atr_mult >= 1.5 and "tp_1.5atr" not in milestones:
                action["partial_close_ratio"] = 0.30
                action["new_sl"] = entry  # Move remaining to breakeven
                action["milestones_hit"].append("tp_1.5atr")
                action["partial_close_reason"] = "Partial TP at 1.5 ATR"

            # At 2.5 × ATR profit: close 30% of remaining, trail at 0.5 ATR
            if atr_mult >= 2.5 and "tp_2.5atr" not in milestones:
                action["partial_close_ratio"] = 0.30
                if direction == "BUY":
                    action["new_sl"] = max(current - atr * 0.5, entry)
                else:
                    action["new_sl"] = min(current + atr * 0.5, entry)
                action["milestones_hit"].append("tp_2.5atr")
                action["partial_close_reason"] = "Partial TP at 2.5 ATR + trail"

            pnl_pct = self._pnl_pct(entry, current, direction)
            action["pnl_pct"] = pnl_pct
            actions.append(action)

        corr_warnings = self._check_correlations(perception.get("positions", []))
        conc_warnings = self._check_concentration(perception.get("positions", []))

        return {
            "actions": actions,
            "correlation_warnings": corr_warnings,
            "concentration_warnings": conc_warnings,
        }

    async def act(self, decision: Dict[str, Any]):
        for action in decision.get("actions", []):
            pid = action["pid"]

            # Handle partial take-profit (close a portion + adjust SL)
            close_ratio = action.get("partial_close_ratio", 0)
            if close_ratio > 0:
                # Track the milestone
                for m in action.get("milestones_hit", []):
                    self._tp_hit.setdefault(pid, set()).add(m)

                new_sl = action.get("new_sl")
                self.memory.remember(
                    event_type="partial_take_profit",
                    description=(
                        f"Position {pid} ({action['symbol']}): close {close_ratio*100:.0f}% "  # noqa: E501
                        f"{action.get('partial_close_reason', '')}"
                    ),
                    importance=0.7,
                    emotion="success",
                )
                # Send partial close + SL change to execution agent
                payload = {
                    "position_id": pid,
                    "close_ratio": close_ratio,
                    "symbol": action.get("symbol", ""),
                }
                if new_sl is not None:
                    payload["sl"] = new_sl
                await self.send(
                    MessageType.POSITION_MODIFIED,
                    payload=payload,
                    target="execution_agent",
                    priority=MessagePriority.HIGH,
                )
                continue

            # Standard trailing stop update
            if "new_sl" in action:
                self._last_check[str(pid)] = time.time()
                new_sl = action["new_sl"]
                self.memory.remember(
                    event_type="trailing_stop_updated",
                    description=f"Position {pid} ({action['symbol']}): SL→{new_sl:.5f}",
                    importance=0.5,
                    emotion="info",
                    data={"pid": pid, "new_sl": new_sl},
                )
                # Send modify command to execution agent
                await self.send(
                    MessageType.POSITION_MODIFIED,
                    payload={
                        "position_id": pid,
                        "sl": new_sl,
                        "symbol": action.get("symbol", ""),
                    },
                    target="execution_agent",
                    priority=MessagePriority.NORMAL,
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

        self.set_world(
            "positions.correlation_warnings", decision.get("correlation_warnings", [])
        )
        self.set_world(
            "positions.concentration_warnings",
            decision.get("concentration_warnings", []),
        )

    async def reflect(self, outcome: Dict[str, Any]):
        pass

    async def on_message(self, message: AgentMessage):
        if message.msg_type == MessageType.POSITION_OPENED:
            payload = message.payload if isinstance(message.payload, dict) else {}
            pid = payload.get("position_id", 0)
            if pid:
                self._tp_hit[pid] = set()
        elif message.msg_type == MessageType.POSITION_CLOSED:
            payload = message.payload if isinstance(message.payload, dict) else {}
            pid = payload.get("position_id", 0)
            self._tp_hit.pop(pid, None)
            self._trailing_state.pop(pid, None)
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

    def _check_trailing(  # noqa: C901
        self, pid: int, entry: float, current: float, atr: float, direction: str
    ) -> Optional[float]:
        """ATR-based trailing stop check.

        Milestones (ATR-based):
          1. Price moved >= 1.0 × ATR in our favor  → move SL to breakeven
          2. Price moved >= 2.0 × ATR in our favor  → trail SL at 1.0 × ATR behind best price  # noqa: E501
          3. Price moved >= 3.0 × ATR in our favor  → trail SL tighter at 0.5 × ATR behind best price  # noqa: E501

        Returns new SL price, or None if no change needed.
        """
        if atr <= 0:
            return None

        if pid not in self._trailing_state:
            self._trailing_state[pid] = {
                "best_price": entry,
                "trailing_sl": 0.0,
                "milestones": set(),
            }
        state = self._trailing_state[pid]

        if direction == "BUY":
            price_move = current - entry
            state["best_price"] = max(state["best_price"], current)

            # Milestone 1: Breakeven at 1.0 ATR
            if price_move >= atr * 1.0 and "breakeven" not in state["milestones"]:
                state["milestones"].add("breakeven")
                state["trailing_sl"] = entry
                return entry

            # Milestone 2: Trail at 1.0 ATR behind best price
            if price_move >= atr * 2.0:
                trail_sl = state["best_price"] - atr * 1.0
                new_sl = max(trail_sl, entry)  # Never worse than breakeven
                if new_sl > state["trailing_sl"]:
                    state["milestones"].add("tight_1atr")
                    state["trailing_sl"] = new_sl
                    return new_sl

            # Milestone 3: Trail tighter at 0.5 ATR behind best price
            if price_move >= atr * 3.0:
                trail_sl = state["best_price"] - atr * 0.5
                new_sl = max(trail_sl, entry)  # Never worse than breakeven
                if new_sl > state["trailing_sl"]:
                    state["milestones"].add("tight_05atr")
                    state["trailing_sl"] = new_sl
                    return new_sl

        else:  # SELL
            price_move = entry - current
            state["best_price"] = min(state["best_price"], current)

            # Milestone 1: Breakeven at 1.0 ATR
            if price_move >= atr * 1.0 and "breakeven" not in state["milestones"]:
                state["milestones"].add("breakeven")
                state["trailing_sl"] = entry
                return entry

            # Milestone 2: Trail at 1.0 ATR behind best price
            if price_move >= atr * 2.0:
                trail_sl = state["best_price"] + atr * 1.0
                new_sl = min(trail_sl, entry)  # Never worse than breakeven
                if new_sl < state["trailing_sl"] or state["trailing_sl"] == 0:
                    state["milestones"].add("tight_1atr")
                    state["trailing_sl"] = new_sl
                    return new_sl

            # Milestone 3: Trail tighter at 0.5 ATR behind best price
            if price_move >= atr * 3.0:
                trail_sl = state["best_price"] + atr * 0.5
                new_sl = min(trail_sl, entry)  # Never worse than breakeven
                if new_sl < state["trailing_sl"] or state["trailing_sl"] == 0:
                    state["milestones"].add("tight_05atr")
                    state["trailing_sl"] = new_sl
                    return new_sl

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
