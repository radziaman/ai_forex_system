from __future__ import annotations
import time
from typing import Dict, Any

from agentic.core.base_agent import BaseAgent
from agentic.core.agent_message import MessageType, MessagePriority
from agentic.core.agent_consciousness import ConsciousnessLevel


class DriftAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="drift_agent",
            role="Concept Drift Monitor",
            purpose="Detect distribution shifts across all symbols and broadcast drift alerts",  # noqa: E501
            domain="monitoring",
            capabilities={
                "drift_detection",
                "adwin_monitoring",
                "feature_drift",
                "performance_drift",
                "broadcast_alerts",
            },
            tick_interval=10.0,
            consciousness_level=ConsciousnessLevel.REFLECTIVE,
        )
        self._drift_monitors: Dict[str, Any] = {}
        self._symbols: list = []
        self.subscribe(MessageType.EXECUTION_RESULT)
        self.subscribe(MessageType.FEATURES_READY)
        self.subscribe(MessageType.DIAGNOSTIC_REQUEST)

    async def _on_start(self):
        from rts_ai_fx.drift_detector import DriftMonitor

        # Only monitor drift for actively traded symbols — major FX + XAUUSD
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
        self._symbols = ACTIVE_SYMBOLS
        for sym in ACTIVE_SYMBOLS:
            self._drift_monitors[sym] = DriftMonitor()
        self.log_state(f"Monitoring drift for {len(ACTIVE_SYMBOLS)} symbols")

    async def perceive(self) -> Dict[str, Any]:
        drifted_symbols = []
        for sym, monitor in self._drift_monitors.items():
            if monitor.retrain_triggered:
                drifted_symbols.append(sym)
        return {
            "drifted_symbols": drifted_symbols,
            "total_monitored": len(self._drift_monitors),
        }

    async def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        actions = {}
        if perception.get("drifted_symbols"):
            actions["broadcast_drift"] = perception["drifted_symbols"]
        return actions

    async def act(self, decision: Dict[str, Any]):
        if decision.get("broadcast_drift"):
            symbols = decision["broadcast_drift"]
            self.set_world("signal.drifted_symbols", len(symbols))
            self.set_world("signal.drifted_list", symbols)
            for sym in symbols:
                self._drift_monitors[sym].reset()
            await self.send(
                MessageType.MODEL_UPDATE,
                payload={
                    "action": "drift_detected",
                    "symbols": symbols,
                    "count": len(symbols),
                    "timestamp": time.time(),
                },
                priority=MessagePriority.HIGH,
            )

    async def record_prediction(self, symbol: str, prediction: float, actual: float):
        mon = self._drift_monitors.get(symbol)
        if mon:
            mon.update(prediction, actual)

    async def reflect(self, outcome: Dict[str, Any]):
        pass

    async def on_message(self, message):
        if message.msg_type == MessageType.EXECUTION_RESULT:
            payload = message.payload if isinstance(message.payload, dict) else {}
            sym = payload.get("symbol", "")
            pred = payload.get("predicted_price", 0)
            actual = payload.get("filled_price", 0)
            if sym and pred > 0 and actual > 0:
                await self.record_prediction(sym, pred, actual)
        elif message.msg_type == MessageType.DIAGNOSTIC_REQUEST:
            await self.send(
                MessageType.DIAGNOSTIC_RESULT,
                payload={
                    "agent": self.name,
                    "drifted": [
                        s
                        for s, m in self._drift_monitors.items()
                        if m.retrain_triggered
                    ],
                    "total": len(self._drift_monitors),
                },
                target=message.source_agent,
            )
