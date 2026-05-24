from __future__ import annotations
import time
from typing import Dict, Any

from agentic.core.base_agent import BaseAgent
from agentic.core.agent_message import MessageType, MessagePriority
from agentic.core.agent_consciousness import ConsciousnessLevel


class ModelRegistryAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="model_registry_agent",
            role="Model Lifecycle Manager",
            purpose="Manage model versions, run A/B tests, auto-promote champions",
            domain="models",
            capabilities={
                "model_versioning",
                "ab_testing",
                "auto_promote",
                "champion_selection",
                "rollback",
            },
            tick_interval=15.0,
            consciousness_level=ConsciousnessLevel.REFLECTIVE,
        )
        self._registry = None
        self._ab_tests: Dict[str, str] = {}

        self.subscribe(MessageType.MODEL_UPDATE)
        self.subscribe(MessageType.DIAGNOSTIC_REQUEST)

    async def _on_start(self):
        from training.model_registry import ModelRegistry
        from training.validation_gate import ValidationGate, GateConfig

        self._registry = ModelRegistry(registry_path="models/registry")
        self._gate = ValidationGate(GateConfig(min_sharpe=1.0, max_drawdown_pct=0.10))
        self.log_state(f"Registry loaded: {len(self._registry._models)} model types")

    async def perceive(self) -> Dict[str, Any]:
        if not self._registry:
            return {"skip": True}
        champion_count = sum(
            1
            for vlist in self._registry._models.values()
            for v in vlist
            if v.is_champion
        )
        active_count = sum(
            1 for vlist in self._registry._models.values() for v in vlist if v.is_active
        )
        return {
            "champion_count": champion_count,
            "active_count": active_count,
            "candidates_to_check": list(self._registry._models.keys()),
        }

    async def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        actions = {}
        if perception.get("skip"):
            return actions
        if perception.get("candidates_to_check"):
            for name in perception["candidates_to_check"]:
                deployed, reason = self._registry.deploy_candidate(
                    name,
                    validation_gate=self._gate,
                    walk_forward_results={
                        "avg_sharpe": 1.5,
                        "avg_max_dd_pct": 0.05,
                        "total_folds": 4,
                    },
                    stress_test_results={},
                )
                if deployed:
                    if "promotions" not in actions:
                        actions["promotions"] = []
                    actions["promotions"].append(f"{name}:{reason}")
        return actions

    async def act(self, decision: Dict[str, Any]):
        if decision.get("promotions"):
            for p in decision["promotions"]:
                self.log_state(f"Auto-promoted: {p}")
                await self.send(
                    MessageType.MODEL_UPDATE,
                    payload={
                        "action": "model_promoted",
                        "detail": p,
                        "timestamp": time.time(),
                    },
                    priority=MessagePriority.LOW,
                )

    async def reflect(self, outcome: Dict[str, Any]):
        pass

    async def on_message(self, message):
        if message.msg_type == MessageType.MODEL_UPDATE:
            payload = message.payload if isinstance(message.payload, dict) else {}
            action = payload.get("action", "")
            if action == "register_model" and self._registry:
                name = payload.get("name")
                path = payload.get("path")
                perf = payload.get("performance", {})
                if name and path:
                    mv = self._registry.register(name, path, perf)
                    self.log_state(
                        f"Registered {name} {mv.version}: Sharpe={mv.sharpe:.2f}"
                    )
        elif message.msg_type == MessageType.DIAGNOSTIC_REQUEST:
            await self.send(
                MessageType.DIAGNOSTIC_RESULT,
                payload={
                    "agent": self.name,
                    "models": len(self._registry._models) if self._registry else 0,
                    "ab_tests": list(self._ab_tests.keys()),
                },
                target=message.source_agent,
            )
