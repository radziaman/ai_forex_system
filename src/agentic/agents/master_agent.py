"""
Master Agent — G9: human-in-loop halt, G17: hierarchy awareness, G22: error escalation handling.  # noqa: E501
"""

from __future__ import annotations
import time
from typing import Dict, List, Any

from agentic.core.base_agent import BaseAgent
from agentic.core.agent_message import (
    AgentMessage,
    MessageType,
    MessagePriority,
    AgentIntention,
)
from agentic.core.agent_consciousness import ConsciousnessLevel
from agentic.core.agent_registry import get_agent_registry


class MasterAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="master_agent",
            role="System Orchestrator",
            purpose="Orchestrate all agents, monitor health, escalate errors, coordinate healing",  # noqa: E501
            domain="orchestration",
            capabilities={
                "system_health_monitoring",
                "agent_coordination",
                "anomaly_detection",
                "self_healing",
                "directive_issuance",
                "global_state_management",
                "error_escalation",  # G22
                "human_in_loop_approval",  # G9
            },
            tick_interval=10.0,
            consciousness_level=ConsciousnessLevel.META,
        )
        self._system_health: float = 1.0
        self._last_diagnostics = 0.0
        self._directives_issued = 0
        self._error_log: List[Dict] = []  # G22
        self._pending_approvals: Dict[str, Dict] = {}  # G9

        self.subscribe(MessageType.AGENT_HEARTBEAT)
        self.subscribe(MessageType.RISK_ALERT)
        self.subscribe(MessageType.CIRCUIT_BREAKER)
        self.subscribe(MessageType.AGENT_ERROR)  # G22
        self.subscribe(MessageType.DIAGNOSTIC_REQUEST)

    async def _on_start(self):
        self.set_world("master.health", 1.0)
        self.set_world("master.status", "active")
        self.set_world("master.started_at", time.time())
        self.log_state("System orchestrator active")

    async def perceive(self) -> Dict[str, Any]:
        registry = get_agent_registry()
        report = registry.health_report()
        alive = report["alive"]
        total = report["total"]
        health_pct = (alive / max(total, 1)) * 100

        dead_agents = {
            k: v
            for k, v in report.get("agents", {}).items()
            if not v.get("alive", True) and k != self.name
        }

        # G17: Check domain-level health
        domain_health = {}
        for domain, info in report.get("by_domain", {}).items():
            domain_health[domain] = f"{info['alive']}/{info['total']} alive"

        return {
            "total_agents": total,
            "alive": alive,
            "health_pct": health_pct,
            "dead_agents": dead_agents,
            "domain_health": domain_health,
            "pending_approvals": len(self._pending_approvals),
        }

    async def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        self._system_health = perception["health_pct"] / 100.0
        self.set_world("master.health", self._system_health)

        actions: Dict[str, Any] = {"directives": [], "alerts": [], "reports": {}}

        # G22: Track domain health
        actions["reports"]["domain_health"] = perception.get("domain_health", {})

        # Check for dead agents
        for name, info in perception.get("dead_agents", {}).items():
            age = info.get("last_heartbeat_age", 0)
            if age > 120:
                actions["directives"].append(
                    {
                        "target": name,
                        "action": "restart",
                        "reason": f"no heartbeat for {age:.0f}s",
                    }
                )

        # Periodic diagnostics
        if time.time() - self._last_diagnostics > 300:
            actions["run_diagnostics"] = True
            self._last_diagnostics = time.time()

        # Update global health from performance
        perf = self.get_world("performance.stats", {})
        if perf.get("sharpe", 0) < -1.0:
            self._system_health = max(0.3, self._system_health - 0.2)

        return actions

    async def act(self, decision: Dict[str, Any]):
        for directive in decision.get("directives", []):
            self._directives_issued += 1
            await self.send(
                MessageType.AGENT_DIRECTIVE,
                payload={"action": directive["action"], "reason": directive["reason"]},
                target=directive["target"],
                priority=MessagePriority.HIGH,
                intention=AgentIntention(
                    primary_goal=f"restart {directive['target']}",
                    reasoning=directive["reason"],
                    expected_outcome="agent resumes normal operation",
                    confidence=0.8,
                ),
            )
        if decision.get("run_diagnostics"):
            await self._run_system_diagnostics()

    async def reflect(self, outcome: Dict[str, Any]):
        self.set_world("master.system_health", round(self._system_health, 3))
        self.set_world("master.directives_issued", self._directives_issued)
        self.set_world(
            "master.domain_health", outcome.get("reports", {}).get("domain_health", {})
        )

    async def on_message(self, message: AgentMessage):
        if message.msg_type == MessageType.AGENT_ERROR:
            # G22: Handle error escalations from any agent
            await self._on_agent_error(message)

        elif message.msg_type == MessageType.RISK_ALERT:
            payload = message.payload if isinstance(message.payload, dict) else {}

            # G9: Human-in-loop approval required for halt
            if payload.get("requires_confirmation"):
                approval_id = f"halt_{int(time.time())}"
                self._pending_approvals[approval_id] = {
                    "message": message,
                    "timestamp": time.time(),
                    "approved": False,
                    "timeout": 300,
                }
                self.log_state(
                    f"HALT requires human approval (id={approval_id}) — waiting 5min",
                    "critical",
                )
                self.memory.remember(
                    event_type="human_approval_pending",
                    description=f"Halt pending human approval #{approval_id}",
                    importance=1.0,
                    emotion="warning",
                )
                return

            # Automatic halt (non-human path)
            alert_type = payload.get("type", "")
            if alert_type == "halt":
                await self.send(
                    MessageType.AGENT_DIRECTIVE,
                    payload={"action": "close_all", "reason": "risk_alert_halt"},
                    target="execution_agent",
                    priority=MessagePriority.CRITICAL,
                )

        elif message.msg_type == MessageType.DIAGNOSTIC_REQUEST:
            registry = get_agent_registry()
            await self.send(
                MessageType.DIAGNOSTIC_RESULT,
                payload={
                    "agent": self.name,
                    "system_health": self._system_health,
                    "agents_alive": registry.health_report()["alive"],
                    "total_agents": registry.health_report()["total"],
                    "errors_logged": len(self._error_log),
                    "pending_approvals": len(self._pending_approvals),
                },
                target=message.source_agent,
            )

    # G22: Handle agent errors
    async def _on_agent_error(self, message: AgentMessage):
        payload = message.payload if isinstance(message.payload, dict) else {}
        error_entry = {
            "source": payload.get("source", "unknown"),
            "error": payload.get("error", "unknown"),
            "consecutive": payload.get("consecutive_errors", 0),
            "timestamp": payload.get("timestamp", time.time()),
        }
        self._error_log.append(error_entry)
        if len(self._error_log) > 100:
            self._error_log = self._error_log[-100:]

        source = error_entry["source"]
        consecutive = error_entry["consecutive"]
        self.log_state(
            f"Error from {source}: {consecutive} consecutive failures", "warning"
        )

        # Escalate if critical
        if consecutive >= 5:
            self.log_state(
                f"CRITICAL: {source} has {consecutive} consecutive errors", "critical"
            )
            await self.send(
                MessageType.RISK_ALERT,
                payload={
                    "type": "agent_critical",
                    "agent": source,
                    "reason": f"{consecutive} consecutive failures",
                    "requires_confirmation": True,
                },
                priority=MessagePriority.CRITICAL,
            )

        # G17: Notify domain supervisor if hierarchy exists
        registry = get_agent_registry()
        agent_reg = registry.get(source)
        if agent_reg and agent_reg.supervisor:
            await self.send(
                MessageType.AGENT_DIRECTIVE,
                payload={
                    "action": "investigate_agent",
                    "agent": source,
                    "error": error_entry["error"],
                },
                target=agent_reg.supervisor,
                priority=MessagePriority.HIGH,
            )

    async def _run_system_diagnostics(self):
        self.log_state("Running system-wide diagnostics...")
        registry = get_agent_registry()
        report = registry.health_report()

        dead = [
            k
            for k, v in report.get("agents", {}).items()
            if not v.get("alive", True) and k != self.name
        ]
        degraded = [
            k
            for k, v in report.get("agents", {}).items()
            if v.get("health", 1) < 0.5 and k != self.name
        ]
        error_sources = list(set(e["source"] for e in self._error_log[-50:]))

        if dead:
            self.memory.remember(
                event_type="diagnostic_dead_agents",
                description=f"Dead: {dead}",
                importance=0.8,
                emotion="warning",
            )
        if degraded:
            self.memory.remember(
                event_type="diagnostic_degraded",
                description=f"Degraded: {degraded}",
                importance=0.6,
                emotion="warning",
            )

        self.set_world(
            "master.last_diagnostics",
            {
                "time": time.time(),
                "dead": dead,
                "degraded": degraded,
                "error_sources": error_sources,
                "health": self._system_health,
                "pending_approvals": len(self._pending_approvals),
            },
        )
