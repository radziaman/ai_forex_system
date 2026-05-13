"""
Agent Registry — directory service with heartbeat health, capability discovery (G6), and hierarchy (G17).
"""

from __future__ import annotations
import time
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class AgentRegistration:
    name: str
    role: str
    domain: str
    capabilities: Set[str]
    dependencies: List[str]
    supervisor: str = ""  # G17: hierarchical supervision
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    health: float = 1.0
    is_active: bool = True
    cycle_count: int = 0
    emotional_state: str = "calm"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_alive(self) -> bool:
        return time.time() - self.last_heartbeat < 60


class AgentRegistry:
    """
    Central directory with:
    - G6: Capability-based agent discovery
    - G17: Hierarchical domain supervision
    - Heartbeat health monitoring
    """

    def __init__(self):
        self._agents: Dict[str, AgentRegistration] = {}
        self._capability_index: Dict[str, List[str]] = {}
        self._domain_index: Dict[str, List[str]] = {}
        self._supervisor_index: Dict[str, List[str]] = {}  # G17: supervisor -> agents
        logger.info("AgentRegistry initialized")

    def register(
        self,
        name: str,
        role: str,
        domain: str,
        capabilities: Set[str],
        dependencies: List[str] = None,
        supervisor: str = "",
    ) -> bool:
        if name in self._agents:
            logger.warning(f"AgentRegistry: {name} already registered")
            return False
        self._agents[name] = AgentRegistration(
            name=name, role=role, domain=domain,
            capabilities=capabilities, dependencies=dependencies or [],
            supervisor=supervisor,
        )
        self._domain_index.setdefault(domain, []).append(name)
        for cap in capabilities:
            self._capability_index.setdefault(cap, []).append(name)
        if supervisor:
            self._supervisor_index.setdefault(supervisor, []).append(name)
        logger.info(f"AgentRegistry: registered [{name}] {role} ({domain})"
                    + (f" supervisor={supervisor}" if supervisor else ""))
        return True

    def unregister(self, name: str):
        if name in self._agents:
            reg = self._agents[name]
            self._domain_index.get(reg.domain, []).remove(name)
            for cap in reg.capabilities:
                if cap in self._capability_index and name in self._capability_index[cap]:
                    self._capability_index[cap].remove(name)
            old_supervisor = reg.supervisor
            if old_supervisor and old_supervisor in self._supervisor_index:
                if name in self._supervisor_index[old_supervisor]:
                    self._supervisor_index[old_supervisor].remove(name)
            del self._agents[name]
            logger.info(f"AgentRegistry: unregistered [{name}]")

    def heartbeat(self, name: str):
        if name in self._agents:
            self._agents[name].last_heartbeat = time.time()
            self._agents[name].is_active = True

    def get(self, name: str) -> Optional[AgentRegistration]:
        return self._agents.get(name)

    def find_by_role(self, role: str) -> List[AgentRegistration]:
        return [self._agents[n] for n in self._role_index.get(role, [])
                if n in self._agents]

    def find_by_domain(self, domain: str) -> List[AgentRegistration]:
        return [self._agents[n] for n in self._domain_index.get(domain, [])
                if n in self._agents]

    # G6: Capability-based discovery
    def find_by_capability(self, capability: str) -> List[AgentRegistration]:
        return [self._agents[n] for n in self._capability_index.get(capability, [])
                if n in self._agents]

    def has_capability(self, name: str, capability: str) -> bool:
        reg = self._agents.get(name)
        return reg is not None and capability in reg.capabilities

    # G17: Hierarchy
    def get_subordinates(self, supervisor_name: str) -> List[AgentRegistration]:
        return [self._agents[n] for n in self._supervisor_index.get(supervisor_name, [])
                if n in self._agents]

    def get_supervisor(self, name: str) -> Optional[AgentRegistration]:
        reg = self._agents.get(name)
        if reg and reg.supervisor:
            return self._agents.get(reg.supervisor)
        return None

    def all_alive(self) -> List[AgentRegistration]:
        return [a for a in self._agents.values() if a.is_alive]

    def all_dead(self) -> List[AgentRegistration]:
        return [a for a in self._agents.values() if not a.is_alive]

    def health_report(self) -> Dict[str, Any]:
        alive = len(self.all_alive())
        dead = len(self.all_dead())
        by_domain = {}
        for domain, names in self._domain_index.items():
            by_domain[domain] = {
                "total": len(names),
                "alive": sum(1 for n in names if n in self._agents and self._agents[n].is_alive),
            }
        return {
            "total": len(self._agents),
            "alive": alive,
            "dead": dead,
            "health_pct": (alive / max(len(self._agents), 1)) * 100,
            "by_domain": by_domain,
            "agents": {
                name: {
                    "role": reg.role, "domain": reg.domain,
                    "health": reg.health, "alive": reg.is_alive,
                    "last_heartbeat_age": time.time() - reg.last_heartbeat,
                    "cycles": reg.cycle_count,
                    "supervisor": reg.supervisor,
                    "emotion": reg.emotional_state,
                }
                for name, reg in self._agents.items()
            },
        }

    def update_health(self, name: str, health: float):
        if name in self._agents:
            self._agents[name].health = health

    def update_emotion(self, name: str, emotion: str):
        if name in self._agents:
            self._agents[name].emotional_state = emotion

    def increment_cycles(self, name: str):
        if name in self._agents:
            self._agents[name].cycle_count += 1


_registry: Optional[AgentRegistry] = None


def get_agent_registry() -> AgentRegistry:
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
    return _registry
