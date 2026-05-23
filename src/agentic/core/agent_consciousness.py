"""
Agent Consciousness — multi-dimensional self-awareness layer.

Tracks:
- Who (identity, role, purpose)
- What (current action, state, cycle phase)
- When (timeline, schedule, cycle count)
- Where (relationships, dependencies)
- Why (intention, reasoning)
- How (metrics, health, efficiency)
- How feels (G15: emotional state: fatigue, stress, confidence, engagement)
- Errors (G22: escalation tracking)
- Resources (G25: cycle time budget)
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Set, Tuple
from enum import Enum, auto


class ConsciousnessLevel(Enum):
    MINIMAL = 0
    BASIC = 1
    AWARE = 2
    REFLECTIVE = 3
    META = 4


class AgentState(Enum):
    BOOTING = auto()
    IDLE = auto()
    PERCEIVING = auto()
    REASONING = auto()
    ACTING = auto()
    REFLECTING = auto()
    COMMUNICATING = auto()
    SLEEPING = auto()
    ERROR = auto()
    HALTED = auto()
    SIMULATING = auto()  # G18


@dataclass
class EmotionalState:
    """G15: Multi-dimensional emotional model."""

    fatigue: float = 0.0  # 0=fresh, 1=exhausted (increases with cycles)
    stress: float = 0.0  # 0=calm, 1=panicked (increases with errors)
    engagement: float = 0.5  # 0=bored, 1=hyperfocused (increases with activity)
    confidence: float = 1.0  # 0=uncertain, 1=certain (decreases with errors)
    curiosity: float = 0.5  # 0=complacent, 1=exploratory (decreases over time)

    def update(
        self,
        cycle_success: bool,
        consecutive_errors: int,
        cycles_since_meaningful: int,
        active: bool,
    ):
        # Fatigue: increases with cycles, resets on meaningful action
        self.fatigue = min(1.0, self.fatigue + 0.001)
        if active:
            self.fatigue = max(0.3, self.fatigue - 0.02)

        # Stress: spikes on errors, decays over time
        if not cycle_success:
            self.stress = min(1.0, self.stress + 0.1 * consecutive_errors)
        else:
            self.stress = max(0.0, self.stress - 0.02)

        # Engagement: increases with activity, decreases with idleness
        if active:
            self.engagement = min(1.0, self.engagement + 0.05)
        else:
            self.engagement = max(0.0, self.engagement - 0.01)

        # Confidence: decreases on errors, recovers on success
        if cycle_success:
            self.confidence = min(1.0, self.confidence + 0.02)
        else:
            self.confidence = max(0.0, self.confidence - 0.1)

        # Curiosity: decays over time, spikes on exploration
        self.curiosity = max(0.1, self.curiosity - 0.005)

    @property
    def overall(self) -> float:
        """Composite emotional health."""
        return (
            self.engagement * 0.3
            + self.confidence * 0.4
            + (1.0 - self.fatigue) * 0.1
            + (1.0 - self.stress) * 0.1
            + self.curiosity * 0.1
        )

    @property
    def dominant(self) -> str:
        names = {
            self.fatigue: "fatigued",
            self.stress: "stressed",
            self.engagement: "engaged",
            self.confidence: "confident",
            self.curiosity: "curious",
        }
        return names[max(names, key=names.get)]

    def summary(self) -> Dict[str, Any]:
        return {
            "fatigue": round(self.fatigue, 3),
            "stress": round(self.stress, 3),
            "engagement": round(self.engagement, 3),
            "confidence": round(self.confidence, 3),
            "curiosity": round(self.curiosity, 3),
            "overall": round(self.overall, 3),
            "dominant": self.dominant,
        }


@dataclass
class AgentIdentity:
    """Who the agent is — immutable."""

    name: str
    role: str
    purpose: str
    version: str = "1.0.0"
    capabilities: Set[str] = field(default_factory=set)
    dependencies: List[str] = field(default_factory=list)
    domain: str = ""

    def describe(self) -> str:
        return f"[{self.name}] {self.role} — {self.purpose}"


@dataclass
class AgentCycleMetrics:
    cycle_id: int = 0
    perceive_ms: float = 0.0
    reason_ms: float = 0.0
    act_ms: float = 0.0
    reflect_ms: float = 0.0
    total_ms: float = 0.0
    messages_sent: int = 0
    messages_received: int = 0
    errors: int = 0
    success: bool = True
    skipped: bool = False  # G11: adaptive cycle skipping

    @property
    def efficiency(self) -> float:
        if self.total_ms == 0:
            return 1.0
        overhead = self.total_ms - (self.perceive_ms + self.act_ms)
        return 1.0 - (overhead / max(self.total_ms, 1))


@dataclass
class AgentConsciousness:
    identity: AgentIdentity
    level: ConsciousnessLevel = ConsciousnessLevel.AWARE

    # Dynamic state
    current_state: AgentState = AgentState.BOOTING
    previous_state: AgentState = AgentState.BOOTING
    state_reason: str = "initializing"
    state_changed_at: float = field(default_factory=time.time)

    # Cycle tracking
    cycle_count: int = 0
    total_cycles: int = 0
    cycle_start: float = 0.0
    cycle_history: List[AgentCycleMetrics] = field(default_factory=list)
    max_cycle_history: int = 100

    # Health
    health_score: float = 1.0
    consecutive_errors: int = 0
    max_consecutive_errors: int = 5
    is_halted: bool = False
    halt_reason: str = ""

    # G15: Emotional state
    emotion: EmotionalState = field(default_factory=EmotionalState)

    # G22: Error escalation
    error_escalation_level: int = 0  # 0=normal, 1=warning, 2=critical, 3=halted
    last_error_time: float = 0.0
    error_history: List[Tuple[float, str]] = field(default_factory=list)
    max_error_history: int = 20

    # G25: Resource governance
    cycle_budget_ms: float = 5000.0  # max ms per cycle before forced skip
    consecutive_overruns: int = 0
    max_consecutive_overruns: int = 3
    last_cycle_overran: bool = False
    skipped_cycles: int = 0

    # G11: Adaptive cycle governance
    cycle_times_ms: List[float] = field(default_factory=list)
    max_cycle_times: int = 20

    # Awareness
    known_agents: Set[str] = field(default_factory=set)
    subscribed_messages: Set[str] = field(default_factory=set)
    current_intention: str = ""
    last_action_summary: str = ""

    # Timing
    uptime_seconds: float = 0.0
    started_at: float = field(default_factory=time.time)
    last_active_at: float = field(default_factory=time.time)
    tick_interval: float = 1.0

    # G18: Simulation mode
    simulation_mode: bool = False

    def start_cycle(self) -> int:
        self.previous_state = self.current_state
        self.current_state = AgentState.PERCEIVING
        self.cycle_count += 1
        self.total_cycles += 1
        self.cycle_start = time.time()
        return self.cycle_count

    def end_cycle(self, metrics: AgentCycleMetrics) -> AgentCycleMetrics:
        metrics.cycle_id = self.cycle_count
        metrics.total_ms = (time.time() - self.cycle_start) * 1000
        self.cycle_history.append(metrics)
        if len(self.cycle_history) > self.max_cycle_history:
            self.cycle_history = self.cycle_history[-self.max_cycle_history :]

        self.current_state = (
            AgentState.IDLE if not self.is_halted else AgentState.HALTED
        )
        self.last_active_at = time.time()
        self.uptime_seconds = self.last_active_at - self.started_at

        # G11: Track cycle times
        self.cycle_times_ms.append(metrics.total_ms)
        if len(self.cycle_times_ms) > self.max_cycle_times:
            self.cycle_times_ms = self.cycle_times_ms[-self.max_cycle_times :]

        # G25: Check for overrun
        if metrics.total_ms > self.cycle_budget_ms:
            self.consecutive_overruns += 1
            self.last_cycle_overran = True
        else:
            self.consecutive_overruns = 0
            self.last_cycle_overran = False

        # G15: Update emotions
        self.emotion.update(
            cycle_success=metrics.success,
            consecutive_errors=self.consecutive_errors,
            cycles_since_meaningful=self.cycle_count % 50,
            active=not metrics.skipped,
        )

        if metrics.success:
            self.consecutive_errors = 0
            self.health_score = min(1.0, self.health_score + 0.05)
        else:
            self.consecutive_errors += 1
            self.health_score = max(0.0, self.health_score - 0.1)
            if self.consecutive_errors >= self.max_consecutive_errors:
                self.halt("max_consecutive_errors_exceeded")

        return metrics

    def record_error(self, error: str):
        self.error_history.append((time.time(), error))
        if len(self.error_history) > self.max_error_history:
            self.error_history = self.error_history[-self.max_error_history :]
        self.last_error_time = time.time()

    def halt(self, reason: str):
        self.is_halted = True
        self.halt_reason = reason
        self.current_state = AgentState.HALTED
        self.health_score = 0.0
        self.error_escalation_level = 3

    def resume(self):
        self.is_halted = False
        self.halt_reason = ""
        self.consecutive_errors = 0
        self.current_state = AgentState.IDLE
        self.health_score = max(0.5, self.health_score)
        self.error_escalation_level = 0

    # G11: Should skip this cycle?
    def should_skip_cycle(self) -> bool:
        if self.last_cycle_overran and self.consecutive_overruns > 0:
            return True
        avg_time = self.avg_cycle_time(5)
        return avg_time > self.tick_interval * 1000 * 1.5

    def avg_cycle_time(self, n: int = 10) -> float:
        recent = self.cycle_history[-n:]
        if not recent:
            return 0.0
        return sum(c.total_ms for c in recent) / len(recent)

    def error_rate(self, n: int = 50) -> float:
        recent = self.cycle_history[-n:]
        if not recent:
            return 0.0
        return sum(1 for c in recent if not c.success) / len(recent)

    def summary(self) -> Dict[str, Any]:
        return {
            "agent": self.identity.name,
            "role": self.identity.role,
            "state": self.current_state.name,
            "health": round(self.health_score, 3),
            "emotion": self.emotion.summary(),
            "cycles": self.total_cycles,
            "skipped_cycles": self.skipped_cycles,
            "uptime_s": round(self.uptime_seconds, 1),
            "avg_cycle_ms": round(self.avg_cycle_time(), 2),
            "error_rate": round(self.error_rate(), 4),
            "escalation": self.error_escalation_level,
            "halted": self.is_halted,
            "known_agents": len(self.known_agents),
            "simulation": self.simulation_mode,
        }
