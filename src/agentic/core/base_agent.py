"""
Base Agent — enhanced with G11 cycle governance, G18 simulation, G22 error escalation, G25 resource limits.  # noqa: E501
"""

from __future__ import annotations
import asyncio
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Any, Set
from loguru import logger

if TYPE_CHECKING:
    from .service_container import ServiceContainer

from .agent_message import AgentMessage, MessageType, MessagePriority, AgentIntention
from .agent_consciousness import AgentConsciousness, AgentIdentity, ConsciousnessLevel
from .agent_consciousness import AgentState, AgentCycleMetrics
from .agent_memory import AgentMemory
from .agent_bus import AgentBus, get_agent_bus
from .agent_registry import AgentRegistry, get_agent_registry
from .world_state import WorldState, get_world_state
from .trace_store import get_trace_store

# Cycle timeout constants
CYCLE_PHASE_TIMEOUT_FACTOR = 0.8  # Each phase times out at 80% of tick_interval
MAX_CONSECUTIVE_FAILURES_FOR_RESTART = 10
BACKOFF_MAX_INTERVAL = 300.0  # 5 minutes max backoff


class BaseAgent(ABC):
    """
    Enhanced base agent.

    G11: Adaptive skip — if cycle takes too long, next cycle is skipped
    G18: Simulation mode — agents can run in sandbox
    G22: Error escalation — consecutive errors reported to master
    G25: Cycle budget — max time per cycle enforced
    """

    def __init__(
        self,
        name: str,
        role: str,
        purpose: str,
        domain: str,
        capabilities: Set[str],
        dependencies: Optional[List[str]] = None,
        tick_interval: float = 1.0,
        consciousness_level: ConsciousnessLevel = ConsciousnessLevel.REFLECTIVE,
        persist_memory: bool = True,
        supervisor: str = "",  # G17
        container: Optional["ServiceContainer"] = None,  # NEW: DI container
    ):
        self.name = name
        self.identity = AgentIdentity(
            name=name,
            role=role,
            purpose=purpose,
            capabilities=capabilities,
            dependencies=dependencies or [],
            domain=domain,
        )
        self.consciousness = AgentConsciousness(
            identity=self.identity,
            level=consciousness_level,
            tick_interval=tick_interval,
        )
        self.memory = AgentMemory(
            agent_name=name,
            persist_path=f"data/agent_memory/{name}" if persist_memory else None,
        )
        self.bus: AgentBus = get_agent_bus()
        self.registry: AgentRegistry = get_agent_registry()
        self.world: WorldState = get_world_state()

        self._container = container
        self._inbox: asyncio.Queue = asyncio.Queue()
        self._outbox: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._main_task: Optional[asyncio.Task] = None
        self._message_task: Optional[asyncio.Task] = None
        self._cycle_metrics: Optional[AgentCycleMetrics] = None
        self._subscriptions: List[MessageType] = []
        self._supervisor = supervisor
        self._last_escalation_time: float = 0.0
        self._current_trace_id = ""  # Set by message handlers when processing a trace

        self.bus.set_registry(self.registry)  # G6: wire registry for capability routing

        logger.info(f"[{name}] Agent initialized — {role}")

    async def start(self):
        self._running = True
        self.consciousness.current_state = AgentState.IDLE
        self.consciousness.started_at = time.time()
        self.registry.register(
            name=self.name,
            role=self.identity.role,
            domain=self.identity.domain,
            capabilities=self.identity.capabilities,
            dependencies=self.identity.dependencies,
            supervisor=self._supervisor,
        )
        self._setup_subscriptions()
        self._message_task = asyncio.create_task(self._message_loop())
        await self._on_start()
        await self.publish_status("started")
        self._main_task = asyncio.create_task(self._cycle_loop())
        logger.info(
            f"[{self.name}] Agent started (interval={self.consciousness.tick_interval}s)"  # noqa: E501
        )

    async def stop(self):
        self._running = False
        await self.publish_status("stopped")
        await self._on_stop()
        if self._message_task:
            self._message_task.cancel()
        try:
            if self._main_task:
                self._main_task.cancel()
                await self._main_task
        except (asyncio.CancelledError, RuntimeError):
            pass
        self.registry.unregister(self.name)
        self.memory.save()
        logger.info(
            f"[{self.name}] Agent stopped ({self.consciousness.total_cycles} cycles)"
        )

    async def send(
        self,
        msg_type: MessageType,
        payload: Any = None,
        target: str = "",
        target_capability: str = "",
        priority: MessagePriority = MessagePriority.NORMAL,
        intention: Optional[AgentIntention] = None,
        requires_ack: bool = False,
        trace_id: str = "",  # for distributed tracing
    ) -> bool:
        msg = AgentMessage(
            msg_type=msg_type,
            priority=priority,
            source_agent=self.name,
            target_agent=target,
            target_capability=target_capability,
            payload=payload,
            requires_ack=requires_ack,
            trace_id=trace_id or self._current_trace_id,
            intention=intention
            or AgentIntention(
                primary_goal=self.consciousness.current_intention or "autonomous",
                reasoning="no explicit reasoning provided",
                expected_outcome="system progress",
            ),
            agent_state_snapshot=self.consciousness.summary(),
            world_state_snapshot=self.world.snapshot(),
        )
        await self.bus.publish(msg)
        self.consciousness.last_action_summary = (
            f"sent {msg_type.name} to {target or target_capability or '*'}"
        )

        # G5: Wait for ack if required
        if requires_ack:
            return await self.bus.wait_for_ack(msg, timeout=5.0)
        return True

    def subscribe(self, msg_type: MessageType):
        self._subscriptions.append(msg_type)
        self.bus.subscribe(msg_type, self._receive)
        self.consciousness.subscribed_messages.add(msg_type.name)

    def log_state(self, message: str, level: str = "info"):
        getattr(logger, level, logger.info)(f"[{self.name}] {message}")

    # ------------------------------------------------------------------
    # Core Cycle
    # ------------------------------------------------------------------

    @abstractmethod
    async def perceive(self) -> Dict[str, Any]: ...

    @abstractmethod
    async def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]: ...

    @abstractmethod
    async def act(self, decision: Dict[str, Any]): ...

    @abstractmethod
    async def reflect(self, outcome: Dict[str, Any]): ...

    async def on_message(self, message: AgentMessage):
        self._inbox.put_nowait(message)

    async def _on_start(self):
        pass

    async def _on_stop(self):
        pass

    # ------------------------------------------------------------------
    # Internal Lifecycle
    # ------------------------------------------------------------------

    def _setup_subscriptions(self):
        for msg_type in self._subscriptions:
            self.bus.subscribe(msg_type, self._receive)

    async def _receive(self, message: AgentMessage):
        self.consciousness.known_agents.add(message.source_agent)
        self.registry.heartbeat(message.source_agent)
        try:
            await self.on_message(message)
        except Exception as e:
            logger.warning(f"[{self.name}] on_message error: {e}")

    async def _cycle_loop(self):  # noqa: C901
        while self._running:
            try:
                cycle_start = time.time()
                self.consciousness.start_cycle()
                self.registry.heartbeat(self.name)
                self.registry.increment_cycles(self.name)

                metrics = AgentCycleMetrics()

                # G11: Adaptive skip — if previous cycles are too slow, skip
                if self.consciousness.should_skip_cycle():
                    metrics.skipped = True
                    self.consciousness.skipped_cycles += 1
                    self.consciousness.end_cycle(metrics)
                    await asyncio.sleep(self.consciousness.tick_interval)
                    continue

                # G18: In simulation mode, skip real I/O
                if self.consciousness.simulation_mode:
                    perception = {"simulation": True, "skip": True}
                    metrics.success = True
                    self.consciousness.end_cycle(metrics)
                    await asyncio.sleep(self.consciousness.tick_interval)
                    continue

                # 1. Perceive (with timeout)
                t0 = time.time()
                phase_timeout = max(
                    1.0, self.consciousness.tick_interval * CYCLE_PHASE_TIMEOUT_FACTOR
                )
                try:
                    perception = await asyncio.wait_for(
                        self.perceive(),
                        timeout=phase_timeout,
                    )
                    metrics.perceive_ms = (time.time() - t0) * 1000
                except asyncio.TimeoutError:
                    logger.warning(
                        f"[{self.name}] perceive() timed out after {phase_timeout:.1f}s"
                    )
                    self.consciousness.record_error("perceive_timeout")
                    await self._handle_cycle_failure()
                    continue

                if not perception or perception.get("skip", False):
                    metrics.success = True
                    self.consciousness.end_cycle(metrics)
                    await asyncio.sleep(self.consciousness.tick_interval)
                    continue

                # 2. Reason (with timeout)
                t0 = time.time()
                try:
                    decision = await asyncio.wait_for(
                        self.reason(perception),
                        timeout=phase_timeout,
                    )
                    metrics.reason_ms = (time.time() - t0) * 1000
                except asyncio.TimeoutError:
                    logger.warning(
                        f"[{self.name}] reason() timed out after {phase_timeout:.1f}s"
                    )
                    self.consciousness.record_error("reason_timeout")
                    await self._handle_cycle_failure()
                    continue

                if not decision or decision.get("skip", False):
                    metrics.success = True
                    self.consciousness.end_cycle(metrics)
                    await asyncio.sleep(self.consciousness.tick_interval)
                    continue

                # G25: Check cycle budget before acting
                elapsed_ms = (time.time() - cycle_start) * 1000
                if elapsed_ms > self.consciousness.cycle_budget_ms:
                    metrics.success = True
                    metrics.skipped = True
                    self.consciousness.end_cycle(metrics)
                    continue

                # 3. Act (with timeout)
                self.consciousness.current_state = AgentState.ACTING
                t0 = time.time()
                try:
                    await asyncio.wait_for(
                        self.act(decision),
                        timeout=phase_timeout,
                    )
                    metrics.act_ms = (time.time() - t0) * 1000
                except asyncio.TimeoutError:
                    logger.warning(
                        f"[{self.name}] act() timed out after {phase_timeout:.1f}s"
                    )
                    self.consciousness.record_error("act_timeout")
                    await self._handle_cycle_failure()
                    continue

                while not self._outbox.empty():
                    msg = await self._outbox.get()
                    await self.bus.publish(msg)
                    metrics.messages_sent += 1

                # Record trace for this cycle
                if self._current_trace_id:
                    get_trace_store().record(
                        trace_id=self._current_trace_id,
                        agent=self.name,
                        event="cycle_complete",
                        detail={
                            "decision_keys": (
                                list(decision.keys())
                                if isinstance(decision, dict)
                                else []
                            )
                        },
                        cycle_id=self.consciousness.cycle_count,
                    )

                # 4. Reflect
                self.consciousness.current_state = AgentState.REFLECTING
                t0 = time.time()
                await self.reflect(decision)
                metrics.reflect_ms = (time.time() - t0) * 1000

                metrics.messages_received = self._inbox.qsize()
                metrics.success = True
                self.consciousness.end_cycle(metrics)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[{self.name}] Cycle error: {e}")
                metrics = AgentCycleMetrics(success=False, errors=1)
                self.consciousness.end_cycle(metrics)
                self.consciousness.record_error(str(e))

                # G22: Escalate to master on consecutive errors
                if (
                    self.consciousness.consecutive_errors >= 2
                    and time.time() - self._last_escalation_time > 60
                ):
                    self._last_escalation_time = time.time()
                    await self._escalate_error(str(e))

            finally:
                sleep_time = max(
                    0, self.consciousness.tick_interval - (time.time() - cycle_start)
                )
                await asyncio.sleep(sleep_time)

    async def _handle_cycle_failure(self):
        """Handle a cycle failure with exponential backoff and auto-restart."""
        self.consciousness.consecutive_errors += 1
        self.consciousness.health_score = max(
            0.0, self.consciousness.health_score - 0.1
        )

        # Exponential backoff: double interval up to max
        new_interval = min(
            self.consciousness.tick_interval * 2.0,
            BACKOFF_MAX_INTERVAL,
        )
        self.consciousness.tick_interval = new_interval

        logger.warning(
            f"[{self.name}] Cycle failed "
            f"({self.consciousness.consecutive_errors} consecutive). "
            f"Backing off to {new_interval:.1f}s interval."
        )

        # G22: Escalate to master on repeated failures
        if (
            self.consciousness.consecutive_errors >= 2
            and time.time() - self._last_escalation_time > 60
        ):
            self._last_escalation_time = time.time()
            await self._escalate_error(
                f"{self.consciousness.consecutive_errors} consecutive cycle failures"
            )

        # Auto-restart after threshold exceeded
        if (
            self.consciousness.consecutive_errors
            >= MAX_CONSECUTIVE_FAILURES_FOR_RESTART
        ):
            logger.error(
                f"[{self.name}] "
                f"{self.consciousness.consecutive_errors} consecutive failures"
                f" — restarting agent"
            )
            await self.stop()
            await asyncio.sleep(2.0)
            await self.start()
            return

        # Record the error
        metrics = AgentCycleMetrics(success=False, errors=1)
        self.consciousness.end_cycle(metrics)
        await asyncio.sleep(self.consciousness.tick_interval)

    async def _message_loop(self):
        while self._running:
            try:
                message = await asyncio.wait_for(self._inbox.get(), timeout=1.0)
                if message and hasattr(message, "msg_type"):
                    self.memory.remember(
                        event_type=f"msg:{message.msg_type.name}",
                        description=f"Received from {message.source_agent}",
                        importance=0.3,
                    )
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.warning(f"[{self.name}] Message loop error: {e}")

    # ------------------------------------------------------------------
    # G22: Error escalation
    # ------------------------------------------------------------------

    async def _escalate_error(self, error: str):
        await self.send(
            MessageType.AGENT_ERROR,
            payload={
                "error": error,
                "source": self.name,
                "consecutive_errors": self.consciousness.consecutive_errors,
                "timestamp": time.time(),
            },
            target="master_agent",
            priority=MessagePriority.HIGH,
            intention=AgentIntention(
                primary_goal="escalate error to master agent",
                reasoning=f"{self.consciousness.consecutive_errors} consecutive errors: {error}",  # noqa: E501
                expected_outcome="master agent investigates and coordinates healing",
                confidence=0.9,
            ),
        )
        self.registry.update_health(self.name, self.consciousness.health_score)
        self.registry.update_emotion(self.name, self.consciousness.emotion.dominant)

    # ------------------------------------------------------------------
    # G18: Simulation mode
    # ------------------------------------------------------------------

    def enable_simulation(self):
        self.consciousness.simulation_mode = True
        self.log_state("Simulation mode enabled", "warning")

    def disable_simulation(self):
        self.consciousness.simulation_mode = False
        self.log_state("Simulation mode disabled")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def publish_status(self, status: str):
        await self.send(
            MessageType.AGENT_HEARTBEAT,
            payload={
                "status": status,
                "health": self.consciousness.health_score,
                "emotion": self.consciousness.emotion.dominant,
            },
            priority=MessagePriority.LOW,
        )

    def get_world(self, key: str, default: Any = None) -> Any:
        return self.world.get(key, default)

    def set_world(self, key: str, value: Any, ttl: float = 0.0):
        self.world.set(key, value, updated_by=self.name, ttl=ttl)

    def update_world(self, key: str, **kwargs):
        self.world.update(key, **kwargs)

    @property
    def container(self) -> "ServiceContainer":
        """Get the service container, creating a default if not injected."""
        if self._container is None:
            from .service_container import get_container

            self._container = get_container()
        return self._container

    def get_inbox_size(self) -> int:
        return self._inbox.qsize()

    async def drain_inbox(self) -> List[AgentMessage]:
        msgs = []
        while not self._inbox.empty():
            try:
                msgs.append(self._inbox.get_nowait())
            except asyncio.QueueEmpty:
                break
        return msgs

    def status_summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "role": self.identity.role,
            "domain": self.identity.domain,
            "consciousness": self.consciousness.summary(),
            "memory": self.memory.summary(),
            "inbox_size": self.get_inbox_size(),
        }
