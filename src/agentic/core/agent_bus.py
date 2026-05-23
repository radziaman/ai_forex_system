"""
Agent Bus — inter-agent communication backbone with priority queuing, capability routing,  # noqa: E501
parallel dispatch, payload validation, and guaranteed delivery.
"""

from __future__ import annotations
import asyncio
import time
from typing import Dict, List, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field
from loguru import logger

from .agent_message import AgentMessage, MessageType, MessagePriority, validate_payload


@dataclass
class BusStats:
    total_messages: int = 0
    messages_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    messages_by_source: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    messages_dropped: int = 0
    avg_latency_ms: float = 0.0
    queue_sizes: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    active_subscribers: int = 0
    workers: int = 1
    ack_count: int = 0
    routing_count: int = 0
    validation_errors: int = 0


class AgentBus:
    """
    Async message bus with:
    - G4: Priority-based queues (CRITICAL queue processed first)
    - G5: Delivery acknowledgment (ACK messages)
    - G6: Capability-based routing (resolve target via registry)
    - G10: Parallel workers (configurable count)
    - G13: Payload schema validation before dispatch
    """

    def __init__(self, max_history: int = 10000, n_workers: int = 2):
        self._queues: Dict[MessagePriority, asyncio.Queue] = {
            p: asyncio.Queue() for p in MessagePriority
        }
        self._subscribers: Dict[MessageType, List[Callable]] = defaultdict(list)
        self._async_subscribers: Dict[MessageType, List[Callable]] = defaultdict(list)
        self._message_history: deque = deque(maxlen=max_history)
        self._dead_letter_queue: deque = deque(maxlen=1000)
        self._running = False
        self._worker_tasks: List[asyncio.Task] = []
        self._n_workers = n_workers
        self._pending_acks: Dict[str, asyncio.Event] = {}
        self._registry = None  # set by set_registry
        self._stats = BusStats(workers=n_workers)
        self._latency_samples: deque = deque(maxlen=1000)

        logger.info(f"AgentBus initialized ({n_workers} workers)")

    def set_registry(self, registry):
        """Set agent registry for capability-based routing (G6)."""
        self._registry = registry

    def subscribe(self, msg_type: MessageType, callback: Callable):
        if asyncio.iscoroutinefunction(callback):
            self._async_subscribers[msg_type].append(callback)
        else:
            self._subscribers[msg_type].append(callback)
        logger.debug(f"AgentBus: subscribed to {msg_type.name}")

    def unsubscribe(self, msg_type: MessageType, callback: Callable):
        if asyncio.iscoroutinefunction(callback):
            if callback in self._async_subscribers.get(msg_type, []):
                self._async_subscribers[msg_type].remove(callback)
        else:
            if callback in self._subscribers.get(msg_type, []):
                self._subscribers[msg_type].remove(callback)

    async def publish(self, message: AgentMessage):
        if not self._running:
            logger.warning("AgentBus: publish called before start")
            return

        # G13: Validate payload
        valid, error = validate_payload(message.msg_type, message.payload)
        if not valid:
            self._stats.validation_errors += 1
            logger.warning(f"AgentBus: schema validation failed: {error}")
            self._dead_letter_queue.append(
                {
                    "reason": f"schema_error: {error}",
                    "message": str(message),
                    "timestamp": time.time(),
                }
            )
            return

        # G6: Resolve capability routing
        if message.target_capability and self._registry:
            agents = self._registry.find_by_capability(message.target_capability)
            if agents:
                message.target_agent = agents[0].name
                self._stats.routing_count += 1
                logger.debug(
                    f"AgentBus: routed {message.msg_type.name} via capability "
                    f"'{message.target_capability}' → {message.target_agent}"
                )
            else:
                logger.warning(
                    f"AgentBus: no agent with capability '{message.target_capability}'"
                )

        # G4: Queue by priority
        queue = self._queues.get(message.priority, self._queues[MessagePriority.NORMAL])
        await queue.put(message)
        self._stats.total_messages += 1
        self._stats.messages_by_type[message.msg_type.name] += 1
        self._stats.messages_by_source[message.source_agent] += 1

    async def wait_for_ack(self, message: AgentMessage, timeout: float = 5.0) -> bool:
        """G5: Block until ACK received or timeout."""
        if not message.requires_ack:
            return True
        event = asyncio.Event()
        self._pending_acks[message.msg_id] = event
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            self._dead_letter_queue.append(
                {
                    "reason": "ack_timeout",
                    "message": str(message),
                    "timestamp": time.time(),
                }
            )
            return False
        finally:
            self._pending_acks.pop(message.msg_id, None)

    async def start(self):
        self._running = True
        for p in MessagePriority:
            self._queues[p] = asyncio.Queue()
        # G10: Start multiple workers
        self._worker_tasks = [
            asyncio.create_task(self._process_loop(worker_id))
            for worker_id in range(self._n_workers)
        ]
        logger.info(f"AgentBus started ({self._n_workers} workers)")

    async def stop(self):
        self._running = False
        for task in self._worker_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        logger.info(
            f"AgentBus stopped ({self._stats.total_messages} messages processed, "
            f"{self._stats.ack_count} acks, {self._stats.routing_count} route)"
        )

    async def _process_loop(self, worker_id: int):
        while self._running:
            try:
                # G4: Process higher priority queues first
                message = None
                for priority in sorted(
                    MessagePriority, key=lambda p: p.value, reverse=True
                ):
                    try:
                        message = await asyncio.wait_for(
                            self._queues[priority].get(), timeout=0.05
                        )
                        break
                    except asyncio.TimeoutError:
                        continue

                if message is None:
                    await asyncio.sleep(0.01)
                    continue

                await self._dispatch(message, worker_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"AgentBus worker {worker_id} error: {e}")

    async def _dispatch(self, message: AgentMessage, worker_id: int):  # noqa: C901
        if message.is_expired:
            self._dead_letter_queue.append(
                {
                    "reason": "expired",
                    "message": str(message),
                    "timestamp": time.time(),
                    "worker": worker_id,
                }
            )
            self._stats.messages_dropped += 1
            return

        start = time.time()

        # Sync subscribers
        for cb in self._subscribers.get(message.msg_type, []):
            try:
                cb(message)
            except Exception as e:
                logger.warning(f"AgentBus sync error: {e}")

        # Async subscribers
        tasks = []
        for cb in self._async_subscribers.get(message.msg_type, []):
            tasks.append(self._safe_dispatch_async(cb, message, worker_id))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # G5: Send ACK if required
        if message.requires_ack:
            ack = message.ack_message()
            ack.source_agent = "agentbus"
            await self.publish(ack)
            self._stats.ack_count += 1

            # Wake up waiter
            if message.msg_id in self._pending_acks:
                self._pending_acks[message.msg_id].set()

        # G5: Handle incoming ACK
        if message.msg_type == MessageType.MESSAGE_ACK:
            ack_for = (
                message.payload.get("ack_for", "")
                if isinstance(message.payload, dict)
                else ""
            )
            if ack_for in self._pending_acks:
                self._pending_acks[ack_for].set()

        # Record history
        self._message_history.append(message)

        latency = (time.time() - start) * 1000
        self._latency_samples.append(latency)
        self._stats.avg_latency_ms = sum(self._latency_samples) / len(
            self._latency_samples
        )
        for p, q in self._queues.items():
            self._stats.queue_sizes[p.name] = q.qsize()

    async def _safe_dispatch_async(
        self, cb: Callable, message: AgentMessage, worker_id: int
    ):
        try:
            await cb(message)
        except Exception as e:
            logger.warning(f"AgentBus async dispatch error: {e}")
            self._dead_letter_queue.append(
                {
                    "reason": str(e),
                    "message": str(message),
                    "timestamp": time.time(),
                    "worker": worker_id,
                }
            )

    def get_stats(self) -> Dict:
        self._stats.active_subscribers = sum(
            len(v) for v in self._subscribers.values()
        ) + sum(len(v) for v in self._async_subscribers.values())
        return {
            "total_messages": self._stats.total_messages,
            "messages_by_type": dict(self._stats.messages_by_type),
            "messages_by_source": dict(self._stats.messages_by_source),
            "dropped": self._stats.messages_dropped,
            "avg_latency_ms": round(self._stats.avg_latency_ms, 3),
            "queue_sizes": dict(self._stats.queue_sizes),
            "active_subscribers": self._stats.active_subscribers,
            "workers": self._n_workers,
            "ack_count": self._stats.ack_count,
            "routing_count": self._stats.routing_count,
            "validation_errors": self._stats.validation_errors,
            "history_size": len(self._message_history),
            "dead_letter_count": len(self._dead_letter_queue),
        }

    def get_recent_messages(
        self, msg_type: Optional[MessageType] = None, n: int = 20
    ) -> List[AgentMessage]:
        if msg_type:
            return [
                m
                for m in list(self._message_history)[-n * 5 :]
                if m.msg_type == msg_type
            ][-n:]
        return list(self._message_history)[-n:]

    def get_dead_letters(self) -> List[Dict]:
        return list(self._dead_letter_queue)


# Global singleton
_bus: Optional[AgentBus] = None


def get_agent_bus(n_workers: int = 2) -> AgentBus:
    global _bus
    if _bus is None:
        _bus = AgentBus(n_workers=n_workers)
    return _bus
