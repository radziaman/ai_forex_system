"""Lightweight pub/sub event bus — replaces AgentBus for pipeline communication.

Simple publish/subscribe event system. No ACKs, no priority queues,
no capability routing, no agent consciousness.

Events:
    "tick" -> (symbol, bid, ask, volume, timestamp)
    "features_ready" -> (symbol, features, timestamp)
    "regime_changed" -> (from_regime, to_regime)
    "signal_generated" -> (signal_dict)
    "risk_approved" -> (signal, volume, sl, tp)
    "risk_rejected" -> (signal, reason)
    "execution_result" -> (result_dict)
    "position_opened" -> (position_dict)
    "position_closed" -> (position_dict)
    "drift_detected" -> (symbol, metric, score)
    "health_check" -> (health_dict)
"""

import asyncio
from typing import Callable, Dict, List, Any, Optional, Tuple
from collections import defaultdict
from loguru import logger


class EventBus:
    """Lightweight pub/sub event bus.
    No ACKs, no priority queues, no capability routing.
    """

    def __init__(self):
        self._handlers: Dict[str, List[Tuple[int, Callable]]] = defaultdict(list)

    def on(
        self, event: str, handler: Optional[Callable] = None, priority: int = 10
    ) -> Callable:
        """Subscribe to an event. Can be used as a decorator.

        Usage:
            # Direct subscription
            bus.on("event", handler)

            # Decorator
            @bus.on("event")
            async def handler(**data):
                ...
        """
        if handler is None:
            # Used as decorator: @bus.on("event")
            def decorator(fn: Callable) -> Callable:
                self._handlers[event].append((priority, fn))
                return fn

            return decorator
        self._handlers[event].append((priority, handler))
        return handler

    def once(
        self, event: str, handler: Optional[Callable] = None, priority: int = 10
    ) -> Callable:
        """Subscribe to an event for a single invocation.
        Auto-unsubscribes after the first emit.

        Can be used as a decorator:

            @bus.once("event")
            async def handler(**data):
                ...
        """
        if handler is None:

            def decorator(fn: Callable) -> Callable:
                async def wrapper(**data):
                    self.off(event, wrapper)
                    await fn(**data)

                self._handlers[event].append((priority, wrapper))
                return fn

            return decorator

        async def wrapper(**data):
            self.off(event, wrapper)
            await handler(**data)

        self._handlers[event].append((priority, wrapper))
        return handler

    def off(self, event: str, handler: Callable) -> None:
        """Unsubscribe from an event."""
        if event not in self._handlers:
            return
        self._handlers[event] = [
            (p, h) for (p, h) in self._handlers[event] if h is not handler
        ]

    async def emit(self, event: str, **data: Any) -> None:
        """Emit an event to all subscribers.

        Handlers are sorted by priority (highest first). Equal-priority
        handlers preserve insertion order. Errors in individual handlers
        are caught and logged without affecting other handlers.
        """
        handlers = self._handlers.get(event, [])
        for _priority, handler in sorted(handlers, key=lambda x: -x[0]):
            try:
                await handler(**data)
            except Exception:
                logger.exception(f"EventBus handler error for '{event}'")

    async def wait_for(self, event: str, timeout: float = 5.0) -> Dict[str, Any]:
        """Wait for the next occurrence of *event*.

        Returns the data dict of the first matching emit().
        Raises asyncio.TimeoutError if *event* doesn't fire within *timeout* seconds.
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        async def handler(**data):
            if not future.done():
                future.set_result(data)

        self.on(event, handler)
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            self.off(event, handler)

    def clear(self) -> None:
        """Clear all handlers (useful in tests)."""
        self._handlers.clear()
