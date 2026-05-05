"""
Event-Driven Architecture — Replaces polling loops with async events.
Provides loose coupling between system components via publish/subscribe.
"""
import asyncio
import time
from typing import Dict, List, Callable, Any, Optional, Coroutine
from dataclasses import dataclass, field
from loguru import logger
from enum import Enum


class EventType(str, Enum):
    """All possible system events."""
    # Market events
    TICK = "tick"
    BAR_CLOSED = "bar_closed"
    PRICE_UPDATE = "price_update"
    
    # Regime events
    REGIME_CHANGE = "regime_change"
    REGIME_CONFIRMED = "regime_confirmed"
    
    # Trading signals
    TRADE_SIGNAL = "trade_signal"
    TRADE_APPROVED = "trade_approved"
    TRADE_REJECTED = "trade_rejected"
    
    # Execution events
    ORDER_FILLED = "order_filled"
    ORDER_REJECTED = "order_rejected"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    SL_HIT = "sl_hit"
    TP_HIT = "tp_hit"
    
    # Risk events
    RISK_ALERT = "risk_alert"
    DRAWDOWN_WARNING = "drawdown_warning"
    KILL_SWITCH = "kill_switch"
    DAILY_LOSS = "daily_loss"
    
    # System events
    CIRCUIT_BREAKER = "circuit_breaker"
    TOXIC_FLOW = "toxic_flow"
    MARKET_STRESS = "market_stress"
    
    # Data events
    FEATURES_READY = "features_ready"
    MODEL_PREDICTION = "model_prediction"
    
    # Validation events
    BACKTEST_COMPLETE = "backtest_complete"
    WALK_FORWARD_COMPLETE = "walk_forward_complete"


@dataclass
class Event:
    """An event with payload."""
    type: EventType
    data: Any = None
    timestamp: float = field(default_factory=time.time)
    source: str = ""
    priority: int = 0  # Higher = higher priority


class TradingEventBus:
    """
    Event-driven architecture for loose coupling.
    Components publish events; others subscribe to what they need.
    """
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._async_subscribers: Dict[EventType, List[Callable]] = {}
        self._event_history: List[Event] = []
        self._max_history = 1000
        self._event_stats: Dict[EventType, int] = {}
        self._running = False
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        
    def subscribe(self, event_type: EventType, callback: Callable):
        """Subscribe to an event (sync or async callback)."""
        if asyncio.iscoroutinefunction(callback):
            if event_type not in self._async_subscribers:
                self._async_subscribers[event_type] = []
            self._async_subscribers[event_type].append(callback)
        else:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)
        logger.debug(f"Subscribed to {event_type.value}")
        
    def unsubscribe(self, event_type: EventType, callback: Callable):
        """Unsubscribe from an event."""
        if asyncio.iscoroutinefunction(callback):
            if event_type in self._async_subscribers:
                self._async_subscribers[event_type] = [
                    c for c in self._async_subscribers[event_type] if c != callback
                ]
        else:
            if event_type in self._subscribers:
                self._subscribers[event_type] = [
                    c for c in self._subscribers[event_type] if c != callback
                ]
    
    async def emit(self, event_type: EventType, data: Any = None, source: str = ""):
        """Emit an event (non-blocking)."""
        event = Event(type=event_type, data=data, source=source)
        await self._event_queue.put(event)
        self._event_stats[event_type] = self._event_stats.get(event_type, 0) + 1
        
    def emit_sync(self, event_type: EventType, data: Any = None, source: str = ""):
        """Emit an event synchronously (for non-async contexts)."""
        event = Event(type=event_type, data=data, source=source)
        asyncio.create_task(self._event_queue.put(event))
        self._event_stats[event_type] = self._event_stats.get(event_type, 0) + 1
        
    async def start(self):
        """Start the event processing worker."""
        self._running = True
        self._worker_task = asyncio.create_task(self._process_events())
        logger.info("Event bus started")
        
    async def stop(self):
        """Stop the event processing worker."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("Event bus stopped")
        
    async def _process_events(self):
        """Main event processing loop."""
        while self._running:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=0.1)
                await self._handle_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Event processing error: {e}")
                
    async def _handle_event(self, event: Event):
        """Dispatch event to subscribers."""
        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]
            
        # Call sync subscribers
        if event.type in self._subscribers:
            for callback in self._subscribers[event.type]:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Sync subscriber error: {e}")
                    
        # Call async subscribers
        if event.type in self._async_subscribers:
            for callback in self._async_subscribers[event.type]:
                try:
                    await callback(event)
                except Exception as e:
                    logger.error(f"Async subscriber error: {e}")
                    
    def get_stats(self) -> Dict:
        """Get event bus statistics."""
        return {
            "total_events": sum(self._event_stats.values()),
            "event_types": {k.value: v for k, v in self._event_stats.items()},
            "queue_size": self._event_queue.qsize(),
            "history_size": len(self._event_history),
        }
    
    def get_recent_events(self, event_type: Optional[EventType] = None, n: int = 10) -> List[Event]:
        """Get recent events, optionally filtered by type."""
        if event_type:
            filtered = [e for e in self._event_history if e.type == event_type]
        else:
            filtered = self._event_history
        return filtered[-n:]


# Global event bus instance
_event_bus: Optional[TradingEventBus] = None


def get_event_bus() -> TradingEventBus:
    """Get or create the global event bus."""
    global _event_bus
    if _event_bus is None:
        _event_bus = TradingEventBus()
    return _event_bus
