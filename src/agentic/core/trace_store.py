"""
Trace Store — distributed tracing for inter-agent messages.

Each signal/trade generates a trace_id that propagates through the
agent chain. The TraceStore records every hop: which agent received
the message, what it did, and where it forwarded it.

Usage:
    store = get_trace_store()
    store.record(trace_id, "signal_agent", "received", {"symbol": "EURUSD"})
    store.record(trace_id, "risk_agent", "approved", {"volume": 0.1})
    trace = store.get_trace(trace_id)
"""

import time
from typing import Dict, List, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class TraceEvent:
    agent: str
    event: str
    detail: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    cycle_id: int = 0

    def to_dict(self) -> Dict:
        return {
            "agent": self.agent,
            "event": self.event,
            "detail": self.detail,
            "timestamp": self.timestamp,
            "age_s": round(time.time() - self.timestamp, 3),
        }


class TraceStore:
    """Stores trace events for inter-agent message debugging.

    Traces expire after max_age_seconds to prevent memory leaks.
    """

    def __init__(self, max_traces: int = 1000, max_age_seconds: float = 3600.0):
        self._traces: Dict[str, List[TraceEvent]] = defaultdict(list)
        self._max_traces = max_traces
        self._max_age = max_age_seconds
        self._total_events = 0

    def record(
        self,
        trace_id: str,
        agent: str,
        event: str,
        detail: Optional[Dict[str, Any]] = None,
        cycle_id: int = 0,
    ):
        """Record a trace event."""
        self._evict_expired()
        event_obj = TraceEvent(
            agent=agent,
            event=event,
            detail=detail or {},
            cycle_id=cycle_id,
        )
        self._traces[trace_id].append(event_obj)
        self._total_events += 1
        logger.debug(f"Trace [{trace_id[:8]}]: {agent} \u2192 {event}")

    def get_trace(self, trace_id: str) -> List[TraceEvent]:
        """Get all events for a trace_id."""
        self._evict_expired()
        return list(self._traces.get(trace_id, []))

    def get_trace_ids(self) -> List[str]:
        """Get all active trace IDs."""
        self._evict_expired()
        return list(self._traces.keys())

    def get_trace_summary(self, trace_id: str) -> Dict:
        """Get a human-readable summary of a trace."""
        events = self.get_trace(trace_id)
        if not events:
            return {"trace_id": trace_id, "events": [], "duration_s": 0.0}

        start = events[0].timestamp
        end = events[-1].timestamp if len(events) > 1 else start
        agents = list(dict.fromkeys(e.agent for e in events))

        return {
            "trace_id": trace_id,
            "agents": agents,
            "hop_count": len(events),
            "unique_agents": len(agents),
            "duration_s": round(end - start, 3),
            "events": [e.to_dict() for e in events],
        }

    def get_recent_traces(self, n: int = 10) -> List[Dict]:
        """Get summaries of the N most recent traces."""
        self._evict_expired()
        sorted_ids = sorted(
            self._traces.keys(),
            key=lambda tid: self._traces[tid][-1].timestamp if self._traces[tid] else 0,
            reverse=True,
        )[:n]
        return [self.get_trace_summary(tid) for tid in sorted_ids]

    def clear_expired(self):
        """Manually trigger expiration cleanup."""
        self._evict_expired()

    def get_stats(self) -> Dict:
        return {
            "active_traces": len(self._traces),
            "total_events": self._total_events,
            "max_traces": self._max_traces,
            "max_age_seconds": self._max_age,
        }

    def _evict_expired(self):
        """Remove traces that have exceeded max_age."""
        now = time.time()
        expired = [
            tid
            for tid, events in self._traces.items()
            if events and (now - events[-1].timestamp) > self._max_age
        ]
        for tid in expired:
            del self._traces[tid]
        # Also enforce max_traces limit
        if len(self._traces) > self._max_traces:
            sorted_ids = sorted(
                self._traces.keys(),
                key=lambda tid: (
                    self._traces[tid][-1].timestamp if self._traces[tid] else 0
                ),
            )
            for tid in sorted_ids[: len(self._traces) - self._max_traces]:
                del self._traces[tid]


# Global singleton
_store: Optional[TraceStore] = None


def get_trace_store() -> TraceStore:
    global _store
    if _store is None:
        _store = TraceStore()
    return _store


def reset_trace_store():
    """Reset the trace store (useful in tests)."""
    global _store
    _store = None
