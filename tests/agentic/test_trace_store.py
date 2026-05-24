"""Tests for the TraceStore distributed tracing system."""

import time
import pytest
from agentic.core.trace_store import (
    TraceStore,
    TraceEvent,
    get_trace_store,
    reset_trace_store,
)


class TestTraceEvent:
    def test_to_dict_contains_all_fields(self):
        event = TraceEvent(
            agent="test_agent", event="test_event", detail={"key": "val"}
        )
        d = event.to_dict()
        assert d["agent"] == "test_agent"
        assert d["event"] == "test_event"
        assert d["detail"] == {"key": "val"}
        assert "timestamp" in d
        assert "age_s" in d


class TestTraceStore:
    def test_record_and_retrieve(self):
        store = TraceStore()
        store.record("trace_001", "agent_a", "received", {"symbol": "EURUSD"})
        store.record("trace_001", "agent_b", "approved", {"volume": 0.1})
        events = store.get_trace("trace_001")
        assert len(events) == 2
        assert events[0].agent == "agent_a"
        assert events[1].agent == "agent_b"

    def test_get_trace_returns_empty_for_unknown(self):
        store = TraceStore()
        assert store.get_trace("nonexistent") == []

    def test_trace_summary(self):
        store = TraceStore()
        store.record("trace_002", "agent_a", "start")
        store.record("trace_002", "agent_b", "end")
        summary = store.get_trace_summary("trace_002")
        assert summary["trace_id"] == "trace_002"
        assert summary["hop_count"] == 2
        assert "agent_a" in summary["agents"]
        assert summary["duration_s"] >= 0

    def test_get_trace_ids(self):
        store = TraceStore()
        store.record("tid1", "a", "e1")
        store.record("tid2", "b", "e2")
        ids = store.get_trace_ids()
        assert "tid1" in ids
        assert "tid2" in ids

    def test_expiry_removes_old_traces(self):
        store = TraceStore(max_age_seconds=0.001)  # 1ms TTL
        store.record("old_trace", "agent_a", "event")
        time.sleep(0.01)
        assert store.get_trace("old_trace") == []

    def test_get_stats(self):
        store = TraceStore(max_traces=500, max_age_seconds=7200)
        store.record("s1", "a", "e1")
        stats = store.get_stats()
        assert stats["active_traces"] == 1
        assert stats["total_events"] == 1
        assert stats["max_traces"] == 500

    def test_recent_traces_ordered_by_time(self):
        store = TraceStore()
        store.record("first", "a", "e1")
        time.sleep(0.001)
        store.record("second", "b", "e2")
        recent = store.get_recent_traces(n=2)
        assert len(recent) == 2
        # Most recent first
        assert recent[0]["trace_id"] == "second"

    def test_global_singleton(self):
        reset_trace_store()
        s1 = get_trace_store()
        s2 = get_trace_store()
        assert s1 is s2
        reset_trace_store()

    def test_max_traces_enforced(self):
        store = TraceStore(max_traces=3)
        for i in range(5):
            store.record(f"trace_{i}", "agent", f"event_{i}")
        assert len(store.get_trace_ids()) <= 3
