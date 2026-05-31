"""Tests for EventBus."""

import asyncio

import pytest
from pipeline.event_bus import EventBus


@pytest.mark.asyncio
async def test_subscribe_and_emit():
    bus = EventBus()
    received = []

    async def handler(**data):
        received.append(data)

    bus.on("test_event", handler)
    await bus.emit("test_event", foo=1, bar="hello")

    assert len(received) == 1
    assert received[0] == {"foo": 1, "bar": "hello"}


@pytest.mark.asyncio
async def test_multiple_handlers():
    bus = EventBus()
    results = []

    async def h1(**data):
        results.append("h1")

    async def h2(**data):
        results.append("h2")

    bus.on("evt", h1)
    bus.on("evt", h2)
    await bus.emit("evt")

    assert results == ["h1", "h2"]


@pytest.mark.asyncio
async def test_unsubscribe():
    bus = EventBus()
    results = []

    async def handler(**data):
        results.append("called")

    bus.on("evt", handler)
    bus.off("evt", handler)
    await bus.emit("evt")

    assert results == []


@pytest.mark.asyncio
async def test_handler_error_does_not_bubble():
    bus = EventBus()

    async def failing(**data):
        raise ValueError("oops")

    async def working(**data):
        pass

    bus.on("evt", failing)
    bus.on("evt", working)

    # Should not raise despite failing handler
    await bus.emit("evt")


@pytest.mark.asyncio
async def test_clear():
    bus = EventBus()
    results = []

    async def handler(**data):
        results.append("called")

    bus.on("evt", handler)
    bus.clear()
    await bus.emit("evt")

    assert results == []


@pytest.mark.asyncio
async def test_decorator_usage():
    bus = EventBus()
    results = []

    @bus.on("decorator_event")
    async def handler(**data):
        results.append(data.get("value"))

    await bus.emit("decorator_event", value=42)
    assert results == [42]


@pytest.mark.asyncio
async def test_event_off_nonexistent():
    bus = EventBus()
    # Should not raise
    bus.off("nonexistent", lambda **x: None)


@pytest.mark.asyncio
async def test_emit_no_handlers():
    bus = EventBus()
    # Should not raise
    await bus.emit("orphan_event", data="value")


@pytest.mark.asyncio
async def test_priority_ordering():
    """Higher priority handlers execute first."""
    bus = EventBus()
    results = []

    async def low(**data):
        results.append("low")

    async def medium(**data):
        results.append("medium")

    async def high(**data):
        results.append("high")

    bus.on("evt", low, priority=5)
    bus.on("evt", medium, priority=10)
    bus.on("evt", high, priority=20)

    await bus.emit("evt")
    assert results == ["high", "medium", "low"]


@pytest.mark.asyncio
async def test_priority_same_orders_by_insertion():
    """Same-priority handlers execute in insertion order (stable sort)."""
    bus = EventBus()
    results = []

    async def first(**data):
        results.append("first")

    async def second(**data):
        results.append("second")

    async def third(**data):
        results.append("third")

    bus.on("evt", first, priority=10)
    bus.on("evt", second, priority=10)
    bus.on("evt", third, priority=10)

    await bus.emit("evt")
    assert results == ["first", "second", "third"]


@pytest.mark.asyncio
async def test_once_auto_unsubscribes():
    """once() handler fires only once."""
    bus = EventBus()
    results = []

    async def handler(**data):
        results.append("called")

    bus.once("evt", handler)
    await bus.emit("evt")
    await bus.emit("evt")  # second emit should not trigger

    assert results == ["called"]


@pytest.mark.asyncio
async def test_once_decorator():
    """Decorator form of once() fires only once."""
    bus = EventBus()
    results = []

    @bus.once("evt")
    async def handler(**data):
        results.append("called")

    await bus.emit("evt")
    await bus.emit("evt")  # should only fire once

    assert results == ["called"]


@pytest.mark.asyncio
async def test_wait_for_basic():
    """wait_for() receives data from next emit()."""
    bus = EventBus()

    async def delayed_emit():
        await asyncio.sleep(0.02)
        await bus.emit("custom_event", value=42)

    asyncio.create_task(delayed_emit())
    result = await bus.wait_for("custom_event", timeout=1.0)
    assert result == {"value": 42}


@pytest.mark.asyncio
async def test_wait_for_timeout():
    """wait_for() raises TimeoutError when event doesn't fire."""
    bus = EventBus()

    with pytest.raises(asyncio.TimeoutError):
        await bus.wait_for("never_fires", timeout=0.1)
