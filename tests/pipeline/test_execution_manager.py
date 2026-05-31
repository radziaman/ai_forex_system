"""Tests for ExecutionManager."""

import pytest
from infrastructure.config import AppConfig
from infrastructure.secrets import Secrets
from pipeline.event_bus import EventBus
from pipeline.pipeline_context import PipelineContext
from pipeline.execution_manager import ExecutionManager, ManagedPosition


def _make_ctx(**kwargs) -> PipelineContext:
    """Create a minimal PipelineContext for testing."""
    defaults = {
        "config": AppConfig(),
        "secrets": Secrets(),
        "bus": EventBus(),
    }
    defaults.update(kwargs)
    return PipelineContext(**defaults)


class FakeTradingConfig:
    mode = "PAPER"


class FakeConfig:
    trading = FakeTradingConfig()


@pytest.mark.asyncio
async def test_execution_manager_init():
    ctx = _make_ctx()
    em = ExecutionManager(ctx)
    assert not em.is_alive
    assert em.total_positions == 0


@pytest.mark.asyncio
async def test_execution_manager_start():
    ctx = _make_ctx()
    em = ExecutionManager(ctx)
    await em.start()
    assert em.is_alive
    await em.stop()


@pytest.mark.asyncio
async def test_execute_order_paper():
    ctx = _make_ctx()
    em = ExecutionManager(ctx)
    await em.start()

    result = await em.execute_order(
        symbol="EURUSD",
        direction="BUY",
        volume=0.01,
        sl_price=1.1100,
        tp_price=1.1300,
        signal_price=1.1200,
        confidence=0.8,
        timestamp=1234567890.0,
    )

    assert result["success"]
    assert result["position_id"] > 0
    assert em.total_positions == 1
    assert len(em.open_positions) == 1

    await em.stop()


@pytest.mark.asyncio
async def test_close_position():
    ctx = _make_ctx()
    em = ExecutionManager(ctx)
    await em.start()

    # Open a position
    result = await em.execute_order(
        symbol="EURUSD",
        direction="SELL",
        volume=0.02,
        sl_price=1.1300,
        tp_price=1.1100,
        signal_price=1.1200,
    )
    pid = result["position_id"]

    events = []

    async def closed_handler(**data):
        events.append(data)

    ctx.bus.on("position_closed", closed_handler)

    close_result = await em.close_position(pid, "test_close", 1.1150)
    assert close_result["success"]
    assert len(events) == 1
    assert events[0]["position_id"] == pid
    # SELL at 1.12, close at 1.1150 (price went down, short is profitable)
    pnl = events[0]["pnl"]
    assert pnl >= 0, f"Expected non-negative PnL for profitable short, got {pnl}"

    # Position no longer open
    assert len(em.open_positions) == 0

    await em.stop()


@pytest.mark.asyncio
async def test_close_nonexistent_position():
    ctx = _make_ctx()
    em = ExecutionManager(ctx)
    await em.start()

    result = await em.close_position(99999, "not_found")
    assert not result["success"]

    await em.stop()


@pytest.mark.asyncio
async def test_check_positions_hits_stop_loss():
    ctx = _make_ctx()
    em = ExecutionManager(ctx)
    await em.start()

    # Open a BUY position with SL at 1.1100
    result = await em.execute_order(
        symbol="EURUSD",
        direction="BUY",
        volume=0.01,
        sl_price=1.1100,
        tp_price=1.1300,
        signal_price=1.1200,
    )
    pid = result["position_id"]

    # Simulate price dropping below SL (override _get_current_price)
    em._get_current_price = lambda sym: 1.1050

    await em.check_positions()

    # Position should be closed
    assert len(em.open_positions) == 0

    await em.stop()


@pytest.mark.asyncio
async def test_risk_approved_triggers_execution():
    ctx = _make_ctx()
    em = ExecutionManager(ctx)
    await em.start()

    events = []

    async def opened_handler(**data):
        events.append(data)

    ctx.bus.on("position_opened", opened_handler)

    await ctx.bus.emit(
        "risk_approved",
        signal={
            "symbol": "EURUSD",
            "direction": "BUY",
            "confidence": 0.8,
            "price": 1.12,
        },
        volume=0.01,
        sl_price=1.11,
        tp_price=1.13,
    )

    assert len(events) > 0
    assert events[0]["symbol"] == "EURUSD"
    assert events[0]["direction"] == "BUY"

    await em.stop()


@pytest.mark.asyncio
async def test_update_trailing_stop_long():
    ctx = _make_ctx()
    em = ExecutionManager(ctx)
    await em.start()

    result = await em.execute_order(
        symbol="EURUSD",
        direction="BUY",
        volume=0.01,
        sl_price=1.1100,
        tp_price=1.1300,
        signal_price=1.1200,
    )
    pid = result["position_id"]
    pos = em._positions[pid]
    original_sl = pos.sl

    # Price goes up
    em._get_current_price = lambda sym: 1.1250
    await em.update_trailing_stop(pid)

    assert pos.sl > original_sl

    await em.stop()


@pytest.mark.asyncio
async def test_update_trailing_stop_short():
    ctx = _make_ctx()
    em = ExecutionManager(ctx)
    await em.start()

    result = await em.execute_order(
        symbol="EURUSD",
        direction="SELL",
        volume=0.01,
        sl_price=1.1300,
        tp_price=1.1100,
        signal_price=1.1200,
    )
    pid = result["position_id"]
    pos = em._positions[pid]
    original_sl = pos.sl

    # Price goes down
    em._get_current_price = lambda sym: 1.1150
    await em.update_trailing_stop(pid)

    assert pos.sl < original_sl

    await em.stop()


def test_calculate_pnl():
    ctx = _make_ctx()
    em = ExecutionManager(ctx)

    pos = ManagedPosition(
        position_id=1,
        symbol="EURUSD",
        direction="BUY",
        volume=1.0,
        entry_price=1.12,
        sl=1.11,
        tp=1.14,
        timestamp=0.0,
    )

    pnl = em._calculate_pnl(pos, 1.13)
    assert pnl == 0.01

    pos2 = ManagedPosition(
        position_id=2,
        symbol="EURUSD",
        direction="SELL",
        volume=1.0,
        entry_price=1.12,
        sl=1.14,
        tp=1.10,
        timestamp=0.0,
    )

    pnl2 = em._calculate_pnl(pos2, 1.11)
    assert pnl2 == 0.01
