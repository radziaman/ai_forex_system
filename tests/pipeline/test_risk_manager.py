"""Tests for RiskManager."""

import pytest
from infrastructure.config import AppConfig
from infrastructure.secrets import Secrets
from pipeline.event_bus import EventBus
from pipeline.pipeline_context import PipelineContext
from pipeline.risk_manager import RiskManager


def _make_ctx(**kwargs) -> PipelineContext:
    """Create a minimal PipelineContext for testing."""
    defaults = {
        "config": AppConfig(),
        "secrets": Secrets(),
        "bus": EventBus(),
    }
    defaults.update(kwargs)
    return PipelineContext(**defaults)


class FakeRiskConfig:
    max_risk_per_trade = 0.02
    max_drawdown = 0.10
    kelly_fraction = 0.25
    sl_atr_multiplier = 2.0
    tp_atr_multiplier = 4.0
    max_consecutive_losses = 5
    max_daily_loss = 0.05
    max_positions = 5


class FakeConfig:
    risk = FakeRiskConfig()


@pytest.mark.asyncio
async def test_risk_manager_init():
    ctx = _make_ctx()
    rm = RiskManager(ctx)
    assert not rm.is_alive
    assert not rm.halted


@pytest.mark.asyncio
async def test_risk_manager_start():
    ctx = _make_ctx()
    rm = RiskManager(ctx)
    await rm.start()
    assert rm.is_alive
    assert rm.effective_kelly == 0.25
    assert rm.effective_risk == 0.02
    await rm.stop()


@pytest.mark.asyncio
async def test_assess_trade_approved():
    ctx = _make_ctx()
    rm = RiskManager(ctx)
    await rm.start()

    signal = {
        "symbol": "EURUSD",
        "direction": "BUY",
        "confidence": 0.8,
        "price": 1.1200,
        "regime": "ranging",
        "timestamp": 1234567890.0,
    }

    result = await rm.assess_trade(signal)
    assert result["approved"]
    assert result["volume"] > 0
    assert result["sl_price"] > 0
    assert result["tp_price"] > 0
    assert "reason" in result

    await rm.stop()


@pytest.mark.asyncio
async def test_assess_trade_halted():
    ctx = _make_ctx()
    rm = RiskManager(ctx)
    await rm.start()
    rm.halted = True

    result = await rm.assess_trade(
        {"symbol": "EURUSD", "direction": "BUY", "confidence": 0.8, "price": 1.12}
    )
    assert not result["approved"]
    assert result["reason"] == "system_halted"

    await rm.stop()


@pytest.mark.asyncio
async def test_assess_trade_low_confidence_with_circuit_breaker():
    """When circuit breaker sets high confidence threshold, low-confidence trades reject."""
    ctx = _make_ctx()
    rm = RiskManager(ctx)
    await rm.start()

    # Override circuit breaker
    class FakeCB:
        def check_market_health(self, symbol, tick):
            class FakeSnapshot:
                confidence_threshold = 0.9

            return False, "", FakeSnapshot()

    rm._circuit_breaker = FakeCB()

    result = await rm.assess_trade(
        {"symbol": "EURUSD", "direction": "BUY", "confidence": 0.5, "price": 1.12}
    )
    assert not result["approved"]
    assert "low_confidence" in result["reason"]

    await rm.stop()


@pytest.mark.asyncio
async def test_assess_trade_consecutive_losses():
    ctx = _make_ctx()
    rm = RiskManager(ctx)
    await rm.start()
    rm._consecutive_losses = 5

    result = await rm.assess_trade(
        {"symbol": "EURUSD", "direction": "BUY", "confidence": 0.8, "price": 1.12}
    )
    assert not result["approved"]

    await rm.stop()


@pytest.mark.asyncio
async def test_position_closed_updates_consecutive_losses():
    ctx = _make_ctx()
    rm = RiskManager(ctx)
    await rm.start()

    await rm._on_position_closed(pnl=-10.0, symbol="EURUSD")
    assert rm._consecutive_losses == 1

    await rm._on_position_closed(pnl=5.0, symbol="EURUSD")
    assert rm._consecutive_losses == 0

    await rm.stop()


@pytest.mark.asyncio
async def test_signal_generated_event_triggers_assessment():
    ctx = _make_ctx()
    rm = RiskManager(ctx)
    await rm.start()

    events = []

    async def approved_handler(**data):
        events.append(("approved", data.get("volume")))

    async def rejected_handler(**data):
        events.append(("rejected", data.get("reason")))

    ctx.bus.on("risk_approved", approved_handler)
    ctx.bus.on("risk_rejected", rejected_handler)

    await ctx.bus.emit(
        "signal_generated",
        symbol="EURUSD",
        direction="BUY",
        confidence=0.8,
        price=1.12,
        regime="ranging",
        timestamp=1234567890.0,
        expert_outputs={},
    )

    assert len(events) > 0

    await rm.stop()


def test_recent_win_rate():
    ctx = _make_ctx()
    rm = RiskManager(ctx)
    assert rm._recent_win_rate() is None

    rm._trade_pnls.extend([10, -5, 15, -3, 8])
    wr = rm._recent_win_rate()
    assert wr is not None
    assert wr == 0.6  # 3 wins out of 5
