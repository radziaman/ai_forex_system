"""Tests for SignalEngine."""

import pytest
from infrastructure.config import AppConfig
from infrastructure.secrets import Secrets
from pipeline.event_bus import EventBus
from pipeline.pipeline_context import PipelineContext
from pipeline.signal_engine import SignalEngine


def _make_ctx(**kwargs) -> PipelineContext:
    """Create a minimal PipelineContext for testing."""
    defaults = {
        "config": AppConfig(),
        "secrets": Secrets(),
        "bus": EventBus(),
    }
    defaults.update(kwargs)
    return PipelineContext(**defaults)


class FakeConfig:
    features = type(
        "Features",
        (),
        {"lookback": 10, "timeframes": ["1h"], "use_microstructure": True},
    )()


@pytest.mark.asyncio
async def test_signal_engine_init():
    ctx = _make_ctx()
    engine = SignalEngine(ctx)
    assert not engine.is_alive
    assert engine._signal_count == 0


@pytest.mark.asyncio
async def test_signal_engine_start_creates_ensemble():
    ctx = _make_ctx()
    engine = SignalEngine(ctx)
    await engine.start()
    assert engine.is_alive
    assert engine._ensemble is not None
    await engine.stop()


@pytest.mark.asyncio
async def test_signal_engine_stop():
    ctx = _make_ctx()
    engine = SignalEngine(ctx)
    await engine.start()
    await engine.stop()
    # Should not raise


@pytest.mark.asyncio
async def test_signal_engine_on_tick_without_data_manager():
    """With no data_manager, tick should produce no signal."""
    ctx = _make_ctx()
    engine = SignalEngine(ctx)
    await engine.start()

    signals = []

    async def signal_handler(**data):
        signals.append(data)

    ctx.bus.on("signal_generated", signal_handler)

    await engine._on_tick(
        symbol="EURUSD",
        bid=1.1200,
        ask=1.1201,
        volume=100,
        timestamp=1234567890.0,
    )

    # Without data manager, features will be None -> no signal
    assert len(signals) == 0
    await engine.stop()


@pytest.mark.asyncio
async def test_generate_signal_without_ensemble():
    ctx = _make_ctx()
    engine = SignalEngine(ctx)
    # Don't start — ensemble won't be initialized

    signal = await engine._generate_signal(
        symbol="EURUSD",
        features=None,
        regime="ranging",
    )

    assert signal["symbol"] == "EURUSD"
    assert signal["direction"] == "HOLD"
    assert signal["confidence"] == 0.0


@pytest.mark.asyncio
async def test_register_experts_sets_ensemble():
    ctx = _make_ctx()
    engine = SignalEngine(ctx)
    await engine.start()

    assert engine._ensemble is not None
    assert len(engine._ensemble.experts) > 0
    await engine.stop()


def test_get_current_session():
    ctx = _make_ctx()
    engine = SignalEngine(ctx)
    session = engine._get_current_session()
    assert session in ("asia", "london", "overlap", "newyork")


def test_features_to_ppo_state():
    ctx = _make_ctx()
    engine = SignalEngine(ctx)
    import numpy as np

    # 1D input
    x1d = np.ones(49)
    result = engine._features_to_ppo_state(x1d)
    assert result.shape == (49,)

    # 2D input
    x2d = np.ones((10, 49))
    result = engine._features_to_ppo_state(x2d)
    assert result.shape == (49,)

    # 3D input
    x3d = np.ones((1, 10, 49))
    result = engine._features_to_ppo_state(x3d)
    assert result.shape == (49,)

    # Short input should pad
    x_short = np.ones(20)
    result = engine._features_to_ppo_state(x_short)
    assert result.shape == (49,)


def test_alpha_regime_for():
    ctx = _make_ctx()
    engine = SignalEngine(ctx)
    assert engine._alpha_regime_for("stat_arb") == "ranging"
    assert engine._alpha_regime_for("carry_trade") == "trending"
    assert engine._alpha_regime_for("unknown") == "ranging"


@pytest.mark.asyncio
async def test_on_position_closed_no_info():
    """Should not raise when no position info cached."""
    ctx = _make_ctx()
    engine = SignalEngine(ctx)
    await engine._on_position_closed(pnl=10.0, symbol="EURUSD", position_id=999)


@pytest.mark.asyncio
async def test_on_execution_result():
    """Should not raise."""
    ctx = _make_ctx()
    engine = SignalEngine(ctx)
    await engine._on_execution_result(
        symbol="EURUSD",
        position_id=1,
        filled_price=1.1205,
        signal_price=1.1200,
        direction="BUY",
    )


@pytest.mark.asyncio
async def test_rule_breakout_prediction_no_data():
    ctx = _make_ctx()
    engine = SignalEngine(ctx)
    result = engine._rule_breakout_prediction(None)
    assert result == 0.0


@pytest.mark.asyncio
async def test_rule_mean_rev_prediction_no_data():
    ctx = _make_ctx()
    engine = SignalEngine(ctx)
    result = engine._rule_mean_rev_prediction(None)
    assert result == 0.0
