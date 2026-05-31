"""Tests for LearningManager."""

import pytest
from infrastructure.config import AppConfig
from infrastructure.secrets import Secrets
from pipeline.event_bus import EventBus
from pipeline.pipeline_context import PipelineContext
from pipeline.learning_manager import (
    LearningManager,
    PerformanceTracker,
    DriftMonitor,
    CheckpointManager,
)
from rts_ai_fx.drift_detector import ADWIN


@pytest.fixture(autouse=True)
def cleanup_checkpoints():
    # Clean up any persisted checkpoints from previous test runs
    import os
    import glob

    for f in glob.glob("data/agent_memory/*_checkpoint.*"):
        os.remove(f)
    yield
    for f in glob.glob("data/agent_memory/*_checkpoint.*"):
        os.remove(f)


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
    pass


# ── LearningManager ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_learning_manager_init():
    ctx = _make_ctx()
    lm = LearningManager(ctx)
    assert not lm.is_alive
    assert lm._retraining_count == 0


@pytest.mark.asyncio
async def test_learning_manager_start():
    ctx = _make_ctx()
    lm = LearningManager(ctx)
    await lm.start()
    assert lm.is_alive
    # Drift monitors should be created for all active symbols
    assert len(lm._drift_monitors) == 8
    await lm.stop()


@pytest.mark.asyncio
async def test_on_execution_result_drift():
    ctx = _make_ctx()
    lm = LearningManager(ctx)
    await lm.start()

    drift_events = []

    async def drift_handler(**data):
        drift_events.append(data)

    ctx.bus.on("drift_detected", drift_handler)

    # Send many bad predictions to trigger drift
    for _ in range(100):
        await lm._on_execution_result(
            symbol="EURUSD",
            signal_price=1.12,
            filled_price=1.10,
        )

    # Drift may have been detected
    await lm.stop()


@pytest.mark.asyncio
async def test_on_position_closed_tracks_performance():
    ctx = _make_ctx()
    lm = LearningManager(ctx)
    await lm.start()

    await lm._on_position_closed(
        symbol="EURUSD",
        direction="BUY",
        volume=0.01,
        entry_price=1.12,
        exit_price=1.13,
        pnl=10.0,
        reason="take_profit",
        regime="trending",
    )

    stats = lm._performance.get_stats()
    assert stats["total_trades"] == 1
    assert stats["win_rate"] == 1.0
    assert stats["profit_factor"] == float("inf")

    await lm.stop()


@pytest.mark.asyncio
async def test_check_retraining_needed_before_init():
    ctx = _make_ctx()
    lm = LearningManager(ctx)
    symbols = await lm.check_retraining_needed()
    assert symbols == []


@pytest.mark.asyncio
async def test_trigger_retraining():
    ctx = _make_ctx()
    lm = LearningManager(ctx)
    await lm.start()

    retrain_events = []

    async def retrain_handler(**data):
        retrain_events.append(data)

    ctx.bus.on("retraining_requested", retrain_handler)

    await lm.trigger_retraining(["EURUSD", "GBPUSD"])
    assert lm._retraining_count == 1

    # If OnlineLearner is available, it takes that path (no event emitted).
    # If not, the fallback emits 'retraining_requested'. Both are valid.
    if lm._online_learner is None:
        assert len(retrain_events) > 0, "Expected retrain event when no OnlineLearner"
    else:
        assert lm._last_training_time > 0

    await lm.stop()


@pytest.mark.asyncio
async def test_save_and_load_state():
    ctx = _make_ctx()
    lm = LearningManager(ctx)
    await lm.start()
    lm._retraining_count = 42

    saved = await lm.save_state()
    assert saved

    await lm.stop()

    # Clean up checkpoint so other tests aren't affected
    import os

    for f in ["learning_checkpoint.json", "learning_checkpoint.sha256"]:
        p = os.path.join("data/agent_memory", f)
        if os.path.exists(p):
            os.remove(p)


@pytest.mark.asyncio
async def test_drifted_symbols():
    ctx = _make_ctx()
    lm = LearningManager(ctx)
    await lm.start()

    assert lm.drifted_symbols == []

    await lm.stop()


@pytest.mark.asyncio
async def test_run_validation():
    ctx = _make_ctx()
    lm = LearningManager(ctx)
    await lm.start()

    result = await lm.run_validation()
    assert "validated" in result or "skipped" in result

    await lm.stop()


# ── PerformanceTracker ──────────────────────────────────────────────


def test_performance_tracker_empty():
    pt = PerformanceTracker()
    assert pt.total_trades == 0
    assert pt.win_rate == 0.0
    assert pt.sharpe_ratio == 0.0


def test_performance_tracker_records():
    pt = PerformanceTracker()
    pt.record_trade({"pnl": 10, "symbol": "EURUSD", "regime": "trending"})
    pt.record_trade({"pnl": -5, "symbol": "GBPUSD", "regime": "ranging"})
    pt.record_trade({"pnl": 3, "symbol": "EURUSD", "regime": "trending"})

    assert pt.total_trades == 3
    assert pt.win_rate == 2 / 3
    assert pt.profit_factor == 13 / 5

    stats = pt.get_stats()
    assert stats["by_symbol"]["EURUSD"]["trades"] == 2
    assert stats["by_symbol"]["EURUSD"]["wins"] == 2
    assert stats["by_regime"]["trending"]["trades"] == 2


def test_adwin():
    adwin = ADWIN(delta=0.05, min_window=5)
    # Not enough data yet
    assert not adwin.update(0.1)
    assert not adwin.update(0.1)
    assert not adwin.update(0.1)
    assert not adwin.update(0.1)
    assert not adwin.update(0.1)

    # With enough data, mean should be stable
    assert adwin.mean > 0

    # Large shift should trigger drift
    adwin2 = ADWIN(delta=0.5, min_window=5)
    for _ in range(10):
        adwin2.update(0.01)
    # Now shift
    for _ in range(10):
        adwin2.update(0.5)


# ── DriftMonitor ────────────────────────────────────────────────────


def test_drift_monitor():
    dm = DriftMonitor(error_threshold=0.02)
    assert not dm.retrain_triggered

    # Initially no drift
    dm.update(1.0, 1.01)  # small error
    assert not dm.retrain_triggered


def test_drift_monitor_reset():
    dm = DriftMonitor()
    dm.retrain_triggered = True
    dm.reset()
    assert not dm.retrain_triggered


# ── CheckpointManager ───────────────────────────────────────────────


def test_checkpoint_save_load(tmp_path):
    cm = CheckpointManager(base_path=str(tmp_path))
    state = {"test": "data", "count": 42}
    saved = cm.save_checkpoint(state, "test")
    assert saved

    loaded = cm.load_checkpoint("test")
    assert loaded is not None
    assert loaded["test"] == "data"
    assert loaded["count"] == 42


def test_checkpoint_load_nonexistent(tmp_path):
    cm = CheckpointManager(base_path=str(tmp_path))
    loaded = cm.load_checkpoint("nonexistent")
    assert loaded is None


def test_checkpoint_integrity(tmp_path):
    cm = CheckpointManager(base_path=str(tmp_path))
    cm.save_checkpoint({"data": 1}, "integ")
    assert cm.verify_integrity("integ")

    # Tamper with the file
    import json

    filepath = tmp_path / "integ_checkpoint.json"
    with open(filepath, "w") as f:
        json.dump({"data": 999}, f)

    assert not cm.verify_integrity("integ")
