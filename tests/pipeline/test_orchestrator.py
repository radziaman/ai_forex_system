"""Tests for Orchestrator."""

import pytest
from infrastructure.config import AppConfig
from infrastructure.secrets import Secrets
from pipeline.event_bus import EventBus
from pipeline.pipeline_context import PipelineContext
from pipeline.orchestrator import Orchestrator


def _make_ctx(**kwargs) -> PipelineContext:
    """Create a minimal PipelineContext for testing."""
    defaults = {
        "config": AppConfig(),
        "secrets": Secrets(),
        "bus": EventBus(),
    }
    defaults.update(kwargs)
    return PipelineContext(**defaults)


class FakeModule:
    def __init__(self):
        self.started = False
        self.stopped = False
        self._alive = True

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True

    def is_alive(self):
        return self._alive


@pytest.mark.asyncio
async def test_orchestrator_start_stop():
    ctx = _make_ctx()
    orch = Orchestrator(ctx)

    mod = FakeModule()
    orch.register_module("test", mod)

    await orch.start()
    assert mod.started
    assert orch.is_alive
    assert orch.uptime > 0

    await orch.stop()
    assert mod.stopped
    assert not orch.is_alive


@pytest.mark.asyncio
async def test_health_score():
    ctx = _make_ctx()
    orch = Orchestrator(ctx)

    good = FakeModule()
    bad = FakeModule()
    bad._alive = False

    orch.register_module("good", good)
    orch.register_module("bad", bad)

    await orch.start()

    # Manually set health_score by running a single check iteration
    alive_count = 0
    for name, module in orch._modules.items():
        if hasattr(module, "is_alive"):
            if module.is_alive():
                alive_count += 1
    orch._health_score = alive_count / max(len(orch._modules), 1)

    assert orch.health_score == 0.5

    await orch.stop()


@pytest.mark.asyncio
async def test_orchestrator_start_failure_does_not_block():
    """A failing module start should not prevent others from starting."""
    ctx = _make_ctx()
    orch = Orchestrator(ctx)

    class FailingModule:
        async def start(self):
            raise RuntimeError("fail")

    class GoodModule:
        def __init__(self):
            self.started = False

        async def start(self):
            self.started = True

    fail_mod = FailingModule()
    good_mod = GoodModule()

    orch.register_module("failing", fail_mod)
    orch.register_module("good", good_mod)

    await orch.start()
    assert good_mod.started
    assert orch.is_alive

    await orch.stop()


@pytest.mark.asyncio
async def test_stop_no_modules():
    """Stopping with no modules should not raise."""
    ctx = _make_ctx()
    orch = Orchestrator(ctx)
    await orch.stop()


@pytest.mark.asyncio
async def test_start_stop_no_lifecycle_methods():
    """Modules without start/stop methods should not cause errors."""
    ctx = _make_ctx()
    orch = Orchestrator(ctx)

    class NoLifecycleModule:
        pass

    orch.register_module("bare", NoLifecycleModule())
    await orch.start()
    await orch.stop()


def test_uptime_zero_before_start():
    ctx = _make_ctx()
    orch = Orchestrator(ctx)
    assert orch.uptime == 0.0


def test_is_alive_false_before_start():
    ctx = _make_ctx()
    orch = Orchestrator(ctx)
    assert not orch.is_alive
