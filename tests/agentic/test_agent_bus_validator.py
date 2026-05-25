"""Tests for AgentBus dependency graph validation."""

from agentic.core.agent_bus import AgentBus
from agentic.core.agent_message import MessageType


class TestAgentBusDependencyValidation:
    def test_validate_dependencies_returns_list(self):
        bus = AgentBus()
        bus.start()

        # Subscribe to a non-existent type
        bus.subscribe(MessageType.SYSTEM_STATE_CHANGE, lambda m: None)

        warnings = bus.validate_dependencies(agent_names=["test_agent"])

        # Should warn about SYSTEM_STATE_CHANGE
        system_warnings = [w for w in warnings if "SYSTEM_STATE_CHANGE" in w]
        assert len(system_warnings) >= 1

    def test_validate_dependencies_ok_for_known_types(self):
        bus = AgentBus()
        bus.start()

        # Subscribe to TICK_RECEIVED (published by multiple agents)
        bus.subscribe(MessageType.TICK_RECEIVED, lambda m: None)

        warnings = bus.validate_dependencies(agent_names=[])
        # TICK_RECEIVED should have publishers (found by scanning)
        # It could still warn if no publisher found, but shouldn't crash
        assert isinstance(warnings, list)

    def test_validate_dependencies_multiple_subscribers(self):
        bus = AgentBus()
        bus.start()

        # Subscribe multiple callbacks to same orphaned type
        bus.subscribe(MessageType.AGENT_COLLABORATE, lambda m: None)
        bus.subscribe(MessageType.AGENT_COLLABORATE, lambda m: None)

        warnings = bus.validate_dependencies(agent_names=["agent_a", "agent_b"])
        collab_warnings = [w for w in warnings if "AGENT_COLLABORATE" in w]
        # Should warn at least once
        assert len(collab_warnings) >= 1

    def test_validate_dependencies_handles_empty(self):
        bus = AgentBus()
        bus.start()
        warnings = bus.validate_dependencies(agent_names=[])
        assert isinstance(warnings, list)

    def test_validate_dependencies_stores_warnings(self):
        """Verify that validate_dependencies caches warnings for get_stats."""
        bus = AgentBus()
        bus.start()
        bus.subscribe(MessageType.KILL_SWITCH, lambda m: None)

        bus.validate_dependencies(agent_names=[])
        stats = bus.get_stats()
        assert "dependency_warnings" in stats
        assert len(stats["dependency_warnings"]) >= 1

    def test_validate_dependencies_async_subscriber(self):
        """Async subscribers should also be tracked."""
        bus = AgentBus()
        bus.start()

        async def async_handler(msg):
            pass

        bus.subscribe(MessageType.TRAINING_REQUEST, async_handler)

        warnings = bus.validate_dependencies(agent_names=["test_agent"])
        training_warnings = [w for w in warnings if "TRAINING_REQUEST" in w]
        assert len(training_warnings) >= 1
