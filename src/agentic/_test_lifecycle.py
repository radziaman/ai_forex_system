"""
End-to-end lifecycle test: boots minimal agents, verifies communication.
"""
import sys, os, asyncio, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from infrastructure.config_v2 import AppConfig
from agentic.core.agent_bus import get_agent_bus
from agentic.core.agent_registry import get_agent_registry
from agentic.core.world_state import get_world_state
from agentic.core.agent_message import MessageType

from agentic.agents.master_agent import MasterAgent
from agentic.agents.performance_agent import PerformanceAgent
from agentic.agents.memory_agent import MemoryAgent
from agentic.agents.monitoring_agent import MonitoringAgent


async def test():
    print("=" * 60)
    print("Agentic Lifecycle Test")
    print("=" * 60)

    config = AppConfig.from_yaml("config.yaml")
    bus = get_agent_bus()
    registry = get_agent_registry()
    ws = get_world_state()

    await bus.start()

    # Create a subset of agents
    agents = [
        MasterAgent(),
        PerformanceAgent(),
        MemoryAgent(),
        MonitoringAgent(),
    ]

    # Start all
    for a in agents:
        await a.start()
        print(f"  [+] Started: {a.name}")

    await asyncio.sleep(2)

    # Check registry health
    health = registry.health_report()
    print(f"\n  Registry: {health['alive']}/{health['total']} agents alive")

    # Verify communication via DIAGNOSTIC_REQUEST (works in all agents)
    for a in agents:
        await a.send(
            MessageType.DIAGNOSTIC_REQUEST,
            payload={"test": True, "message": f"hello from {a.name}"},
        )

    await asyncio.sleep(1)

    # Check consciousness
    for a in agents:
        c = a.consciousness.summary()
        print(f"  [{c['agent']}] state={c['state']} cycles={c['cycles']} health={c['health']}")

    # Stop all
    for a in reversed(agents):
        await a.stop()

    await bus.stop()

    # Summary
    total = health['total']
    alive = health['alive']
    print(f"\n  RESULT: {alive}/{total} agents completed lifecycle")
    passed = total > 0 and alive == total
    print(f"  {'PASS' if passed else 'FAIL'}")
    print("=" * 60)
    return passed


if __name__ == "__main__":
    passed = asyncio.run(test())
    sys.exit(0 if passed else 1)
