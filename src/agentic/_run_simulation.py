"""
Run RTS: Agentic Moneybot System Elite in paper mode with real data.
Booting all 15 agents, processing market data for 60 seconds.
"""
import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from infrastructure.config_v2 import AppConfig
from infrastructure.secrets import Secrets
from agentic.main_agentic import AgenticOrchestrator


async def main():
    config = AppConfig.from_yaml('config.yaml')
    secrets = Secrets()
    orch = AgenticOrchestrator(config, secrets, mode='paper', initial_balance=100000.0)
    orch.build_agents()
    orch._print_agent_table()

    await orch.bus.start()
    orch._init_world_state()

    start_tasks = [agent.start() for agent in orch.agents]
    await asyncio.gather(*start_tasks)

    duration = 90
    interval = 15
    steps = duration // interval

    print()
    print("=" * 60)
    print(f"  SIMULATION RUNNING — {duration}s with real data")
    print(f"  Agents: {len(orch.agents)}")
    print("=" * 60)

    for i in range(steps):
        await asyncio.sleep(interval)
        health = orch.registry.health_report()
        bus = orch.bus.get_stats()
        regime = orch.world.get('regime.current', '?')
        positions = orch.world.get('account.open_positions', 0)
        perf = orch.world.get('performance.stats', {})
        sharpe = perf.get('sharpe', 0)
        pnl = perf.get('total_pnl', 0)
        trades = perf.get('total_trades', 0)

        # Check agent consciousness for interesting stats
        agent_states = {}
        for a in orch.agents:
            c = a.consciousness.summary()
            agent_states[a.name] = {
                'cycles': c['cycles'],
                'state': c['state'],
                'emotion': c.get('emotion', {}).get('dominant', '?'),
            }

        top_agents = sorted(agent_states.items(), key=lambda x: x[1]['cycles'], reverse=True)[:5]
        top_str = ' '.join(f"{n}:{s['cycles']}c" for n, s in top_agents)

        print(f"  [{i+1}/{steps}] "
              f"agents={health['alive']}/{health['total']} "
              f"msgs={bus['total_messages']} "
              f"dropped={bus['dropped']} "
              f"regime={regime} "
              f"pos={positions} "
              f"trades={trades} "
              f"sharpe={sharpe:.2f} "
              f"pnl=${pnl:+.2f}")
        print(f"           top: {top_str}")
        print(f"           bus: {bus['avg_latency_ms']:.1f}ms avg, "
              f"{bus['queue_sizes']} queue, "
              f"{bus['ack_count']} acks, "
              f"{bus['validation_errors']} val_errs")

        if regime != '?' and regime != 'unknown':
            print(f"           regime detected: {regime}")

    print()
    print("=" * 60)
    print("  FINAL STATE")
    print("=" * 60)
    health = orch.registry.health_report()
    bus = orch.bus.get_stats()
    perf = orch.world.get('performance.stats', {})

    for name, reg in health.get('agents', {}).items():
        c = next((a.consciousness for a in orch.agents if a.name == name), None)
        emotion = c.emotion.dominant if c else '?'
        cycles = c.total_cycles if c else 0
        print(f"  [{reg['domain']:>14}] {name:<22} cycles={cycles:<4} "
              f"health={reg['health']:.1f} emotion={emotion}")

    print()
    print(f"  Bus: {bus['total_messages']} msgs, {bus['dropped']} dropped, "
          f"{bus['avg_latency_ms']:.2f}ms avg")
    print(f"  Acks: {bus['ack_count']}, Routes: {bus['routing_count']}, "
          f"Val errors: {bus['validation_errors']}")
    print(f"  Performance: {perf}")

    for agent in reversed(orch.agents):
        await agent.stop()
    await orch.bus.stop()
    print("  System shutdown complete")
    print("=" * 60)


if __name__ == '__main__':
    asyncio.run(main())
