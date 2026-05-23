"""
Run RTS: Agentic FX System Elite with real-live streaming data.
Boots all 15 agents, connects to cTrader for live ticks, processes data through full pipeline.  # noqa: E501
"""

import sys
import os
import asyncio
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from infrastructure.config_v2 import AppConfig  # noqa: E402
from infrastructure.secrets import Secrets  # noqa: E402
from agentic.main_agentic import AgenticOrchestrator  # noqa: E402


async def main():
    config = AppConfig.from_yaml("config.yaml")
    secrets = Secrets()
    orch = AgenticOrchestrator(config, secrets, mode="live", initial_balance=100000.0)
    orch.build_agents()
    orch._print_agent_table()

    await orch.bus.start()
    orch._init_world_state()

    start_tasks = [agent.start() for agent in orch.agents]
    await asyncio.gather(*start_tasks)

    duration = 180  # 3 minutes
    interval = 15
    steps = duration // interval

    print()
    print("=" * 70)
    print(f"  REAL-LIVE SIMULATION — {duration}s with cTrader streaming")
    print(f"  {len(orch.agents)} agents, processing live data pipeline")
    print("=" * 70)

    for i in range(steps):
        await asyncio.sleep(interval)

        _ = time.time()
        health = orch.registry.health_report()
        bus = orch.bus.get_stats()
        ws = orch.world

        # Pipeline status
        regime = ws.get("regime.current", "?")
        positions = ws.get("account.open_positions", 0)
        tick_rate = ws.get("data.tick_rate", 0)
        fresh_data = ws.get("data.freshness", {})
        perf = ws.get("performance.stats", {})
        sharpe = perf.get("sharpe", 0)
        pnl = perf.get("total_pnl", 0)
        trades = perf.get("total_trades", 0)
        symbols_with_data = (
            fresh_data.get("fresh", 0) if isinstance(fresh_data, dict) else 0
        )

        # Agent cycle counts
        cycles = {a.name: a.consciousness.total_cycles for a in orch.agents}
        top5 = sorted(cycles.items(), key=lambda x: x[1], reverse=True)[:5]
        top_str = " ".join(f"{n}:{c}c" for n, c in top5)

        # Emotional states
        emotions = {a.name: a.consciousness.emotion.dominant for a in orch.agents}

        # Connection status
        exec_connected = ws.get("execution.connected", "?")
        telegram_ok = ws.get("monitoring.telegram_health", "?")

        print(f"  [{i+1}/{steps}] +{i*interval}s")
        print(
            f"    agents: {health['alive']}/{health['total']} alive  |  "
            f"connected: exec={exec_connected} telegram={telegram_ok}"
        )
        print(
            f"    bus: {bus['total_messages']} msgs ({bus['avg_latency_ms']:.1f}ms avg)  |  "  # noqa: E501
            f"dropped={bus['dropped']}  acks={bus['ack_count']}  routes={bus['routing_count']}"  # noqa: E501
        )
        print(
            f"    data: {tick_rate} ticks/s  |  {symbols_with_data}/24 symbols fresh  |  "  # noqa: E501
            f"regime={regime}"
        )
        print(
            f"    pipeline: {trades} trades  |  "
            f"{positions} pos  |  sharpe={sharpe:.2f}  |  pnl=${pnl:+.2f}"
        )
        print(f"    top cycles: {top_str}")
        print(
            f"    emotions: \
data={emotions.get('data_agent','?')} \
signal={emotions.get('signal_agent','?')} \
risk={emotions.get('risk_agent','?')} \
exec={emotions.get('execution_agent','?')} \
master={emotions.get('master_agent','?')}"
        )

        # Check for signals
        signal_count = ws.get("signal.last.EURUSD", None)
        if signal_count:
            print(f"    last signal: {ws.get('signal.last.EURUSD', {})}")

        live_prices = []
        for sym in ["EURUSD", "GBPUSD", "XAUUSD", "BTCUSD"]:
            price = ws.get(f"data.price.{sym}", 0)
            if price > 0:
                live_prices.append(f"{sym}={price:.5f}")
        if live_prices:
            print(f"    prices: {' '.join(live_prices)}")

    print()
    print("=" * 70)
    print("  FINAL STATE REPORT")
    print("=" * 70)

    health = orch.registry.health_report()
    bus = orch.bus.get_stats()

    print("\n  AGENT SUMMARY:")
    for name, reg in sorted(health.get("agents", {}).items(), key=lambda x: x[0]):
        c = next((a.consciousness for a in orch.agents if a.name == name), None)
        if c:
            e = c.emotion.summary()
            print(
                f"  [{reg['domain']:>14}] {name:<22} "
                f"cycles={c.total_cycles:<4} health={c.health_score:.2f} "
                f"emotion={e['dominant']} ({e['overall']:.2f}) "
                f"uptime={c.uptime_seconds:.0f}s"
            )

    print(
        f"\n  BUS: {bus['total_messages']} total msgs, {bus['dropped']} dropped, "
        f"{bus['ack_count']} acks, {bus['routing_count']} routes, "
        f"{bus['validation_errors']} val_errs, {bus['avg_latency_ms']:.2f}ms avg latency"  # noqa: E501
    )
    print(f"  QUEUES: {bus['queue_sizes']}")
    print(f"  DEAD LETTERS: {bus['dead_letter_count']}")

    perf = ws.get("performance.stats", {})
    if perf:
        print(f"\n  PERFORMANCE: {perf}")

    for agent in reversed(orch.agents):
        await agent.stop()
    await orch.bus.stop()
    print("\n  System shutdown complete.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
