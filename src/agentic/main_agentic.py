"""
RTS: Agentic Moneybot System Elite — Agentic Architecture Entry Point.

Boots all autonomous agents, establishes communication,
and lets them work together as a distributed intelligence.
"""

from __future__ import annotations
import asyncio
import signal
import sys
import os
import time
from typing import List
from loguru import logger

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # Required for loading Keras 2 models
os.environ["TF_USE_LEGACY_ADAM"] = "1"  # 10x faster Adam on Apple Silicon M1/M2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infrastructure.config_v2 import AppConfig  # noqa: E402
from infrastructure.secrets import Secrets  # noqa: E402

from agentic.core.agent_bus import get_agent_bus  # noqa: E402
from agentic.core.agent_registry import get_agent_registry  # noqa: E402
from agentic.core.world_state import get_world_state  # noqa: E402
from agentic.core.agent_consciousness import ConsciousnessLevel  # noqa: E402

from agentic.agents.data_agent import DataAgent  # noqa: E402
from agentic.agents.feature_agent import FeatureAgent  # noqa: E402
from agentic.agents.regime_agent import RegimeAgent  # noqa: E402
from agentic.agents.signal_agent import SignalAgent  # noqa: E402
from agentic.agents.risk_agent import RiskAgent  # noqa: E402
from agentic.agents.execution_agent import ExecutionAgent  # noqa: E402
from agentic.agents.position_agent import PositionAgent  # noqa: E402
from agentic.agents.performance_agent import PerformanceAgent  # noqa: E402
from agentic.agents.adaptive_risk_agent import AdaptiveRiskAgent  # noqa: E402
from agentic.agents.master_agent import MasterAgent  # noqa: E402
from agentic.agents.validation_agent import ValidationAgent  # noqa: E402
from agentic.agents.monitoring_agent import MonitoringAgent  # noqa: E402
from agentic.agents.connection_agent import ConnectionAgent  # noqa: E402
from agentic.agents.learning_agent import LearningAgent  # noqa: E402
from agentic.agents.memory_agent import MemoryAgent  # noqa: E402
from agentic.agents.model_registry_agent import ModelRegistryAgent  # noqa: E402
from agentic.agents.drift_agent import DriftAgent  # noqa: E402
from agentic.agents.circuit_breaker_agent import CircuitBreakerAgent  # noqa: E402
from agentic.agents.cost_agent import CostAgent  # noqa: E402
from agentic.agents.screener_agent import InstrumentScreenerAgent  # noqa: E402

from data.data_manager import SYMBOLS  # noqa: E402

HELP = """
Agentic Trading System — Autonomous Mode

  python -m agentic.main_agentic --mode paper
    Run all 20 agents autonomously in paper trading mode

  python -m agentic.main_agentic --mode live
    Run in live mode with cTrader broker connection

  python -m agentic.main_agentic --status
    Print system status summary and exit
"""


class AgenticOrchestrator:
    """
    Boots and manages all autonomous agents.

    Each agent runs independently on its own interval.
    They communicate through the AgentBus.
    They share state through the WorldState.
    They discover each other through the AgentRegistry.
    """

    def __init__(
        self,
        config: AppConfig,
        secrets: Secrets,
        mode: str = "paper",
        initial_balance: float = 100_000.0,
    ):
        self.config = config
        self.secrets = secrets
        self.mode = mode
        self.initial_balance = initial_balance

        # Core infrastructure
        self.bus = get_agent_bus(n_workers=2)
        self.registry = get_agent_registry()
        self.world = get_world_state()
        self.bus.set_registry(self.registry)  # G6: wire capability routing

        # All agents
        self.agents: List = []
        self.running = False
        self.simulation_mode = False  # G18
        self.timeout = 0  # Auto-shutdown timeout (seconds, 0 = no limit)

        logger.info("=" * 60)
        logger.info("  Agentic System Elite — Booting Autonomous Agents")
        logger.info("=" * 60)

    def build_agents(self):
        """Create all 15 autonomous agents with G17 supervisor hierarchy."""

        # Shared config knowledge
        for k, v in [
            ("config.max_positions", self.config.trading.max_positions),
            ("config.max_drawdown", self.config.trading.max_drawdown),
            ("config.kelly_fraction", self.config.trading.kelly_fraction),
            ("config.lookback", self.config.features.lookback),
            ("config.timeframes", self.config.features.timeframes),
            ("config.mode", self.mode),
            ("config.initial_balance", self.initial_balance),
        ]:
            self.world.set(k, v, ttl=86400)

        # G17: Supervisor hierarchy — master_agent supervises all domain supervisors
        # Domain supervisors report to master; individual agents report to domain supervisors  # noqa: E501

        # ── Tier 1: Data & Intelligence (supervisor: master_agent) ──

        self.data_agent = DataAgent(self.config)
        self.data_agent.consciousness.identity.purpose = (
            "I ingest market data from all sources, maintain OHLCV across 5 timeframes, "  # noqa: E501
            "detect and heal data gaps, and publish fresh features when bars close."
        )
        self.data_agent.consciousness.identity.capabilities.add("tick_ingestion")
        self.agents.append(self.data_agent)

        self.feature_agent = FeatureAgent(self.config)
        self.feature_agent.consciousness.identity.purpose = "I transform raw OHLCV into normalized features. I bridge data and intelligence."  # noqa: E501
        self.agents.append(self.feature_agent)

        self.regime_agent = RegimeAgent(self.config)
        self.regime_agent.consciousness.identity.purpose = "I detect market hidden state via HMM: trending, ranging, volatile, or crisis."  # noqa: E501
        self.agents.append(self.regime_agent)

        self.signal_agent = SignalAgent(self.config)
        self.signal_agent.consciousness.identity.purpose = "I fuse PPO, LSTM-CNN, rule-based experts into high-conviction signals via MoE ensemble."  # noqa: E501
        self.agents.append(self.signal_agent)

        # ── Tier 2: Risk (supervisor: master_agent) ──

        self.risk_agent = RiskAgent(self.config, self.initial_balance)
        self.risk_agent.consciousness.identity.purpose = (
            "I am the gatekeeper. Kelly sizing, VaR, drawdown checks, circuit breakers. "  # noqa: E501
            "No trade passes without my approval."
        )
        self.agents.append(self.risk_agent)

        self.adaptive_risk_agent = AdaptiveRiskAgent(
            base_kelly=self.config.trading.kelly_fraction,
            base_risk=self.config.trading.max_risk_per_trade,
        )
        self.adaptive_risk_agent.consciousness.identity.purpose = (
            "I am the risk thermostat. I dynamically adjust sizing based on "
            "volatility, drawdown, and win rate trends."
        )
        self.agents.append(self.adaptive_risk_agent)

        # ── Tier 3: Execution (supervisor: master_agent) ──

        self.execution_agent = ExecutionAgent(
            self.config, self.secrets, self.initial_balance
        )
        self.execution_agent.consciousness.identity.purpose = (
            "I send approved orders to the broker and stream live ticks back to data_agent. "  # noqa: E501
            "I track every position through its lifecycle."
        )
        self.agents.append(self.execution_agent)

        self.position_agent = PositionAgent()
        self.position_agent.consciousness.identity.purpose = "I manage trailing stops, partial closes, and correlation risk for all open positions."  # noqa: E501
        self.agents.append(self.position_agent)

        # ── Tier 4: Analytics (supervisor: master_agent) ──

        self.performance_agent = PerformanceAgent()
        self.performance_agent.consciousness.identity.purpose = "I track every trade. Sharpe, profit factor, win rate, per-symbol analytics."  # noqa: E501
        self.agents.append(self.performance_agent)

        self.validation_agent = ValidationAgent()
        self.validation_agent.consciousness.identity.purpose = "I prove edge via walk-forward, Monte Carlo, stress tests, and A/B comparisons."  # noqa: E501
        self.agents.append(self.validation_agent)

        # ── Tier 5: Infrastructure (supervisor: master_agent) ──

        self.connection_agent = ConnectionAgent(SYMBOLS)
        self.connection_agent.consciousness.identity.purpose = (
            "I keep broker connection alive. Auto-reconnect with exponential backoff."
        )
        self.agents.append(self.connection_agent)

        self.monitoring_agent = MonitoringAgent(self.secrets)
        self.monitoring_agent.consciousness.identity.purpose = (
            "I send Telegram alerts, update dashboard, maintain audit log."
        )
        self.agents.append(self.monitoring_agent)

        self.learning_agent = LearningAgent()
        self.learning_agent.consciousness.identity.purpose = (
            "I monitor for drift and trigger retraining when performance degrades."
        )
        self.agents.append(self.learning_agent)

        self.memory_agent = MemoryAgent()
        self.memory_agent.consciousness.identity.purpose = (
            "I persist system state with integrity checks for crash recovery."
        )
        self.agents.append(self.memory_agent)

        # ── Tier 6: Specialized Services (supervisor: master_agent) ──

        self.model_registry_agent = ModelRegistryAgent()
        self.model_registry_agent.consciousness.identity.purpose = (
            "I manage model versions, run A/B tests, and auto-promote champions."
        )
        self.agents.append(self.model_registry_agent)

        self.drift_agent = DriftAgent()
        self.drift_agent.consciousness.identity.purpose = (
            "I detect concept drift across all symbols and broadcast alerts."
        )
        self.agents.append(self.drift_agent)

        self.circuit_breaker_agent = CircuitBreakerAgent()
        self.circuit_breaker_agent.consciousness.identity.purpose = "I monitor market stress independently and halt trading during disorderly conditions."  # noqa: E501
        self.agents.append(self.circuit_breaker_agent)

        self.cost_agent = CostAgent()
        self.cost_agent.consciousness.identity.purpose = (
            "I track live spreads, estimate costs, and warn on unfavorable conditions."
        )
        self.agents.append(self.cost_agent)

        # ── Screener Agent (autonomous instrument discovery) ──

        self.screener_agent = InstrumentScreenerAgent(scan_interval_hours=24)
        self.screener_agent.consciousness.identity.purpose = (
            "I autonomously scan all instruments for tradeable edges. "
            "I publish only those that pass statistical thresholds."
        )
        self.agents.append(self.screener_agent)

        # ── Tier 0: Orchestration (conducts all agents) ──

        self.master_agent = MasterAgent()
        self.master_agent.consciousness.identity.purpose = "I conduct the orchestra. Monitor health, escalate errors, coordinate healing."  # noqa: E501
        self.master_agent.consciousness.level = ConsciousnessLevel.META
        self.agents.append(self.master_agent)

        # World state
        self.world.set("agentic.total_agents", len(self.agents))
        self.world.set("agentic.agent_names", [a.name for a in self.agents])
        self.world.set("agentic.simulation_mode", self.simulation_mode)

        logger.info(f"[orchestrator] Built {len(self.agents)} autonomous agents")

    async def start(self):
        logger.info("")
        logger.info(f"  {'=' * 56}")
        logger.info(f"  Agentic System Elite — Starting {len(self.agents)} Agents")
        logger.info(f"  Mode: {self.mode.upper()}")
        logger.info(f"  {'=' * 56}")
        logger.info("")

        # 1. Start infrastructure
        await self.bus.start()

        # 2. Set initial world state
        self._init_world_state()

        # 3. Build and start all agents
        self.build_agents()

        # G18: Enable simulation mode for all agents if flag is set
        if self.simulation_mode:
            logger.info("[orchestrator] SIMULATION MODE — agents will skip real I/O")
            for agent in self.agents:
                agent.enable_simulation()
            self.world.set("agentic.simulation_mode", True)

        start_tasks = [agent.start() for agent in self.agents]
        await asyncio.gather(*start_tasks)

        self.running = True
        logger.info(f"[orchestrator] All {len(self.agents)} agents running")
        logger.info(f"[orchestrator] Agent domains: {self._list_domains()}")

        # 4. Main loop — let agents run, log status periodically
        self._print_agent_table()
        last_log = 0.0
        boot_time = time.time()

        try:
            while self.running:
                await asyncio.sleep(10)

                # --timeout: auto-shutdown after N seconds
                if self.timeout > 0 and time.time() - boot_time > self.timeout:
                    logger.info(
                        f"[orchestrator] Timeout ({self.timeout}s) reached — shutting down"  # noqa: E501
                    )
                    break

                # Log status every 60s
                if time.time() - last_log > 60:
                    last_log = time.time()
                    health = self.registry.health_report()
                    perf = self.world.get("performance.stats", {})

                    regime = self.world.get("regime.current", "?")
                    positions = self.world.get("account.open_positions", 0)
                    balance = self.world.get("account.balance", 0)
                    equity = self.world.get("account.equity", 0)
                    sharpe = perf.get("sharpe", 0)
                    pnl = perf.get("total_pnl", 0)

                    logger.info(
                        f"[status] agents={health['alive']}/{health['total']} "
                        f"health={health['health_pct']:.0f}% "
                        f"bal=${balance:,.0f} "
                        f"eq=${equity:,.0f} "
                        f"regime={regime} "
                        f"pos={positions} "
                        f"sharpe={sharpe:.2f} "
                        f"pnl=${pnl:+.2f}"
                    )

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[orchestrator] Fatal error: {e}")
        finally:
            await self.stop()

    async def stop(self):
        logger.info("[orchestrator] Shutting down all agents...")
        self.running = False

        # Stop agents in reverse order (orchestrator last)
        for agent in reversed(self.agents):
            try:
                await agent.stop()
                logger.info(f"  [x] {agent.name} stopped")
            except Exception as e:
                logger.warning(f"  [!] {agent.name} stop error: {e}")

        await self.bus.stop()
        logger.info("[orchestrator] All agents stopped. System offline.")

    def _init_world_state(self):
        self.world.set("account.balance", self.initial_balance)
        self.world.set("account.equity", self.initial_balance)
        self.world.set("account.margin", 0)
        self.world.set("account.open_positions", 0)
        self.world.set("data.primary_symbol", SYMBOLS[0] if SYMBOLS else "EURUSD")
        self.world.set("system.mode", self.mode)
        self.world.set("system.boot_time", time.time())
        self.world.set("system.timeout", self.timeout)

    def _list_domains(self) -> str:
        domains = set(a.identity.domain for a in self.agents)
        return ", ".join(sorted(domains))

    def _print_agent_table(self):
        logger.info("")
        logger.info(f"  {'Agent':<22} {'Role':<28} {'Interval':<10}")
        logger.info(f"  {'-'*22} {'-'*28} {'-'*10}")
        for agent in sorted(self.agents, key=lambda a: a.identity.domain):
            logger.info(
                f"  {agent.name:<22} {agent.identity.role:<28} "
                f"{agent.consciousness.tick_interval:<10.1f}s"
            )
        logger.info("")


def cmd_status():
    """Print agentic system status."""
    config = AppConfig.from_yaml("config.yaml")
    secrets = Secrets()
    orch = AgenticOrchestrator(config, secrets)
    orch.build_agents()
    orch._print_agent_table()
    logger.info(f"\nTotal agents: {len(orch.agents)}")
    logger.info("System version: 4.0.0-agentic")
    logger.info(f"World state variables: {len(orch.world.snapshot())}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m agentic.main_agentic",
        description="RTS: Agentic Moneybot System Elite — Agentic Architecture",
        epilog=HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--mode", choices=["paper", "live", "dry-run"], default="paper")
    parser.add_argument("--status", action="store_true", help="Print system status")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without executing",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",  # G18
        help="Run agents in simulation mode (no real I/O)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Auto-shutdown after N seconds (0 = no limit)",
    )

    args = parser.parse_args()

    if args.status:
        cmd_status()
        return

    config = AppConfig.from_yaml(args.config)
    secrets = Secrets()
    mode = "dry-run" if args.dry_run else args.mode

    orch = AgenticOrchestrator(
        config,
        secrets,
        mode=mode,
        initial_balance=args.capital,
    )
    orch.simulation_mode = args.simulate
    orch.timeout = args.timeout

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig:
            try:
                loop.add_signal_handler(sig, lambda: asyncio.ensure_future(orch.stop()))
            except NotImplementedError:
                pass  # Windows

    try:
        loop.run_until_complete(orch.start())
    except (KeyboardInterrupt, asyncio.CancelledError):
        loop.run_until_complete(orch.stop())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
