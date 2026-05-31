"""
RTS: AI Forex Trading System — Pipeline Entry Point.
Replaces the 20-agent swarm with a clean 5-module pipeline.
"""

import asyncio
import signal
import os
from loguru import logger

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_USE_LEGACY_ADAM"] = "1"

# Note: Run with `python -m pipeline.main` after `pip install -e .`
# or set PYTHONPATH to include src/

from infrastructure.config import AppConfig  # noqa: E402
from infrastructure.config_watcher import ConfigWatcher  # noqa: E402
from infrastructure.secrets import Secrets  # noqa: E402
from data.data_manager import DataManager  # noqa: E402
from pipeline.event_bus import EventBus  # noqa: E402
from pipeline.pipeline_context import PipelineContext  # noqa: E402
from pipeline.orchestrator import Orchestrator  # noqa: E402
from pipeline.signal_engine import SignalEngine  # noqa: E402
from pipeline.risk_manager import RiskManager  # noqa: E402
from pipeline.execution_manager import ExecutionManager  # noqa: E402
from pipeline.learning_manager import LearningManager
from pipeline.attribution_manager import AttributionManager  # noqa: E402
from pipeline.health_monitor import HealthMonitor  # noqa: E402
from pipeline.symbol_discovery import SymbolDiscovery  # noqa: E402
from pipeline.strategy_discovery import StrategyDiscovery  # noqa: E402


HELP = """
RTS AI Forex Trading System — Pipeline Mode

  python -m pipeline.main --mode paper
    Run in paper trading mode (default)

  python -m pipeline.main --mode live
    Run with cTrader broker connection

  python -m pipeline.main --mode backtest
    Run in backtest-only mode
"""


async def main(config, secrets, mode="paper"):
    # Create shared services
    data_manager = DataManager(historical_path=config.data.historical_path)
    bus = EventBus()

    # Create pipeline context
    ctx = PipelineContext(
        config=config,
        secrets=secrets,
        bus=bus,
        data_manager=data_manager,
    )

    # Create pipeline modules
    orchestrator = Orchestrator(ctx)
    signal_engine = SignalEngine(ctx)
    risk_manager = RiskManager(ctx)
    execution_manager = ExecutionManager(ctx, mode=mode)
    learning_manager = LearningManager(ctx)

    # Performance attribution
    attribution_manager = AttributionManager(
        event_bus=bus,
        slippage_estimate=getattr(config, "slippage_estimate", 0.0001),
    )
    ctx.attribution_manager = attribution_manager

    # ================================================================
    # Autonomous capability modules
    # ================================================================

    # Self-healing: monitors module heartbeats, auto-recovers crashes
    health_monitor = HealthMonitor(
        event_bus=bus,
        check_interval=10.0,
        heartbeat_timeout=30.0,
        max_failures=3,
    )
    ctx.health_monitor = health_monitor

    # Auto-symbol discovery: scans candidates, picks best by liquidity
    symbol_discovery = SymbolDiscovery(
        event_bus=bus,
        data_manager=data_manager,
        max_symbols=11,
        scan_interval=3600.0,
        warmup_ticks=100,
    )
    ctx.symbol_discovery = symbol_discovery

    # Auto-strategy discovery: parameter sweeps, promotes winners
    strategy_discovery = StrategyDiscovery(
        event_bus=bus,
        expert_registry=getattr(signal_engine, "expert_registry", None),
        min_eval_trades=20,
        sharpe_threshold=1.0,
        win_rate_threshold=0.55,
    )
    ctx.strategy_discovery = strategy_discovery

    # State checkpointing: periodic save for crash recovery
    # Ensemble is lazily loaded by SignalEngine — CheckpointManager
    # will call ensemble.save_state() on each checkpoint cycle
    from pipeline.checkpoint_manager import CheckpointManager

    checkpoint_manager = CheckpointManager(
        event_bus=bus,
        pipeline_ctx=ctx,  # Lazily resolves ensemble via ctx
        checkpoint_dir="data/checkpoints",
        save_interval=60.0,
        auto_load=True,
    )
    ctx.checkpoint_manager = checkpoint_manager

    # Register modules with orchestrator
    orchestrator.register_module("signal_engine", signal_engine)
    orchestrator.register_module("risk_manager", risk_manager)
    orchestrator.register_module("execution_manager", execution_manager)
    orchestrator.register_module("learning_manager", learning_manager)
    orchestrator.register_module("attribution_manager", attribution_manager)
    orchestrator.register_module("health_monitor", health_monitor)
    orchestrator.register_module("symbol_discovery", symbol_discovery)
    orchestrator.register_module("strategy_discovery", strategy_discovery)
    orchestrator.register_module("checkpoint_manager", checkpoint_manager)

    # Config hot-reload watcher (opt-out via config)
    if getattr(config, "enable_config_watch", True):
        config_watcher = ConfigWatcher(
            event_bus=bus,
            config_path="config.yaml",
            poll_interval=10.0,
            config=config,
        )
        orchestrator.register_module("config_watcher", config_watcher)

    # Start pipeline
    await orchestrator.start()

    logger.info("=" * 60)
    logger.info("  RTS AI Forex Trading System — Pipeline Active")
    logger.info(f"  Mode: {mode.upper()}")
    logger.info("=" * 60)

    # Wait for shutdown signal
    stop_event = asyncio.Event()

    def _signal_handler():
        logger.info("Shutdown signal received...")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows compatibility
            pass

    await stop_event.wait()
    await orchestrator.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RTS AI Forex Trading System")
    parser.add_argument(
        "--mode", default="paper", choices=["paper", "live", "backtest"]
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = AppConfig.from_yaml(args.config)
    secrets = Secrets()

    try:
        asyncio.run(main(config, secrets, mode=args.mode))
    except KeyboardInterrupt:
        logger.info("Shutdown complete")
