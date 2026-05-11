"""RTS: AI Moneybot System Elite — v2 Service Architecture Entry Point."""
import asyncio
import signal
import sys
import os
import time
from loguru import logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from infrastructure.config_v2 import AppConfig
from infrastructure.secrets import Secrets
from infrastructure.event_bus import get_event_bus, EventType
from infrastructure.service_registry import ServiceRegistry
from infrastructure.system_info import collect_report, VERSION
from services.data_pipeline import DataPipeline
from services.signal_engine import SignalEngine
from services.risk_gatekeeper import RiskGatekeeper
from services.execution_service import ExecutionService
from services.monitoring_service import MonitoringService
from data.data_manager import SYMBOLS, BASE_PRICES


HELP_EPILOG = """
Examples:

  python -m src.main_v2 --mode paper
    Run in paper trading mode with $100K simulated capital

  python -m src.main_v2 --mode live
    Run in live mode (connects to cTrader with configured account)

  python -m src.main_v2 --info
    Print system configuration summary and exit

  python -m src.main_v2 --check
    Run readiness check on all components and exit

  python -m src.main_v2 --mode paper --capital 50000
    Paper trade with $50K starting capital

Files:
  config.yaml              System configuration (trading, AI, risk, features)
  .env                     API credentials (never committed to git)
  data/logs/moneybot.log   Log output
  models/                  Trained model weights
  data/dukascopy_cache/    Historical tick data
"""


class TradingOrchestrator:
    """Thin loop: manages services, health checks, graceful shutdown."""

    def __init__(self, config: AppConfig, secrets: Secrets, mode: str = "paper",
                 initial_balance: float = 100_000.0):
        self.config = config
        self.secrets = secrets
        self.mode = mode
        self.initial_balance = initial_balance
        self.registry = ServiceRegistry()
        self.event_bus = get_event_bus()
        self.running = False
        self.cycle_counter = 0
        self._tick_count = 0
        self._bar_close_count = 0
        self._signal_count = 0
        self._trade_count = 0
        self._last_status_time = 0.0
        self._init_logging()
        self._init_services()

    def _init_logging(self):
        logger.remove()
        log_path = self.config.data.logs_path.rstrip("/") + "/moneybot_v2.log"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        fmt = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
        logger.add(log_path, rotation=self.config.logging.rotation,
                   retention=self.config.logging.retention, format=fmt, colorize=False)
        logger.add(sys.stdout, format=fmt, colorize=True, level="INFO")

    def _init_services(self):
        self.data_pipeline = DataPipeline(self.config)
        self.signal_engine = SignalEngine(self.config)
        self.risk_gatekeeper = RiskGatekeeper(self.config, initial_balance=self.initial_balance)
        self.execution_service = ExecutionService(self.config, self.secrets, self.data_pipeline,
                                                   initial_balance=self.initial_balance)
        self.monitoring = MonitoringService(self.config, self.secrets)

        for svc in [self.data_pipeline, self.signal_engine, self.risk_gatekeeper,
                    self.execution_service, self.monitoring]:
            self.registry.register(svc)

        self.event_bus.subscribe(EventType.FEATURES_READY, self._on_features_ready)

    async def _on_features_ready(self, event):
        """Features → Signal → Risk → Execution pipeline."""
        update = event.data
        if update is None:
            return
        symbol = update.symbol

        # STAGE 1: Signal generation
        logger.info(f"[signal] {symbol}: computing signal from features "
                    f"(price={update.price:.5f}, bars={len(update.ohlcv) if update.ohlcv is not None else 0})")

        signal = self.signal_engine.on_features(update)
        if signal is None:
            return
        self._signal_count += 1
        logger.info(f"[signal] {symbol}: {signal.direction.value} "
                    f"confidence={signal.confidence:.3f} regime={signal.regime.value}")
        self.monitoring.on_signal(signal)

        # STAGE 2: Risk gate
        acc = await self.execution_service.get_account_info()
        atr = self.data_pipeline.get_atr(symbol)
        open_positions = self.execution_service.get_open_positions()
        logger.info(f"[risk] {symbol}: evaluating signal — "
                    f"atr={atr:.5f} pos_open={len(open_positions)} "
                    f"bal=${acc.get('balance',0):,.0f} eq=${acc.get('equity',0):,.0f}")

        decision = self.risk_gatekeeper.evaluate(
            signal, acc.get("balance", 100_000), acc.get("equity", 100_000),
            acc.get("margin", 0), atr, len(open_positions),
        )
        if decision is None:
            logger.info(f"[risk] {symbol}: REJECTED (risk gate)")
            return
        logger.info(f"[risk] {symbol}: APPROVED — "
                    f"vol={decision.volume:.0f} sl={decision.sl_price:.5f} tp={decision.tp_price:.5f}")

        # STAGE 3: Execution
        result = await self.execution_service.execute(decision)
        if result:
            self.monitoring.on_execution(result, signal)
            if result.success:
                self._trade_count += 1
                logger.success(f"[trade] {symbol}: EXECUTED — "
                              f"{signal.direction.value} {decision.volume:.0f} @ {result.filled_price:.5f}")
                self.signal_engine.on_trade_result(symbol, signal.price, result.filled_price or signal.price)
            else:
                logger.error(f"[trade] {symbol}: FAILED — {result.error}")

    async def _check_new_bars(self) -> int:
        """Check for new 1m bars across all symbols. Returns count of bars closed."""
        closed = 0
        for sym in SYMBOLS:
            df = self.data_pipeline.get_ohlcv(sym, "1m")
            if df is not None and not df.empty:
                current_ts = float(df["timestamp"].iloc[-1])
                last_ts = getattr(self, f"_last_bar_{sym}", 0)
                if current_ts != last_ts:
                    setattr(self, f"_last_bar_{sym}", current_ts)
                    logger.debug(f"[data] {sym}: new 1m bar @ ts={current_ts}")
                    self.data_pipeline.on_bar_close(sym)
                    closed += 1
        if closed > 0:
            logger.info(f"[data] {closed}/{len(SYMBOLS)} symbols had new bars")
        return closed

    def _print_banner(self):
        w = 56
        logger.info("")
        logger.info("=" * w)
        logger.info(f"  RTS: AI Moneybot System Elite v{VERSION}")
        logger.info(f"  Mode: {self.mode.upper()}")
        logger.info("=" * w)
        forex_count = len([s for s in SYMBOLS if s in ("EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF","NZDUSD","EURJPY","GBPJPY","EURGBP")])
        crypto_count = len([s for s in SYMBOLS if s in ("BTCUSD","ETHUSD","LTCUSD","XRPUSD")])
        logger.info(f"  Symbols:  {len(SYMBOLS)} total  |  {forex_count} forex  |  {crypto_count} crypto  |  rest commodities/indices")
        logger.info(f"  Timeframes:  {', '.join(self.config.features.timeframes)}")
        logger.info(f"  Lookback:  {self.config.features.lookback} bars")
        logger.info(f"  Capital:   ${self.config.trading.max_risk_per_trade*100:.0f}% risk per trade")
        logger.info(f"  Max DD:    {self.config.trading.max_drawdown:.0%}")
        logger.info(f"  Kelly:     {self.config.trading.kelly_fraction:.0%} fraction")
        logger.info(f"  Services:  {len(self.registry.all)} active")
        logger.info("=" * w)

    async def start(self):
        self._print_banner()

        errors = self.config.validate()
        if errors:
            logger.error(f"Configuration validation failed:")
            for e in errors:
                logger.error(f"  - {e}")
            logger.error("Fix config.yaml and restart")
            return

        await self.event_bus.start()
        await self.registry.start_all()
        self.running = True

        logger.info("")
        logger.info(f"  [+] All services running. Monitoring {len(SYMBOLS)} symbols...")
        logger.info(f"  [+] Press Ctrl+C to stop gracefully")
        logger.info("")

        last_heartbeat = 0.0
        last_detail = 0.0

        while self.running:
            try:
                self.cycle_counter += 1
                now = time.time()

                # Check for new bars — this is where features get computed
                bars_closed = await self._check_new_bars()
                self._bar_close_count += bars_closed

                # --- Status line every 10s: tells user what bot is doing ---
                if now - last_detail > 10:
                    last_detail = now
                    tick_rate = self.data_pipeline.tick_counter
                    self.data_pipeline.tick_counter = 0
                    # Check data freshness for first few symbols
                    fresh_symbols = []
                    for sym in SYMBOLS[:5]:
                        price = self.data_pipeline.get_price(sym)
                        df = self.data_pipeline.get_ohlcv(sym, "1h")
                        bars = len(df) if df is not None else 0
                        fresh_symbols.append(f"{sym}={price:.5f}({bars}b)")
                    logger.info(f"[status] cycle={self.cycle_counter} "
                               f"ticks/s={tick_rate} "
                               f"bars_closed={bars_closed} "
                               f"signals={self._signal_count} "
                               f"trades={self._trade_count}  |  "
                               f"{'  '.join(fresh_symbols[:3])}")

                # --- Heartbeat every 60s: account-level summary ---
                if now - last_heartbeat > 60:
                    last_heartbeat = now
                    pos = len(self.execution_service.get_open_positions())
                    acc = await self.execution_service.get_account_info()
                    bal = acc.get("balance", 0)
                    eq = acc.get("equity", 0)
                    pnl = eq - bal
                    logger.info(f"[heartbeat] cycle={self.cycle_counter}  "
                               f"pos={pos}  bal=${bal:,.0f}  eq=${eq:,.0f}  "
                               f"upnl=${pnl:+,.0f}  "
                               f"total_signals={self._signal_count}  "
                               f"total_trades={self._trade_count}")

                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[cycle error] {e}")
                await asyncio.sleep(5)

    async def stop(self):
        logger.info("")
        logger.info("Shutting down...")
        self.running = False
        pos = len(self.execution_service.get_open_positions())
        if pos > 0:
            logger.info(f"Closing {pos} open position(s)...")
            await self.execution_service.close_all("shutdown")
            logger.info("All positions closed")
        await self.event_bus.stop()
        await self.registry.stop_all()
        logger.info("RTS: AI Moneybot System Elite — stopped.")


def cmd_info(args):
    """Print system information and exit."""
    config = AppConfig.from_yaml(args.config)
    report = collect_report(config, mode=args.mode, balance=args.capital)
    print(report.print_summary())


def cmd_check(args):
    """Run readiness check and exit."""
    config = AppConfig.from_yaml(args.config)
    report = collect_report(config, mode=args.mode, balance=args.capital)
    ok, output = report.print_check()
    print(output)
    sys.exit(0 if ok else 1)


def cmd_run(args):
    """Run the trading bot."""
    config = AppConfig.from_yaml(args.config)
    secrets = Secrets()
    orch = TradingOrchestrator(config, secrets, mode=args.mode, initial_balance=args.capital)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig:
            try:
                loop.add_signal_handler(sig, lambda: asyncio.ensure_future(orch.stop()))
            except NotImplementedError:
                pass

    try:
        loop.run_until_complete(orch.start())
    except KeyboardInterrupt:
        loop.run_until_complete(orch.stop())
    finally:
        loop.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        prog="python -m src.main_v2",
        description="RTS: AI Moneybot System Elite — Multi-pair AI forex trading system",
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config.yaml",
                        help="Path to configuration YAML (default: config.yaml)")
    parser.add_argument("--capital", type=float, default=100_000.0,
                        help="Starting capital in USD (default: 100000)")
    parser.add_argument("--mode", choices=["paper", "live"], default="paper",
                        help="Trading mode: paper (simulated) or live (real broker)")
    parser.add_argument("--info", action="store_true",
                        help="Print system configuration summary and exit")
    parser.add_argument("--check", action="store_true",
                        help="Run readiness check on all components and exit")
    parser.add_argument("--version", action="store_true",
                        help="Print version and exit")

    args = parser.parse_args()

    if args.version:
        print(f"RTS: AI Moneybot System Elite v{VERSION}")
        sys.exit(0)

    if args.info:
        cmd_info(args)
        sys.exit(0)

    if args.check:
        cmd_check(args)
        sys.exit(0)

    cmd_run(args)


if __name__ == "__main__":
    main()
