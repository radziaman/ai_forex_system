"""RTS: AI Moneybot System Elite — Service Architecture Entry Point (v2)."""

import asyncio
import signal
import sys
import os
import time
from typing import Dict
from loguru import logger

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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
from services.position_persistence import PositionPersistence
from services.connection_manager import ConnectionManager
from services.position_reconciler import PositionReconciler
from agents import AdaptiveRiskManager, PositionManager, TradeManager, DataAgent
from data.data_manager import SYMBOLS, DUKASCOPE_SYMBOLS

HELP_EPILOG = """
Examples:

  python -m src.main --mode paper
    Run in paper trading mode with $100K simulated capital

  python -m src.main --mode live
    Run in live mode (connects to cTrader with configured account)

  python -m src.main --info
    Print system configuration summary and exit

  python -m src.main --check
    Run readiness check on all components and exit

  python -m src.main --mode paper --capital 50000
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
        self.registry = ServiceRegistry()
        self.event_bus = get_event_bus()
        self.running = False
        self.cycle_counter = 0
        self._tick_count = 0
        self._bar_close_count = 0
        self._signal_count = 0
        self._trade_count = 0
        self._last_status_time = 0.0
        self._last_bar_timestamps: Dict[str, float] = {}
        self._init_logging()
        self._init_services()

    def _init_logging(self):
        logger.remove()
        log_path = self.config.data.logs_path.rstrip("/") + "/moneybot.log"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        fmt = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
        logger.add(
            log_path,
            rotation=self.config.logging.rotation,
            retention=self.config.logging.retention,
            format=fmt,
            colorize=False,
        )
        logger.add(sys.stdout, format=fmt, colorize=True, level="INFO")

    def _init_services(self):
        self.data_pipeline = DataPipeline(self.config)
        self.signal_engine = SignalEngine(self.config)
        self.risk_gatekeeper = RiskGatekeeper(
            self.config, initial_balance=self.initial_balance
        )
        self.execution_service = ExecutionService(
            self.config,
            self.secrets,
            self.data_pipeline,
            risk_manager=self.risk_gatekeeper.risk_manager,
            initial_balance=self.initial_balance,
        )
        self.monitoring = MonitoringService(self.config, self.secrets)
        self.persistence = PositionPersistence()

        # Agentic agents
        self.adaptive_risk = AdaptiveRiskManager(
            self.risk_gatekeeper.risk_manager,
            self.config,
        )
        self.position_manager = PositionManager(
            self.execution_service.engine,
            self.data_pipeline,
        )
        self.trade_manager = TradeManager()
        self.data_agent = DataAgent(
            self.data_pipeline,
            SYMBOLS,
        )

        # Connection manager with auto-reconnect
        self.connection_manager = ConnectionManager(
            getattr(self.execution_service, 'ctrader', None)
            or getattr(self.execution_service, 'engine', None)
            or self,
            SYMBOLS,
        )

        # Position reconciler
        self.reconciler = PositionReconciler(
            self.execution_service.engine,
            self.persistence,
        )

        for svc in [
            self.data_pipeline,
            self.signal_engine,
            self.risk_gatekeeper,
            self.execution_service,
            self.monitoring,
        ]:
            self.registry.register(svc)

        self.event_bus.subscribe(EventType.FEATURES_READY, self._on_features_ready)

    async def _on_features_ready(self, event):
        """Features → Signal → Risk → Execution pipeline."""
        update = event.data
        if update is None:
            return
        symbol = update.symbol

        # STAGE 1: Signal generation
        logger.info(
            f"[signal] {symbol}: computing signal from features "
            f"(price={update.price:.5f}, bars={len(update.ohlcv) if update.ohlcv is not None else 0})"
        )

        signal = self.signal_engine.on_features(update)
        if signal is None:
            return
        self._signal_count += 1
        logger.info(
            f"[signal] {symbol}: {signal.direction.value} "
            f"confidence={signal.confidence:.3f} regime={signal.regime.value}"
        )
        self.monitoring.on_signal(signal)

        # STAGE 2: Risk gate
        acc = await self.execution_service.get_account_info()
        atr = self.data_pipeline.get_atr(symbol)
        open_positions = self.execution_service.get_open_positions()
        logger.info(
            f"[risk] {symbol}: evaluating signal — "
            f"atr={atr:.5f} pos_open={len(open_positions)} "
            f"bal=${acc.get('balance',0):,.0f} eq=${acc.get('equity',0):,.0f}"
        )

        decision = self.risk_gatekeeper.evaluate(
            signal,
            acc.get("balance", 100_000),
            acc.get("equity", 100_000),
            acc.get("margin", 0),
            atr,
            len(open_positions),
        )
        if decision is None:
            logger.info(f"[risk] {symbol}: REJECTED (risk gate)")
            return
        logger.info(
            f"[risk] {symbol}: APPROVED — "
            f"vol={decision.volume:.0f} sl={decision.sl_price:.5f} tp={decision.tp_price:.5f}"
        )

        # STAGE 3: Execution (skip in dry-run mode)
        if self.mode == "dry-run":
            logger.info(
                f"[trade] {symbol}: DRY-RUN — would execute "
                f"{signal.direction.value} {decision.volume:.2f} @ {signal.price:.5f}"
            )
            return

        result = await self.execution_service.execute(decision)
        if result:
            self.monitoring.on_execution(
                result, signal, volume=decision.volume, atr=atr,
            )
            if result.success:
                self._trade_count += 1
                logger.success(
                    f"[trade] {symbol}: EXECUTED — "
                    f"{signal.direction.value} {decision.volume:.2f} @ {result.filled_price:.5f}"
                )
                self.trade_manager.record_trade({
                    "symbol": symbol,
                    "direction": signal.direction.value,
                    "volume": decision.volume,
                    "entry": result.filled_price or signal.price,
                    "regime": signal.regime.value,
                    "confidence": signal.confidence,
                    "timestamp": time.time(),
                })
                self.signal_engine.on_trade_result(
                    symbol, signal.price, result.filled_price or signal.price
                )
            else:
                logger.error(f"[trade] {symbol}: FAILED — {result.error}")

    async def _load_historical_data(self):
        """Load historical OHLCV data from cache/yFinance, log what's available."""
        dm = self.data_pipeline.data_manager
        logger.info(f"[data] Loading historical data for {len(SYMBOLS)} symbols...")

        # 1. Load from Dukascopy BI5 cache (fast, local)
        dukas_loaded = dm.load_from_dukascopy_cache(max_hours=168)
        logger.info(
            f"[data] Dukascopy cache: {dukas_loaded}/{len(DUKASCOPE_SYMBOLS)} symbols loaded"
        )

        # 2. For symbols without data, try yFinance fallback
        yfin_loaded = 0
        for sym in SYMBOLS:
            df = dm.get_ohlcv(sym, "1h")
            if df is None or len(df) < 50:
                if dm.try_alternative_source(sym, "1h", days=60):
                    yfin_loaded += 1
        if yfin_loaded > 0:
            logger.info(f"[data] yFinance fallback: {yfin_loaded} symbols loaded")

        # 3. Report data freshness
        total_with_data = 0
        total_bars = 0
        for sym in SYMBOLS:
            df = dm.get_ohlcv(sym, "1h")
            if df is not None:
                bars = len(df)
                total_bars += bars
                if bars > 50:
                    total_with_data += 1
                    last_ts = float(df["timestamp"].iloc[-1])
                    age_mins = (time.time() - last_ts) / 60
                    logger.debug(
                        f"[data] {sym}: {bars} bars, last bar {age_mins:.0f}min ago"
                    )

        logger.info(
            f"[data] Ready: {total_with_data}/{len(SYMBOLS)} symbols have {total_bars} total bars"
        )

        # 4. Generate other timeframes from 1h data by resampling
        tfs_needed = [tf for tf in self.config.features.timeframes if tf != "1h"]
        if tfs_needed and total_with_data > 0:
            generated = 0
            for sym in SYMBOLS:
                df_1h = dm.get_ohlcv(sym, "1h")
                if df_1h is None or len(df_1h) < 30:
                    continue
                for tf in tfs_needed:
                    if tf == "4h":
                        res = df_1h.copy()
                        res["timestamp"] = (res["timestamp"] // 14400) * 14400
                        resampled = (
                            res.groupby("timestamp")
                            .agg(
                                open=("open", "first"),
                                high=("high", "max"),
                                low=("low", "min"),
                                close=("close", "last"),
                                volume=("volume", "sum"),
                            )
                            .reset_index()
                        )
                        dm.ohlcv[sym][tf] = resampled
                    elif tf == "15m":
                        df_5m = dm.get_ohlcv(sym, "5m")
                        if df_5m is not None and len(df_5m) >= 3:
                            res = df_5m.copy()
                            res["timestamp"] = (res["timestamp"] // 900) * 900
                            resampled = (
                                res.groupby("timestamp")
                                .agg(
                                    open=("open", "first"),
                                    high=("high", "max"),
                                    low=("low", "min"),
                                    close=("close", "last"),
                                    volume=("volume", "sum"),
                                )
                                .reset_index()
                            )
                            dm.ohlcv[sym][tf] = resampled
                        else:
                            from_1m = dm.get_ohlcv(sym, "1m")
                            if from_1m is not None and len(from_1m) >= 15:
                                res = from_1m.copy()
                                res["timestamp"] = (res["timestamp"] // 900) * 900
                                resampled = (
                                    res.groupby("timestamp")
                                    .agg(
                                        open=("open", "first"),
                                        high=("high", "max"),
                                        low=("low", "min"),
                                        close=("close", "last"),
                                        volume=("volume", "sum"),
                                    )
                                    .reset_index()
                                )
                                dm.ohlcv[sym][tf] = resampled
                generated += 1
            if generated > 0:
                logger.info(
                    f"[data] Generated {len(tfs_needed)} timeframes for {generated} symbols"
                )

        # 5. Fit feature pipeline scalers on loaded data
        if total_with_data > 0:
            self.data_pipeline.feature_pipeline.fit_all(dm.ohlcv)
            self.data_pipeline.feature_pipeline.save_normalization()
            logger.info(f"[data] Feature pipeline fitted on {total_with_data} symbols")
        else:
            loaded = self.data_pipeline.feature_pipeline.load_normalization()
            if loaded:
                logger.info("[data] Feature normalization loaded from disk")

    async def _check_new_bars(self) -> int:
        """Check for new 1m bars across all symbols. Returns count of bars closed."""
        closed = 0
        for sym in SYMBOLS:
            df = self.data_pipeline.get_ohlcv(sym, "1m")
            if df is not None and not df.empty:
                current_ts = float(df["timestamp"].iloc[-1])
                last_ts = self._last_bar_timestamps.get(sym, 0)
                if current_ts != last_ts:
                    self._last_bar_timestamps[sym] = current_ts
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
        forex_count = len(
            [
                s
                for s in SYMBOLS
                if s
                in (
                    "EURUSD",
                    "GBPUSD",
                    "USDJPY",
                    "AUDUSD",
                    "USDCAD",
                    "USDCHF",
                    "NZDUSD",
                    "EURJPY",
                    "GBPJPY",
                    "EURGBP",
                )
            ]
        )
        crypto_count = len(
            [s for s in SYMBOLS if s in ("BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD")]
        )
        logger.info(
            f"  Symbols:  {len(SYMBOLS)} total  |  {forex_count} forex  |  {crypto_count} crypto"
        )
        logger.info(f"  Timeframes:  {', '.join(self.config.features.timeframes)}")
        logger.info(f"  Lookback:  {self.config.features.lookback} bars")
        logger.info(
            f"  Capital:   ${self.config.trading.max_risk_per_trade*100:.0f}% risk per trade"
        )
        logger.info(f"  Max DD:    {self.config.trading.max_drawdown:.0%}")
        logger.info(f"  Kelly:     {self.config.trading.kelly_fraction:.0%} fraction")
        logger.info(f"  Services:  {len(self.registry.all)} active")
        logger.info("=" * w)

    async def start(self):
        self._print_banner()

        errors = self.config.validate()
        if errors:
            logger.error("Configuration validation failed:")
            for e in errors:
                logger.error(f"  - {e}")
            logger.error("Fix config.yaml and restart")
            return

        await self.event_bus.start()
        await self.registry.start_all()

        # Load historical data so OHLCV is populated for signal generation
        await self._load_historical_data()

        # In live mode, sync risk initial_balance to actual broker balance
        if self.mode == "live":
            try:
                acc = await self.execution_service.get_account_info()
                broker_balance = acc.get("balance", 0)
                if broker_balance > 0:
                    self.risk_gatekeeper.risk_manager.initial_balance = broker_balance
                    self.risk_gatekeeper.risk_manager.peak_balance = broker_balance
                    self.risk_gatekeeper.risk_manager.kill_switch_triggered = False
                    logger.info(
                        f"[risk] Synced to broker balance: ${broker_balance:,.2f}"
                    )
            except Exception as e:
                logger.warning(f"[risk] Could not sync broker balance: {e}")

        # Restore persisted state if available
        try:
            saved_risk = self.persistence.load_risk_state(self.risk_gatekeeper.risk_manager)
            saved_positions = self.persistence.load_positions()
            if saved_positions:
                logger.info(f"[persist] Restored {len(saved_positions)} positions from disk")
            if saved_risk:
                logger.info("[persist] Restored risk manager state")
        except Exception as e:
            logger.warning(f"[persist] State restoration skipped: {e}")

        # Position reconciliation
        if self.mode == "live":
            try:
                await self.reconciler.reconcile()
                await self.reconciler.reconcile_account_balance(
                    self.risk_gatekeeper.risk_manager
                )
            except Exception as e:
                logger.warning(f"[reconcile] Failed: {e}")

        # Start connection manager
        try:
            await self.connection_manager.start()
        except Exception as e:
            logger.warning(f"[conn] Connection manager start failed: {e}")

        self.running = True

        logger.info("")
        logger.info(f"  [+] All services running. Monitoring {len(SYMBOLS)} symbols...")
        logger.info("  [+] Press Ctrl+C to stop gracefully")
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
                    # Show per-symbol data freshness (price + bar count + data source)
                    from data.data_manager import BASE_PRICES

                    live_count = 0
                    total_count = 0
                    for sym in SYMBOLS:
                        live = self.data_pipeline.get_live_price(sym)
                        if live != BASE_PRICES.get(sym, 0):
                            live_count += 1
                        total_count += 1
                    fresh_symbols = []
                    for sym in ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD"]:
                        live = self.data_pipeline.get_live_price(sym)
                        df = self.data_pipeline.get_ohlcv(sym, "1h")
                        bars = len(df) if df is not None else 0
                        is_live = live != BASE_PRICES.get(sym, 0)
                        tag = "LIVE" if is_live else "HIST"
                        fresh_symbols.append(f"{sym}={live:.5f}({bars}b,{tag})")
                    logger.info(
                        f"[status] cycle={self.cycle_counter} "
                        f"ticks/s={tick_rate} "
                        f"bars/min={bars_closed} "
                        f"sig={self._signal_count} "
                        f"trd={self._trade_count}  |  "
                        f"{'  '.join(fresh_symbols)}"
                    )

                # --- Agent ticks ---
                try:
                    await self.position_manager.tick()
                except Exception as e:
                    logger.debug(f"[agent] position_manager: {e}")

                data_health = self.data_agent.tick()

                # Adaptive risk update (use first symbol's price as proxy)
                sample_price = self.data_pipeline.get_price(SYMBOLS[0]) if SYMBOLS else 1.0
                sample_atr = self.data_pipeline.get_atr(SYMBOLS[0]) if SYMBOLS else 0.001
                acc_info = await self.execution_service.get_account_info()
                raw_regime = self.signal_engine._regimes.get(SYMBOLS[0], "ranging") if SYMBOLS else "ranging"
                current_regime = raw_regime.value if hasattr(raw_regime, "value") else str(raw_regime)
                self.adaptive_risk.update(
                    acc_info.get("balance", self.initial_balance),
                    acc_info.get("equity", self.initial_balance),
                    sample_atr,
                    sample_price,
                    regime=current_regime,
                )

                # --- Heartbeat every 60s: account-level + agent summary ---
                if now - last_heartbeat > 60:
                    last_heartbeat = now
                    pos = len(self.execution_service.get_open_positions())
                    bal = acc_info.get("balance", 0)
                    eq = acc_info.get("equity", 0)
                    pnl = eq - bal
                    agent_status = self.adaptive_risk.get_status()
                    trade_summary = self.trade_manager.get_trade_summary()
                    logger.info(
                        f"[heartbeat] cycle={self.cycle_counter}  "
                        f"pos={pos}  bal=${bal:,.0f}  eq=${eq:,.0f}  "
                        f"upnl=${pnl:+,.0f}  "
                        f"signals={self._signal_count}  "
                        f"trades={self._trade_count}  |  "
                        f"kelly={agent_status['effective_kelly']:.2f}  "
                        f"risk={agent_status['effective_risk']:.3f}  "
                        f"dd={agent_status['drawdown_regime']}  "
                        f"vol={agent_status['volatility_regime']}  |  "
                        f"data={data_health['fresh_pct']:.0f}% "
                        f"[{data_health['fresh_count']}/{data_health['total']}]"
                    )

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

        # Persist state before closing
        try:
            self.persistence.save_all(
                self.execution_service.engine,
                self.risk_gatekeeper.risk_manager,
            )
            logger.info("[persist] State saved")
        except Exception as e:
            logger.warning(f"[persist] Could not save state: {e}")

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


def _shutdown(loop, orch):
    """Safely stop the orchestrator from any thread."""
    if loop.is_running():
        asyncio.run_coroutine_threadsafe(orch.stop(), loop)

def cmd_run(args):
    """Run the trading bot."""
    config = AppConfig.from_yaml(args.config)
    secrets = Secrets()
    mode = "dry-run" if args.dry_run else args.mode
    orch = TradingOrchestrator(config, secrets, mode=mode, initial_balance=args.capital)

    if mode == "dry-run":
        logger.info("=" * 56)
        logger.info("  DRY-RUN MODE: signals generated, orders LOGGED only")
        logger.info("  No real or paper orders will be executed")
        logger.info("=" * 56)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Platform-agnostic signal handling: use signal.signal for Windows compat
    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig:
            try:
                loop.add_signal_handler(sig, lambda: asyncio.ensure_future(orch.stop()))
            except NotImplementedError:
                signal.signal(sig, lambda *_: _shutdown(loop, orch))

    try:
        loop.run_until_complete(orch.start())
    except (KeyboardInterrupt, asyncio.CancelledError):
        loop.run_until_complete(orch.stop())
    finally:
        loop.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description="RTS: AI Moneybot System Elite — Multi-pair AI forex trading system",
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration YAML (default: config.yaml)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100_000.0,
        help="Starting capital in USD (default: 100000)",
    )
    parser.add_argument(
        "--mode",
        choices=["paper", "live", "dry-run"],
        default="paper",
        help="Trading mode: paper (simulated), live (real broker), or dry-run (log only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry-run mode: full pipeline runs but no orders are sent",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print system configuration summary and exit",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run readiness check on all components and exit",
    )
    parser.add_argument("--version", action="store_true", help="Print version and exit")

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
