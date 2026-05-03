"""
RTS AI Forex Trading System — Elite Edition.
Multi-pair async trading loop with 7 major forex pairs, per-symbol regime
detection, MoE ensemble, economic calendar gating, FinBERT sentiment,
and institutional-grade risk management.
"""
import asyncio
import json
import os
import signal
import sys
import time
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from infrastructure.config import Config
from infrastructure.secrets import Secrets
from risk.manager import RiskManager, RiskParameters, TrailingStopManager
from execution.engine import ExecutionEngine
from execution.cost_model import CostModel
from api.base import PriceTick
from api.provider_factory import create_execution_provider
from rts_ai_fx.features_unified import FeaturePipeline
from rts_ai_fx.regime_detector import HMMRegimeDetector
from rts_ai_fx.ensemble import MoEEnsemble
from rts_ai_fx.drift_detector import DriftMonitor
from data.data_manager import DataManager, SYMBOLS, BASE_PRICES
from data.economic_calendar import EconomicCalendar
from data.alternative_data import AlternativeDataProvider
from ai.sentiment import SentimentAnalyzer
from validation.walk_forward import PurgedWalkForward
from validation.monte_carlo import MonteCarloSigTest
from backtest.vectorized_backtester import VectorizedBacktester
from notifications.telegram import TelegramNotifier
from dashboard.app import app, broadcast_update, update_state, latest_state

try:
    from loguru import logger as loguru_logger
    logger = loguru_logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# Pairs that share a common base — prevents double-counting correlated entries
CORRELATED_GROUPS = [
    {"EURUSD", "GBPUSD", "EURGBP"},
    {"AUDUSD", "NZDUSD"},
    {"USDCAD", "USDCHF"},
]


class RTSForexBot:
    """Multi-pair AI forex trading bot with full institutional toolchain."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = Config()
        self.config.config_path = config_path
        self.config._load_yaml()
        self.secrets = Secrets()

        logger.remove()
        logger.add(sys.stdout, level=self.config.logging.level)
        logger.add(
            self.config.data.logs_path.rstrip("/") + "/moneybot.log",
            rotation=self.config.logging.rotation,
            retention=self.config.logging.retention,
        )

        self._init_ctrader()
        self._init_risk()
        self._init_data()
        self._init_models()
        self._init_intelligence()
        self._init_notifications()
        self._init_validation()
        self._init_monitoring()

        self.trade_count = 0
        self.is_running = False
        self._last_snapshot_time = 0.0
        self._current_sentiment = 0.0
        self._regimes: Dict[str, str] = {}
        self._trade_decisions: Dict[str, dict] = {}

        logger.info("=" * 60)
        logger.info("  RTS AI Forex Trading System v4.0 (Multi-Pair)")
        logger.info("  Pairs: EURUSD | GBPUSD | USDJPY | AUDUSD")
        logger.info("         USDCAD | USDCHF | NZDUSD")
        logger.info("  Economic Calendar | FinBERT Sentiment | Alt Data")
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_ctrader(self):
        self.execution, self.data_provider = create_execution_provider(self.secrets)
        self.execution.on_price = self._on_market_data
        self.ctrader = getattr(self.execution, 'raw', self.execution)

    def _init_risk(self):
        params = RiskParameters(
            max_risk_per_trade=self.config.trading.max_risk_per_trade,
            max_drawdown=self.config.trading.max_drawdown,
            max_margin_usage=self.config.trading.max_margin_usage,
        )
        self.risk = RiskManager(params, initial_balance=100_000.0)
        self.trailing_stop = TrailingStopManager(
            tp_pcts=[0.01, 0.02, 0.03],
            trail_atr_mult=self.config.trading.sl_atr_multiplier,
        )
        self.cost_model = CostModel(commission_per_lot=self.config.trading.commission_per_lot)
        self.execution = ExecutionEngine(self.ctrader, self.risk, None)
        self.execution.cost_model = self.cost_model

    def _init_data(self):
        self.data_manager = DataManager(
            historical_path=self.config.data.historical_path,
        )
        self.feature_pipeline = FeaturePipeline(
            lookback=30, timeframes=["1h", "4h", "1d"],
            use_microstructure=True, use_cross_asset=False,
        )
        self.regime_detector = HMMRegimeDetector(n_regimes=4, lookback=60)

    def _init_models(self):
        self.ensemble = MoEEnsemble()
        self.ensemble.add_expert(
            name="rule_based",
            predict_fn=self._rule_prediction,
            confidence_fn=lambda X: 0.7,
            regime="ranging",
        )
        self.active_models: Dict[str, Any] = {}
        self.drift_monitors: Dict[str, DriftMonitor] = {}

    def _rule_prediction(self, X: np.ndarray) -> float:
        try:
            last = X[-1] if X.ndim == 2 else X
            mom = float(last[-5]) if len(last) > 5 and not np.isnan(last[-5]) else 0.0
            return 1.12 + mom * 0.001
        except Exception:
            return 1.12

    def _init_intelligence(self):
        self.economic_calendar = EconomicCalendar(
            cache_path="data/alternative_data/economic_calendar.json",
        )
        self.sentiment_analyzer = SentimentAnalyzer(use_finbert=True, cache_ttl=300)
        self.alternative_data = AlternativeDataProvider(
            fred_api_key=self.secrets.fred_api_key if hasattr(self.secrets, 'fred_api_key') else "",
            cache_dir="data/alternative_data",
        )
        logger.info("Intelligence layer initialized")

    def _init_notifications(self):
        self.notifier = TelegramNotifier(
            bot_token=self.secrets.telegram_bot_token,
            chat_id=self.secrets.telegram_chat_id,
            send_trade_alerts=True,
            send_daily_summary=True,
            send_risk_alerts=True,
        )
        self._last_daily_summary_day = -1

        # Wire trade close callback
        self.risk._on_trade_close = self._on_trade_closed

    def _on_trade_closed(self, trade):
        hold_time = time.time() - trade.timestamp
        self.notifier.trade_closed(
            symbol=trade.symbol,
            direction=trade.direction,
            entry=trade.entry_price,
            exit=trade.exit_price,
            pnl=trade.pnl,
            reason=trade.reason,
            hold_time=hold_time,
        )

    def _init_validation(self):
        self.walk_forward = PurgedWalkForward(
            n_folds=6, test_window=252, embargo=10, min_train_window=504,
        )
        self.monte_carlo = MonteCarloSigTest(n_permutations=10000, alpha=0.05)
        self.vectorized_backtester = VectorizedBacktester(
            spread_pips=0.5, commission_per_lot=7.0, slippage_model="moderate",
        )
        logger.info("Validation layer initialized")

    def _init_monitoring(self):
        self._alert_cooldown = {}
        self._last_health_check = 0.0

    # ================================================================
    # MAIN LOOP
    # ================================================================

    async def start(self):
        logger.info("Starting RTS Forex Bot...")
        result = await self.ctrader.start()
        if not result:
            logger.warning("cTrader in simulation mode — no real connection")

        # Load historical data for ALL symbols
        await self._load_historical_data()

        # Fit regime detector per symbol
        for sym in SYMBOLS:
            df = self.data_manager.get_ohlcv(sym, "1h")
            if df is not None and len(df) > 200:
                rd = HMMRegimeDetector(n_regimes=4, lookback=60)
                rd.fit(df)
                self._regimes[sym] = rd.detect_regime(df)
                logger.info(f"HMM regime detector fitted for {sym}")

        # Fit feature pipeline on ALL symbols (per-symbol normalization)
        self.feature_pipeline.fit_all(self.data_manager.ohlcv)
        logger.info("Feature pipeline fitted on all symbols")

        # Fetch intelligence
        try:
            self.economic_calendar.fetch(days_forward=7)
            logger.info("Economic calendar loaded")
        except Exception:
            pass
        try:
            self.sentiment_analyzer.get_latest(force_refresh=True)
            logger.info("Sentiment analyzer warmed up")
        except Exception:
            pass
        try:
            self.alternative_data.fetch_all()
            logger.info("Alternative data pre-fetched")
        except Exception:
            pass

        # Start WebSocket streaming
        if hasattr(self.execution, 'stream_prices'):
            try:
                await self.execution.stream_prices(SYMBOLS)
            except NotImplementedError:
                pass

        self._start_dashboard()
        self.is_running = True
        logger.info("Bot is LIVE — monitoring 7 pairs")

        cycle_counter = 0
        while self.is_running:
            try:
                await self._trading_cycle()
                await asyncio.sleep(1.0)
                cycle_counter += 1

                # Periodic refresh
                if cycle_counter % 300 == 0:
                    try:
                        self.sentiment_analyzer.get_latest(force_refresh=True)
                    except Exception:
                        pass
                if cycle_counter % 3600 == 0:
                    try:
                        self.economic_calendar.fetch(days_forward=7)
                        self.alternative_data.fetch_all()
                    except Exception:
                        pass

                # Daily summary (send once per day)
                today = pd.Timestamp.now().day
                if today != self._last_daily_summary_day:
                    self._last_daily_summary_day = today
                    balance = self.risk.initial_balance + self.risk.daily_pnl
                    regime_summary = ", ".join(
                        f"{s}:{r}" for s, r in sorted(self._regimes.items())[:7]
                    )
                    self.notifier.daily_summary(
                        trades_today=self.risk.daily_trades,
                        wins=self.risk.wins,
                        losses=self.risk.losses,
                        pnl=self.risk.daily_pnl,
                        balance=balance,
                        open_positions=len(self.execution.get_open_positions()),
                        regime_summary=regime_summary,
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cycle error: {e}")
                await asyncio.sleep(5.0)

    async def _load_historical_data(self):
        # Load all 7 symbols from data provider
        if self.data_provider is not None:
            try:
                for sym in SYMBOLS:
                    ohlcv = await self.data_provider.fetch_ohlcv(
                        sym, "1h", "2025-06-01", "2026-03-31"
                    )
                    if ohlcv:
                        df = pd.DataFrame([{
                            "timestamp": o.timestamp, "open": o.open, "high": o.high,
                            "low": o.low, "close": o.close, "volume": o.volume,
                        } for o in ohlcv])
                        if len(df) > 100:
                            self.data_manager.ohlcv[sym]["1h"] = df
                            logger.info(f"Loaded {len(df)} 1h bars for {sym}")
            except Exception as e:
                logger.warning(f"Data provider failed: {e}")

        # Fallback: load/generate per symbol
        for sym in SYMBOLS:
            for tf in ["1h", "4h", "1d"]:
                df = self.data_manager.get_ohlcv(sym, tf)
                if df is None or len(df) < 100:
                    self.data_manager.load_historical(sym, tf, days=365)

    def _start_dashboard(self):
        import threading
        import uvicorn
        d = self.config.dashboard
        t = threading.Thread(
            target=lambda: uvicorn.run(app, host=d.host, port=d.port, log_level="error"),
            daemon=True,
        )
        t.start()
        logger.info(f"Dashboard: http://{d.host}:{d.port}")

    # ================================================================
    # TRADING CYCLE — iterates over all 7 pairs
    # ================================================================

    async def _trading_cycle(self):
        # 1. Economic Calendar Gate (global)
        suppressed, event = self.economic_calendar.is_suppressed()
        if suppressed and event:
            logger.info(f"Trading suppressed: {event.title}")
            self._update_dashboard({"balance": 100000}, "SUPPRESSED")
            return

        # 2. Gather global intelligence
        external_signals = self._gather_external_signals()

        # 3. Evaluate each symbol
        self._trade_decisions = {}
        account_info = await self.execution.get_account_info()
        if account_info is None:
            return

        for sym in SYMBOLS:
            decision = await self._evaluate_symbol(sym, account_info, external_signals)
            if decision:
                self._trade_decisions[sym] = decision

        # 4. Apply correlation filter — prevent opposing trades on correlated pairs
        self._apply_correlation_filter()

        # 5. Execute approved trades
        for sym, decision in self._trade_decisions.items():
            if decision.get("approved") and not decision.get("blocked_by_correlation"):
                await self._execute_trade(sym, decision, account_info)

        # 6. Update dashboard
        self._update_dashboard(account_info, "trading")

    def _gather_external_signals(self) -> np.ndarray:
        try:
            sentiment_vec = self.sentiment_analyzer.get_feature_vector()
        except Exception:
            sentiment_vec = np.zeros(11, dtype=np.float32)
        try:
            alt_data_vec = self.alternative_data.get_full_feature_vector()
        except Exception:
            alt_data_vec = np.zeros(38, dtype=np.float32)
        return np.concatenate([sentiment_vec, alt_data_vec])

    async def _evaluate_symbol(
        self, symbol: str, account_info: dict, external_signals: np.ndarray,
    ) -> Optional[dict]:
        # Get OHLCV for this symbol
        ohlcv = self.data_manager.get_ohlcv_dict(symbol)
        price = self.data_manager.get_price(symbol, "1h")
        atr = self.data_manager.get_atr(symbol, "1h", 14)
        df_1h = self.data_manager.get_ohlcv(symbol, "1h")

        if df_1h is None or len(df_1h) < 60:
            return None

        # Detect regime
        if symbol not in self._regimes or self.regime_detector is None:
            rd = HMMRegimeDetector(n_regimes=4, lookback=60)
            rd.fit(df_1h)
            regime = rd.detect_regime(df_1h)
            self._regimes[symbol] = regime
        else:
            # Reuse or re-detect
            regime = self.regime_detector.detect_regime(df_1h)
            self._regimes[symbol] = regime

        if not HMMRegimeDetector.should_trade_static(regime):
            return None

        # Feature extraction
        tick_buffer = self.data_manager.get_tick_buffer(symbol, 1000)
        features = self.feature_pipeline.transform(
            self.data_manager.ohlcv, symbol=symbol,
            tick_buffer=tick_buffer,
            external_signals=external_signals,
        )
        if features is None:
            return None

        # Ensemble inference
        ensemble_pred = self.ensemble.predict(features, regime=regime)
        confidence = ensemble_pred.confidence
        should_trade, direction, agreement = self.ensemble.should_trade(
            ensemble_pred, price, min_confidence=0.65,
        )

        # Sentiment-adjusted confidence
        if should_trade and abs(self._current_sentiment) > 0.3:
            sent_direction = 1 if self._current_sentiment > 0 else 0
            if sent_direction == (1 if direction == "BUY" else 0):
                confidence *= 1.1
            else:
                confidence *= 0.9
            confidence = min(max(confidence, 0.0), 1.0)

        if not should_trade:
            return None

        # Regime params
        regime_params = HMMRegimeDetector.get_regime_params_static(regime)
        adjusted_conf = confidence * regime_params["pos_mult"]

        if self._current_sentiment < -0.5:
            adjusted_conf *= 0.5

        # Kelly sizing
        volume = self.risk.calculate_kelly_size(
            account_info.get("balance", 100_000), price, atr, adjusted_conf,
        )
        volume = max(volume, 100)

        return {
            "symbol": symbol,
            "direction": direction,
            "volume": volume,
            "price": price,
            "atr": atr,
            "confidence": confidence,
            "regime": regime,
            "regime_params": regime_params,
            "approved": True,
            "blocked_by_correlation": False,
        }

    def _apply_correlation_filter(self):
        """Prevent opening trades on correlated pairs that conflict."""
        approved_symbols = {d["symbol"] for d in self._trade_decisions.values()
                           if d.get("approved")}
        if len(approved_symbols) < 2:
            return

        for group in CORRELATED_GROUPS:
            active = [sym for sym in approved_symbols if sym in group]
            if len(active) < 2:
                continue
            dirs = [self._trade_decisions[sym]["direction"] for sym in active]
            # If any pair in a correlated group goes opposite direction, block the weaker signal
            if len(set(dirs)) > 1:
                confidences = [self._trade_decisions[sym]["confidence"] for sym in active]
                min_idx = np.argmin(confidences)
                self._trade_decisions[active[min_idx]]["blocked_by_correlation"] = True
                self._trade_decisions[active[min_idx]]["approved"] = False
                logger.info(
                    f"Correlation filter: blocked {active[min_idx]} "
                    f"(conflict in {group})"
                )

    async def _execute_trade(self, symbol: str, decision: dict, account_info: dict):
        direction = decision["direction"]
        volume = decision["volume"]
        price = decision["price"]
        atr = decision["atr"]
        regime_params = decision["regime_params"]
        confidence = decision["confidence"]
        regime = decision["regime"]

        # Risk checks
        balance = account_info.get("balance", 100_000)
        equity = account_info.get("equity", 100_000)
        margin = account_info.get("margin", 0)
        daily_pnl = self.risk.daily_pnl

        approved, reason = self.risk.pre_trade_checks(balance, equity, margin, daily_pnl)
        if not approved:
            logger.info(f"{symbol} trade blocked: {reason}")
            self._send_alert(f"{symbol} blocked: {reason}", level="warning")
            self.notifier.risk_warning(
                f"Trade blocked for {symbol}",
                details={"reason": reason, "balance": balance, "daily_pnl": daily_pnl},
            )
            return

        # Place order
        trade = await self.execution.open_position(
            symbol=symbol,
            direction=direction,
            volume=volume,
            sl=price - atr * regime_params["sl_pct"] / (atr / price + 1e-8),
            tp=price + atr * regime_params["tp_pct"] / (atr / price + 1e-8),
            reason=f"regime={regime} conf={confidence:.2f} sent={self._current_sentiment:.2f}",
        )
        if trade:
            self.trade_count += 1
            self._send_alert(
                f"{direction} {volume:.0f} {symbol} @ {price:.5f} (regime={regime})",
                level="success",
            )
            self.notifier.trade_opened(
                symbol=symbol, direction=direction, volume=volume,
                price=price, regime=regime, confidence=confidence, atr=atr,
            )

        # Update drift monitors
        for name, model_data in self.active_models.items():
            if "drift" in model_data:
                model_data["drift"].update(price, price)

    # ================================================================
    # HELPERS
    # ================================================================

    def _on_market_data(self, tick: PriceTick):
        self.data_manager.update_tick(tick.symbol, tick.bid, tick.ask, tick.volume)

    def _send_alert(self, message: str, level: str = "info"):
        logger.info(f"[ALERT] {message}")
        self.notifier.send(message, level=level)

    def _update_dashboard(self, account_info: dict, regime: str = "trading"):
        positions = self.execution.get_open_positions()
        suppressed, event = self.economic_calendar.is_suppressed()
        upcoming = self.economic_calendar.get_upcoming_events(hours=24)

        prices = self.data_manager.all_prices()
        regime_str = ", ".join(
            f"{sym}:{reg}" for sym, reg in list(self._regimes.items())[:7]
        )

        update_state(
            balance=account_info.get("balance", 100000),
            equity=account_info.get("equity", 100000),
            margin=account_info.get("margin", 0),
            free_margin=account_info.get("free_margin", 100000),
            initial_balance=100000,
            total_trades=self.trade_count,
            win_rate=self.risk.get_win_rate(),
            mode=self.risk.mode,
            regime=regime_str,
            open_positions=positions,
            trade_history=self.execution.get_trade_history(20),
            market_data={
                "prices": prices,
                "spread": 0.0,
            },
            ai_metrics={
                "regime": regime_str,
                "sentiment": round(self._current_sentiment, 3),
                "econ_suppressed": suppressed,
                "econ_next_event": event.title if event else "",
                "upcoming_events": len(upcoming),
                "active_decisions": len(self._trade_decisions),
                "var": self.risk.var(),
                "cvar": self.risk.cvar(),
            },
        )
        try:
            asyncio.run_coroutine_threadsafe(
                broadcast_update(latest_state), asyncio.get_event_loop()
            )
        except Exception:
            pass

    async def stop(self):
        logger.info("Stopping bot...")
        self.is_running = False
        await self.execution.close_all_positions("Shutdown")
        await self.ctrader.disconnect()
        logger.info("Bot stopped.")

    # ================================================================
    # VALIDATION COMMANDS
    # ================================================================

    async def run_walk_forward(self, prices: np.ndarray, features: np.ndarray):
        logger.info("Starting walk-forward validation...")
        results = self.walk_forward.run(prices, lambda *a: [], features)
        summary = PurgedWalkForward.summary(results)
        logger.info(f"Walk-Forward Summary: {json.dumps(summary, indent=2)}")
        return summary

    async def run_monte_carlo_test(self, trades: List[Dict], by_regime: Dict[str, List[Dict]] = None):
        logger.info("Running Monte Carlo significance tests...")
        if by_regime:
            return self.monte_carlo.run_battery(by_regime, trades)
        return self.monte_carlo.test(trades)

    async def run_vectorized_backtest(self, prices: np.ndarray, features: np.ndarray, atr: np.ndarray = None):
        logger.info("Running vectorized backtest...")
        results = self.vectorized_backtester.run_with_sensitivity(
            prices, lambda p, f: np.zeros(len(p)), features, atr,
        )
        for key, r in results.items():
            logger.info(f"  {key}: {r.to_dict()}")
        return results


# ================================================================
# Update HMMRegimeDetector with static methods for multi-pair use
# ================================================================

# Patch in static methods for multi-pair convenience
HMMRegimeDetector.should_trade_static = staticmethod(lambda regime: HMMRegimeDetector.get_regime_params_static(regime)["pos_mult"] > 0)
HMMRegimeDetector.get_regime_params_static = staticmethod(HMMRegimeDetector.get_regime_params)


# ================================================================
# ENTRY POINT
# ================================================================

def signal_handler(sig, frame):
    if "bot" in globals():
        asyncio.create_task(bot.stop())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RTS AI Forex Trading System")
    parser.add_argument("--mode", choices=["live", "backtest", "validate", "train"], default="live")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--mc-test", action="store_true")
    parser.add_argument("--bt-sensitivity", action="store_true")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    for d in ["data/logs", "data/trades", "models", "data/alternative_data"]:
        os.makedirs(d, exist_ok=True)

    bot = RTSForexBot(args.config)

    if args.mode in ("backtest",) or args.bt_sensitivity:
        prices = bot.data_manager.get_ohlcv("EURUSD", "1h")
        if prices is not None and len(prices) > 100:
            p = prices["close"].values
            f = bot.feature_pipeline.transform(bot.data_manager.ohlcv, "EURUSD")
            if args.bt_sensitivity:
                asyncio.run(bot.run_vectorized_backtest(p, f))
    elif args.mode in ("validate",) or args.walk_forward or args.mc_test:
        prices = bot.data_manager.get_ohlcv("EURUSD", "1h")
        if prices is not None and len(prices) > 100:
            p = prices["close"].values
            if args.walk_forward:
                asyncio.run(bot.run_walk_forward(p, None))
            if args.mc_test:
                trades = bot.execution.get_trade_history(1000)
                asyncio.run(bot.run_monte_carlo_test(trades))
    else:
        asyncio.run(bot.start())
