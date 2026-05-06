"""
RTS AI Forex Trading System -- Elite Edition.
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
from data.market_session import MarketSession
from data.economic_calendar import EconomicCalendar
from data.alternative_data import AlternativeDataProvider
from data.order_flow import OrderFlowAnalyzer, GammaExposureMapper, DarkPoolDetector
from data.toxic_flow import ToxicFlowDetector, ToxicFlowSnapshot
from data.smart_money import COTAnalyzer, COTSnapshot
from ai.sentiment import SentimentAnalyzer
from ai.regime_agents import RegimeSpecialistSystem, REGIME_CONFIGS
from ai.maml_agent import MAMLAgent
from ai.behavioral_sentiment import BehavioralSentimentAI
from rts_ai_fx.causal_features import CausalFeatureSelector
from rts_ai_fx.attention_fusion import AttentionFusionPipeline
from execution.algo_executor import AlgoExecutor, ExecutionAlgoConfig
from risk.circuit_breaker import CircuitBreaker, MarketStressSnapshot
from infrastructure.event_bus import TradingEventBus, EventType, get_event_bus
from training.model_registry import ModelRegistry, ABTestConfig
from validation.walk_forward import PurgedWalkForward
from validation.monte_carlo import MonteCarloSigTest
from backtest.vectorized_backtester import VectorizedBacktester
from notifications.telegram import TelegramNotifier
from training.online_learner import OnlineLearner
from dashboard.app import app, broadcast_update, update_state, latest_state

try:
    from loguru import logger as loguru_logger
    logger = loguru_logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# Pairs that share a common base -- prevents double-counting correlated entries
CORRELATED_GROUPS = [
    {"EURUSD", "GBPUSD", "EURGBP"},
    {"AUDUSD", "NZDUSD"},
    {"USDCAD", "USDCHF"},
    {"EURJPY", "GBPJPY", "USDJPY"},
]

# Asset class detection
CRYPTO_PAIRS = {"BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD"}
INDEX_PAIRS = {"US500", "US30", "USTEC", "UK100", "DE40"}
ENERGY_PAIRS = {"XTIUSD", "XBRUSD", "XNGUSD"}
METAL_PAIRS = {"XAUUSD", "XAGUSD"}


def get_asset_class(symbol: str) -> str:
    """Detect asset class from symbol name."""
    sym = symbol.upper()
    if sym in CRYPTO_PAIRS:
        return "crypto"
    elif sym in INDEX_PAIRS:
        return "index"
    elif sym in ENERGY_PAIRS or sym in METAL_PAIRS:
        return "commodity"
    elif "JPY" in sym or "USD" in sym or "EUR" in sym or "GBP" in sym:
        return "forex"
    return "unknown"


class RTSForexBot:
    """Multi-pair AI forex trading bot with full institutional toolchain."""

    def __init__(self, config_path: str = "config.yaml", mode: str = "paper", initial_balance: float = 100000.0):
        self.config = Config()
        self.config.config_path = config_path
        self.config._load_yaml()
        self.secrets = Secrets()
        self.mode = mode
        self.initial_balance = initial_balance

        logger.remove()
        # Only log to file to avoid Windows console encoding issues
        # Disable all stdout/stderr output with icons
        logger.add(
            self.config.data.logs_path.rstrip("/") + "/moneybot.log",
            rotation=self.config.logging.rotation,
            retention=self.config.logging.retention,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            colorize=False,
            enqueue=False,
            serialize=False,
        )

        self._init_ctrader()
        self._init_data()    # data_manager first, needed by ExecutionEngine
        self._init_risk()    # ExecutionEngine needs data_manager
        self._init_models()
        self._init_intelligence()
        self._init_notifications()
        self._init_validation()
        self._init_monitoring()
        self._init_enhancements()  # New: All 11 enhancements

        self.trade_count = 0
        self.is_running = False
        self._last_snapshot_time = 0.0
        self._current_sentiment = 0.0
        self._regimes: Dict[str, str] = {}
        self._trade_decisions: Dict[str, dict] = {}
        self._toxic_flows: Dict[str, ToxicFlowSnapshot] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._cot_data: Dict[str, COTSnapshot] = {}
        self._attn_fusion: Optional[AttentionFusionPipeline] = None

        logger.info("=" * 60)
        logger.info("  RTS AI Forex Trading System v4.0 (Multi-Pair)")
        logger.info("  Pairs: EURUSD | GBPUSD | USDJPY | AUDUSD")
        logger.info("         USDCAD | USDCHF | NZDUSD")
        logger.info("  Economic Calendar | FinBERT Sentiment | Alt Data")
        logger.info("=" * 60)

    def _init_enhancements(self):
        """Initialize all 11 enhancement modules."""
        logger.info("=" * 60)
        logger.info("  Initializing 11 Enhancement Modules")
        logger.info("=" * 60)

        # 1. Toxic Flow Detection
        self.toxic_detector = ToxicFlowDetector(lookback=100, bucket_size=1000)
        logger.info("[OK] Toxic Flow Detector (VPIN) initialized")

        # 2. Multi-Agent Regime Specialist System
        state_dim = 55 * 30  # 55 features * 30 lookback
        self.regime_system = RegimeSpecialistSystem(state_dim=state_dim, n_actions=5)
        logger.info("[OK] Regime Specialist System initialized")

        # 3. Causal Feature Selection
        self.causal_selector = CausalFeatureSelector(max_lag=5, alpha=0.01)
        logger.info("[OK] Causal Feature Selector initialized")

        # 4. Smart Money Tracking (COT)
        self.cot_analyzer = COTAnalyzer(cache_dir="data/alternative_data")
        logger.info("[OK] Smart Money Tracker (COT) initialized")

        # 5. Flash Crash & Circuit Breakers
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        for sym in SYMBOLS:
            self.circuit_breakers[sym] = CircuitBreaker(
                price_velocity_threshold=0.005,
                spread_multiplier_threshold=5.0,
                volume_spike_multiplier=10.0,
            )
        logger.info(f"[OK] Circuit Breakers initialized for {len(SYMBOLS)} symbols")

        # 6. Meta-Learning Agent
        self.maml_agent = MAMLAgent(input_dim=55 * 30, inner_lr=0.01, meta_lr=0.001)
        maml_path = "models/maml_agent.pth"
        if os.path.exists(maml_path):
            self.maml_agent.load(maml_path)
            logger.info("MAML agent loaded from checkpoint")
        logger.info("[OK] Meta-Learning Agent initialized")

        # 7. Execution Algorithm Library
        self.algo_executor = AlgoExecutor(self.ctrader, self.data_manager, self.cost_model)
        logger.info("[OK] Execution Algorithm Library initialized")

        # 8. Event-Driven Architecture
        self.event_bus = get_event_bus()
        # Subscribe to key events (sync operation, safe to do here)
        self.event_bus.subscribe(EventType.TOXIC_FLOW, self._on_toxic_flow)
        self.event_bus.subscribe(EventType.CIRCUIT_BREAKER, self._on_circuit_breaker)
        self.event_bus.subscribe(EventType.MARKET_STRESS, self._on_market_stress)
        self.event_bus.subscribe(EventType.RISK_ALERT, self._on_risk_alert)
        logger.info("[OK] Event-Driven Architecture initialized")

        # 9. Model Registry & A/B Testing
        self.model_registry = ModelRegistry(registry_path="models/registry")
        logger.info("[OK] Model Registry initialized")

        # 10. Multi-Timeframe Attention Fusion
        tf_dims = {tf: 55 for tf in self.feature_pipeline.timeframes} if self.feature_pipeline else {"1h": 55}
        self.attention_fusion = AttentionFusionPipeline(
            timeframes=list(tf_dims.keys()),
            lookback=30,
        )
        if self.feature_pipeline:
            self.attention_fusion.init_model(tf_dims, hidden_dim=256)
        logger.info("[OK] Multi-Timeframe Attention Fusion initialized")

        # 11. Social Media / Satellite / News / Behavioral Sentiment
        self.behavioral_ai = BehavioralSentimentAI(
            use_transformers=True,
            cache_ttl=300,
            max_posts=200,
            satellite_enabled=True,
            onchain_enabled=True,
        )
        logger.info("[OK] Behavioral Sentiment AI initialized")

        logger.info("=" * 60)
        logger.info("  All 11 Enhancement Modules Active")
        logger.info("=" * 60)

    async def _on_toxic_flow(self, event):
        """Handle toxic flow detection event."""
        snapshot: ToxicFlowSnapshot = event.data
        if snapshot.is_toxic:
            logger.warning(
                f"TOXIC FLOW: VPIN={snapshot.vpin:.2f} | "
                f"Level={snapshot.toxicity_level} | "
                f"Direction={snapshot.informed_direction}"
            )
            self.notifier.send(
                f"Toxic flow detected! VPIN={snapshot.vpin:.2f}",
                level="warning",
            )

    async def _on_circuit_breaker(self, event):
        """Handle circuit breaker event."""
        snapshot: MarketStressSnapshot = event.data
        if snapshot.should_halt:
            logger.error(f"CIRCUIT BREAKER: {snapshot.halt_reason}")
            self.notifier.send(
                f"Trading halted: {snapshot.halt_reason}",
                level="error",
            )

    async def _on_market_stress(self, event):
        """Handle market stress event."""
        snapshot: MarketStressSnapshot = event.data
        if not snapshot.is_healthy:
            logger.warning(f"Market stress: {snapshot.stress_level}")

    async def _on_risk_alert(self, event):
        """Handle risk alert event."""
        alert_data = event.data
        logger.warning(f"RISK ALERT: {alert_data}")
        self.notifier.risk_warning(
            "Risk alert triggered",
            details=alert_data,
        )

    # ------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------

    def _init_ctrader(self):
        self.execution, self.data_provider = create_execution_provider(self.secrets)
        self.execution.on_price = self._on_market_data
        self.ctrader = getattr(self.execution, 'raw', self.execution)
        # Use Dukascopy for real-time data if available
        if self.data_provider:
            logger.info(f"Data provider: {self.data_provider.__class__.__name__}")

    def _init_risk(self):
        params = RiskParameters(
            max_risk_per_trade=self.config.trading.max_risk_per_trade,
            max_drawdown=self.config.trading.max_drawdown,
            max_margin_usage=self.config.trading.max_margin_usage,
        )
        self.risk = RiskManager(params, initial_balance=self.initial_balance)
        self.risk.mode = "PAPER" if self.mode == "paper" else "LIVE"
        self.trailing_stop = TrailingStopManager(
            tp_pcts=[0.01, 0.02, 0.03],
            trail_atr_mult=self.config.trading.sl_atr_multiplier,
        )
        self.cost_model = CostModel(commission_per_lot=self.config.trading.commission_per_lot)
        # ExecutionEngine wraps the execution provider with SL/TP monitoring
        self.execution = ExecutionEngine(self.ctrader, self.risk, self.data_manager)
        self.execution.cost_model = self.cost_model

    def _init_data(self):
        self.data_manager = DataManager(
            historical_path=self.config.data.historical_path,
        )
        self.feature_pipeline = FeaturePipeline(
            lookback=30, timeframes=["1h"],
            use_microstructure=True, use_cross_asset=False,
        )
        self.regime_detector = HMMRegimeDetector(n_regimes=4, lookback=60)

    def _init_models(self):
        self.ensemble = MoEEnsemble()
        # Rule-based expert (baseline)
        self.ensemble.add_expert(
            name="rule_based",
            predict_fn=self._rule_prediction,
            confidence_fn=lambda X: 0.7,
            regime="ranging",
        )
        # PPO RL Agent expert
        try:
            from ai.rl_agent import PPOAgent
            state_dim = 55 * 30  # 55 features * 30 lookback
            self.ppo_agent = PPOAgent(state_dim=state_dim, n_actions=5)
            ppo_path = "models/ppo_agent.pth"
            if os.path.exists(ppo_path):
                self.ppo_agent.load(ppo_path)
                logger.info("PPO Agent loaded from checkpoint")
            self.ensemble.add_expert(
                name="ppo_agent",
                predict_fn=self._ppo_prediction,
                confidence_fn=self._ppo_confidence,
                regime="trending",
            )
        except Exception as e:
            logger.warning(f"PPO Agent init failed: {e}")
            self.ppo_agent = None
        # LSTM-CNN expert
        try:
            from rts_ai_fx.model import LSTMCNNHybrid
            self.lstm_model = LSTMCNNHybrid(lookback=30, n_features=55)
            lstm_path = "models/lstm_cnn.h5"
            if os.path.exists(lstm_path):
                self.lstm_model.load(lstm_path)
                logger.info("LSTM-CNN model loaded")
            self.ensemble.add_expert(
                name="lstm_cnn",
                predict_fn=self._lstm_prediction,
                confidence_fn=self._lstm_confidence,
                regime="ranging",
            )
        except Exception as e:
            logger.warning(f"LSTM-CNN init failed: {e}")
            self.lstm_model = None
        self.active_models: Dict[str, Any] = {}
        self.drift_monitors: Dict[str, DriftMonitor] = {}

    def _rule_prediction(self, X: np.ndarray) -> float:
        try:
            last = X[-1] if X.ndim == 2 else X
            mom = float(last[-5]) if len(last) > 5 and not np.isnan(last[-5]) else 0.0
            return 1.12 + mom * 0.001
        except Exception:
            return 1.12

    def _ppo_prediction(self, X: np.ndarray, symbol: str = "EURUSD") -> float:
        """PPO agent prediction - returns predicted price for given symbol."""
        if self.ppo_agent is None:
            return self.data_manager.get_price(symbol, "1h") or 1.12
        try:
            state = X.flatten() if X.ndim > 1 else X
            action, sl_raw, tp_raw, size_raw, info = self.ppo_agent.select_action(state)
            # Action mapping: 0=HOLD, 1=BUY, 2=SELL, 3=CLOSE, 4=MODIFY
            price = self.data_manager.get_price(symbol, "1h")
            if price is None:
                return 1.12
            if action == 1:  # BUY
                return price * (1 + 0.001 * sl_raw)
            elif action == 2:  # SELL
                return price * (1 - 0.001 * sl_raw)
            return price
        except Exception as e:
            logger.debug(f"PPO prediction error for {symbol}: {e}")
            return 1.12

    def _ppo_confidence(self, X: np.ndarray) -> float:
        """PPO agent confidence based on action probabilities."""
        if self.ppo_agent is None:
            return 0.5
        try:
            import torch
            state = X.flatten() if X.ndim > 1 else X
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.ppo_agent.device)
            with torch.no_grad():
                act_logits, _, _, value = self.ppo_agent.actor(state_t)
                probs = torch.softmax(act_logits, dim=1)
                confidence = probs.max().item()
            return float(confidence)
        except Exception:
            return 0.5

    def _lstm_prediction(self, X: np.ndarray) -> float:
        """LSTM-CNN model prediction."""
        if self.lstm_model is None or self.lstm_model.model is None:
            return 1.12
        try:
            pred = self.lstm_model.predict(X.reshape(1, 30, 55))
            return float(pred[0, 0])
        except Exception as e:
            logger.debug(f"LSTM prediction error: {e}")
            return 1.12

    def _lstm_confidence(self, X: np.ndarray) -> float:
        """LSTM-CNN confidence (based on prediction magnitude)."""
        if self.lstm_model is None:
            return 0.5
        try:
            pred = self.lstm_model.predict(X.reshape(1, 30, 55))
            conf = min(abs(pred[0, 0] - 1.12) * 100, 1.0)
            return float(conf)
        except Exception:
            return 0.5

    def _init_intelligence(self):
        self.economic_calendar = EconomicCalendar(
            cache_path="data/alternative_data/economic_calendar.json",
        )
        self.sentiment_analyzer = SentimentAnalyzer(use_finbert=True, cache_ttl=300)
        self.alternative_data = AlternativeDataProvider(
            fred_api_key=self.secrets.fred_api_key if hasattr(self.secrets, 'fred_api_key') else "",
            cache_dir="data/alternative_data",
        )
        self.online_learner = OnlineLearner(
            model_dir="models",
            retrain_cooldown_hours=4.0,
            min_trades_before_retrain=50,
            lstm_epochs=30,
            clf_epochs=20,
            notify_callback=lambda msg: self.notifier.send(msg, level="info"),
        )
        # Institutional features
        self.order_flow_analyzer = OrderFlowAnalyzer(n_levels=20)
        self.gamma_mapper = GammaExposureMapper()
        self.dark_pool_detector = DarkPoolDetector()
        logger.info("Intelligence layer initialized (including institutional features)")

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

    async def _start_dukascopy_stream(self):
        """Start real-time Dukascopy tick streaming."""
        if not self.data_provider:
            return
        try:
            await self.data_provider.stream_prices(SYMBOLS, self._on_dukascopy_tick)
            logger.info(f"Dukascopy streaming started for {len(SYMBOLS)} symbols")
        except NotImplementedError:
            logger.warning("Dukascopy streaming not available, using polling")
            # Fallback to polling - will be started in async start() method
            # asyncio.create_task(self._poll_dukascopy_prices())  # MOVED to start()

    async def _poll_dukascopy_prices(self):
        """Poll Dukascopy for latest prices."""
        while self.is_running:
            for sym in SYMBOLS:
                try:
                    tick = self.data_provider.get_latest_price(sym)
                    if tick:
                        self._on_dukascopy_tick(tick)
                except Exception as e:
                    logger.debug(f"Poll error for {sym}: {e}")
            await asyncio.sleep(1.0)

    def _on_dukascopy_tick(self, tick: PriceTick):
        """Handle incoming Dukascopy tick data."""
        self.data_manager.update_tick(tick.symbol, tick.bid, tick.ask, tick.volume, tick.timestamp)
        # Forward to execution engine if needed
        if hasattr(self.execution, 'on_market_data'):
            self.execution.on_market_data(tick)

    async def start(self):
        logger.info("Starting RTS Forex Bot...")
        
        # Start event bus now that we have an event loop
        if hasattr(self, 'event_bus') and self.event_bus:
            await self.event_bus.start()
            logger.info("[OK] Event-Driven Architecture started")
        
        result = await self.ctrader.start()
        if not result:
            logger.warning("cTrader in simulation mode -- no real connection")

        # Subscribe to Level II DOM for all symbols (institutional feature)
        # Try Open API first, fall back to FIX if available
        if hasattr(self.ctrader, 'subscribe_depth'):
            # Open API client
            for sym in SYMBOLS:
                from api.ctrader_client import SYMBOL_MAP
                sym_id = SYMBOL_MAP.get(sym)
                if sym_id:
                    await self.ctrader.subscribe_depth(sym_id)
            self.ctrader.on_depth_update = self._on_depth_update
            logger.info(f"Level II DOM (Open API) subscribed for {len(SYMBOLS)} symbols")
        elif hasattr(self.ctrader, 'raw') and hasattr(self.ctrader.raw, 'on_market_data'):
            # FIX client - wire market data to DataManager
            self.ctrader.raw.on_market_data = self._on_fix_market_data
            # Subscribe to market data
            if hasattr(self.ctrader, 'start'):
                await self.ctrader.start()
            logger.info(f"Level II DOM (FIX API) wired for {len(SYMBOLS)} symbols")

        # Load historical data for ALL symbols from Dukascopy
        await self._load_historical_data()

        # Fit regime detector per symbol + init drift monitors
        for sym in SYMBOLS:
            df = self.data_manager.get_ohlcv(sym, "1h")
            if df is not None and len(df) > 200:
                rd = HMMRegimeDetector(n_regimes=4, lookback=60)
                rd.fit(df)
                self._regimes[sym] = rd.detect_regime(df)
                logger.info(f"HMM regime detector fitted for {sym}")
                # Fit shared regime detector on first available symbol
                if not hasattr(self.regime_detector, 'model') or self.regime_detector.model is None:
                    self.regime_detector.fit(df)
            # Initialize drift monitor for online learning
            self.active_models[sym] = {
                "drift": DriftMonitor(error_threshold=0.02),
            }

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

        # Start Dukascopy real-time streaming
        await self._start_dukascopy_stream()

        # Fallback: Also try cTrader streaming if available
        if hasattr(self.execution, 'stream_prices'):
            try:
                await self.execution.stream_prices(SYMBOLS)
            except NotImplementedError:
                pass

        self._start_dashboard()
        self.is_running = True
        logger.info("Bot is LIVE -- monitoring 7 pairs with Dukascopy data")

        cycle_counter = 0
        last_sentiment_refresh = 0.0
        last_calendar_fetch = 0.0
        last_summary_day = pd.Timestamp.now().day
        last_data_download = 0.0
        download_queue = [s for s in SYMBOLS if self.data_manager.get_ohlcv(s, "1h") is None or len(self.data_manager.get_ohlcv(s, "1h")) < 50]
        if download_queue:
            logger.info(f"Background downloader: {len(download_queue)} symbols queued")

        while self.is_running:
            try:
                cycle_start = time.time()
                
                await self._trading_cycle()
                
                # Adaptive sleep to maintain ~1 second cycles (with jitter protection)
                cycle_time = time.time() - cycle_start
                sleep_time = max(1.0 - cycle_time, 0.5)  # Min 0.5s to avoid hammering
                await asyncio.sleep(sleep_time)
                cycle_counter += 1

                # Periodic refresh (time-based)
                now = time.time()
                if now - last_sentiment_refresh > 300:  # Every 5 minutes
                    try:
                        snapshot = self.sentiment_analyzer.get_latest(force_refresh=True)
                        self._current_sentiment = snapshot.overall_score if hasattr(snapshot, 'overall_score') else float(snapshot)
                        last_sentiment_refresh = now
                    except Exception:
                        pass
                if now - last_calendar_fetch > 3600:  # Every hour
                    try:
                        self.economic_calendar.fetch(days_forward=7)
                        self.alternative_data.fetch_all()
                        last_calendar_fetch = now
                    except Exception:
                        pass

                # Daily summary (send once per day)
                today = pd.Timestamp.now().day
                if today != last_summary_day:
                    last_summary_day = today
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

                # Background data downloader -- download one missing symbol per cycle
                if download_queue and time.time() - last_data_download > 30:
                    sym = download_queue[0]
                    logger.info(f"Background downloading {sym} from Dukascopy...")
                    last_data_download = time.time()
                    success = False
                    try:
                        from data.dukascopy_realtime import DukascopyProvider, CACHE_DIR
                        from datetime import datetime, timezone, timedelta

                        # Check for cached data first
                        cache_files = sorted(CACHE_DIR.glob(f"{sym}_*_*.bi5"))
                        if cache_files:
                            import lzma, struct, pandas as pd
                            ticks = []
                            for cf in cache_files[-720:]:
                                raw = cf.read_bytes()
                                try: decompressed = lzma.decompress(raw)
                                except: decompressed = raw
                                parts = cf.stem.split("_")
                                dt_str, hour_str = parts[-2], parts[-1]
                                base_ts = datetime.strptime(dt_str, "%Y%m%d").timestamp() + int(hour_str) * 3600
                                for i in range(0, len(decompressed), 20):
                                    chunk = decompressed[i:i+20]
                                    if len(chunk) < 20: break
                                    ms = struct.unpack(">I", chunk[0:4])[0]
                                    ask_raw = struct.unpack(">I", chunk[4:8])[0]
                                    bid_raw = struct.unpack(">I", chunk[8:12])[0]
                                    ticks.append((base_ts + ms/1000.0, bid_raw/100000.0, ask_raw/100000.0))
                            if len(ticks) > 50:
                                ticks.sort(key=lambda t: t[0])
                                df = pd.DataFrame(ticks, columns=["timestamp","bid","ask"])
                                df["bar"] = (df["timestamp"] // 3600).astype(int)
                                bars = df.groupby("bar").agg(open=("bid","first"),high=("bid","max"),
                                    low=("bid","min"),close=("bid","last"),volume=("ask","count")).reset_index()
                                bars["timestamp"] = bars["bar"] * 3600
                                bars = bars.drop(columns=["bar"])
                                self.data_manager.ohlcv[sym]["1h"] = bars
                                if len(bars) > 200:
                                    rd = HMMRegimeDetector(n_regimes=4, lookback=60)
                                    rd.fit(bars)
                                    self._regimes[sym] = rd.detect_regime(bars)
                                self.feature_pipeline.fit_all(self.data_manager.ohlcv)
                                logger.info(f"... Background loaded {sym}: {len(bars)} bars (cached)")
                                success = True
                        else:
                            # Download from network (end=yesterday avoids current-day timeouts)
                            provider = DukascopyProvider(cache=True)
                            end = datetime.now(timezone.utc) - timedelta(days=1)
                            start = end - timedelta(days=30)
                            ohlcv = await provider.fetch_ohlcv(sym, "1h",
                                start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
                            await provider.close()
                            if ohlcv and len(ohlcv) > 50:
                                df = pd.DataFrame([{
                                    "timestamp": o.timestamp, "open": o.open, "high": o.high,
                                    "low": o.low, "close": o.close, "volume": o.volume,
                                } for o in ohlcv])
                                self.data_manager.ohlcv[sym]["1h"] = df
                                if len(df) > 200:
                                    rd = HMMRegimeDetector(n_regimes=4, lookback=60)
                                    rd.fit(df)
                                    self._regimes[sym] = rd.detect_regime(df)
                                self.feature_pipeline.fit_all(self.data_manager.ohlcv)
                                logger.info(f"... Background download complete for {sym}: {len(df)} bars")
                                success = True
                    except Exception as e:
                        logger.warning(f"Background download failed for {sym}: {e}")
                    
                    if success:
                        download_queue.pop(0)
                        self._send_alert(f"Data loaded for {sym}", level="info")
                    if not download_queue:
                        logger.info("... Background downloader: all symbols complete")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cycle error: {e}")
                await asyncio.sleep(5.0)

    async def _load_historical_data(self):
        loaded = 0
        from pathlib import Path
        from data.dukascopy_realtime import CACHE_DIR
        import lzma, struct
        from datetime import datetime

        logger.info("Loading cached historical data from Dukascopy...")

        for sym in SYMBOLS:
            if self.data_manager.get_ohlcv(sym, "1h") is not None and len(self.data_manager.get_ohlcv(sym, "1h")) > 200:
                loaded += 1
                continue

            cache_files = sorted(CACHE_DIR.glob(f"{sym}_*_*.bi5"))
            if not cache_files:
                logger.debug(f"No cached data for {sym}")
                continue

            ticks = []
            for cf in cache_files[-168:]:
                try:
                    raw = cf.read_bytes()
                    decompressed = lzma.decompress(raw)
                except Exception:
                    decompressed = raw
                parts = cf.stem.split("_")
                dt_str, hour_str = parts[-2], parts[-1]
                base_ts = datetime.strptime(dt_str, "%Y%m%d").timestamp() + int(hour_str) * 3600
                for i in range(0, len(decompressed), 20):
                    chunk = decompressed[i:i+20]
                    if len(chunk) < 20:
                        break
                    ms = struct.unpack(">I", chunk[0:4])[0]
                    ask_raw = struct.unpack(">I", chunk[4:8])[0]
                    bid_raw = struct.unpack(">I", chunk[8:12])[0]
                    ticks.append((base_ts + ms/1000.0, bid_raw/100000.0, ask_raw/100000.0))

            if len(ticks) < 10:
                logger.debug(f"Too few cached ticks for {sym}")
                continue

            ticks.sort(key=lambda t: t[0])
            df = pd.DataFrame(ticks, columns=["timestamp", "bid", "ask"])
            df["bar"] = (df["timestamp"] // 3600).astype(int)
            bars = df.groupby("bar").agg(
                open=("bid", "first"), high=("bid", "max"), low=("bid", "min"),
                close=("bid", "last"), volume=("ask", "count")
            ).reset_index()
            bars["timestamp"] = bars["bar"] * 3600
            bars = bars.drop(columns=["bar"])
            self.data_manager.ohlcv[sym]["1h"] = bars
            loaded += 1
            logger.info(f"Loaded {len(bars)} 1h bars for {sym} (cached)")

        if loaded < len(SYMBOLS):
            logger.warning(f"Only {loaded}/{len(SYMBOLS)} pairs in cache. Background downloader will fetch remaining.")
        else:
            logger.info(f"[OK] All {loaded} pairs loaded from Dukascopy cache")

    def _start_dashboard(self):
        """Dashboard disabled for now - use log files for monitoring."""
        d = self.config.dashboard
        logger.info(f"Dashboard disabled - monitor via logs at data/logs/moneybot.log")
        # import threading
        # import uvicorn
        # d = self.config.dashboard
        # self._dashboard_loop = None
        # 
        # def run_dashboard():
        #     loop = asyncio.new_event_loop()
        #     asyncio.set_event_loop(loop)
        #     self._dashboard_loop = loop
        #     uvicorn.run(app, host=d.host, port=d.port, log_level="error")
        # 
        # t = threading.Thread(target=run_dashboard, daemon=True)
        # t.start()
        # 
        # import time
        # for _ in range(50):
        #     if self._dashboard_loop:
        #         break
        #     time.sleep(0.1)
        # 
        # if self._dashboard_loop:
        #     logger.info(f"Dashboard: http://{d.host}:{d.port}")
        # else:
        #     logger.warning(f"Dashboard may have failed to start on {d.host}:{d.port}")

    # ================================================================
    # TRADING CYCLE -- iterates over all 7 pairs
    # ================================================================

    async def _trading_cycle(self):
        # 0. Market Session & Liquidity Check
        if MarketSession.is_weekend():
            logger.info("Weekend -- market closed")
            self._update_dashboard({"balance": 100000}, "CLOSED")
            await asyncio.sleep(3600)  # Sleep 1 hour on weekends
            return

        if not MarketSession.is_market_open():
            logger.info("Market closed (outside 24/5 window)")
            self._update_dashboard({"balance": 100000}, "CLOSED")
            await asyncio.sleep(1800)  # Sleep 30 min
            return

        should_pause, pause_reason = MarketSession.should_pause_trading()
        if should_pause:
            logger.info(f"Trading paused: {pause_reason}")
            self._update_dashboard({"balance": 100000}, "PAUSED")
            # Still check existing positions even if not trading new
            await self._manage_positions()
            return

        # Log active sessions
        active_sessions = MarketSession.get_active_sessions()
        is_liquid, liquidity_reason = MarketSession.is_high_liquidity()
        if not is_liquid:
            logger.debug(f"Low liquidity: {liquidity_reason}")

        # 0.5 NEW: Circuit Breaker Check (Enhancement #5)
        for sym in SYMBOLS:
            tick = self.data_manager.latest_snapshot.get(sym)
            if tick:
                should_halt, reason, snapshot = self.circuit_breakers[sym].check_market_health(
                    sym, {"bid": getattr(tick, 'bid', 0), "ask": getattr(tick, 'ask', 0),
                           "price": getattr(tick, 'bid', 0), "volume": getattr(tick, 'volume', 0)}
                )
                if should_halt:
                    await self.event_bus.emit(
                        EventType.CIRCUIT_BREAKER,
                        {"symbol": sym, "reason": reason, "snapshot": snapshot},
                        source="circuit_breaker"
                    )
                    if sym in self.circuit_breakers:
                        logger.error(f"CIRCUIT BREAKER: {sym} trading halted: {reason}")
                    continue

        # 1. Economic Calendar Gate (global)
        suppressed, event = self.economic_calendar.is_suppressed()
        if suppressed and event:
            logger.info(f"Trading suppressed: {event.title}")
            self._update_dashboard({"balance": 100000}, "SUPPRESSED")
            # Still manage existing positions
            await self._manage_positions()
            return

        # 2. Gather global intelligence (Enhanced with Behavioral AI #11)
        external_signals = self._gather_external_signals()
        
        # NEW: Get behavioral sentiment snapshot
        try:
            behavioral_snap = self.behavioral_ai.analyze_social_media()
            self._behavioral_sentiment = behavioral_snap.overall_score
            logger.debug(f"Behavioral sentiment: {behavioral_snap.overall_score:.2f} (fear/greed={behavioral_snap.fear_greed_index:.0f})")
        except Exception:
            self._behavioral_sentiment = 0.0

        # 3. Evaluate each symbol (Enhanced with Regime Specialists #2 & Attention #10)
        self._trade_decisions = {}
        account_info = await self.execution.get_account_info()
        if account_info is None:
            return

        for sym in SYMBOLS:
            decision = await self._evaluate_symbol_enhanced(sym, account_info, external_signals)
            if decision:
                # Add spread check
                spread_pips = self._get_current_spread(sym)
                decision["spread_pips"] = spread_pips
                decision["is_acceptable_spread"] = self.cost_model.get_spread_warning_level(sym, spread_pips) == "normal"
                
                # NEW: Toxic flow check (Enhancement #1)
                if sym in self.toxic_detectors:
                    toxic_snap = self.toxic_detector.update(
                        {"price": self.data_manager.get_price(sym, "1h") or 1.12,
                         "mid": self.data_manager.get_price(sym, "1h") or 1.12,
                         "volume": 1}
                    )
                    decision["toxic_flow"] = toxic_snap.is_toxic
                    decision["toxic_vpin"] = toxic_snap.vpin
                    if toxic_snap.is_toxic:
                        decision["approved"] = False
                        decision["rejection_reason"] = f"toxic_flow_vpin={toxic_snap.vpin:.2f}"
                        await self.event_bus.emit(
                            EventType.TOXIC_FLOW,
                            {"symbol": sym, "snapshot": toxic_snap},
                            source="toxic_detector"
                        )
                
                # NEW: Smart money check (Enhancement #4)
                if decision.get("approved") and sym in self.cot_data:
                    cot_snap = self.cot_analyzer.fetch_latest(sym)
                    should_block, reason = self.cot_analyzer.get_trading_signal(
                        sym, decision.get("direction", "BUY")
                    )
                    if should_block:
                        decision["approved"] = False
                        decision["rejection_reason"] = reason
                        logger.info(f"{sym} trade blocked: {reason}")
                
                self._trade_decisions[sym] = decision

        # 4. Apply correlation filter -- prevent opposing trades on correlated pairs
        self._apply_correlation_filter()

        # 5. Execute approved trades (with spread verification)
        for sym, decision in self._trade_decisions.items():
            if decision.get("approved") and not decision.get("blocked_by_correlation"):
                # Double-check spread before execution
                if not decision.get("is_acceptable_spread", True):
                    logger.warning(f"{sym} trade blocked: spread too wide ({decision.get('spread_pips', 0):.1f} pips)")
                    continue
                await self._execute_trade_enhanced(sym, decision, account_info)

        # 6. Manage existing positions -- check SL/TP against current prices
        await self._manage_positions()

        # 7. Check if retraining is needed for any pair
        for sym in SYMBOLS:
            if self.online_learner.should_retrain(sym, self.risk.total_trades):
                def fetch_fn(p):
                    df = self.data_manager.get_ohlcv(p, "1h")
                    if df is not None and len(df) > 200:
                        return df
                    return None
                self.online_learner.request_retrain(sym, fetch_fn, self.feature_pipeline)

        # 8. Update dashboard (Enhanced with new metrics)
        self._update_dashboard_enhanced(account_info, "trading")

    async def _evaluate_symbol_enhanced(
        self, symbol: str, account_info: dict, external_signals: np.ndarray,
    ) -> Optional[dict]:
        """Enhanced symbol evaluation with all 11 modules."""
        # Get OHLCV for this symbol
        ohlcv = self.data_manager.get_ohlcv_dict(symbol)
        price = self.data_manager.get_price(symbol, "1h")
        atr = self.data_manager.get_atr(symbol, "1h", 14)
        df_1h = self.data_manager.get_ohlcv(symbol, "1h")

        if df_1h is None or len(df_1h) < 60:
            return None

        # Detect regime (Enhanced with Regime Specialist #2)
        if symbol not in self._regimes or self.regime_detector is None:
            rd = HMMRegimeDetector(n_regimes=4, lookback=60)
            rd.fit(df_1h)
            regime = rd.detect_regime(df_1h)
            self._regimes[symbol] = regime
        else:
            regime = self.regime_detector.detect_regime(df_1h)
            self._regimes[symbol] = regime

        if not HMMRegimeDetector.should_trade_static(regime):
            return None

        # Feature extraction (Enhanced with Causal Features #3 & Attention Fusion #10)
        tick_buffer = self.data_manager.get_tick_buffer(symbol, 1000)
        
        # Use causal feature selection if available
        features_df = None
        if self.causal_selector and len(df_1h) > 50:
            try:
                # Prepare target for causal analysis
                if "close" in df_1h.columns and len(df_1h) > 10:
                    target = df_1h["close"].pct_change().shift(-1).dropna()
                    if len(target) > 50:
                        features_processed = compute_features(df_1h.iloc[:-1].copy())
                        causal_features = self.causal_selector.fit_transform(
                            features_processed, target
                        )
                        features_df = causal_features
            except Exception:
                pass
        
        # Transform features
        if features_df is None:
            features = self.feature_pipeline.transform(
                self.data_manager.ohlcv, symbol=symbol,
                tick_buffer=tick_buffer,
                external_signals=external_signals,
            )
        else:
            # Use attention fusion if available
            if self.attention_fusion:
                try:
                    tf_data = {}
                    for tf in self.feature_pipeline.timeframes:
                        df_tf = self.data_manager.get_ohlcv(symbol, tf)
                        if df_tf is not None and len(df_tf) >= 30:
                            processed = compute_features(df_tf.copy())
                            tf_data[tf] = processed.values[-30:]
                    fused, attn_info = self.attention_fusion.fuse(tf_data)
                    features = fused
                except Exception:
                    features = None
            else:
                features = None

        if features is None:
            return None

        # Ensemble inference (Enhanced with Regime Specialist #2)
        # Use regime specialist system instead of basic ensemble
        if hasattr(self, 'regime_system') and self.regime_system:
            try:
                state = features.flatten() if features.ndim > 1 else features
                action, sl_raw, tp_raw, size_raw, info = self.regime_system.select_action(
                    state, regime
                )
                # Convert action to prediction
                pred_price = price * (1 + 0.001 * sl_raw) if action == 1 else price * (1 - 0.001 * sl_raw)
                confidence = info.get("confidence", 0.5)
                direction = "BUY" if action in [1, 3] else "SELL" if action in [2, 4] else "HOLD"
                should_trade = direction != "HOLD"
            except Exception as e:
                logger.debug(f"Regime specialist error: {e}")
                # Fallback to basic ensemble
                ensemble_pred = self.ensemble.predict(features, regime=regime)
                confidence = ensemble_pred.confidence
                should_trade, direction, agreement = self.ensemble.should_trade(
                    ensemble_pred, price, min_confidence=0.65,
                )
                pred_price = ensemble_pred.price
        else:
            # Basic ensemble (original)
            ensemble_pred = self.ensemble.predict(features, regime=regime)
            confidence = ensemble_pred.confidence
            should_trade, direction, agreement = self.ensemble.should_trade(
                ensemble_pred, price, min_confidence=0.65,
            )
            pred_price = ensemble_pred.price

        # Sentiment-adjusted confidence (Enhanced with Behavioral AI #11)
        if should_trade and abs(self._behavioral_sentiment) > 0.3:
            sent_direction = 1 if self._behavioral_sentiment > 0 else 0
            if sent_direction == (1 if direction == "BUY" else 0):
                confidence *= 1.1
            else:
                confidence *= 0.9
            confidence = min(max(confidence, 0.0), 1.0)

        if not should_trade:
            return None

        # Regime params (Enhanced with Regime Specialist #2)
        if hasattr(self, 'regime_system') and self.regime_system:
            regime_params = self.regime_system.get_regime_params(regime)
        else:
            regime_params = HMMRegimeDetector.get_regime_params_static(regime)
        
        adjusted_conf = confidence * regime_params["pos_mult"]

        if self._current_sentiment < -0.5:
            adjusted_conf *= 0.5

        # Kelly sizing
        volume = self.risk.calculate_kelly_size(
            account_info.get("balance", 100_000), price, atr, adjusted_conf,
        )
        volume = max(volume, 1)  # Minimum 1 unit

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
            "model_prediction": pred_price,
            # NEW: Enhanced fields
            "behavioral_sentiment": self._behavioral_sentiment,
            "toxic_vpin": self._toxic_flows.get(symbol, ToxicFlowSnapshot()).vpin,
            "cot_signal": self._cot_data.get(symbol, COTSnapshot()).institutional_signal if hasattr(self, '_cot_data') else "neutral",
        }

    async def _execute_trade_enhanced(self, symbol: str, decision: dict, account_info: dict):
        """Enhanced trade execution with Execution Algorithms #7."""
        direction = decision["direction"]
        volume = decision["volume"]
        price = decision["price"]
        atr = decision["atr"]
        regime_params = decision["regime_params"]
        confidence = decision["confidence"]
        regime = decision["regime"]
        spread_pips = decision.get("spread_pips", 1.0)

        # Final spread check at execution time
        current_spread = self._get_current_spread(symbol)
        if current_spread > spread_pips * 3:
            logger.warning(f"{symbol} trade aborted: spread widened to {current_spread:.1f} pips")
            return

        # Liquidity check
        active_sessions = MarketSession.get_active_sessions()
        if not active_sessions:
            logger.info(f"{symbol} trade delayed: no active trading session")
            return

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
            await self.event_bus.emit(
                EventType.RISK_ALERT,
                {"symbol": symbol, "reason": reason},
                source="risk_manager"
            )
            return

        # Apply cost model
        cost = self.cost_model.calculate(
            symbol=symbol,
            direction=direction,
            volume=volume,
            price=price,
            atr=atr,
            actual_spread_pips=current_spread,
        )
        if not cost.is_acceptable:
            logger.warning(f"{symbol} trade blocked: {cost.rejection_reason}")
            return

        # Place order with ATR-based SL/TP (direction-aware!)
        if direction == "BUY":
            sl_price = price - atr * regime_params.get("sl_atr", 1.5)
            tp_price = price + atr * regime_params.get("tp_atr", 3.0)
        else:  # SELL -- SL above entry, TP below entry
            sl_price = price + atr * regime_params.get("sl_atr", 1.5)
            tp_price = price - atr * regime_params.get("tp_atr", 3.0)

        # NEW: Use Execution Algorithms for large orders (Enhancement #7)
        if volume > 100000 and hasattr(self, 'algo_executor'):  # Large order
            logger.info(f"Using TWAP for large order: {volume:.0f} {symbol}")
            await self.algo_executor.twap_execution(
                symbol=symbol,
                side=direction,
                volume=volume,
                duration_minutes=30,
            )
            trade = {"symbol": symbol, "direction": direction, "volume": volume}
        else:
            trade = await self.execution.open_position(
                symbol=symbol,
                direction=direction,
                volume=volume,
                sl=sl_price,
                tp=tp_price,
                reason=f"regime={regime} conf={confidence:.2f} sent={self._current_sentiment:.2f} spread={current_spread:.1f} behavioral={decision.get('behavioral_sentiment', 0):.2f}",
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

        # Update drift monitor for this symbol with (model_prediction, actual_price)
        model_prediction = decision.get("model_prediction")
        if model_prediction and model_prediction > 0:
            model_data = self.active_models.get(symbol)
            if model_data and "drift" in model_data:
                drifted = model_data["drift"].update(model_prediction, price)
                if drifted:
                    logger.info(f"Drift detected for {symbol}: error={model_data['drift'].error_detector.mean:.4f}")
                    self.online_learner.on_drift_detected(symbol, model_data["drift"].error_detector.drift_count)

    def _get_current_spread(self, symbol: str) -> float:
        """Get current spread in pips for a symbol."""
        try:
            tick = self.data_manager.latest_snapshot.get(symbol)
            if tick and hasattr(tick, 'bid') and hasattr(tick, 'ask'):
                pip_size = 0.0001 if "JPY" not in symbol.upper() else 0.01
                spread = (tick.ask - tick.bid) / pip_size
                return float(spread)
            # Fallback to CostModel
            return self.cost_model.pip_to_price(symbol) * 10000  # Rough estimate
        except Exception:
            return 1.0

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
        volume = max(volume, 1)  # Minimum 1 unit

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
            "model_prediction": ensemble_pred.price,
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
        spread_pips = decision.get("spread_pips", 1.0)

        # Final spread check at execution time
        current_spread = self._get_current_spread(symbol)
        if current_spread > spread_pips * 3:
            logger.warning(f"{symbol} trade aborted: spread widened to {current_spread:.1f} pips")
            return

        # Liquidity check
        active_sessions = MarketSession.get_active_sessions()
        if not active_sessions:
            logger.info(f"{symbol} trade delayed: no active trading session")
            return

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

        # Apply cost model
        cost = self.cost_model.calculate(
            symbol=symbol,
            direction=direction,
            volume=volume,
            price=price,
            atr=atr,
            actual_spread_pips=current_spread,
        )
        if not cost.is_acceptable:
            logger.warning(f"{symbol} trade blocked: {cost.rejection_reason}")
            return

        # Place order with ATR-based SL/TP (direction-aware!)
        if direction == "BUY":
            sl_price = price - atr * regime_params.get("sl_atr", 1.5)
            tp_price = price + atr * regime_params.get("tp_atr", 3.0)
        else:  # SELL -- SL above entry, TP below entry
            sl_price = price + atr * regime_params.get("sl_atr", 1.5)
            tp_price = price - atr * regime_params.get("tp_atr", 3.0)
        trade = await self.execution.open_position(
            symbol=symbol,
            direction=direction,
            volume=volume,
            sl=sl_price,
            tp=tp_price,
            reason=f"regime={regime} conf={confidence:.2f} sent={self._current_sentiment:.2f} spread={current_spread:.1f}",
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

        # Update drift monitor for this symbol with (model_prediction, actual_price)
        model_prediction = decision.get("model_prediction")
        if model_prediction and model_prediction > 0:
            model_data = self.active_models.get(symbol)
            if model_data and "drift" in model_data:
                drifted = model_data["drift"].update(model_prediction, price)
                if drifted:
                    logger.info(f"Drift detected for {symbol}: error={model_data['drift'].error_detector.mean:.4f}")
                    self.online_learner.on_drift_detected(symbol, model_data["drift"].error_detector.drift_count)

    async def _manage_positions(self):
        """Check all open positions against current prices. Close if SL/TP hit."""
        positions = list(self.execution.open_positions.values())
        if not positions:
            return

        for trade in positions:
            try:
                sym = trade.symbol
                df = self.data_manager.get_ohlcv(sym, "1h")
                if df is None or len(df) < 3:
                    continue
                price = float(df["close"].iloc[-1])
                high = float(df["high"].iloc[-1])
                low = float(df["low"].iloc[-1])

                sl_hit = False
                tp_hit = False
                reason = ""

                if trade.direction == "BUY":
                    if low <= trade.sl:
                        sl_hit = True
                        reason = "Stop Loss"
                    elif high >= trade.tp:
                        tp_hit = True
                        reason = "Take Profit"
                else:
                    if high >= trade.sl:
                        sl_hit = True
                        reason = "Stop Loss"
                    elif low <= trade.tp:
                        tp_hit = True
                        reason = "Take Profit"

                if sl_hit or tp_hit:
                    exit_price = trade.sl if sl_hit else trade.tp
                    pip_size = 0.01 if "JPY" in sym.upper() else 0.0001
                    raw_pips = ((1 if trade.direction == "BUY" else -1) * (exit_price - trade.entry_price)) / pip_size
                    lots = trade.volume / 100000.0
                    pnl_usd = raw_pips * lots * 10.0
                    logger.info(f"Closing {sym} {trade.direction} at SL/TP @ {exit_price:.5f}: {reason}")
                    await self.execution.close_position(trade.position_id, reason, exit_price)
                    self.notifier.trade_closed(
                        symbol=sym, direction=trade.direction,
                        entry=trade.entry_price, exit=exit_price,
                        pnl=pnl_usd, reason=reason, hold_time=time.time() - trade.timestamp,
                    )
            except Exception as e:
                logger.debug(f"Position mgmt error for {getattr(trade,'symbol','?')}: {e}")

    # ================================================================
    # HELPERS
    # ================================================================

    def _on_market_data(self, tick: PriceTick):
        self.data_manager.update_tick(tick.symbol, tick.bid, tick.ask, tick.volume)
        # Feed to order flow analyzer
        self.order_flow_analyzer.update_tick(
            tick.symbol, tick.bid, tick.ask, tick.volume,
            tick.timestamp if hasattr(tick, 'timestamp') else time.time()
        )
        # Update gamma mapper with spot price
        self.gamma_mapper.update_spot(tick.symbol, (tick.bid + tick.ask) / 2.0)

    def _on_fix_market_data(self, fix_data):
        """Handle market data from FIX API and convert to MarketDepthData."""
        from src.data.data_manager import MarketDepthData

        if fix_data.bid > 0 or fix_data.ask > 0:
            md = MarketDepthData(
                symbol=fix_data.symbol or f"ID_{fix_data.symbol_id}",
                bids=[(fix_data.bid, fix_data.bid_size)] if fix_data.bid > 0 else [],
                asks=[(fix_data.ask, fix_data.ask_size)] if fix_data.ask > 0 else [],
                timestamp=fix_data.timestamp,
            )
            self.data_manager.update_market_depth(md)

            if fix_data.symbol:
                price = fix_data.bid if fix_data.bid > 0 else fix_data.ask
                self.data_manager.update_price(fix_data.symbol, price, "1h")

    def _on_depth_update(self, depth):
        """Handle Level II DOM updates from cTrader."""
        try:
            self.data_manager.update_market_depth(depth)
            # Log DOM imbalance for institutional analysis
            for sym in SYMBOLS:
                imbalance = self.data_manager.get_dom_imbalance(sym, levels=5)
                if abs(imbalance) > 0.3:  # Significant imbalance
                    logger.debug(f"DOM imbalance {sym}: {imbalance:.2f}")
        except Exception as e:
            logger.debug(f"DOM update handler error: {e}")

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
            dl = getattr(self, '_dashboard_loop', None)
            if dl:
                asyncio.run_coroutine_threadsafe(broadcast_update(latest_state), dl)
        except Exception:
            pass

    def _update_dashboard_enhanced(self, account_info: dict, regime: str = "trading"):
        """Enhanced dashboard update with all 11 new module metrics."""
        positions = self.execution.get_open_positions()
        suppressed, event = self.economic_calendar.is_suppressed()
        upcoming = self.economic_calendar.get_upcoming_events(hours=24)

        prices = self.data_manager.all_prices()
        regime_str = ", ".join(
            f"{sym}:{reg}" for sym, reg in list(self._regimes.items())[:7]
        )

        # Gather enhanced metrics from all 11 modules
        toxic_metrics = {}
        for sym in SYMBOLS[:5]:  # First 5 symbols
            if sym in self.toxic_detectors:
                snap = self.toxic_detectors[sym].get_snapshot()
                toxic_metrics[sym] = {
                    "vpin": round(snap.vpin, 3),
                    "is_toxic": snap.is_toxic,
                }

        circuit_metrics = {}
        for sym in SYMBOLS[:5]:
            if sym in self.circuit_breakers:
                snap = self.circuit_breakers[sym].get_snapshot()
                circuit_metrics[sym] = {
                    "state": snap.state.value,
                    "consecutive_bad": snap.consecutive_bad_ticks,
                }

        # Behavioral sentiment
        behavioral_sentiment = getattr(self, '_behavioral_sentiment', 0.0)

        # COT data
        cot_metrics = {}
        for sym in SYMBOLS[:5]:
            if sym in self.cot_data:
                snap = self.cot_data[sym]
                cot_metrics[sym] = {
                    "signal": snap.institutional_signal,
                    "positioning": snap.net_positioning,
                }

        # Regime specialist info
        regime_specialist_info = {}
        if hasattr(self, 'regime_system') and self.regime_system:
            try:
                regime_specialist_info = {
                    "current_regime": self.regime_system.current_regime,
                    "specialist_stats": {
                        k: {"trades": v.total_trades, "win_rate": v.win_rate}
                        for k, v in self.regime_system.specialists.items()
                    },
                }
            except Exception:
                pass

        # Execution algorithm stats
        algo_stats = {}
        if hasattr(self, 'algo_executor'):
            try:
                algo_stats = self.algo_executor.get_statistics()
            except Exception:
                pass

        # Model registry stats
        registry_stats = {}
        if hasattr(self, 'model_registry'):
            try:
                registry_stats = self.model_registry.get_statistics()
            except Exception:
                pass

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
                "behavioral_sentiment": round(behavioral_sentiment, 3),
                "econ_suppressed": suppressed,
                "econ_next_event": event.title if event else "",
                "upcoming_events": len(upcoming),
                "active_decisions": len(self._trade_decisions),
                "var": self.risk.var(),
                "cvar": self.risk.cvar(),
                # Enhanced module metrics
                "toxic_flow": toxic_metrics,
                "circuit_breakers": circuit_metrics,
                "cot_smart_money": cot_metrics,
                "regime_specialists": regime_specialist_info,
                "execution_algos": algo_stats,
                "model_registry": registry_stats,
            },
        )
        try:
            dl = getattr(self, '_dashboard_loop', None)
            if dl:
                asyncio.run_coroutine_threadsafe(broadcast_update(latest_state), dl)
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
_REGIME_PARAMS = {
    "trending": {"sl_atr": 2.0, "tp_atr": 4.0, "pos_mult": 1.0, "min_conf": 0.60},
    "ranging": {"sl_atr": 1.5, "tp_atr": 3.0, "pos_mult": 1.0, "min_conf": 0.65},
    "volatile": {"sl_atr": 2.5, "tp_atr": 5.0, "pos_mult": 0.5, "min_conf": 0.75},
    "crisis": {"sl_atr": 1.0, "tp_atr": 2.0, "pos_mult": 0.0, "min_conf": 0.95},
}
HMMRegimeDetector.should_trade_static = staticmethod(lambda regime: _REGIME_PARAMS.get(regime, _REGIME_PARAMS["ranging"])["pos_mult"] > 0)
HMMRegimeDetector.get_regime_params_static = staticmethod(lambda regime: _REGIME_PARAMS.get(regime, _REGIME_PARAMS["ranging"]))


# ================================================================
# ENTRY POINT
# ================================================================

def signal_handler(sig, frame):
    if "bot" in globals():
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.call_soon_threadsafe(lambda: asyncio.ensure_future(bot.stop()))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RTS: AI Moneybot System Elite")
    parser.add_argument("--mode", choices=["live", "paper", "backtest", "validate", "train"], default="paper")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--capital", type=float, default=100000.0, help="Starting capital")
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--mc-test", action="store_true")
    parser.add_argument("--bt-sensitivity", action="store_true")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    for d in ["data/logs", "data/trades", "models", "data/alternative_data"]:
        os.makedirs(d, exist_ok=True)

    bot = RTSForexBot(args.config, mode=args.mode, initial_balance=args.capital)

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

