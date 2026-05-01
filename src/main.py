"""
RTS AI Forex Trading System — Elite Edition.
Asynchronous trading loop with multi-timeframe ML, HMM regime detection,
uncertainty quantification, concept drift detection, and institutional-grade risk.
"""
import asyncio
import os
import signal
import sys
from typing import Optional, Dict, Any

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
from data.data_manager import DataManager
from dashboard.app import app, broadcast_update, update_state, latest_state

try:
    from loguru import logger as loguru_logger
    logger = loguru_logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class RTSForexBot:
    """Elite-grade AI forex trading bot."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = Config()
        self.config.config_path = config_path
        self.config._load_yaml()
        self.secrets = Secrets()

        # Configure logging
        logger.remove()
        logger.add(sys.stdout, level=self.config.logging.level)
        logger.add(
            self.config.data.logs_path.rstrip("/") + "/moneybot.log",
            rotation=self.config.logging.rotation,
            retention=self.config.logging.retention,
        )

        # Initialize components
        self._init_ctrader()
        self._init_risk()
        self._init_data()
        self._init_models()
        self._init_monitoring()

        self.trade_count = 0
        self.is_running = False
        self._last_snapshot_time = 0.0

        logger.info("=" * 60)
        logger.info("  RTS AI Forex Trading System v4.0 (Elite)")
        logger.info("  Multi-Timeframe ML | HMM Regime | Uncertainty Quant")
        logger.info("=" * 60)

    def _init_ctrader(self):
        self.execution, self.data_provider = create_execution_provider(self.secrets)
        self.execution.on_price = self._on_market_data
        # For backward compatibility, expose as self.ctrader
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
        self.cost_model = CostModel(
            commission_per_lot=self.config.trading.commission_per_lot,
        )
        self.execution = ExecutionEngine(self.ctrader, self.risk, None)
        self.execution.cost_model = self.cost_model

    def _init_data(self):
        self.data_manager = DataManager(
            historical_path=self.config.data.historical_path,
        )
        self.feature_pipeline = FeaturePipeline(
            lookback=30,
            timeframes=["1h", "4h", "1d"],
        )
        self.regime_detector = HMMRegimeDetector(n_regimes=4, lookback=60)

    def _init_models(self):
        self.ensemble = MoEEnsemble()
        # Rule-based fallback expert for simulation (uses RSI/MACD cross signals)
        self.ensemble.add_expert(
            name="rule_based",
            predict_fn=self._rule_prediction,
            confidence_fn=lambda X: 0.7,
            regime="ranging",
        )
        self.active_models: Dict[str, Any] = {}
        self.drift_monitors: Dict[str, DriftMonitor] = {}

    def _rule_prediction(self, X: np.ndarray) -> float:
        """Simple rule-based prediction using last feature values for simulation."""
        try:
            # X is (lookback, n_features) — use last row
            last = X[-1] if X.ndim == 2 else X
            # Features are z-score normalized; positive RSI-like means upward bias
            # Index-based heuristic: use last few price changes if available
            if len(last) > 5:
                mom = float(last[-5]) if not np.isnan(last[-5]) else 0.0
            else:
                mom = 0.0
            price = 1.12 + mom * 0.001
            return price
        except Exception:
            return 1.12

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

        # Load historical data
        await self._load_historical_data()

        # Fit regime detector
        primary_df = self.data_manager.ohlcv.get("1h")
        if primary_df is not None and len(primary_df) > 200:
            self.regime_detector.fit(primary_df)
            logger.info("HMM regime detector fitted")

        # Fit feature pipeline
        self.feature_pipeline.fit(self.data_manager.ohlcv)
        logger.info("Feature pipeline fitted")

        # Start WebSocket streaming if available
        if hasattr(self.execution, 'stream_prices'):
            symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]
            try:
                await self.execution.stream_prices(symbols)
            except NotImplementedError:
                pass

        self._start_dashboard()

        self.is_running = True
        logger.info("Bot is LIVE")

        while self.is_running:
            try:
                await self._trading_cycle()
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cycle error: {e}")
                await asyncio.sleep(5.0)

    async def _load_historical_data(self):
        # Try the dedicated data provider first (Dukascopy or OANDA)
        if self.data_provider is not None:
            try:
                symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]
                for sym in symbols:
                    ohlcv = await self.data_provider.fetch_ohlcv(
                        sym, "1h", "2025-06-01", "2026-03-31"
                    )
                    if ohlcv:
                        df = pd.DataFrame([{
                            "timestamp": o.timestamp, "open": o.open, "high": o.high,
                            "low": o.low, "close": o.close, "volume": o.volume,
                        } for o in ohlcv])
                        if len(df) > 100:
                            self.data_manager.ohlcv["1h"] = df
                            logger.info(f"Loaded {len(df)} 1h bars for {sym} via data provider")
                            break
            except Exception as e:
                logger.warning(f"Data provider failed, using fallback: {e}")

        # Fallback to synthetic data
        for tf in ["1h", "4h", "1d"]:
            if tf not in self.data_manager.ohlcv or len(self.data_manager.ohlcv[tf]) < 100:
                logger.info(f"Generating {tf} data...")
                self.data_manager.load_historical(tf, days=365)

    def _start_dashboard(self):
        import threading
        import uvicorn
        d = self.config.dashboard
        t = threading.Thread(
            target=lambda: uvicorn.run(
                app, host=d.host, port=d.port, log_level="error"
            ),
            daemon=True,
        )
        t.start()
        logger.info(f"Dashboard: http://{d.host}:{d.port}")

    async def _trading_cycle(self):
        # 1. Get market snapshot
        snapshot = await self._get_snapshot()
        if snapshot is None:
            return

        # 2. Detect regime
        primary = self.data_manager.ohlcv.get("1h")
        if primary is not None and len(primary) > 60:
            regime = self.regime_detector.detect_regime(primary)
        else:
            regime = "ranging"

        if not self.regime_detector.should_trade(regime):
            self._update_dashboard(snapshot, regime)
            return

        # 3. Generate features
        features = self.feature_pipeline.transform(self.data_manager.ohlcv)
        if features is None:
            return
        features_flat = features.flatten()

        # 4. Model inference with uncertainty
        ensemble_pred = self.ensemble.predict(features, regime=regime)
        confidence = ensemble_pred.confidence

        # 5. Decide whether to trade
        should_trade, direction, agreement = self.ensemble.should_trade(
            ensemble_pred, snapshot.get("price", 1.12), min_confidence=0.65
        )

        if not should_trade:
            self._update_dashboard(snapshot, regime)
            return

        # 6. Risk checks
        approved, reason = self.risk.pre_trade_checks(
            snapshot.get("balance", 100_000),
            snapshot.get("equity", 100_000),
            snapshot.get("margin", 0),
            snapshot.get("daily_pnl", 0),
        )
        if not approved:
            self._send_alert(f"Trade blocked: {reason}")
            self._update_dashboard(snapshot, regime)
            return

        # 7. Calculate position size
        atr = snapshot.get("atr", 0.001)
        price = snapshot.get("price", 1.12)
        regime_params = self.regime_detector.get_regime_params(regime)
        adjusted_conf = confidence * regime_params["pos_mult"]
        volume = self.risk.calculate_kelly_size(
            snapshot.get("balance", 100_000), price, atr, adjusted_conf
        )
        volume = max(volume, 100)

        # 8. Execute trade
        trade = await self.execution.open_position(
            symbol=snapshot.get("symbol", "EURUSD"),
            direction=direction,
            volume=volume,
            sl=price - atr * regime_params["sl_pct"] / (atr / price + 1e-8),
            tp=price + atr * regime_params["tp_pct"] / (atr / price + 1e-8),
            reason=f"regime={regime} conf={confidence:.2f}",
        )

        if trade:
            self.trade_count += 1
            self._send_alert(
                f"{direction} {volume:.0f} {snapshot.get('symbol', 'EURUSD')} @ {price:.5f}"
            )

        # 9. Update drift monitors
        for name, model_data in self.active_models.items():
            if "drift" in model_data:
                pred_val = ensemble_pred.expert_outputs.get(name, {}).get("prediction", price)
                model_data["drift"].update(pred_val, price)

        self._update_dashboard(snapshot, regime)

    async def _get_snapshot(self) -> Optional[Dict[str, Any]]:
        acc = await self.execution.get_account_info()
        if acc is None:
            return None
        positions = self.execution.get_open_positions()
        primary = self.data_manager.ohlcv.get("1h")
        price = primary["close"].iloc[-1] if primary is not None and len(primary) > 0 else 1.12
        atr_val = 0.0
        if primary is not None and len(primary) > 15:
            tr = pd.concat([
                primary["high"] - primary["low"],
                (primary["high"] - primary["close"].shift()).abs(),
                (primary["low"] - primary["close"].shift()).abs(),
            ], axis=1).max(1)
            atr_val = tr.iloc[-14:].mean() if len(tr) >= 14 else 0.001
        self.risk.daily_pnl = sum(
            p.get("unrealized_pnl", 0) for p in positions
        )
        self.risk.update_price_history(price)
        return {
            "balance": acc.balance,
            "equity": acc.equity,
            "margin": acc.margin,
            "free_margin": acc.free_margin,
            "price": price,
            "atr": max(atr_val, 0.0001),
            "daily_pnl": self.risk.daily_pnl,
            "positions": positions,
            "symbol": "EURUSD",
        }

    def _on_market_data(self, tick: PriceTick):
        self.data_manager.update_tick(tick.symbol, tick.bid, tick.ask, tick.volume)

    def _send_alert(self, message: str):
        logger.info(f"[ALERT] {message}")
        # Telegram integration placeholder
        # if self.secrets.telegram_bot_token:
        #     import requests
        #     requests.post(
        #         f"https://api.telegram.org/bot{self.secrets.telegram_bot_token}/sendMessage",
        #         json={"chat_id": self.secrets.telegram_chat_id, "text": message},
        #     )

    def _update_dashboard(self, snapshot: Dict[str, Any], regime: str = "ranging"):
        positions = self.execution.get_open_positions()
        update_state(
            balance=snapshot.get("balance", 100000),
            equity=snapshot.get("equity", 100000),
            margin=snapshot.get("margin", 0),
            free_margin=snapshot.get("free_margin", 100000),
            initial_balance=100000,
            total_trades=self.trade_count,
            win_rate=self.risk.get_win_rate(),
            mode=self.risk.mode,
            regime=regime,
            open_positions=positions,
            trade_history=self.execution.get_trade_history(20),
            market_data={
                "bid": snapshot.get("price", 0),
                "ask": snapshot.get("price", 0),
                "spread": 0.0,
            },
            ai_metrics={
                "regime": regime,
                "confidence": 0.0,
                "action": "HOLD",
                "var": self.risk.var(),
                "cvar": self.risk.cvar(),
            },
        )
        try:
            asyncio.run_coroutine_threadsafe(broadcast_update(latest_state), asyncio.get_event_loop())
        except Exception:
            pass

    async def stop(self):
        logger.info("Stopping bot...")
        self.is_running = False
        await self.execution.close_all_positions("Shutdown")
        await self.ctrader.disconnect()
        logger.info("Bot stopped.")


def signal_handler(sig, frame):
    if "bot" in globals():
        asyncio.create_task(bot.stop())


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    os.makedirs("data/logs", exist_ok=True)
    os.makedirs("data/trades", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    bot = RTSForexBot("config.yaml")
    asyncio.run(bot.start())
