"""ExpertRegistry — owns all trading strategies and their prediction/confidence methods.

SignalEngine delegates to this for expert registration and predictions,
removing ~590 lines of strategy logic from the engine.

Data flow:
    1. SignalEngine creates ExpertRegistry and calls update_context()
       with current OHLCV, symbol, and session.
    2. SignalEngine calls get_strategy_specs() to register all experts.
    3. On each tick, prediction methods access context via cached fields.
"""

import time
import numpy as np
from typing import Dict, Any, List, Callable
from loguru import logger


class ExpertRegistry:
    """Owns all trading strategies and their prediction/confidence methods.

    SignalEngine delegates to this for expert registration and predictions.

    The registry maintains its own context (last_df, current_symbol, session)
    which is synced by SignalEngine via update_context() before predictions.
    """

    # SL/TP multipliers by strategy name (copied from signal_strategies.py)
    STRATEGY_SL_TP: Dict[str, tuple] = {
        "ppo_trending": (2.0, 4.0),
        "ppo_ranging": (1.5, 3.0),
        "ppo_volatile": (2.5, 5.0),
        "ppo_crisis": (1.0, 2.0),
        "lstm_cnn": (2.0, 4.0),
        "rule_breakout": (2.0, 5.0),
        "rule_mean_rev": (1.5, 2.0),
        "bb_squeeze": (2.0, 3.0),
        "ts_momentum": (2.5, 4.0),
        "vol_mean_rev": (1.5, 2.0),
        "orderflow": (1.5, 2.5),
        "macro_sentiment": (1.5, 2.5),
        "social_sentiment": (1.5, 2.0),
        "xgboost": (2.0, 4.0),
    }

    def __init__(
        self,
        data_manager: Any = None,
        feature_pipeline: Any = None,
    ):
        self._last_df: Any = None
        self._current_symbol: str = ""
        self._current_session: str = "asia"
        self._data_manager = data_manager
        self._feature_pipeline = feature_pipeline
        self._alpha_strategies: Dict[str, Any] = {}
        self._cross_sectional_alpha: Any = None

    def update_context(
        self,
        last_df: Any = None,
        current_symbol: str = "",
        current_session: str = "asia",
    ) -> None:
        """Sync trading context from SignalEngine before predictions."""
        if last_df is not None:
            self._last_df = last_df
        if current_symbol:
            self._current_symbol = current_symbol
        if current_session:
            self._current_session = current_session

    # ── Strategy loading ─────────────────────────────────────────

    def load_alpha_strategies(self) -> None:
        """Load alpha strategies from the registry (Phase 12/13)."""
        try:
            from ai.alpha_strategies import StrategyRegistry as AlphaStrategyRegistry
            from ai.alpha_strategies.stat_arb import StatArbStrategy
            from ai.alpha_strategies.carry_trade import CarryTradeStrategy
            from ai.alpha_strategies.event_driven import EventDrivenStrategy
            from ai.alpha_strategies.vol_expansion import VolExpansionStrategy
            from ai.alpha_strategies.order_flow_momentum import (
                OrderFlowMomentumStrategy,
            )

            registry = AlphaStrategyRegistry()
            registry.register("stat_arb", StatArbStrategy)
            registry.register("carry_trade", CarryTradeStrategy)
            registry.register("event_driven", EventDrivenStrategy)
            registry.register("vol_expansion", VolExpansionStrategy)
            registry.register("order_flow_momentum", OrderFlowMomentumStrategy)

            registry.map_regime("trending", ["stat_arb", "carry_trade"])
            registry.map_regime("ranging", ["event_driven", "order_flow_momentum"])
            registry.map_regime("volatile", ["vol_expansion", "event_driven"])
            registry.map_regime("crisis", ["event_driven"])

            for name, strat_class in registry.get_all().items():
                instance = strat_class(symbol=self._current_symbol, params={})
                self._alpha_strategies[name] = instance
        except Exception:
            self._alpha_strategies = {}

        try:
            from rts_ai_fx.cross_sectional_alpha import CrossSectionalAlpha

            self._cross_sectional_alpha = CrossSectionalAlpha()
        except Exception:
            self._cross_sectional_alpha = None

    def get_strategy_specs(self) -> List[Dict]:
        """Build strategy spec dicts for all experts.

        Each spec's predict_fn is a closure that calls through
        to the appropriate method on this registry.
        """
        specs: List[Dict] = []

        # Phase 1: 4 regime-specific PPO experts (using RegimeAgentManager)
        try:
            from ai.regime_agents import RegimeAgentManager

            regime_manager = RegimeAgentManager()
            for regime in ["trending", "ranging", "volatile", "crisis"]:
                specs.append(
                    {
                        "name": f"ppo_{regime}",
                        "predict_fn": lambda X, r=regime: self.ppo_prediction(
                            X, regime=r, manager=regime_manager
                        ),
                        "confidence_fn": lambda X, r=regime: self.ppo_confidence(
                            X, regime=r, manager=regime_manager
                        ),
                        "regime": regime,
                    }
                )
        except Exception:
            logger.debug("[signal_engine] PPO regime agents not available")

        # Phase 2: Rule-based experts
        specs.append(
            {
                "name": "rule_breakout",
                "predict_fn": self.rule_breakout_prediction,
                "confidence_fn": lambda X: 0.65,
                "regime": "trending",
            }
        )
        specs.append(
            {
                "name": "rule_mean_rev",
                "predict_fn": self.rule_mean_rev_prediction,
                "confidence_fn": lambda X: 0.6,
                "regime": "ranging",
            }
        )

        # Phase 6: Research-backed forex strategies
        specs.append(
            {
                "name": "bb_squeeze",
                "predict_fn": self.bb_squeeze_prediction,
                "confidence_fn": lambda X: 0.55,
                "regime": "volatile",
            }
        )
        specs.append(
            {
                "name": "ts_momentum",
                "predict_fn": self.ts_momentum_prediction,
                "confidence_fn": lambda X: 0.6,
                "regime": "trending",
            }
        )
        specs.append(
            {
                "name": "vol_mean_rev",
                "predict_fn": self.vol_mean_rev_prediction,
                "confidence_fn": lambda X: 0.55,
                "regime": "volatile",
            }
        )

        # Phase 7: Order flow / CVD expert
        specs.append(
            {
                "name": "orderflow",
                "predict_fn": self.orderflow_prediction,
                "confidence_fn": self.orderflow_confidence,
                "regime": "ranging",
            }
        )

        # Phase 8: Macro sentiment expert
        specs.append(
            {
                "name": "macro_sentiment",
                "predict_fn": self.macro_sentiment_prediction,
                "confidence_fn": lambda X: 0.5,
                "regime": "ranging",
            }
        )

        # Phase 9: Social sentiment expert
        specs.append(
            {
                "name": "social_sentiment",
                "predict_fn": self.social_sentiment_prediction,
                "confidence_fn": lambda X: 0.5,
                "regime": "ranging",
            }
        )

        # Phase 10: XGBoost expert
        specs.append(
            {
                "name": "xgboost",
                "predict_fn": self.xgboost_prediction,
                "confidence_fn": lambda X: 0.5,
                "regime": "ranging",
            }
        )

        # Alpha strategy experts
        for name in self._alpha_strategies:
            specs.append(
                {
                    "name": f"alpha_{name}",
                    "predict_fn": self.make_alpha_predict_fn(name),
                    "confidence_fn": self.make_alpha_confidence_fn(name),
                    "regime": self.alpha_regime_for(name),
                }
            )

        return specs

    # ── PPO predictions ──────────────────────────────────────────────

    def ppo_prediction(
        self,
        X: np.ndarray,
        regime: str = "ranging",
        manager: Any = None,
    ) -> float:
        """Get PPO agent prediction for a given regime."""
        if manager is None:
            return 0.0
        try:
            state = self.features_to_ppo_state(X)
            action, *_ = manager.select_action(state, regime=regime)
            pred = 0.001 if action in (1, 3) else -0.001 if action in (2, 4) else 0.0
            return float(np.clip(pred, -0.01, 0.01)) if np.isfinite(pred) else 0.0
        except Exception:
            return 0.0

    def ppo_confidence(
        self,
        X: np.ndarray,
        regime: str = "ranging",
        manager: Any = None,
    ) -> float:
        """Get confidence from PPO agent."""
        if manager is None:
            return 0.5
        try:
            state = self.features_to_ppo_state(X)
            _action, *_ = manager.select_action(state, regime=regime)
            return 0.6
        except Exception:
            return 0.5

    def features_to_ppo_state(self, X: np.ndarray) -> np.ndarray:
        """Extract a flat state vector from multi-bar feature matrix."""
        X_arr = np.asarray(X)
        if X_arr.ndim == 3:
            X_arr = X_arr[0]
        if X_arr.ndim == 2:
            state = X_arr[-1, :]
        else:
            state = X_arr.flatten()
        target_dim = 49
        state = state[:target_dim]
        if len(state) < target_dim:
            state = np.pad(state, (0, target_dim - len(state)))
        return state.astype(np.float32)

    # ── Rule-based experts ───────────────────────────────────────────

    def get_current_session(self) -> str:
        """Determine current trading session based on UTC hour."""
        import datetime

        utc_hour = datetime.datetime.now(datetime.timezone.utc).hour
        if 7 <= utc_hour < 12:
            return "london"
        elif 12 <= utc_hour < 16:
            return "overlap"
        elif 16 <= utc_hour < 21:
            return "newyork"
        elif 21 <= utc_hour or utc_hour < 7:
            return "asia"
        return "asia"

    def compute_atr(self, period: int = 14) -> float:
        """Compute ATR from cached DataFrame."""
        if self._last_df is None or len(self._last_df) < period + 1:
            return 0.0
        df = self._last_df
        high_low = df["high"] - df["low"]
        atr = high_low.rolling(period).mean()
        return (
            float(atr.iloc[-1]) if not atr.empty and not np.isnan(atr.iloc[-1]) else 0.0
        )

    def rule_breakout_prediction(self, X: np.ndarray) -> float:
        """Session-aware breakout: 3-bar during London, 5-bar during NY, skip Asia."""
        if self._last_df is None or len(self._last_df) < 10:
            return 0.0
        session = self._current_session
        if session == "asia":
            return 0.0
        try:
            df = self._last_df
            close = float(df["close"].iloc[-1])
            atr = self.compute_atr(14)
            if atr <= 0:
                return 0.0
            lookback_bars = 3 if session == "london" else 5
            high_n = float(df["high"].iloc[-(lookback_bars + 1) : -1].max())
            low_n = float(df["low"].iloc[-(lookback_bars + 1) : -1].min())
            if close > high_n + 0.5 * atr:
                return 0.005
            if close < low_n - 0.5 * atr:
                return -0.005
            return 0.0
        except Exception:
            return 0.0

    def rule_mean_rev_prediction(self, X: np.ndarray) -> float:
        """Session-aware mean reversion with adjusted z-score thresholds."""
        if self._last_df is None or len(self._last_df) < 25:
            return 0.0
        session = self._current_session
        try:
            df = self._last_df
            close_series = df["close"]
            close_val = float(close_series.iloc[-1])
            sma20 = float(close_series.rolling(20).mean().iloc[-1])
            std20 = float(close_series.rolling(20).std().iloc[-1])
            if std20 <= 0:
                return 0.0
            z = (close_val - sma20) / std20
            if session == "asia":
                z_threshold = 1.2
            elif session == "overlap":
                z_threshold = 2.0
            else:
                z_threshold = 1.5
            # RSI computation
            rsi = None
            if "rsi_14" in df.columns:
                rsi_val = df["rsi_14"].iloc[-1]
                if not np.isnan(rsi_val):
                    rsi = float(rsi_val)
            if rsi is None:
                delta = close_series.diff()
                gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
                loss = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
                if loss and loss > 0:
                    rsi = float(100 - 100 / (1 + gain / loss))
                else:
                    rsi = 50.0
            if z < -z_threshold and rsi < 40:
                return 0.005
            if z > z_threshold and rsi > 60:
                return -0.005
            return 0.0
        except Exception:
            return 0.0

    def bb_squeeze_prediction(self, X: np.ndarray) -> float:
        """Bollinger Band Squeeze (Bollinger 2002)."""
        if self._last_df is None or len(self._last_df) < 105:
            return 0.0
        try:
            df = self._last_df
            if "bb_width" not in df.columns or "bb_mid" not in df.columns:
                return 0.0
            bb_width = df["bb_width"].values
            i = len(bb_width) - 1
            if i < 100:
                return 0.0
            bb_100_min = np.min(bb_width[max(0, i - 100) : i + 1])
            bb_current = bb_width[i]
            bb_prev = bb_width[i - 1] if i > 0 else bb_current
            if np.isnan(bb_current) or np.isnan(bb_prev) or np.isnan(bb_100_min):
                return 0.0
            if bb_current <= bb_100_min * 1.01 and bb_current > bb_prev:
                close = float(df["close"].iloc[-1])
                bb_mid = float(df["bb_mid"].iloc[-1])
                if close > bb_mid:
                    return 0.005
                elif close < bb_mid:
                    return -0.005
            return 0.0
        except Exception:
            return 0.0

    def ts_momentum_prediction(self, X: np.ndarray) -> float:
        """Time Series Momentum (Moskowitz, Ooi & Pedersen 2012)."""
        if self._last_df is None or len(self._last_df) < 15:
            return 0.0
        try:
            df = self._last_df
            if not all(col in df.columns for col in ["mom_1", "mom_5", "mom_10"]):
                return 0.0
            m1 = float(df["mom_1"].iloc[-1])
            m5 = float(df["mom_5"].iloc[-1])
            m10 = float(df["mom_10"].iloc[-1])
            if np.isnan(m1) or np.isnan(m5) or np.isnan(m10):
                return 0.0
            if m5 > 0 and m10 > 0 and m1 > 0:
                return 0.005
            if m5 < 0 and m10 < 0 and m1 < 0:
                return -0.005
            return 0.0
        except Exception:
            return 0.0

    def vol_mean_rev_prediction(self, X: np.ndarray) -> float:
        """Volatility Mean Reversion (Bollerslev, Tauchen & Zhou 2009)."""
        if self._last_df is None or len(self._last_df) < 20:
            return 0.0
        try:
            df = self._last_df
            if "vol_ratio" not in df.columns or "rsi_14" not in df.columns:
                return 0.0
            vr = float(df["vol_ratio"].iloc[-1])
            rsi = float(df["rsi_14"].iloc[-1])
            if np.isnan(vr) or np.isnan(rsi):
                return 0.0
            if vr > 1.5:
                if rsi < 30:
                    return 0.005
                if rsi > 70:
                    return -0.005
            return 0.0
        except Exception:
            return 0.0

    def orderflow_prediction(self, X: np.ndarray) -> float:
        """Order flow / CVD-based prediction (Evans & Lyons 2002)."""
        try:
            if not self._current_symbol:
                return 0.0
            of: Dict[str, Any] = {}
            if self._data_manager is not None:
                of = getattr(
                    self._data_manager,
                    "get_orderflow",
                    lambda sym: {},
                )(self._current_symbol)
            cvd = of.get("cvd", 0.0)
            cvd_slope = of.get("cvd_slope", 0.0)
            dom_imb = of.get("dom_imbalance", 0.0)
            signal = 0.0
            if abs(cvd_slope) > 0.1:
                signal += cvd_slope * 0.002
            if abs(dom_imb) > 0.6:
                signal -= dom_imb * 0.003
            if abs(cvd) > 500:
                signal += (cvd / 5000.0) * 0.001
            return float(np.clip(signal, -0.008, 0.008))
        except Exception:
            return 0.0

    def orderflow_confidence(self, X: np.ndarray) -> float:
        """Confidence based on order flow signal extremity."""
        try:
            of: Dict[str, Any] = {}
            if self._data_manager is not None:
                of = getattr(
                    self._data_manager,
                    "get_orderflow",
                    lambda sym: {},
                )(self._current_symbol)
            cvd = abs(of.get("cvd", 0.0))
            dom_imb = abs(of.get("dom_imbalance", 0.0))
            confidence = min(0.4 + (cvd / 2000.0) * 0.3 + dom_imb * 0.3, 0.75)
            return float(confidence)
        except Exception:
            return 0.4

    def macro_sentiment_prediction(self, X: np.ndarray) -> float:
        """Macroeconomic sentiment prediction (Andersen et al. 2003)."""
        try:
            macro: Dict[str, Any] = {}
            if self._data_manager is not None:
                macro = getattr(
                    self._data_manager,
                    "get_macro",
                    lambda: {},
                )()
            upcoming = macro.get("upcoming_events", [])
            if not upcoming:
                return 0.0
            now = time.time()
            for ev in upcoming:
                event_ts = ev.get("timestamp", 0)
                hours_until = (event_ts - now) / 3600
                if ev.get("impact") == "high" and 0 < hours_until < 2:
                    return 0.0
            recent = [ev for ev in upcoming if ev.get("timestamp", 0) > now - 3600]
            if recent:
                return -0.002
            return 0.0
        except Exception:
            return 0.0

    def social_sentiment_prediction(self, X: np.ndarray) -> float:
        """Social media sentiment prediction (contrarian/follow-trend)."""
        try:
            sentiment: Dict[str, Any] = {}
            if self._data_manager is not None:
                sentiment = getattr(
                    self._data_manager,
                    "get_sentiment",
                    lambda: {},
                )()
            score = sentiment.get("score", 0.0)
            confidence = sentiment.get("confidence", 0.0)
            if confidence < 0.3:
                return 0.0
            if abs(score) > 0.6:
                return float(-score * 0.003)
            elif abs(score) > 0.2:
                return float(score * 0.002)
            return 0.0
        except Exception:
            return 0.0

    def xgboost_prediction(self, X: np.ndarray) -> float:
        """XGBoost gradient-boosted trees prediction."""
        try:
            sym = self._current_symbol
            if not sym:
                return 0.0
            import os

            model_path = f"models/xgboost_{sym}.pkl"
            if not os.path.exists(model_path):
                return 0.0
            import joblib

            gbm_model = joblib.load(model_path)
            X_arr = np.asarray(X).reshape(1, -1)
            if X_arr.ndim == 3:
                X_arr = X_arr.reshape(X_arr.shape[0], -1)
            pred = gbm_model.predict(X_arr)[0]
            return float(np.clip(pred, -0.01, 0.01))
        except Exception:
            return 0.0

    # ── Alpha strategy helpers ───────────────────────────────────────

    def make_alpha_predict_fn(self, name: str) -> Callable:
        """Create a predict closure for an alpha strategy."""
        return lambda X: self.alpha_prediction(X, name)

    def make_alpha_confidence_fn(self, name: str) -> Callable:
        """Create a confidence closure for an alpha strategy."""
        return lambda X: self.alpha_confidence(X, name)

    def alpha_regime_for(self, name: str) -> str:
        mapping = {
            "stat_arb": "ranging",
            "carry_trade": "trending",
            "event_driven": "volatile",
            "vol_expansion": "volatile",
            "order_flow_momentum": "ranging",
        }
        return mapping.get(name, "ranging")

    def alpha_prediction(self, X: np.ndarray, name: str) -> float:
        """Convert alpha strategy signal direction to price-like prediction."""
        strategy = self._alpha_strategies.get(name)
        if strategy is None or self._last_df is None:
            return 0.0
        try:
            sig = strategy.generate_signal(self._last_df)
            direction = sig.get("direction", "HOLD")
            conf = sig.get("confidence", 0.0)
            if direction == "BUY":
                return 0.005 * conf
            elif direction == "SELL":
                return -0.005 * conf
            return 0.0
        except Exception:
            return 0.0

    def alpha_confidence(self, X: np.ndarray, name: str) -> float:
        """Return alpha strategy confidence."""
        strategy = self._alpha_strategies.get(name)
        if strategy is None or self._last_df is None:
            return 0.5
        try:
            sig = strategy.generate_signal(self._last_df)
            return float(sig.get("confidence", 0.5))
        except Exception:
            return 0.5
