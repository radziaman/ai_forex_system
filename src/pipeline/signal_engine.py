"""SignalEngine — features -> HMM regime detection -> MoE ensemble -> trading signals.

Replaces: FeatureAgent + RegimeAgent + SignalAgent.

Data flow:
    1. Receive tick data -> compute 49-dim features
    2. Detect market regime (HMM: trending/ranging/volatile/crisis)
    3. Run MoE ensemble (28 experts with regime gating, Elo, Sharpe weighting)
    4. Generate signal with confidence score
    5. Emit 'signal_generated' event

Compared to the old architecture, this is a single module instead of:
- feature_agent.py (perceive->reason->act->reflect cycle)
- regime_agent.py (HMM detection with agent overhead)
- signal_agent.py (1446-line monolith with consciousness)
"""

import time
import numpy as np
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque
from loguru import logger

from .pipeline_context import PipelineContext
from .expert_registry import ExpertRegistry


class SignalEngine:
    """End-to-end signal generation pipeline."""

    ACTIVE_SYMBOLS = [
        "EURUSD",
        "GBPUSD",
        "USDJPY",
        "AUDUSD",
        "USDCAD",
        "USDCHF",
        "NZDUSD",
        "XAUUSD",
    ]

    # Backward-compat reference — logic now lives in ExpertRegistry
    STRATEGY_SL_TP: Dict[str, tuple] = ExpertRegistry.STRATEGY_SL_TP

    def __init__(
        self,
        ctx: PipelineContext,
        ensemble: Any = None,
    ):
        self.ctx = ctx
        self.config = ctx.config
        self.bus = ctx.bus
        self._ensemble = (
            ensemble or ctx.ensemble
        )  # MoEEnsemble instance (injected or created)
        self._data_manager = ctx.data_manager  # DataManager instance (injected)
        self._feature_pipeline: Any = None
        self._regime_detector: Any = None
        self._models_loaded = False
        self._signal_count = 0

        # ExpertRegistry — owns all strategy prediction/confidence methods
        self._expert_registry: Optional[ExpertRegistry] = None

        # Rule-based expert state (synced to ExpertRegistry via update_context)
        self._last_df: Any = None  # cached DataFrame for rule-based experts
        self._last_price: float = 0.0
        self._current_symbol: str = "EURUSD"
        self._current_session: str = "asia"

        # G8: Track outcomes per expert for online learning
        self._expert_outcomes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self._expert_pnl: Dict[str, float] = defaultdict(float)
        self._expert_trades: Dict[str, int] = defaultdict(int)

        # G8 enhanced: Elo ratings, Sharpe per expert, decay tracking
        self._elo_ratings: Dict[str, float] = defaultdict(lambda: 1200.0)
        self._expert_returns: Dict[str, List[float]] = defaultdict(list)
        self._expert_sharpes: Dict[str, float] = defaultdict(float)
        self._expert_last_trade: Dict[str, float] = defaultdict(float)
        self._avg_trade_pnl: float = 0.0
        self._trade_count_for_avg: int = 0

        # Track position_id -> {symbol, expert_outputs, direction} for PnL at close
        self._position_info: Dict[int, Dict] = {}

        # Phase 4.3: Execution quality feedback — confidence threshold
        self._min_confidence_threshold: float = 0.55

        # ATR-based dynamic prediction threshold
        self._atr_threshold_multiplier: float = getattr(
            self.config.trading, "atr_threshold_multiplier", 0.1
        )

        # Subscribe to events
        self.bus.on("tick", self._on_tick)
        self.bus.on("execution_result", self._on_execution_result)
        self.bus.on("position_closed", self._on_position_closed)
        self.bus.on("execution_quality", self._on_execution_quality)

    async def start(self) -> None:
        """Initialize the signal engine: feature pipeline, regime detector, ensemble."""
        await self._ensure_initialized()

    async def _ensure_initialized(self) -> None:
        """Lazy init on first use — loads models only when needed."""
        if self._models_loaded:
            return
        from rts_ai_fx.features_unified import FeaturePipeline
        from rts_ai_fx.regime_detector import HMMRegimeDetector
        from rts_ai_fx.ensemble import MoEEnsemble

        features_config = getattr(self.config, "features", self.config)
        self._feature_pipeline = FeaturePipeline(
            lookback=getattr(features_config, "lookback", 30),
            timeframes=getattr(features_config, "timeframes", ["1h"]),
            use_microstructure=getattr(features_config, "use_microstructure", True),
        )
        self.ctx.feature_pipeline = self._feature_pipeline
        self._regime_detector = HMMRegimeDetector()
        self.ctx.regime_detector = self._regime_detector

        # Create ExpertRegistry and sync current context
        self._expert_registry = ExpertRegistry(
            data_manager=self._data_manager,
            feature_pipeline=self._feature_pipeline,
        )
        self._expert_registry.update_context(
            last_df=self._last_df,
            current_symbol=self._current_symbol,
            current_session=self._current_session,
        )

        if self._ensemble is None:
            self._ensemble = MoEEnsemble()
            self.ctx.ensemble = self._ensemble
            self._register_experts()

        self._models_loaded = True
        logger.info(
            f"[signal_engine] Models loaded (lazy) — "
            f"{len(self._ensemble.experts) if self._ensemble else 0} experts registered"
        )

    def _register_experts(self) -> None:
        """Register all ensemble experts.

        Delegates strategy spec generation to ExpertRegistry.
        Also initialises alpha research pipeline and meta-learning orchestrator.
        """
        if not self._ensemble:
            return
        self._ensemble.experts = []
        self._ensemble.elo_ratings = {}

        # Delegate strategy loading and spec building to ExpertRegistry
        if self._expert_registry is not None:
            self._expert_registry.load_alpha_strategies()
            specs = self._expert_registry.get_strategy_specs()
        else:
            specs = []

        for spec in specs:
            self._ensemble.add_expert(
                name=spec["name"],
                predict_fn=spec["predict_fn"],
                confidence_fn=spec["confidence_fn"],
                regime=spec["regime"],
            )

        # Alpha research pipeline (Phase 14)
        try:
            from ai.alpha_research import AlphaResearchPipeline, AlphaPipelineConfig

            self._alpha_pipeline = AlphaResearchPipeline(
                config=AlphaPipelineConfig(min_ic=0.02, min_ir=0.5, min_t_stat=2.0),
                registry_path="models/signal_registry.json",
            )
        except Exception:
            self._alpha_pipeline = None

        # Meta-learning orchestrator (Phase 15)
        try:
            from ai.maml_scaler import MetaLearningOrchestrator, MetaConfig

            self._meta_learner = MetaLearningOrchestrator(
                config=MetaConfig(
                    inner_lr=0.01, inner_steps=5, adaptation_threshold=0.15
                )
            )
            for sym in self.ACTIVE_SYMBOLS:
                self._meta_learner.register_model(sym)
        except Exception:
            self._meta_learner = None

        logger.info(f"[signal_engine] Registered {len(specs)} experts")

    # ── Backward-compatible delegates ────────────────────────────────
    # These forward to ExpertRegistry; kept for existing tests that
    # call private methods on SignalEngine directly.

    def _get_current_session(self) -> str:
        """Delegate to ExpertRegistry.get_current_session()."""
        if self._expert_registry is not None:
            return self._expert_registry.get_current_session()
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

    def _features_to_ppo_state(self, X: np.ndarray) -> np.ndarray:
        """Delegate to ExpertRegistry.features_to_ppo_state()."""
        if self._expert_registry is not None:
            return self._expert_registry.features_to_ppo_state(X)
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

    def _alpha_regime_for(self, name: str) -> str:
        """Delegate to ExpertRegistry.alpha_regime_for()."""
        if self._expert_registry is not None:
            return self._expert_registry.alpha_regime_for(name)
        mapping = {
            "stat_arb": "ranging",
            "carry_trade": "trending",
            "event_driven": "volatile",
            "vol_expansion": "volatile",
            "order_flow_momentum": "ranging",
        }
        return mapping.get(name, "ranging")

    def _rule_breakout_prediction(self, X: np.ndarray) -> float:
        """Delegate to ExpertRegistry.rule_breakout_prediction()."""
        if self._expert_registry is not None and self._last_df is not None:
            return self._expert_registry.rule_breakout_prediction(X)
        return 0.0

    def _rule_mean_rev_prediction(self, X: np.ndarray) -> float:
        """Delegate to ExpertRegistry.rule_mean_rev_prediction()."""
        if self._expert_registry is not None and self._last_df is not None:
            return self._expert_registry.rule_mean_rev_prediction(X)
        return 0.0

    def _compute_current_atr(self, symbol: str, timeframe: str = "1h") -> float:
        """Compute current ATR for dynamic prediction threshold.

        Tries, in order:
          1. Extract ``atr_14`` from the feature pipeline's computed features
             if the pipeline has already stored them on ``_last_df``.
          2. Compute ATR from the last 14 bars of OHLCV data cached in
             ``_last_df``.
          3. Fall back to the original hardcoded threshold (0.0003) if no
             data is available.

        Args:
            symbol: Forex symbol (e.g. "EURUSD").
            timeframe: OHLCV timeframe (unused, kept for API compat; uses
                       the cached ``_last_df`` which is at 1h).

        Returns:
            ATR value as float, always > 0.
        """
        df = self._last_df
        if df is not None and not df.empty:
            # 1. Use pre-computed atr_14 if available
            if "atr_14" in df.columns:
                atr_val = df["atr_14"].iloc[-1]
                if not (np.isnan(atr_val) or np.isinf(atr_val)) and atr_val > 0:
                    return float(atr_val)

            # 2. Compute ATR from OHLCV data (last 14 bars)
            if all(c in df.columns for c in ("high", "low", "close")):
                high = df["high"].values
                low = df["low"].values
                close = df["close"].values
                if len(high) >= 15:
                    # True Range = max(high-low, |high-prev_close|, |low-prev_close|)
                    prev_close = close[:-1]
                    tr = np.maximum(
                        high[1:] - low[1:],
                        np.maximum(
                            np.abs(high[1:] - prev_close),
                            np.abs(low[1:] - prev_close),
                        ),
                    )
                    atr_val = float(np.mean(tr[-14:]))
                    if not np.isnan(atr_val) and np.isfinite(atr_val) and atr_val > 0:
                        return atr_val

        # 3. Fallback — original hardcoded threshold
        logger.debug(
            f"[signal_engine] No ATR data for {symbol}, falling back "
            f"to default 0.0003"
        )
        return 0.0003

    # ── Event handlers ───────────────────────────────────────────────

    async def _on_tick(
        self,
        symbol: str,
        bid: float,
        ask: float,
        volume: float,
        timestamp: float,
    ) -> None:
        """Process a tick: compute features, detect regime, generate signal."""
        await self._ensure_initialized()  # First tick triggers model load
        self._current_symbol = symbol
        if self._expert_registry is not None:
            self._current_session = self._expert_registry.get_current_session()
        else:
            self._current_session = "asia"

        # Sync trading context to ExpertRegistry for prediction methods
        if self._expert_registry is not None:
            self._expert_registry.update_context(
                last_df=self._last_df,
                current_symbol=self._current_symbol,
                current_session=self._current_session,
            )

        # Run expert decay periodically (every ~100 ticks)
        if self._signal_count % 100 == 0 and self._signal_count > 0:
            self._decay_experts()

        features = await self._compute_features(symbol)
        if features is None:
            return

        # Detect regime
        regime = None
        if self._regime_detector is not None and self._last_df is not None:
            try:
                regime = self._regime_detector.detect_regime(self._last_df)
                if regime:
                    await self.bus.emit(
                        "regime_changed",
                        from_regime=str(
                            getattr(self._regime_detector, "current_regime", "?")
                        ),
                        to_regime=regime,
                    )
            except Exception:
                regime = "ranging"
        else:
            regime = "ranging"

        # Generate signal
        signal = await self._generate_signal(symbol, features, regime)
        if signal and signal.get("direction") != "HOLD":
            # Apply updated confidence threshold from execution quality feedback
            confidence = signal.get("confidence", 0.0)
            if confidence >= self._min_confidence_threshold:
                # ATR-based dynamic prediction threshold
                expert_outputs = signal.get("expert_outputs")
                if expert_outputs and self._ensemble is not None:
                    from rts_ai_fx.ensemble import EnsemblePrediction

                    # Reconstruct EnsemblePrediction for should_trade
                    pred = EnsemblePrediction()
                    pred.price = signal.get("price", 0.0)
                    pred.confidence = confidence
                    pred.direction = signal.get("direction", "HOLD")
                    pred.expert_outputs = expert_outputs

                    atr = self._compute_current_atr(symbol)
                    prediction_threshold = max(
                        0.0001, atr * self._atr_threshold_multiplier
                    )
                    logger.debug(
                        f"Dynamic threshold for {symbol}: "
                        f"{prediction_threshold:.6f} "
                        f"(ATR: {atr:.6f}, "
                        f"multiplier: {self._atr_threshold_multiplier})"
                    )

                    should, _, _ = self._ensemble.should_trade(
                        pred,
                        current_price=bid,
                        min_confidence=self._min_confidence_threshold,
                        prediction_threshold=prediction_threshold,
                    )
                    if not should:
                        logger.debug(
                            f"[signal_engine] ATR threshold filtered "
                            f"{symbol}: prediction too small relative "
                            f"to ATR (threshold={prediction_threshold:.6f})"
                        )
                        return

                self._signal_count += 1
                await self.bus.emit("signal_generated", **signal)

    async def _compute_features(self, symbol: str) -> Optional[np.ndarray]:
        """Compute 49-dim feature vector for a symbol."""
        try:
            if self._feature_pipeline is None:
                return None
            if self._data_manager is None:
                return None

            # Get OHLCV data for the symbol
            lookback = getattr(self._feature_pipeline, "lookback", 30)
            ohlcv = getattr(self._data_manager, "get_ohlcv", lambda sym, tf: None)(
                symbol, "1h"
            )

            if ohlcv is None or (hasattr(ohlcv, "empty") and ohlcv.empty):
                return None

            # Need at least 30 bars for rolling calculations
            if len(ohlcv) < 30:
                return None

            # Cache for rule-based experts
            self._last_df = ohlcv
            if self._expert_registry is not None:
                self._expert_registry.update_context(
                    last_df=self._last_df,
                    current_symbol=self._current_symbol,
                    current_session=self._current_session,
                )

            # Compute features using the unified pipeline
            from rts_ai_fx.features_unified import compute_features

            df_features = compute_features(ohlcv)
            if df_features is None or df_features.empty:
                return None

            # Extract the feature columns as a numpy array
            from rts_ai_fx.features_unified import EXPECTED_FEATURE_DIM

            feature_cols = [
                c
                for c in df_features.columns
                if c
                not in {
                    "timestamp",
                    "datetime",
                    "time",
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                }
            ]
            if len(feature_cols) < EXPECTED_FEATURE_DIM:
                logger.warning(
                    f"[signal_engine] Feature count {len(feature_cols)} "
                    f"< expected {EXPECTED_FEATURE_DIM} for {symbol}"
                )

            feature_values = df_features[feature_cols].values
            if len(feature_values) == 0:
                return None

            # Return the most recent feature vector
            return feature_values[-1].astype(np.float32)

        except Exception as e:
            logger.debug(f"[signal_engine] Feature computation error for {symbol}: {e}")
            return None

    async def _generate_signal(
        self,
        symbol: str,
        features: np.ndarray,
        regime: Optional[str],
    ) -> Dict[str, Any]:
        """Generate a trading signal from the ensemble."""
        try:
            if self._ensemble is None or not self._ensemble.experts:
                return {
                    "symbol": symbol,
                    "direction": "HOLD",
                    "confidence": 0.0,
                    "price": 0.0,
                    "regime": regime or "ranging",
                    "timestamp": time.time(),
                }

            prediction = self._ensemble.predict(features, regime=regime or "ranging")

            return {
                "symbol": symbol,
                "direction": prediction.direction,
                "confidence": prediction.confidence,
                "price": prediction.price,
                "regime": regime or "ranging",
                "timestamp": time.time(),
                "expert_outputs": prediction.expert_outputs,
            }
        except Exception:
            logger.warning(f"[signal_engine] Signal generation error for {symbol}")
            return {
                "symbol": symbol,
                "direction": "HOLD",
                "confidence": 0.0,
                "price": 0.0,
                "regime": regime or "ranging",
                "timestamp": time.time(),
            }

    async def _on_execution_result(self, **data: Any) -> None:
        """Learn from execution outcomes."""
        symbol = data.get("symbol", "")
        _ = data.get("position_id", 0)
        filled_price = data.get("filled_price", 0.0)
        signal_price = data.get("signal_price", 0.0)
        _ = data.get("direction", "")

        # Track expert performance based on execution quality
        if filled_price > 0 and signal_price > 0:
            slippage = abs(filled_price - signal_price)
            if slippage > 0:
                logger.debug(
                    f"[signal_engine] Execution slippage for {symbol}: "
                    f"{slippage:.5f}"
                )

    def _sync_elo_to_ensemble(self) -> None:
        """Sync enhanced Elo ratings back to the ensemble for prediction weighting."""
        if self._ensemble is None:
            return
        if (
            hasattr(self._ensemble, "elo_ratings")
            and self._ensemble.elo_ratings is not None
        ):
            for name, rating in self._elo_ratings.items():
                self._ensemble.elo_ratings[name] = rating

    async def _on_position_closed(self, **data: Any) -> None:
        """Learn from closed position P&L — update Elo ratings and Sharpe."""
        pnl = data.get("pnl", 0)
        symbol = data.get("symbol", "EURUSD")
        position_id = data.get("position_id", 0)

        # Update average trade PnL (used for Elo weighting)
        self._trade_count_for_avg += 1
        n = self._trade_count_for_avg
        self._avg_trade_pnl += (abs(pnl) - self._avg_trade_pnl) / n

        # Look up cached expert info for this position
        info = self._position_info.pop(position_id, None)
        if info is None:
            return

        expert_outputs = info.get("expert_outputs", {})
        _ = info.get("symbol", symbol)

        # Distribute PnL to each expert that contributed
        total_weight = sum(o.get("weight", 1.0) for o in expert_outputs.values())
        if total_weight <= 0:
            total_weight = 1.0

        for expert_name, output in expert_outputs.items():
            weight_share = output.get("weight", 1.0) / total_weight
            confidence = output.get("confidence", 0.5)
            expert_pnl = pnl * weight_share
            self._expert_pnl[expert_name] += expert_pnl
            self._expert_trades[expert_name] += 1

            # Update ensemble Elo rating
            if self._ensemble is not None and hasattr(
                self._ensemble, "update_expert_result"
            ):
                self._ensemble.update_expert_result(expert_name, expert_pnl)

            # Track outcome for win rate calculation
            self._expert_outcomes[expert_name].append(expert_pnl)

            # G8 enhanced: update Elo with P&L-weighted scoring
            self._update_expert_elo(expert_name, expert_pnl, confidence)
            # G8 enhanced: update rolling Sharpe per expert
            self._update_expert_sharpe(expert_name, expert_pnl)
            # Track last trade time for decay
            self._expert_last_trade[expert_name] = time.time()

        # Sync enhanced Elo ratings to ensemble for prediction weighting
        self._sync_elo_to_ensemble()

    def _update_expert_elo(
        self, expert_name: str, pnl: float, confidence: float
    ) -> None:
        """Elo update weighted by P&L magnitude and signal confidence."""
        k_factor = 32.0 * abs(confidence)  # Higher confidence = bigger update
        expected = 0.5  # Neutral expectation
        actual = 1.0 if pnl > 0 else 0.0  # Win/loss
        pnl_weight = (
            min(abs(pnl) / self._avg_trade_pnl, 2.0) if self._avg_trade_pnl > 0 else 1.0
        )

        current = self._elo_ratings.get(expert_name, 1200.0)
        new_elo = current + k_factor * pnl_weight * (actual - expected)
        self._elo_ratings[expert_name] = max(1000.0, min(2000.0, new_elo))

    def _update_expert_sharpe(self, expert_name: str, pnl: float) -> None:
        """Rolling Sharpe per expert."""
        self._expert_returns[expert_name].append(pnl)
        returns = self._expert_returns[expert_name]
        if len(returns) >= 20:
            arr = np.array(returns[-100:])
            std = float(np.std(arr))
            if std > 1e-8:
                sharpe = float(np.mean(arr)) / std * np.sqrt(252)
                self._expert_sharpes[expert_name] = sharpe

    def _decay_experts(self) -> None:
        """Elo decay for inactive experts — decay toward 1200."""
        now = time.time()
        for name in list(self._elo_ratings.keys()):
            last = self._expert_last_trade.get(name, 0)
            if now - last > 86400:  # 24h inactivity
                current = self._elo_ratings[name]
                self._elo_ratings[name] += (1200.0 - current) * 0.01

    async def _on_execution_quality(self, **data: Any) -> None:
        """Raise confidence threshold when execution is poor."""
        multiplier = data.get("slippage_multiplier", 1.0)
        self._min_confidence_threshold = 0.55 * multiplier

    async def stop(self) -> None:
        """Clean shutdown."""
        logger.info(f"[signal_engine] Stopped after {self._signal_count} signals")

    @property
    def is_alive(self) -> bool:
        """Whether the engine is initialized."""
        return self._models_loaded
