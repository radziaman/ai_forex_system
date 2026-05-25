"""
Signal Agent — G8: online learning from trade outcomes, G16: confidence calibration.
"""

from __future__ import annotations
import datetime
import os
import time
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
from collections import defaultdict, deque
from loguru import logger

from agentic.core.base_agent import BaseAgent
from agentic.core.agent_message import (
    MessageType,
    MessagePriority,
    AgentIntention,
    AgentMessage,
)
from agentic.core.agent_consciousness import ConsciousnessLevel

from agentic.agents.signal_strategies import StrategyExecutor, STRATEGY_SL_TP
from agentic.agents.signal_model_loader import ModelLoader
from agentic.agents.signal_kill_switch import KillSwitchMonitor
from agentic.agents.signal_confidence import ConfidenceCalibrator


class SignalAgent(BaseAgent):
    def __init__(self, config, ensemble=None, container=None):
        super().__init__(
            name="signal_agent",
            role="Ensemble Signal Generator",
            purpose="Generate symbol-specific trading signals from multi-model ensemble",  # noqa: E501
            domain="signals",
            capabilities={
                "ensemble_signal_generation",
                "mixture_of_experts",
                "elo_rating",
                "sharpe_weighting",
                "concept_drift_detection",
                "regime_gated_inference",
                "online_learning",
                "confidence_calibration",
                "multi_symbol_models",
            },
            tick_interval=1.0,
            consciousness_level=ConsciousnessLevel.REFLECTIVE,
            container=container,
        )
        self.config = config
        self._injected_ensemble = ensemble  # stored for _on_start
        self.ensemble = None
        self._lstm_models: Dict[str, Any] = {}  # symbol -> loaded LSTM model
        self._tft_models: Dict[str, Any] = {}  # symbol -> loaded TFT model
        self._classifiers: Dict[str, Any] = {}  # symbol -> loaded classifier
        self._model_registry = None
        self._regime_manager = None
        self._ppo_state_dim = 49  # fallback default
        self._drift_monitors: Dict[str, Any] = {}
        self._models_loaded = False
        self._signal_count = 0
        self._last_signal_time = 0.0
        self._rejection_count = 0
        self._rejection_reasons: Dict[str, int] = {}
        self._fallback_warnings: set = set()  # Track which symbols we warned about
        self._alpha_strategies: Dict[str, Any] = {}  # name -> strategy instance
        self._alpha_pipeline = None

        # G8: Track outcomes per expert for online learning
        self._expert_outcomes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self._expert_pnl: Dict[str, float] = defaultdict(float)
        self._expert_trades: Dict[str, int] = defaultdict(int)

        # Phase 2: Cache last feature DataFrame for rule-based experts
        self._last_df: Optional[pd.DataFrame] = None
        self._last_price: float = 0.0
        self._current_symbol: str = "EURUSD"
        self._current_session: str = "asia"

        # Phase 3: Strategy performance tracker
        from agentic.agents.strategy_tracker import (
            StrategyTracker,
            PerSymbolStrategyTracker,
        )

        self._strategy_tracker = StrategyTracker()
        self._per_symbol_tracker = PerSymbolStrategyTracker(min_learn_trades=10)

        # Track position_id -> {symbol, expert_outputs, direction} for PnL at close
        self._position_info: Dict[int, Dict] = {}

        # Attribution engine for per-strategy PnL decomposition
        from validation.attribution import StrategyAttributionEngine

        self._attribution_engine = StrategyAttributionEngine()
        self._attribution_trade_counter = 0

        # Active trading symbols: major FX + XAUUSD only
        self._active_symbols = [
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "AUDUSD",
            "USDCAD",
            "USDCHF",
            "NZDUSD",
            "XAUUSD",
        ]

        # Modular components (extracted concerns)
        self._model_loader = ModelLoader(self._active_symbols, config)
        self._strategy_executor = StrategyExecutor(self)
        self._kill_switch = KillSwitchMonitor(self.send)
        self._confidence = ConfidenceCalibrator()

        self.subscribe(MessageType.FEATURES_READY)
        self.subscribe(MessageType.REGIME_CHANGED)
        self.subscribe(MessageType.MODEL_UPDATE)
        self.subscribe(MessageType.EXECUTION_RESULT)  # G8: learn from outcomes
        self.subscribe(MessageType.POSITION_CLOSED)  # Per-symbol PnL tracking
        self.subscribe(MessageType.RISK_REJECTED)  # Learn from rejected trades
        self.subscribe(MessageType.DIAGNOSTIC_REQUEST)
        self.subscribe(
            MessageType.INSTRUMENTS_UPDATED
        )  # Dynamic symbol selection from screener

    async def _on_start(self):
        self.consciousness.current_intention = (
            "loading AI models and initializing ensemble"
        )

        # Resolve ensemble: injected > container > create
        if self._injected_ensemble is not None:
            self.ensemble = self._injected_ensemble
        elif self.container.has("ensemble"):
            self.ensemble = self.container.get("ensemble")
        else:
            from rts_ai_fx.ensemble import MoEEnsemble

            self.ensemble = MoEEnsemble()

        self.ensemble.use_sharpe_weighting = True
        # Wire strategy tracker for dynamic weight adjustments (Phase 4)
        self.ensemble.set_tracker_weight_fn(self._per_symbol_weight_fn)

        # Load models via ModelLoader
        self._model_loader.set_log_fn(self.log_state)
        self._model_loader.load_all()
        self._regime_manager = self._model_loader.regime_manager
        self._lstm_models = self._model_loader.lstm_models
        self._tft_models = self._model_loader.tft_models
        self._classifiers = self._model_loader.classifiers
        self._model_registry = self._model_loader.model_registry
        self._ppo_state_dim = self._model_loader.ppo_state_dim
        self._fallback_warnings = self._model_loader.fallback_warnings
        self._models_loaded = self._model_loader.models_loaded

        self._register_experts()
        self.set_world("signal.models_loaded", self._models_loaded)
        self.set_world(
            "signal.experts", len(self.ensemble.experts) if self.ensemble else 0
        )
        self.log_state(
            f"Signal engine ready: {'experts loaded' if self._models_loaded else 'rule-based only'}"  # noqa: E501
        )

    def _check_kill_switch(self) -> bool:
        """Return True if kill switch is active in world state."""
        return self._kill_switch.is_active(self.get_world)

    async def _handle_kill_switch(self):
        """Log warning, alert Telegram, and attempt recovery after 30 min."""
        result = await self._kill_switch.handle(self.get_world)
        if result == "request_release":
            self.set_world("risk.kill_switch_release_requested", True)

    def _clear_kill_switch_state(self):
        """Reset local kill switch tracking when world state clears."""
        self._kill_switch.clear()

    def _maybe_publish_attribution(self):
        """Publish attribution report to world state every 5 trades."""
        self._attribution_trade_counter += 1
        if (
            self._attribution_trade_counter % 5 == 0
            and self._attribution_engine is not None
        ):
            report = self._attribution_engine.get_report()
            if report:
                self.set_world("strategy_attribution", report)

    def _per_symbol_weight_fn(self, name: str, regime: str) -> float:
        """Dynamic weight incorporating per-symbol strategy performance."""
        global_weight = self._strategy_tracker.get_weight(name, regime)
        symbol = self._current_symbol
        tracker = self._per_symbol_tracker

        if not tracker.is_strategy_enabled(symbol, name):
            return global_weight * 0.1  # Nearly disabled

        best = tracker.get_best_strategy(symbol, regime)
        if best == name:
            return global_weight * 1.5  # Bonus for best strategy on this symbol
        return global_weight * 0.5  # Penalty for non-best

    async def perceive(self) -> Dict[str, Any]:
        return {"ready": self._models_loaded}

    async def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        return {"should_process": True}

    async def act(self, decision: Dict[str, Any]):
        pass

    async def reflect(self, outcome: Dict[str, Any]):
        if self.consciousness.cycle_count % 100 == 0:
            # G16: Log calibration metrics
            self._log_calibration()
            expert_names = [
                e.name for e in (self.ensemble.experts if self.ensemble else [])
            ]
            self.set_world("signal.experts", expert_names)
            # G8: Log expert performance
            for name, pnl in self._expert_pnl.items():
                trades = self._expert_trades.get(name, 0)
                self.memory.know(f"signal.expert.{name}.pnl", round(pnl, 2), ttl=3600)
                self.memory.know(f"signal.expert.{name}.trades", trades, ttl=3600)

    async def on_message(self, message: AgentMessage):
        if message.msg_type == MessageType.INSTRUMENTS_UPDATED:
            # Dynamically update active symbols from screener
            payload = message.payload if isinstance(message.payload, dict) else {}
            tradeable = payload.get("tradeable", [])
            if tradeable:
                new_symbols = [
                    t.get("ticker", "") for t in tradeable if t.get("ticker")
                ]
                if new_symbols:
                    # Merge screener findings with core symbols.
                    # Only add symbols that are in our trading universe.
                    CORE_SYMBOLS = [
                        "EURUSD",
                        "GBPUSD",
                        "USDJPY",
                        "AUDUSD",
                        "USDCAD",
                        "USDCHF",
                        "NZDUSD",
                        "XAUUSD",
                    ]
                    # Only merge tickers that look like FX pairs or XAU
                    valid_new = [
                        t
                        for t in new_symbols
                        if any(c in t.upper() for c in CORE_SYMBOLS)
                        or "XAU" in t.upper()
                        or "XAG" in t.upper()
                    ]
                    merged = list(dict.fromkeys(CORE_SYMBOLS + valid_new))
                    self._active_symbols = merged
                    self.set_world("signal.active_symbols", merged)
                    if valid_new:
                        self.log_state(
                            f"Screener merged: {len(merged)} symbols "
                            f"(+{len(valid_new)} from screener)"
                        )
            return

        if message.msg_type == MessageType.FEATURES_READY:
            await self._on_features(message)
        elif message.msg_type == MessageType.EXECUTION_RESULT:
            # G8: Learn from trade outcome
            await self._on_execution_result(message)
        elif message.msg_type == MessageType.POSITION_CLOSED:
            # Per-symbol PnL tracking
            await self._on_position_closed(message)
        elif message.msg_type == MessageType.RISK_REJECTED:
            # Track rejection reasons for learning / calibration
            payload = message.payload if isinstance(message.payload, dict) else {}
            reason = payload.get("reason", "unknown")
            signal_info = payload.get("signal", {})
            symbol = signal_info.get("symbol", "unknown")
            self._rejection_count += 1
            if not hasattr(self, "_rejection_reasons"):
                self._rejection_reasons = {}
            self._rejection_reasons[reason] = self._rejection_reasons.get(reason, 0) + 1
            self.set_world(
                f"signal.rejected.{symbol}",
                {
                    "reason": reason,
                    "count": self._rejection_reasons[reason],
                    "timestamp": time.time(),
                },
            )
        elif message.msg_type == MessageType.MODEL_UPDATE:
            await self.send(
                MessageType.DIAGNOSTIC_RESULT,
                payload={
                    "agent": self.name,
                    "models_loaded": self._models_loaded,
                    "n_experts": len(self.ensemble.experts) if self.ensemble else 0,
                    "signals": self._signal_count,
                },
                target=message.source_agent,
            )

    async def _on_features(self, message: AgentMessage):
        # Step 1: Extract and validate payload
        extracted = self._extract_signal_payload(message)
        if extracted is None:
            logger.debug("_on_features: extracted is None")
            return
        logger.debug(
            f"_on_features: {extracted['symbol']} features={extracted['features'].shape}"
        )
        symbol = extracted["symbol"]
        features = extracted["features"]
        df = extracted["df"]
        price = extracted["price"]

        # Task 2: Pre-trade kill switch check
        if self._check_kill_switch():
            await self._handle_kill_switch()
            return
        # Clear stale kill switch state if it was previously active
        self._clear_kill_switch_state()

        # Per-symbol: skip symbols with no profitable strategy yet
        if not self._per_symbol_tracker.is_symbol_tradeable(symbol):
            self.log_state(
                f"Skipping {symbol}: no profitable strategy found yet", "debug"
            )
            return

        # Cache for rule-based experts
        self._cache_signal_state(df, price, symbol)

        # Regime detection
        regime_str = self._detect_regime(df, symbol)
        if not self.ensemble:
            return

        # Ensemble prediction
        pred = self.ensemble.predict(features, regime=regime_str)
        if pred is None or not hasattr(pred, "confidence") or pred.confidence is None:
            logger.debug(
                f"ensemble predict failed for {symbol}: pred type={type(pred).__name__}"
            )
            return

        should_trade, direction_str, agreement = self.ensemble.should_trade(
            pred,
            price,
            min_confidence=0.40,
        )
        if not should_trade:
            logger.debug(
                f"should_trade=False for {symbol}: conf={pred.confidence:.3f} dir={direction_str} agree={agreement:.3f}"
            )
            return

        # ── Transaction cost gate: reject if edge doesn't cover spread ──
        edge = abs(float(getattr(pred, "price", 0))) * float(
            getattr(pred, "confidence", 0.5)
        )
        logger.debug(
            f"Signal gate for {symbol}: edge={edge:.6f} price={getattr(pred,'price',0):.6f} conf={getattr(pred,'confidence',0):.3f}"
        )
        spread_pips = self.get_world(f"data.spread.{symbol}", 0)
        pip_value = self.get_world(f"data.pip_value.{symbol}", 0.10)
        commission = 0.07  # ~$7/lot × 0.01 lots
        # Use calibrated realized slippage if available, else fall back to raw spread
        realized_slippage_bps = self.get_world(f"execution.slippage_bps.{symbol}", None)
        if realized_slippage_bps is not None:
            slippage_cost = realized_slippage_bps / 10_000 * price  # bps → $ per unit
            trade_cost = max(spread_pips * pip_value, slippage_cost) + commission
        else:
            trade_cost = spread_pips * pip_value + commission
        min_edge_ratio = 2.0  # edge must be at least 2× cost
        if edge > 0 and trade_cost > 0 and edge < trade_cost * min_edge_ratio:
            self.set_world(
                f"signal.rejected.{symbol}",
                {
                    "reason": f"edge_too_thin: edge={edge:.4f} cost={trade_cost:.4f}",
                    "count": 1,
                    "timestamp": time.time(),
                },
            )
            self.log_state(
                f"Rejected {symbol}: edge ${edge:.4f} < ${trade_cost:.4f} × {min_edge_ratio}",
                "debug",
            )
            return

        self._signal_count += 1
        self._last_signal_time = time.time()

        # Best expert selection
        best_expert, sl_atr, tp_atr = self._select_best_expert(
            getattr(pred, "expert_outputs", {})
        )

        await self.send(
            MessageType.SIGNAL_GENERATED,
            payload=self._build_signal_message(
                symbol=symbol,
                direction_str=direction_str,
                pred=pred,
                regime_str=regime_str,
                price=price,
                agreement=agreement,
                best_expert=best_expert,
                sl_atr=sl_atr,
                tp_atr=tp_atr,
            ),
            priority=MessagePriority.NORMAL,
            intention=AgentIntention(
                primary_goal=f"generate {direction_str} signal for {symbol}",
                reasoning=f"ensemble conf={pred.confidence:.2f} agree={agreement:.2f} "
                f"strat={best_expert} sl_atr={sl_atr} tp_atr={tp_atr}",
                expected_outcome="risk agent evaluates and gatekeeper approves or rejects",  # noqa: E501
                confidence=float(pred.confidence),
            ),
        )

    def _extract_signal_payload(
        self, message: AgentMessage
    ) -> Optional[Dict[str, Any]]:
        """Extract and validate signal payload from message.

        Returns a dict with symbol, features, df, price, and optionally
        orthogonal signal metadata (sentiment, macro, options, NASA).
        Returns None if features are missing.
        """
        payload = message.payload if isinstance(message.payload, dict) else {}
        symbol = payload.get("symbol", "")
        features = payload.get("features")
        df = payload.get("ohlcv")
        price = payload.get("price", 0)
        if features is None:
            return None
        result: Dict[str, Any] = {
            "symbol": symbol,
            "features": features,
            "df": df,
            "price": price,
        }
        # Orthogonal data from feature_agent (attached as metadata,
        # not fused into feature vector — preserves model input dimensions)
        sentiment = payload.get("sentiment_scores")
        if sentiment:
            result["sentiment_scores"] = sentiment
        ext = payload.get("external_signals")
        if ext:
            result["external_signals"] = ext
        return result

    def _cache_signal_state(self, df, price, symbol):
        """Cache current data for rule-based experts and set trading session."""
        if df is not None:
            self._last_df = df
        if price != 0:
            self._last_price = price
        self._current_symbol = symbol
        self._current_session = self._get_current_session()

    def _detect_regime(self, df, symbol) -> str:
        """Detect market regime using HMM detector or world state fallback."""
        if df is not None and len(df) > 60:
            from rts_ai_fx.regime_detector import HMMRegimeDetector

            detector = HMMRegimeDetector()
            return detector.detect_regime(df)
        return self.get_world(f"regime.{symbol}", "ranging")

    def _select_best_expert(self, expert_outputs):
        """Select the best expert and return its SL/TP multipliers.

        Returns (expert_name, sl_atr, tp_atr).
        """
        if expert_outputs:
            best_expert = max(
                expert_outputs.items(), key=lambda x: x[1].get("weight", 0)
            )[0]
        else:
            best_expert = "unknown"
        sl_atr, tp_atr = STRATEGY_SL_TP.get(best_expert, (2.0, 4.0))
        return best_expert, sl_atr, tp_atr

    def _build_signal_message(
        self,
        symbol,
        direction_str,
        pred,
        regime_str,
        price,
        agreement,
        best_expert,
        sl_atr,
        tp_atr,
    ) -> Dict[str, Any]:
        """Build the SIGNAL_GENERATED message payload."""
        return {
            "symbol": symbol,
            "direction": direction_str,
            "confidence": getattr(pred, "confidence", 0.5),
            "regime": regime_str,
            "session": self._current_session,
            "price": price,
            "agreement": agreement,
            "expert_outputs": getattr(pred, "expert_outputs", {}),
            "ensemble_price": getattr(pred, "price", price),
            "strategy": best_expert,
            "sl_atr": sl_atr,
            "tp_atr": tp_atr,
            "timestamp": time.time(),
        }

    # G8: Learn from execution outcomes
    async def _on_execution_result(self, message: AgentMessage):
        payload = message.payload if isinstance(message.payload, dict) else {}
        if not payload.get("success"):
            return

        symbol = payload.get("symbol", "EURUSD")
        expert_outputs = payload.get("signal", {}).get("expert_outputs", {})
        position_id = payload.get("position_id", 0)

        # Cache position info for PnL tracking at close time
        if position_id and expert_outputs:
            self._position_info[position_id] = {
                "symbol": symbol,
                "expert_outputs": expert_outputs,
                "direction": payload.get("direction", ""),
            }

        # Update Elo ratings based on outcome
        if self.ensemble and payload.get("filled_price") and expert_outputs:
            for expert_name, output in expert_outputs.items():
                was_correct = (
                    payload.get("direction") == "BUY"
                    and output.get("prediction", 0) > 0
                ) or (
                    payload.get("direction") == "SELL"
                    and output.get("prediction", 0) < 0
                )
                self.ensemble.update_elo(expert_name, was_correct)
                self._expert_outcomes[expert_name].append(1 if was_correct else 0)
                self._expert_trades[expert_name] += 1

                # Track which experts participated (PnL recorded at close)
                self._per_symbol_tracker.register(symbol, expert_name)

        # G16: Calibrate confidence vs actual outcomes
        confidence = payload.get("signal", {}).get("confidence", 0.5)
        self._confidence.record_outcome(confidence, bool(payload.get("success")))

    # Per-symbol: track actual PnL when positions close
    async def _on_position_closed(self, message: AgentMessage):
        payload = message.payload if isinstance(message.payload, dict) else {}
        pnl = payload.get("pnl", 0)
        symbol = payload.get("symbol", "EURUSD")
        position_id = payload.get("position_id", 0)

        # Look up cached expert info for this position
        info = self._position_info.pop(position_id, None)
        if info is None:
            return

        expert_outputs = info.get("expert_outputs", {})
        symbol = info.get("symbol", symbol)  # Use cached symbol

        # Distribute PnL to each expert that contributed
        total_weight = sum(o.get("weight", 1.0) for o in expert_outputs.values())
        if total_weight <= 0:
            total_weight = 1.0

        for expert_name, output in expert_outputs.items():
            weight_share = output.get("weight", 1.0) / total_weight
            expert_pnl = pnl * weight_share
            self._expert_pnl[expert_name] += expert_pnl

            if hasattr(self.ensemble, "update_expert_result"):
                self.ensemble.update_expert_result(expert_name, expert_pnl)

            # Global strategy tracker
            self._strategy_tracker.record_trade(expert_name, expert_pnl)

            # Per-symbol strategy tracker
            self._per_symbol_tracker.record_trade(symbol, expert_name, expert_pnl)

            # Attribution engine: decompose PnL
            if self._attribution_engine is not None:
                self._attribution_engine.attribute_trade(
                    {
                        "pnl": expert_pnl,
                        "expected_pnl": expert_pnl * 0.8,
                        "fill_price": payload.get("fill_price", 0.0),
                        "signal_price": payload.get("signal_price", 0.0),
                        "strategy": expert_name,
                        "market_return": 0.0,
                    }
                )

        # Publish tradeable symbols to world state for execution agent
        tradeable = [
            s
            for s in self._active_symbols
            if self._per_symbol_tracker.is_symbol_tradeable(s)
        ]
        self.set_world("signal.tradeable_symbols", tradeable)

        # Publish attribution report to world state every 5 trades
        self._maybe_publish_attribution()

    # G16: Log calibration report
    def _log_calibration(self):
        report = self._confidence.get_report()
        if report:
            self.memory.know("signal.calibration", report, ttl=3600)

    def _load_models(self):  # noqa: C901
        """Load all models via ModelLoader.

        ModelLoader handles PPO regime agents, LSTM, TFT, and classifiers.
        After loading, sync results back to SignalAgent attributes.
        """
        self._model_loader.set_log_fn(self.log_state)
        loaded = self._model_loader.load_all()
        self._regime_manager = self._model_loader.regime_manager
        self._lstm_models = self._model_loader.lstm_models
        self._tft_models = self._model_loader.tft_models
        self._classifiers = self._model_loader.classifiers
        self._model_registry = self._model_loader.model_registry
        self._ppo_state_dim = self._model_loader.ppo_state_dim
        self._fallback_warnings = self._model_loader.fallback_warnings
        self._models_loaded = self._model_loader.models_loaded

        # Publish PPO training status to world state
        has_real_weights = False
        if self._regime_manager:
            has_real_weights = any(
                any(p.norm().item() > 1.0 for p in agent.actor.parameters())
                for agent in self._regime_manager.agents.values()
                if agent
            )
        self.set_world("models.ppo_trained", has_real_weights)
        if not has_real_weights and self._regime_manager:
            self.set_world("models.untrained", True)

    def _register_experts(self):  # noqa: C901
        if not self.ensemble:
            return
        self.ensemble.experts = []
        self.ensemble.elo_ratings = {}

        # Load alpha strategies first (StrategyExecutor.get_all_specs references them)
        self._load_alpha_strategies()

        # Get all strategy specs from StrategyExecutor
        specs = self._strategy_executor.get_all_specs()
        for spec in specs:
            self.ensemble.add_expert(
                name=spec.name,
                predict_fn=spec.predict_fn,
                confidence_fn=spec.confidence_fn,
                regime=spec.regime,
            )
            self._strategy_tracker.register_strategy(spec.name, spec.regime)
            for sym in self._active_symbols:
                self._per_symbol_tracker.register(sym, spec.name, spec.regime)

        # Phase 14: Alpha research pipeline — auto-discover and validate signals
        try:
            from ai.alpha_research import AlphaResearchPipeline, AlphaPipelineConfig

            self._alpha_pipeline = AlphaResearchPipeline(
                config=AlphaPipelineConfig(min_ic=0.02, min_ir=0.5, min_t_stat=2.0),
                registry_path="models/signal_registry.json",
            )
            self.log_state("Alpha research pipeline initialized")
        except Exception as e:
            self.log_state(f"Alpha pipeline not initialized: {e}", "warning")

        # Phase 15: Meta-learning orchestrator for rapid adaptation
        try:
            from ai.maml_scaler import MetaLearningOrchestrator, MetaConfig

            self._meta_learner = MetaLearningOrchestrator(
                config=MetaConfig(
                    inner_lr=0.01, inner_steps=5, adaptation_threshold=0.15
                )
            )
            for sym in self._active_symbols:
                self._meta_learner.register_model(sym)
            self.log_state(
                f"Meta-learner registered for {len(self._active_symbols)} symbols"
            )
        except Exception as e:
            self.log_state(f"Meta-learner not initialized: {e}", "warning")

    def _load_alpha_strategies(self):
        """Load alpha strategies and cross-sectional alpha.

        Populates _alpha_strategies and _cross_sectional_alpha for use
        by StrategyExecutor and the ensemble.
        """
        # Phase 12: Alpha strategies from StrategyRegistry
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

            # Map regimes to strategies for dynamic selection
            registry.map_regime("trending", ["stat_arb", "carry_trade"])
            registry.map_regime("ranging", ["event_driven", "order_flow_momentum"])
            registry.map_regime("volatile", ["vol_expansion", "event_driven"])
            registry.map_regime("crisis", ["event_driven"])

            for name, strat_class in registry.get_all().items():
                instance = strat_class(symbol=self._current_symbol, params={})
                self._alpha_strategies[name] = instance
        except Exception as e:
            self.log_state(f"Alpha strategies not loaded: {e}", "warning")

        # Phase 13: Cross-sectional alpha expert (relative value signals)
        try:
            from rts_ai_fx.cross_sectional_alpha import CrossSectionalAlpha

            self._cross_sectional_alpha = CrossSectionalAlpha()
        except Exception as e:
            self.log_state(f"Cross-sectional alpha not loaded: {e}", "warning")

    def _make_alpha_predict_fn(self, name: str):
        """Closure that routes ensemble calls to the correct alpha strategy."""
        return lambda X, n=name: self._alpha_prediction(X, n)

    def _make_alpha_confidence_fn(self, name: str):
        return lambda X, n=name: self._alpha_confidence(X, n)

    def _alpha_regime_for(self, name: str) -> str:
        mapping = {
            "stat_arb": "ranging",
            "carry_trade": "trending",
            "event_driven": "volatile",
            "vol_expansion": "volatile",
            "order_flow_momentum": "ranging",
        }
        return mapping.get(name, "ranging")

    def _alpha_prediction(self, X: np.ndarray, name: str) -> float:
        """Convert alpha strategy signal direction to a price-like prediction."""
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

    def _alpha_confidence(self, X: np.ndarray, name: str) -> float:
        """Return alpha strategy confidence."""
        strategy = self._alpha_strategies.get(name)
        if strategy is None or self._last_df is None:
            return 0.5
        try:
            sig = strategy.generate_signal(self._last_df)
            return float(sig.get("confidence", 0.5))
        except Exception:
            return 0.5

    def _cross_sectional_predict(self, X: np.ndarray) -> float:
        """Cross-sectional alpha prediction across all symbols."""
        try:
            if (
                not hasattr(self, "_cross_sectional_alpha")
                or self._cross_sectional_alpha is None
            ):
                return 0.0
            prices = {}
            for sym in self._active_symbols:
                bid = self.get_world(f"data.bid.{sym}", 0)
                ask = self.get_world(f"data.ask.{sym}", 0)
                if bid > 0 and ask > 0:
                    prices[sym] = pd.Series([(bid + ask) / 2])
            if len(prices) < 2:
                return 0.0
            signal = self._cross_sectional_alpha.compute_all(prices)
            composite = signal.composite_zscore
            if not composite:
                return 0.0
            return float(np.mean(list(composite.values())))
        except Exception:
            return 0.0

    def _tft_prediction(self, X: np.ndarray) -> float:
        """TFT ensemble prediction for the current symbol."""
        model = self._tft_models.get(self._current_symbol)
        if model is None:
            model = self._tft_models.get("EURUSD") or next(
                iter(self._tft_models.values()), None
            )
        if model is None:
            return 0.0
        try:
            X_arr = np.asarray(X)
            if X_arr.ndim == 2:
                X_arr = X_arr.reshape(1, X_arr.shape[0], X_arr.shape[1])
            elif X_arr.ndim == 1:
                X_arr = X_arr.reshape(1, 1, -1)
            # Build static context: simple symbol + session encoding
            static = self._build_static_context()
            probs = model.predict(X_arr, static)
            prob = float(probs.flatten()[0])
            # Map probability [0,1] to signed prediction centered at 0.5
            return (prob - 0.5) * 0.02
        except Exception:
            return 0.0

    def _tft_confidence(self, X: np.ndarray) -> float:
        """TFT confidence = distance from 0.5 in probability space."""
        model = self._tft_models.get(self._current_symbol)
        if model is None:
            return 0.5
        try:
            X_arr = np.asarray(X)
            if X_arr.ndim == 2:
                X_arr = X_arr.reshape(1, X_arr.shape[0], X_arr.shape[1])
            elif X_arr.ndim == 1:
                X_arr = X_arr.reshape(1, 1, -1)
            static = self._build_static_context()
            probs = model.predict(X_arr, static)
            prob = float(probs.flatten()[0])
            return 0.5 + abs(prob - 0.5)
        except Exception:
            return 0.5

    def _build_static_context(self) -> np.ndarray:
        """Encode current symbol and session as a static vector for TFT."""
        sym = self._current_symbol.upper()
        session = self._current_session
        # Simple one-hot-ish encoding (8 dims)
        vec = np.zeros(8, dtype=np.float32)
        # First 3 dims: symbol hash
        h = hash(sym) % 1000 / 1000.0
        vec[0] = h
        vec[1] = np.sin(h * np.pi * 2)
        vec[2] = np.cos(h * np.pi * 2)
        # Next dims: session encoding
        sessions = ["asia", "london", "overlap", "newyork", "pacific"]
        if session in sessions:
            idx = sessions.index(session)
            vec[3 + idx] = 1.0
        return vec

    def _get_lstm_prediction(self, symbol: str) -> float:
        """Get symbol-specific LSTM prediction."""
        model = self._lstm_models.get(symbol) or self._lstm_models.get("EURUSD")
        if model is None:
            model = next(iter(self._lstm_models.values()), None)
        if model is None:
            return 0.0
        # This is called from _on_features where we have the features
        return 0.0  # Placeholder; actual prediction is done in _on_features

    def _features_to_ppo_state(self, X: np.ndarray) -> np.ndarray:
        """Extract a flat state vector from multi-bar feature matrix matching PPO dims."""  # noqa: E501
        X_arr = np.asarray(X)
        if X_arr.ndim == 3:
            X_arr = X_arr[0]
        if X_arr.ndim == 2:
            state = X_arr[-1, :]
        else:
            state = X_arr.flatten()
        target_dim = (
            self._regime_manager.state_dim
            if self._regime_manager
            else self._ppo_state_dim
        )
        state = state[:target_dim]
        if len(state) < target_dim:
            state = np.pad(state, (0, target_dim - len(state)))
        return state.astype(np.float32)

    def _sane_prediction(
        self, value: float, label: str = "model", bounds: tuple = (-10, 10)
    ) -> float:
        """Fix 4: Sanity-check a model prediction for NaN, Inf, and realistic bounds."""
        if value is None or (hasattr(value, "__len__") and len(value) == 0):
            self.memory.remember(
                f"{label}_prediction_invalid", "None/empty prediction", importance=0.3
            )
            return 0.0
        v = float(value) if hasattr(value, "__float__") else 0.0
        if not np.isfinite(v):
            self.memory.remember(
                f"{label}_prediction_nan",
                f"Non-finite prediction: {v}",
                importance=0.5,
                emotion="warning",
            )
            return 0.0
        lo, hi = bounds
        if v < lo or v > hi:
            return float(np.clip(v, lo, hi))
        return v

    def _ppo_prediction(self, X: np.ndarray, regime: str = "ranging") -> float:
        if self._regime_manager is None:
            return 0.0
        try:
            state = self._features_to_ppo_state(X)
            action, *rest = self._regime_manager.select_action(state, regime=regime)
            pred = 0.001 if action in (1, 3) else -0.001 if action in (2, 4) else 0.0
            return self._sane_prediction(pred, "ppo", (-0.01, 0.01))
        except Exception:
            return 0.0

    def _ppo_confidence(self, X: np.ndarray, regime: str = "ranging") -> float:
        if self._regime_manager is None:
            return 0.5
        try:
            state = self._features_to_ppo_state(X)
            action, *rest = self._regime_manager.select_action(state, regime=regime)
            # rest = (sl_raw, tp_raw, size_raw, info_dict)
            info = rest[3] if len(rest) >= 4 else {}
            action_logits = info.get("action_logits")
            if action_logits is not None:
                logits = np.array(action_logits, dtype=np.float64)
                logits -= logits.max()  # numerical stability
                probs = np.exp(logits) / np.sum(np.exp(logits))
                prob = float(probs[action]) if action < len(probs) else 0.6
            else:
                prob = 0.6
            return float(np.clip(prob, 0.0, 1.0))
        except Exception:
            return 0.5

    # ── Phase 2: Rule-based experts ──────────────────────────────────────

    def _get_current_session(self) -> str:
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

    def _compute_atr(self, period: int = 14) -> float:
        """Compute ATR from cached DataFrame."""
        if self._last_df is None or len(self._last_df) < period + 1:
            return 0.0
        df = self._last_df
        high_low = df["high"] - df["low"]
        atr = high_low.rolling(period).mean()
        return (
            float(atr.iloc[-1]) if not atr.empty and not np.isnan(atr.iloc[-1]) else 0.0
        )

    def _rule_breakout_prediction(self, X: np.ndarray) -> float:
        """Session-aware breakout: 3-bar during London, 5-bar during NY, skip during Asia."""  # noqa: E501
        if self._last_df is None or len(self._last_df) < 10:
            return 0.0

        session = getattr(self, "_current_session", "asia")

        # Asia: don't trade breakouts (low volatility, ranging markets)
        if session == "asia":
            return 0.0

        try:
            df = self._last_df
            close = float(df["close"].iloc[-1])
            atr = self._compute_atr(14)
            if atr <= 0:
                return 0.0

            # London: aggressive 3-bar breakout; elsewhere: standard 5-bar
            lookback_bars = 3 if session == "london" else 5
            high_n = float(df["high"].iloc[-(lookback_bars + 1) : -1].max())
            low_n = float(df["low"].iloc[-(lookback_bars + 1) : -1].min())

            # Long breakout
            if close > high_n + 0.5 * atr:
                return 0.005
            # Short breakout (close < lowest low - 0.5*ATR)
            if close < low_n - 0.5 * atr:
                return -0.005
            return 0.0
        except Exception:
            return 0.0

    def _rule_breakout_confidence(self, X: np.ndarray) -> float:
        return 0.65

    def _rule_mean_rev_prediction(self, X: np.ndarray) -> float:  # noqa: C901
        """Session-aware mean reversion: adjusted z-score thresholds per session."""
        if self._last_df is None or len(self._last_df) < 25:
            return 0.0

        session = getattr(self, "_current_session", "asia")

        try:
            df = self._last_df
            close_series = df["close"]
            close_val = float(close_series.iloc[-1])
            sma20 = float(close_series.rolling(20).mean().iloc[-1])
            std20 = float(close_series.rolling(20).std().iloc[-1])
            if std20 <= 0:
                return 0.0
            z = (close_val - sma20) / std20

            # Session-based z-score thresholds
            if session == "asia":
                z_threshold = 1.2  # Easier to trigger in low vol
            elif session == "overlap":
                z_threshold = 2.0  # Higher threshold to avoid noise
            else:
                z_threshold = 1.5  # Standard (london, ny, pacific)

            # RSI from feature columns or compute
            rsi = None
            if "rsi_14" in df.columns:
                rsi_val = df["rsi_14"].iloc[-1]
                if not np.isnan(rsi_val):
                    rsi = float(rsi_val)
            if rsi is None:
                # Compute simple RSI from price changes
                delta = close_series.diff()
                gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
                loss = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
                if loss and loss > 0:
                    rsi = float(100 - 100 / (1 + gain / loss))
                else:
                    rsi = 50.0
            # Long: oversold
            if z < -z_threshold and rsi < 40:
                return 0.005
            # Short: overbought
            if z > z_threshold and rsi > 60:
                return -0.005
            return 0.0
        except Exception:
            return 0.0

    def _rule_mean_rev_confidence(self, X: np.ndarray) -> float:
        return 0.6

    # ── Phase 6: Research-backed forex strategies ────────────────────────

    def _bb_squeeze_prediction(self, X: np.ndarray) -> float:
        """Bollinger Band Squeeze (Bollinger 2002).

        Detects volatility contraction/expansion cycles:
        - When BB width reaches a 100-period low and starts expanding,
          enter in the direction of the breakout relative to BB mid.
        """
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

            if pd.isna(bb_current) or pd.isna(bb_prev) or pd.isna(bb_100_min):
                return 0.0

            # Squeeze: BB width at/near 100-period low, starting to expand
            if bb_current <= bb_100_min * 1.01 and bb_current > bb_prev:
                close = float(df["close"].iloc[-1])
                bb_mid = float(df["bb_mid"].iloc[-1])
                if close > bb_mid:
                    return 0.005  # Long breakout above mid
                elif close < bb_mid:
                    return -0.005  # Short breakout below mid
            return 0.0
        except Exception:
            return 0.0

    def _ts_momentum_prediction(self, X: np.ndarray) -> float:
        """Time Series Momentum (Moskowitz, Ooi & Pedersen 2012).

        Multi-horizon agreement: requires mom_1, mom_5, and mom_10 to all
        agree on direction before entering a momentum trade.
        """
        if self._last_df is None or len(self._last_df) < 15:
            return 0.0
        try:
            df = self._last_df
            if not all(col in df.columns for col in ["mom_1", "mom_5", "mom_10"]):
                return 0.0

            m1 = float(df["mom_1"].iloc[-1])
            m5 = float(df["mom_5"].iloc[-1])
            m10 = float(df["mom_10"].iloc[-1])

            if pd.isna(m1) or pd.isna(m5) or pd.isna(m10):
                return 0.0

            # All three horizons must agree
            if m5 > 0 and m10 > 0 and m1 > 0:
                return 0.005  # Momentum long
            if m5 < 0 and m10 < 0 and m1 < 0:
                return -0.005  # Momentum short
            return 0.0
        except Exception:
            return 0.0

    def _vol_mean_rev_prediction(self, X: np.ndarray) -> float:
        """Volatility Mean Reversion (Bollerslev, Tauchen & Zhou 2009).

        When volatility spikes (vol_ratio > 1.5) and RSI is extreme,
        expect mean reversion:
        - vol_ratio > 1.5 + RSI < 30 → long (extreme fear = buy)
        - vol_ratio > 1.5 + RSI > 70 → short (extreme greed = sell)
        """
        if self._last_df is None or len(self._last_df) < 20:
            return 0.0
        try:
            df = self._last_df
            if "vol_ratio" not in df.columns or "rsi_14" not in df.columns:
                return 0.0

            vr = float(df["vol_ratio"].iloc[-1])
            rsi = float(df["rsi_14"].iloc[-1])

            if pd.isna(vr) or pd.isna(rsi):
                return 0.0

            if vr > 1.5:  # Volatility 50% above normal — mean reversion likely
                if rsi < 30:
                    return 0.005  # Oversold + vol spike = long
                if rsi > 70:
                    return -0.005  # Overbought + vol spike = short
            return 0.0
        except Exception:
            return 0.0

    # ── Phase 7: Order flow / CVD expert ────────────────────────────────

    def _orderflow_prediction(self, X: np.ndarray) -> float:
        """Order flow / CVD-based prediction (Evans & Lyons 2002).

        Uses Cumulative Volume Delta (CVD) and order book imbalance from
        depth events.  Key signals:
          - CVD rising + price flat → accumulation → BUY
          - CVD falling + price flat → distribution → SELL
          - Extreme DOM imbalance (>0.6 or <-0.6) → mean reversion
          - CVD divergence (price up/CVD down) → reversal SELL

        Data sources (via world state, populated by DataAgent):
          - data.orderflow.{symbol}.cvd
          - data.orderflow.{symbol}.imbalance
          - data.orderflow.{symbol}.dom_imbalance
        """
        try:
            if not hasattr(self, "_current_symbol") or not self._current_symbol:
                return 0.0

            sym = self._current_symbol
            of = self.get_world(f"data.orderflow.{sym}", {})

            cvd = of.get("cvd", 0.0)
            cvd_slope = of.get("cvd_slope", 0.0)
            dom_imb = of.get("dom_imbalance", 0.0)

            signal = 0.0

            # Signal 1: CVD slope direction
            if abs(cvd_slope) > 0.1:
                signal += cvd_slope * 0.002  # Scale to price move

            # Signal 2: Extreme DOM imbalance → mean reversion
            if abs(dom_imb) > 0.6:
                signal -= dom_imb * 0.003  # Fade the imbalance

            # Signal 3: CVD running absolute level beyond normal range
            if abs(cvd) > 500:
                signal += (cvd / 5000.0) * 0.001

            return float(np.clip(signal, -0.008, 0.008))

        except Exception:
            return 0.0

    def _orderflow_confidence(self, X: np.ndarray) -> float:
        """Confidence based on how extreme the order flow signals are."""
        try:
            sym = self._current_symbol
            of = self.get_world(f"data.orderflow.{sym}", {})
            cvd = abs(of.get("cvd", 0.0))
            dom_imb = abs(of.get("dom_imbalance", 0.0))
            # Higher confidence when signals are more extreme
            confidence = min(0.4 + (cvd / 2000.0) * 0.3 + dom_imb * 0.3, 0.75)
            return float(confidence)
        except Exception:
            return 0.4

    # ── Phase 8: Macro sentiment expert ─────────────────────────────────

    def _macro_sentiment_prediction(self, X: np.ndarray) -> float:
        """Macroeconomic sentiment prediction using FRED/economic calendar data.

        (Andersen, Bollerslev, Diebold & Vega 2003)
          - High-impact events (NFP, CPI, rate decisions) cause directional moves
          - Suppress trading before major events (event risk)
          - After event: trade in direction of surprise
        """
        try:
            macro = self.get_world("macro.data", {})
            upcoming = macro.get("upcoming_events", [])

            if not upcoming:
                return 0.0

            # Check if any high-impact event is within 2 hours
            now = time.time()
            for ev in upcoming:
                event_ts = ev.get("timestamp", 0)
                hours_until = (event_ts - now) / 3600
                if ev.get("impact") == "high" and 0 < hours_until < 2:
                    # Suppress: return HOLD before major events
                    return 0.0

            # If we had a recent event (within 1 hour), look for aftermath
            # direction (simplified: direction = event severity × currency strength)
            recent = [ev for ev in upcoming if ev.get("timestamp", 0) > now - 3600]
            if recent:
                # Fade the event: most events cause overshoot then reversion
                return -0.002  # Small mean reversion bias after events

            return 0.0

        except Exception:
            return 0.0

    # ── Phase 9: Social sentiment expert ────────────────────────────────

    def _social_sentiment_prediction(self, X: np.ndarray) -> float:
        """Social media sentiment prediction (Twitter/Reddit).

        Reads sentiment scores published by monitoring_agent from world state.
        Social sentiment is a noisy but leading indicator for retail-driven
        moves, especially in crypto and retail FX pairs.

        Signal logic:
          - Extreme sentiment (>0.6 or <-0.6) → contrarian fade
          - Moderate sentiment (0.2-0.4) → follow trend
          - No data → neutral (0.0)
        """
        try:
            sentiment = self.get_world("sentiment.aggregate", {})
            score = sentiment.get("score", 0.0)
            confidence = sentiment.get("confidence", 0.0)

            if confidence < 0.3:
                return 0.0  # Not enough data

            if abs(score) > 0.6:
                # Extreme sentiment → contrarian fade
                return float(-score * 0.003)
            elif abs(score) > 0.2:
                # Moderate sentiment → follow trend
                return float(score * 0.002)
            return 0.0

        except Exception:
            return 0.0

    # ── Phase 10: XGBoost expert (Chen 2016) ────────────────────────────

    def _xgboost_prediction(self, X: np.ndarray) -> float:
        """XGBoost gradient-boosted trees prediction.

        Ensemble diversity principle (Brown 2005):
        Combining XGBoost (tree-based) with neural network experts
        produces more robust ensemble predictions than either alone.
        """
        try:
            sym = self._current_symbol
            if not sym:
                return 0.0

            # Load gradient boosting model for this symbol (XGBoost or sklearn)
            import joblib
            import os

            model_path = f"models/xgboost_{sym}.pkl"
            if not os.path.exists(model_path):
                return 0.0

            gbm_model = joblib.load(model_path)
            X_arr = np.asarray(X).reshape(1, -1)

            # Model expects 2D (samples, features) — flatten if 3D (lookback, features)
            if X_arr.ndim == 3:
                X_arr = X_arr.reshape(X_arr.shape[0], -1)

            pred = gbm_model.predict(X_arr)[0]
            return float(np.clip(pred, -0.01, 0.01))

        except Exception:
            return 0.0

    # ── Phase 4: LSTM ensemble expert ────────────────────────────────────

    def _lstm_ensemble_prediction(self, X: np.ndarray) -> float:
        """LSTM prediction for the current symbol, used as an ensemble expert."""
        return self._lstm_prediction_for_symbol(X, self._current_symbol)

    # ── Feature helpers ──────────────────────────────────────────────────

    def _features_to_lstm_input(
        self, X: np.ndarray, symbol: str = "EURUSD"
    ) -> np.ndarray:
        """Reshape features for LSTM using the symbol-specific model's dimensions."""
        X_arr = np.asarray(X)
        if X_arr.ndim == 3 and X_arr.shape[0] == 1:
            X_arr = X_arr[0]

        model = self._lstm_models.get(symbol) or next(
            iter(self._lstm_models.values()), None
        )
        if model and model.model:
            expected_lookback = model.model.input_shape[1]
            expected_n_features = model.model.input_shape[2]
        else:
            expected_lookback, expected_n_features = 30, 49

        lookback = min(X_arr.shape[0], expected_lookback) if X_arr.ndim >= 2 else 1
        if X_arr.ndim >= 2:
            trimmed = X_arr[-lookback:, :expected_n_features]
        else:
            trimmed = np.zeros((1, expected_n_features))

        n_features = trimmed.shape[1]
        if n_features < expected_n_features:
            padded = np.pad(trimmed, ((0, 0), (0, expected_n_features - n_features)))
        else:
            padded = trimmed[:, :expected_n_features]
        if padded.shape[0] < expected_lookback:
            padded = np.pad(padded, ((expected_lookback - padded.shape[0], 0), (0, 0)))
        return padded.reshape(1, expected_lookback, expected_n_features).astype(
            np.float32
        )

    def _lstm_prediction_for_symbol(self, X: np.ndarray, symbol: str) -> float:
        """Symbol-specific LSTM prediction. Falls back to EURUSD model if no dedicated model."""  # noqa: E501
        model = self._lstm_models.get(symbol)
        if model is None:
            # Try EURUSD as fallback
            model = self._lstm_models.get("EURUSD") or next(
                iter(self._lstm_models.values()), None
            )
        if model is None:
            return 0.0
        try:
            X_lstm = self._features_to_lstm_input(X, symbol)
            pred = model.predict(X_lstm)
            raw = float(pred[0][0]) if hasattr(pred, "__len__") else float(pred)
            # Sanity: raw must be a finite number in a realistic price range
            if not np.isfinite(raw):
                self.memory.remember(
                    "lstm_nan",
                    "LSTM output non-finite",
                    importance=0.5,
                    emotion="warning",
                )
                return 0.0
            if abs(raw) > 1e6:  # Absurd price
                return 0.0
            return float(np.clip((raw - 1.12) / 1.12, -0.01, 0.01))
        except Exception as e:
            logger.debug(f"LSTM predict error: {e}")
            return 0.0
