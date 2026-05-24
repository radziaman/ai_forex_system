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


# Strategy-specific SL/TP (ATR multipliers) — each strategy has its own optimal
# stop-loss and take-profit to improve profitability.
STRATEGY_SL_TP = {
    "rule_breakout": (2.0, 5.0),  # Trending breakout
    "rule_mean_rev": (1.5, 2.0),  # Ranging mean reversion
    "bb_squeeze": (2.0, 3.0),  # Volatility breakout
    "ts_momentum": (2.5, 4.0),  # Trend momentum (wider SL)
    "vol_mean_rev": (1.5, 2.0),  # Volatility reversion
    "ppo_trending": (2.0, 4.0),  # PPO trending agent
    "ppo_ranging": (1.5, 3.0),  # PPO ranging agent
    "ppo_volatile": (2.5, 5.0),  # PPO volatile agent
    "ppo_crisis": (1.0, 2.0),  # PPO crisis agent
    "lstm_cnn": (2.0, 4.0),  # LSTM model
    "tft_primary": (2.0, 4.0),  # TFT model
    "orderflow": (1.5, 2.5),  # Order flow / CVD (tight SL, moderate TP)
    "macro_sentiment": (2.0, 4.0),  # Macroeconomic / sentiment (wider SL)
    "stat_arb": (1.5, 3.0),  # Statistical arbitrage
    "carry_trade": (2.0, 4.0),  # Carry trade
    "event_driven": (1.0, 2.0),  # Event-driven straddle
    "vol_expansion": (2.5, 5.0),  # Volatility breakout
    "order_flow_momentum": (1.5, 2.5),  # Order flow momentum
}


class SignalAgent(BaseAgent):
    def __init__(self, config):
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
        )
        self.config = config
        self.ensemble = None
        self._lstm_models: Dict[str, Any] = {}  # symbol -> loaded LSTM model
        self._tft_models: Dict[str, Any] = {}  # symbol -> loaded TFT model
        self._classifiers: Dict[str, Any] = {}  # symbol -> loaded classifier
        self._model_registry = None
        self._regime_manager = None
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

        # Kill switch tracking
        self._kill_switch_since: Optional[float] = None
        self._kill_switch_alerted: bool = False

        # G16: Confidence calibration
        self._confidence_buckets: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=500)
        )
        self._calibration_bins = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

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

        # Active symbols: starts with default set, dynamically updated by screener
        from data.data_manager import SYMBOLS

        self._active_symbols = list(SYMBOLS)

    async def _on_start(self):
        self.consciousness.current_intention = (
            "loading AI models and initializing ensemble"
        )
        from rts_ai_fx.ensemble import MoEEnsemble

        self.ensemble = MoEEnsemble()
        self.ensemble.use_sharpe_weighting = True
        # Wire strategy tracker for dynamic weight adjustments (Phase 4)
        self.ensemble.set_tracker_weight_fn(self._per_symbol_weight_fn)
        self._load_models()
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
        active = bool(self.get_world("risk.kill_switch", False))
        if active and self._kill_switch_since is None:
            self._kill_switch_since = time.time()
        return active

    async def _handle_kill_switch(self):
        """Log warning, alert Telegram, and attempt recovery after 30 min."""
        if not self._kill_switch_alerted:
            logger.warning("Kill switch active — all signals blocked")
            await self.send(
                MessageType.RISK_ALERT,
                payload={
                    "type": "kill_switch",
                    "reason": "Kill switch active — all signals blocked",
                },
                priority=MessagePriority.CRITICAL,
            )
            self._kill_switch_alerted = True
        else:
            elapsed = time.time() - self._kill_switch_since
            if elapsed > 30 * 60:
                drawdown = self.get_world("risk.drawdown", 0.0)
                if drawdown < 0.03:
                    logger.info(
                        "Kill switch recovery: drawdown recovered, "
                        "attempting release"
                    )
                    self._kill_switch_since = None
                    self._kill_switch_alerted = False
                    self.set_world("risk.kill_switch_release_requested", True)
                else:
                    logger.warning(
                        f"Kill switch still active after {elapsed / 60:.0f}min, "
                        f"drawdown={drawdown:.1%}"
                    )

    def _clear_kill_switch_state(self):
        """Reset local kill switch tracking when world state clears."""
        self._kill_switch_since = None
        self._kill_switch_alerted = False

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
                    # Merge screener findings with core symbols (don't replace)
                    from data.data_manager import SYMBOLS as CORE_SYMBOLS

                    merged = list(dict.fromkeys(list(CORE_SYMBOLS) + new_symbols))
                    self._active_symbols = merged
                    self.set_world("signal.active_symbols", merged)
                    self.log_state(
                        f"Screener merged: {len(merged)} symbols "
                        f"({len(CORE_SYMBOLS)} core + {len(new_symbols)} screener)"
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
        payload = message.payload if isinstance(message.payload, dict) else {}
        symbol = payload.get("symbol", "")
        features = payload.get("features")
        df = payload.get("ohlcv")
        price = payload.get("price", 0)
        if features is None:
            return

        # Task 2: Pre-trade kill switch check
        if self._check_kill_switch():
            await self._handle_kill_switch()
            return
        if self._kill_switch_since is not None:
            self._clear_kill_switch_state()

        # Per-symbol: skip symbols with no profitable strategy yet
        if not self._per_symbol_tracker.is_symbol_tradeable(symbol):
            self.log_state(
                f"Skipping {symbol}: no profitable strategy found yet", "debug"
            )
            return

        # Phase 2: Cache for rule-based experts
        if df is not None:
            self._last_df = df
        if price != 0:
            self._last_price = price
        # Cache current symbol so LSTM expert can use it
        self._current_symbol = symbol

        # Determine trading session for rule-based expert adjustments
        self._current_session = self._get_current_session()

        if df is not None and len(df) > 60:
            from rts_ai_fx.regime_detector import HMMRegimeDetector

            detector = HMMRegimeDetector()
            regime_str = detector.detect_regime(df)
        else:
            regime_str = self.get_world(f"regime.{symbol}", "ranging")

        if not self.ensemble:
            return

        # Get ensemble prediction (Phase 4: LSTM is now a proper expert in the ensemble)
        pred = self.ensemble.predict(features, regime=regime_str)
        if pred is None or not hasattr(pred, "confidence") or pred.confidence is None:
            return

        should_trade, direction_str, agreement = self.ensemble.should_trade(
            pred,
            price,
            min_confidence=0.40,
        )
        if not should_trade:
            return

        self._signal_count += 1
        self._last_signal_time = time.time()

        # Find the best expert (highest weight) for per-strategy SL/TP
        expert_outputs = getattr(pred, "expert_outputs", {})
        if expert_outputs:
            best_expert = max(
                expert_outputs.items(), key=lambda x: x[1].get("weight", 0)
            )[0]
        else:
            best_expert = "unknown"
        sl_atr, tp_atr = STRATEGY_SL_TP.get(best_expert, (2.0, 4.0))

        await self.send(
            MessageType.SIGNAL_GENERATED,
            payload={
                "symbol": symbol,
                "direction": direction_str,
                "confidence": getattr(pred, "confidence", 0.5),
                "regime": regime_str,
                "session": self._current_session,
                "price": price,
                "agreement": agreement,
                "expert_outputs": expert_outputs,
                "ensemble_price": getattr(pred, "price", price),
                "strategy": best_expert,
                "sl_atr": sl_atr,
                "tp_atr": tp_atr,
                "timestamp": time.time(),
            },
            priority=MessagePriority.NORMAL,
            intention=AgentIntention(
                primary_goal=f"generate {direction_str} signal for {symbol}",
                reasoning=f"ensemble conf={pred.confidence:.2f} agree={agreement:.2f} "
                f"strat={best_expert} sl_atr={sl_atr} tp_atr={tp_atr}",
                expected_outcome="risk agent evaluates and gatekeeper approves or rejects",  # noqa: E501
                confidence=float(pred.confidence),
            ),
        )

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
        if isinstance(confidence, (int, float)):
            for bin_edge in self._calibration_bins:
                if confidence <= bin_edge:
                    self._confidence_buckets[f"bin_{bin_edge}"].append(
                        1 if payload.get("success") else 0
                    )
                    break

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
        report = {}
        for bin_name, outcomes in self._confidence_buckets.items():
            if len(outcomes) >= 5:
                actual_accuracy = sum(outcomes) / len(outcomes)
                expected = float(bin_name.split("_")[1])
                report[bin_name] = {
                    "expected": expected,
                    "actual": round(actual_accuracy, 3),
                    "samples": len(outcomes),
                    "calibration_error": round(abs(actual_accuracy - expected), 3),
                }
        if report:
            self.memory.know("signal.calibration", report, ttl=3600)

    def _load_models(self):  # noqa: C901
        # PPO regime agents (shared across all symbols)
        try:
            from ai.regime_agents import RegimeSpecialistSystem

            self._regime_manager = RegimeSpecialistSystem(state_dim=49, n_actions=5)
            n_agents = len([a for a in self._regime_manager.agents.values() if a])

            has_real_weights = any(
                any(p.norm().item() > 1.0 for p in agent.actor.parameters())
                for agent in self._regime_manager.agents.values()
                if agent
            )
            self.set_world("models.ppo_trained", has_real_weights)
            if not has_real_weights:
                self.log_state(
                    f"Loaded {n_agents} PPO regime agents (untrained — random weights)"
                )
                self.set_world("models.untrained", True)
            else:
                self.log_state(
                    f"Loaded {n_agents} PPO regime agents (trained weights loaded)"
                )
        except Exception as e:
            self.log_state(f"PPO agents not loaded: {e}", "warning")

        # Per-symbol model registry
        try:
            from agentic.agents.model_registry import SymbolModelRegistry

            self._model_registry = SymbolModelRegistry()
            self._model_registry.discover()
            self.log_state(f"Model registry: {self._model_registry.summary()}")
        except Exception as e:
            self.log_state(f"Model registry failed: {e}", "warning")

        # Load LSTM models for all active symbols
        from rts_ai_fx.model import LSTMCNNHybrid

        for sym in self._active_symbols:
            entry = self._model_registry.get_lstm(sym) if self._model_registry else None
            if entry and entry.file_path and os.path.exists(entry.file_path):
                try:
                    loaded = LSTMCNNHybrid.load(entry.file_path)
                    if loaded and loaded.model is not None:
                        self._lstm_models[sym] = loaded
                except Exception:
                    pass
            if sym not in self._lstm_models:
                # Fallback: try the generic model
                for p in [
                    "models/lstm_cnn_model_reloaded.keras",
                    "models/lstm_cnn_EURUSD.keras",
                    "models/lstm_cnn_model.keras",
                ]:
                    if os.path.exists(p):
                        try:
                            loaded = LSTMCNNHybrid.load(p)
                            if loaded and loaded.model is not None:
                                self._lstm_models[sym] = loaded
                                self._fallback_warnings.add(sym)
                                break
                        except Exception:
                            pass

        # Load TFT models for all active symbols
        try:
            from ai.tft_model import TemporalFusionTransformer

            for sym in self._active_symbols:
                tft_path = f"models/tft_{sym}.pt"
                if os.path.exists(tft_path):
                    try:
                        loaded = TemporalFusionTransformer.load(tft_path)
                        self._tft_models[sym] = loaded
                    except Exception:
                        pass
                if sym not in self._tft_models:
                    # Fallback: generic model
                    generic = "models/tft_primary.pt"
                    if os.path.exists(generic):
                        try:
                            loaded = TemporalFusionTransformer.load(generic)
                            self._tft_models[sym] = loaded
                            self._fallback_warnings.add(sym)
                        except Exception:
                            pass
        except Exception as e:
            self.log_state(f"TFT models not loaded: {e}", "warning")

        # Load classifier models per active symbol
        from rts_ai_fx.model import ProfitabilityClassifier

        for sym in self._active_symbols:
            entry = (
                self._model_registry.get_classifier(sym)
                if self._model_registry
                else None
            )
            if entry and entry.file_path and os.path.exists(entry.file_path):
                try:
                    self._classifiers[sym] = ProfitabilityClassifier.load(
                        entry.file_path
                    )
                except Exception:
                    pass
            if sym not in self._classifiers:
                try:
                    clf = ProfitabilityClassifier(lookback=30, n_features=49)
                    self._classifiers[sym] = clf
                except Exception:
                    pass

        # Log summary
        lstm_count = len(self._lstm_models)
        tft_count = len(self._tft_models)
        clf_count = len(self._classifiers)
        lstm_unique = len(set(id(m) for m in self._lstm_models.values()))
        if self._fallback_warnings:
            fallback_list = sorted(self._fallback_warnings)[:5]
            self.log_state(
                f"{lstm_count} symbols have LSTM models ({lstm_unique} unique instances) "  # noqa: E501
                f"— {len(self._fallback_warnings)} symbols fall back to EURUSD "
                f"({', '.join(fallback_list)}{'...' if len(self._fallback_warnings) > 5 else ''})",  # noqa: E501
                level="warning",
            )
        else:
            self.log_state(
                f"{lstm_count} symbols have LSTM models ({lstm_unique} unique instances)"  # noqa: E501
            )
        self.log_state(f"{tft_count} symbols have TFT models")
        self.log_state(f"{clf_count} symbols have classifiers")

        self._models_loaded = self._regime_manager is not None and (
            len(self._lstm_models) > 0 or len(self._tft_models) > 0
        )

    def _register_experts(self):  # noqa: C901
        if not self.ensemble:
            return
        self.ensemble.experts = []
        self.ensemble.elo_ratings = {}

        # Phase 1: 4 regime-specific PPO experts instead of single ppo_regime
        if self._regime_manager:
            for regime in ["trending", "ranging", "volatile", "crisis"]:
                self.ensemble.add_expert(
                    name=f"ppo_{regime}",
                    predict_fn=lambda X, r=regime: self._ppo_prediction(X, regime=r),
                    confidence_fn=lambda X, r=regime: self._ppo_confidence(X, regime=r),
                    regime=regime,
                )
                self._strategy_tracker.register_strategy(f"ppo_{regime}", regime)
                for sym in self._active_symbols:
                    self._per_symbol_tracker.register(sym, f"ppo_{regime}", regime)

        # Phase 4: Register LSTM as a proper expert (no longer a placeholder)
        has_lstm = len(self._lstm_models) > 0
        if has_lstm:
            self.ensemble.add_expert(
                name="lstm_cnn",
                predict_fn=self._lstm_ensemble_prediction,
                confidence_fn=lambda X: 0.6,
                regime="ranging",
            )
            self._strategy_tracker.register_strategy("lstm_cnn", "ranging")
            for sym in self._active_symbols:
                self._per_symbol_tracker.register(sym, "lstm_cnn", "ranging")

        # Phase 2: Replace placeholder rule_based with two real rule-based experts
        self.ensemble.add_expert(
            name="rule_breakout",
            predict_fn=self._rule_breakout_prediction,
            confidence_fn=self._rule_breakout_confidence,
            regime="trending",
        )
        self._strategy_tracker.register_strategy("rule_breakout", "trending")
        for sym in self._active_symbols:
            self._per_symbol_tracker.register(sym, "rule_breakout", "trending")
        self.ensemble.add_expert(
            name="rule_mean_rev",
            predict_fn=self._rule_mean_rev_prediction,
            confidence_fn=self._rule_mean_rev_confidence,
            regime="ranging",
        )
        self._strategy_tracker.register_strategy("rule_mean_rev", "ranging")
        for sym in self._active_symbols:
            self._per_symbol_tracker.register(sym, "rule_mean_rev", "ranging")

        # Phase 6: Research-backed forex strategies
        self.ensemble.add_expert(
            name="bb_squeeze",
            predict_fn=self._bb_squeeze_prediction,
            confidence_fn=lambda X: 0.55,
            regime="volatile",
        )
        self._strategy_tracker.register_strategy("bb_squeeze", "volatile")
        for sym in self._active_symbols:
            self._per_symbol_tracker.register(sym, "bb_squeeze", "volatile")
        self.ensemble.add_expert(
            name="ts_momentum",
            predict_fn=self._ts_momentum_prediction,
            confidence_fn=lambda X: 0.6,
            regime="trending",
        )
        self._strategy_tracker.register_strategy("ts_momentum", "trending")
        for sym in self._active_symbols:
            self._per_symbol_tracker.register(sym, "ts_momentum", "trending")
        self.ensemble.add_expert(
            name="vol_mean_rev",
            predict_fn=self._vol_mean_rev_prediction,
            confidence_fn=lambda X: 0.55,
            regime="volatile",
        )
        self._strategy_tracker.register_strategy("vol_mean_rev", "volatile")
        for sym in self._active_symbols:
            self._per_symbol_tracker.register(sym, "vol_mean_rev", "volatile")

        # Phase 7: Order flow / CVD expert (Evans & Lyons 2002)
        self.ensemble.add_expert(
            name="orderflow",
            predict_fn=self._orderflow_prediction,
            confidence_fn=self._orderflow_confidence,
            regime="ranging",
        )
        self._strategy_tracker.register_strategy("orderflow", "ranging")
        for sym in self._active_symbols:
            self._per_symbol_tracker.register(sym, "orderflow", "ranging")

        # Phase 8: Macro sentiment expert (uses FRED data for event-driven signals)
        self.ensemble.add_expert(
            name="macro_sentiment",
            predict_fn=self._macro_sentiment_prediction,
            confidence_fn=lambda X: 0.5,
            regime="trending",
        )
        self._strategy_tracker.register_strategy("macro_sentiment", "trending")
        for sym in self._active_symbols:
            self._per_symbol_tracker.register(sym, "macro_sentiment", "trending")

        # Phase 9: Social sentiment expert (Twitter/Reddit market sentiment)
        self.ensemble.add_expert(
            name="social_sentiment",
            predict_fn=self._social_sentiment_prediction,
            confidence_fn=lambda X: 0.35,  # Lower confidence — noisy signal
            regime="volatile",
        )
        self._strategy_tracker.register_strategy("social_sentiment", "volatile")
        for sym in self._active_symbols:
            self._per_symbol_tracker.register(sym, "social_sentiment", "volatile")

        # Phase 10: XGBoost expert (Chen 2016) — gradient boosted trees
        # Provides ensemble diversity: XGBoost + Neural networks outperform either alone
        self.ensemble.add_expert(
            name="xgboost",
            predict_fn=self._xgboost_prediction,
            confidence_fn=lambda X: 0.55,
            regime="ranging",
        )
        self._strategy_tracker.register_strategy("xgboost", "ranging")
        for sym in self._active_symbols:
            self._per_symbol_tracker.register(sym, "xgboost", "ranging")

        # Phase 11: TFT primary expert (Temporal Fusion Transformer)
        has_tft = len(self._tft_models) > 0
        if has_tft:
            self.ensemble.add_expert(
                name="tft_primary",
                predict_fn=self._tft_prediction,
                confidence_fn=self._tft_confidence,
                regime="ranging",
            )
            self._strategy_tracker.register_strategy("tft_primary", "ranging")
            for sym in self._active_symbols:
                self._per_symbol_tracker.register(sym, "tft_primary", "ranging")

        # Phase 12: Alpha strategies from StrategyRegistry
        try:
            from ai.alpha_strategies import StrategyRegistry
            from ai.alpha_strategies.stat_arb import StatArbStrategy
            from ai.alpha_strategies.carry_trade import CarryTradeStrategy
            from ai.alpha_strategies.event_driven import EventDrivenStrategy
            from ai.alpha_strategies.vol_expansion import VolExpansionStrategy
            from ai.alpha_strategies.order_flow_momentum import (
                OrderFlowMomentumStrategy,
            )

            registry = StrategyRegistry()
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
                predict_fn = self._make_alpha_predict_fn(name)
                conf_fn = self._make_alpha_confidence_fn(name)
                regime = self._alpha_regime_for(name)
                self.ensemble.add_expert(
                    name=name,
                    predict_fn=predict_fn,
                    confidence_fn=conf_fn,
                    regime=regime,
                )
                self._strategy_tracker.register_strategy(name, regime)
                for sym in self._active_symbols:
                    self._per_symbol_tracker.register(sym, name, regime)
        except Exception as e:
            self.log_state(f"Alpha strategies not registered: {e}", "warning")

        # Phase 13: Cross-sectional alpha expert (relative value signals)
        try:
            from rts_ai_fx.cross_sectional_alpha import CrossSectionalAlpha

            self._cross_sectional_alpha = CrossSectionalAlpha()
            self.ensemble.add_expert(
                name="cross_sectional",
                predict_fn=self._cross_sectional_predict,
                confidence_fn=lambda X: 0.6,
                regime="ranging",
            )
            self._strategy_tracker.register_strategy("cross_sectional", "ranging")
            for sym in self._active_symbols:
                self._per_symbol_tracker.register(sym, "cross_sectional", "ranging")
        except Exception as e:
            self.log_state(f"Cross-sectional alpha not registered: {e}", "warning")

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
        target_dim = self._regime_manager.state_dim if self._regime_manager else 49
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
        """Detect the current forex trading session based on UTC time."""
        utc_hour = datetime.datetime.now(datetime.timezone.utc).hour
        if 7 <= utc_hour < 12:
            return "london"
        elif 12 <= utc_hour < 16:
            return "overlap"
        elif 16 <= utc_hour < 21:
            return "newyork"
        elif 21 <= utc_hour or utc_hour < 7:
            return "asia"
        elif 8 <= utc_hour < 12:
            return "pacific"
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
        try:
            X_arr = np.asarray(X)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(1, 1, -1)
            pred = self._lstm_model.predict(X_arr)
            raw = pred[0][0] if hasattr(pred, "__len__") else pred
            return float(np.clip((raw - 1.12) / 1.12, -0.01, 0.01))
        except Exception:
            return 0.0
