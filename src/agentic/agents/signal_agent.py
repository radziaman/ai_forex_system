"""
Signal Agent — G8: online learning from trade outcomes, G16: confidence calibration.
"""

from __future__ import annotations
import time
import numpy as np
from typing import Dict, List, Optional, Any, Set, Callable
from collections import defaultdict, deque
from loguru import logger

from agentic.core.base_agent import BaseAgent
from agentic.core.agent_message import MessageType, MessagePriority, AgentIntention
from agentic.core.agent_consciousness import ConsciousnessLevel


class SignalAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(
            name="signal_agent",
            role="Ensemble Signal Generator",
            purpose="Generate high-conviction trading signals from multi-model ensemble",
            domain="signals",
            capabilities={
                "ensemble_signal_generation", "mixture_of_experts",
                "elo_rating", "sharpe_weighting", "concept_drift_detection",
                "regime_gated_inference", "online_learning",  # G8
                "confidence_calibration",  # G16
            },
            tick_interval=1.0,
            consciousness_level=ConsciousnessLevel.REFLECTIVE,
        )
        self.config = config
        self.ensemble = None
        self._lstm_model = None
        self._classifier = None
        self._regime_manager = None
        self._drift_monitors: Dict[str, Any] = {}
        self._models_loaded = False
        self._signal_count = 0
        self._last_signal_time = 0.0

        # G8: Track outcomes per expert for online learning
        self._expert_outcomes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self._expert_pnl: Dict[str, float] = defaultdict(float)
        self._expert_trades: Dict[str, int] = defaultdict(int)

        # G16: Confidence calibration
        self._confidence_buckets: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self._calibration_bins = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        self.subscribe(MessageType.FEATURES_READY)
        self.subscribe(MessageType.REGIME_CHANGED)
        self.subscribe(MessageType.MODEL_UPDATE)
        self.subscribe(MessageType.EXECUTION_RESULT)  # G8: learn from outcomes
        self.subscribe(MessageType.DIAGNOSTIC_REQUEST)

    async def _on_start(self):
        self.consciousness.current_intention = "loading AI models and initializing ensemble"
        from rts_ai_fx.ensemble import MoEEnsemble
        self.ensemble = MoEEnsemble()
        self.ensemble.use_sharpe_weighting = True
        self._load_models()
        self._register_experts()
        self.set_world("signal.models_loaded", self._models_loaded)
        self.set_world("signal.experts", len(self.ensemble.experts) if self.ensemble else 0)
        self.log_state(f"Signal engine ready: {'experts loaded' if self._models_loaded else 'rule-based only'}")

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
            expert_names = [e.name for e in (self.ensemble.experts if self.ensemble else [])]
            self.set_world("signal.experts", expert_names)
            # G8: Log expert performance
            for name, pnl in self._expert_pnl.items():
                trades = self._expert_trades.get(name, 0)
                self.memory.know(f"signal.expert.{name}.pnl", round(pnl, 2), ttl=3600)
                self.memory.know(f"signal.expert.{name}.trades", trades, ttl=3600)

    async def on_message(self, message: AgentMessage):
        if message.msg_type == MessageType.FEATURES_READY:
            await self._on_features(message)
        elif message.msg_type == MessageType.EXECUTION_RESULT:
            # G8: Learn from trade outcome
            await self._on_execution_result(message)
        elif message.msg_type == MessageType.MODEL_UPDATE:
            self._load_models()
            self._register_experts()
            self._models_loaded = True
            self.set_world("signal.models_loaded", True)
        elif message.msg_type == MessageType.DIAGNOSTIC_REQUEST:
            await self.send(MessageType.DIAGNOSTIC_RESULT, payload={
                "agent": self.name, "models_loaded": self._models_loaded,
                "n_experts": len(self.ensemble.experts) if self.ensemble else 0,
                "signals": self._signal_count,
            }, target=message.source_agent)

    async def _on_features(self, message: AgentMessage):
        payload = message.payload if isinstance(message.payload, dict) else {}
        symbol = payload.get("symbol", "")
        features = payload.get("features")
        df = payload.get("ohlcv")
        price = payload.get("price", 0)
        if features is None:
            return

        if df is not None and len(df) > 60:
            from rts_ai_fx.regime_detector import HMMRegimeDetector
            detector = HMMRegimeDetector()
            regime_str = detector.detect_regime(df)
        else:
            regime_str = self.get_world(f"regime.{symbol}", "ranging")

        if not self.ensemble:
            return

        pred = self.ensemble.predict(features, regime=regime_str)
        if pred is None or not hasattr(pred, 'confidence') or pred.confidence is None:
            return

        should_trade, direction_str, agreement = self.ensemble.should_trade(
            pred, price, min_confidence=0.40,
        )
        if not should_trade:
            return

        self._signal_count += 1
        self._last_signal_time = time.time()

        await self.send(MessageType.SIGNAL_GENERATED, payload={
            "symbol": symbol, "direction": direction_str,
            "confidence": getattr(pred, 'confidence', 0.5),
            "regime": regime_str, "price": price,
            "agreement": agreement,
            "expert_outputs": getattr(pred, 'expert_outputs', {}),
            "ensemble_price": getattr(pred, 'price', price),
            "timestamp": time.time(),
        }, priority=MessagePriority.NORMAL,
            intention=AgentIntention(
                primary_goal=f"generate {direction_str} signal for {symbol}",
                reasoning=f"ensemble confidence={pred.confidence:.2f}, agreement={agreement:.2f}",
                expected_outcome="risk agent evaluates and gatekeeper approves or rejects",
                confidence=float(pred.confidence),
            ))

    # G8: Learn from execution outcomes
    async def _on_execution_result(self, message: AgentMessage):
        payload = message.payload if isinstance(message.payload, dict) else {}
        if not payload.get("success"):
            return

        signal_msg = message.causal_parent_id
        expert_outputs = payload.get("signal", {}).get("expert_outputs", {})
        out_timestamp = payload.get("timestamp", time.time())

        # Update Elo ratings based on outcome
        if self.ensemble and payload.get("filled_price") and expert_outputs:
            for expert_name, output in expert_outputs.items():
                was_correct = (payload.get("direction") == "BUY" and output.get("prediction", 0) > 0) or \
                              (payload.get("direction") == "SELL" and output.get("prediction", 0) < 0)
                self.ensemble.update_elo(expert_name, was_correct)
                self._expert_outcomes[expert_name].append(1 if was_correct else 0)
                self._expert_trades[expert_name] += 1

                # Track PnL contribution per expert
                pnl = payload.get("pnl", 0)
                if pnl != 0:
                    self._expert_pnl[expert_name] += pnl
                    if hasattr(self.ensemble, 'update_expert_result'):
                        self.ensemble.update_expert_result(expert_name, pnl)

        # G16: Calibrate confidence vs actual outcomes
        confidence = payload.get("signal", {}).get("confidence", 0.5)
        if isinstance(confidence, (int, float)):
            for bin_edge in self._calibration_bins:
                if confidence <= bin_edge:
                    self._confidence_buckets[f"bin_{bin_edge}"].append(
                        1 if payload.get("success") else 0
                    )
                    break

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

    def _load_models(self):
        try:
            from ai.regime_agents import RegimeSpecialistSystem
            self._regime_manager = RegimeSpecialistSystem(state_dim=55, n_actions=5)
            n_agents = len([a for a in self._regime_manager.agents.values() if a])
            self.log_state(f"Loaded {n_agents} PPO regime agents")
        except Exception as e:
            self.log_state(f"PPO agents not loaded: {e}", "warning")
        try:
            from rts_ai_fx.model import LSTMCNNHybrid
            # Try reloaded model first (saved with current TF, verified weights)
            paths = ["models/lstm_cnn_model_reloaded.keras",
                     "models/lstm_cnn_EURUSD.keras",
                     "models/lstm_cnn_model.keras"]
            for p in paths:
                if not __import__('os').path.exists(p):
                    continue
                loaded = LSTMCNNHybrid.load(p)
                if loaded and loaded.model is not None:
                    self._lstm_model = loaded
                    n_layers = len(loaded.model.layers)
                    self.log_state(f"Loaded LSTM-CNN from {p} ({n_layers} layers)")
                    break
        except Exception as e:
            self.log_state(f"LSTM-CNN not loaded: {e}", "warning")
        try:
            from rts_ai_fx.model import ProfitabilityClassifier
            self._classifier = ProfitabilityClassifier(lookback=30, n_features=51)
        except Exception:
            pass
        self._models_loaded = self._regime_manager is not None and self._lstm_model is not None

    def _register_experts(self):
        if not self.ensemble:
            return
        self.ensemble.experts = []
        self.ensemble.elo_ratings = {}
        if self._regime_manager:
            self.ensemble.add_expert(name="ppo_regime",
                predict_fn=self._ppo_prediction, confidence_fn=self._ppo_confidence, regime="ranging")
        if self._lstm_model:
            self.ensemble.add_expert(name="lstm_cnn",
                predict_fn=self._lstm_prediction, confidence_fn=lambda X: 0.6, regime="ranging")
        self.ensemble.add_expert(name="rule_based",
            predict_fn=lambda X, sym="EURUSD": 0.0, confidence_fn=lambda X: 0.7, regime="ranging")

    def _features_to_ppo_state(self, X: np.ndarray) -> np.ndarray:
        """Extract a flat 55-dim state vector from multi-bar feature matrix.
        
        The feature pipeline produces (lookback, n_features).
        PPO was trained on flat 55-dim vectors from the most recent bar.
        We take the last bar's features and trim/pad to 55.
        """
        X_arr = np.asarray(X)
        if X_arr.ndim == 3:
            X_arr = X_arr[0]  # (1, lookback, n_features) -> (lookback, n_features)
        if X_arr.ndim == 2:
            state = X_arr[-1, :]  # Last bar only
        else:
            state = X_arr.flatten()
        state = state[:55]
        if len(state) < 55:
            state = np.pad(state, (0, 55 - len(state)))
        return state.astype(np.float32)

    def _sane_prediction(self, value: float, label: str = "model", bounds: tuple = (-10, 10)) -> float:
        """Fix 4: Sanity-check a model prediction for NaN, Inf, and realistic bounds."""
        if value is None or (hasattr(value, '__len__') and len(value) == 0):
            self.memory.remember(f"{label}_prediction_invalid", f"None/empty prediction", importance=0.3)
            return 0.0
        v = float(value) if hasattr(value, '__float__') else 0.0
        if not np.isfinite(v):
            self.memory.remember(f"{label}_prediction_nan", f"Non-finite prediction: {v}", importance=0.5, emotion="warning")
            return 0.0
        lo, hi = bounds
        if v < lo or v > hi:
            return float(np.clip(v, lo, hi))
        return v

    def _ppo_prediction(self, X: np.ndarray) -> float:
        if self._regime_manager is None:
            return 0.0
        try:
            state = self._features_to_ppo_state(X)
            action, _ = self._regime_manager.select_action(state, regime="ranging")
            pred = 0.001 if action in (1, 3) else -0.001 if action in (2, 4) else 0.0
            return self._sane_prediction(pred, "ppo", (-0.01, 0.01))
        except Exception:
            return 0.0

    def _ppo_confidence(self, X: np.ndarray) -> float:
        if self._regime_manager is None:
            return 0.5
        try:
            state = self._features_to_ppo_state(X)
            action, log_probs = self._regime_manager.select_action(state, regime="ranging")
            prob = float(np.exp(log_probs)) if hasattr(log_probs, '__float__') else 0.6
            return float(np.clip(self._sane_prediction(prob, "ppo_conf", (0, 1)), 0.0, 1.0))
        except Exception:
            return 0.5

    def _features_to_lstm_input(self, X: np.ndarray) -> np.ndarray:
        """Reshape features for LSTM using the loaded model's expected dimensions."""
        X_arr = np.asarray(X)
        if X_arr.ndim == 3 and X_arr.shape[0] == 1:
            X_arr = X_arr[0]
        
        # Read expected dims from the loaded model
        expected_lookback = self._lstm_model.model.input_shape[1] if self._lstm_model.model else 30
        expected_n_features = self._lstm_model.model.input_shape[2] if self._lstm_model.model else 51
        
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
        return padded.reshape(1, expected_lookback, expected_n_features).astype(np.float32)

    def _lstm_prediction(self, X: np.ndarray) -> float:
        """Return return signal from LSTM with sanity bounds."""
        if self._lstm_model is None:
            return 0.0
        try:
            X_lstm = self._features_to_lstm_input(X)
            pred = self._lstm_model.predict(X_lstm, verbose=0)
            raw = float(pred[0][0]) if hasattr(pred, '__len__') else float(pred)
            # Sanity: raw must be a finite number in a realistic price range
            if not np.isfinite(raw):
                self.memory.remember("lstm_nan", "LSTM output non-finite", importance=0.5, emotion="warning")
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
            raw = pred[0][0] if hasattr(pred, '__len__') else pred
            return float(np.clip((raw - 1.12) / 1.12, -0.01, 0.01))
        except Exception:
            return 0.0
