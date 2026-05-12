"""Pure signal generation: features + OHLCV → Signal. No I/O, no risk."""

from typing import Optional, Dict
from loguru import logger
import numpy as np

from infrastructure.service_base import TradingService
from infrastructure.config_v2 import AppConfig
from services import Signal, SignalDirection, Regime, FeatureUpdate
from rts_ai_fx.regime_detector import HMMRegimeDetector
from rts_ai_fx.ensemble import MoEEnsemble
from rts_ai_fx.drift_detector import DriftMonitor
from ai.regime_agents import RegimeSpecialistSystem
from rts_ai_fx.model import LSTMCNNHybrid, ProfitabilityClassifier
from rts_ai_fx.features_unified import FeaturePipeline


class SignalEngine(TradingService):
    """Transforms features into trading signals. Pure — no side effects."""

    def __init__(self, config: AppConfig):
        super().__init__("signal_engine")
        self.config = config
        self.ensemble = MoEEnsemble()
        self.regime_detector = HMMRegimeDetector(n_regimes=4, lookback=60)
        self.drift_monitors: Dict[str, DriftMonitor] = {}
        self._regimes: Dict[str, Regime] = {}
        self._regime_agent_manager: Optional[RegimeSpecialistSystem] = None
        self._lstm_model: Optional[LSTMCNNHybrid] = None
        self._classifier: Optional[ProfitabilityClassifier] = None
        self._feature_pipeline: Optional[FeaturePipeline] = None
        self._models_loaded = False

    async def start(self) -> None:
        self._load_models()
        self._register_experts()
        self._running = True
        if self._models_loaded:
            logger.info(
                f"SignalEngine: {len(self.ensemble.experts)} experts loaded (PPO + LSTM-CNN + rule)"
            )
        else:
            logger.warning("SignalEngine: no ML models found, using rule-based only")

    async def stop(self) -> None:
        self._running = False

    def _load_models(self):
        try:
            self._regime_agent_manager = RegimeSpecialistSystem(
                state_dim=55, n_actions=5
            )
            n_agents = len(
                [a for a in self._regime_agent_manager.agents.values() if a is not None]
            )
            if n_agents > 0:
                logger.info(f"SignalEngine: loaded {n_agents} PPO regime agents")
        except Exception as e:
            logger.warning(f"SignalEngine: could not load PPO agents: {e}")

        try:
            loaded_model = LSTMCNNHybrid.load("models/lstm_cnn_model.keras")
            if loaded_model and loaded_model.model is not None:
                self._lstm_model = loaded_model
                logger.info("SignalEngine: loaded LSTM-CNN model")
        except Exception as e:
            logger.warning(f"SignalEngine: could not load LSTM-CNN: {e}")

        try:
            self._classifier = ProfitabilityClassifier(lookback=30, n_features=51)
        except Exception:
            pass

        self._models_loaded = (
            self._regime_agent_manager is not None and self._lstm_model is not None
        )

    def _register_experts(self):
        if self._regime_agent_manager is not None:
            self.ensemble.add_expert(
                name="ppo_regime",
                predict_fn=self._ppo_prediction,
                confidence_fn=self._ppo_confidence,
                regime="ranging",
            )
            logger.info(f"SignalEngine: registered PPO regime expert")

        if self._lstm_model is not None:
            self.ensemble.add_expert(
                name="lstm_cnn",
                predict_fn=self._lstm_prediction,
                confidence_fn=self._lstm_confidence,
                regime="ranging",
            )
            logger.info(f"SignalEngine: registered LSTM-CNN expert")

        self.ensemble.add_expert(
            name="rule_based",
            predict_fn=self._rule_prediction,
            confidence_fn=lambda X: 0.7,
            regime="ranging",
        )

    def _extract_features_for_model(self, update: FeatureUpdate) -> np.ndarray:
        if update.features is not None:
            features = np.asarray(update.features)
            if features.ndim == 1:
                features = features.reshape(1, -1)
            return features
        if update.ohlcv is not None and self._feature_pipeline is not None:
            features = self._feature_pipeline.transform(update.ohlcv)
            return (
                np.asarray(features).reshape(1, -1)
                if features is not None
                else np.zeros((1, 55))
            )
        return np.zeros((1, 55))

    def _ppo_prediction(self, X: np.ndarray) -> float:
        """Return directional return signal. Positive=buy, negative=sell, zero=hold."""
        if self._regime_agent_manager is None:
            return 0.0
        try:
            X_flat = np.asarray(X).flatten()
            if X_flat.shape[0] < 55:
                X_flat = np.pad(X_flat, (0, 55 - X_flat.shape[0]))
            action, _ = self._regime_agent_manager.select_action(
                X_flat[:55], regime="ranging"
            )
            if action in (1, 3):
                return 0.001
            elif action in (2, 4):
                return -0.001
            return 0.0
        except Exception as e:
            logger.debug(f"PPO predict error: {e}")
            return 0.0

    def _ppo_confidence(self, X: np.ndarray) -> float:
        if self._regime_agent_manager is None:
            return 0.5
        try:
            action, log_probs = self._regime_agent_manager.select_action(
                X.flatten()[:55], regime="ranging"
            )
            prob = float(np.exp(log_probs)) if hasattr(log_probs, "__float__") else 0.6
            return float(np.clip(prob, 0.0, 1.0))
        except Exception:
            return 0.5

    def _lstm_prediction(self, X: np.ndarray) -> float:
        """Return return signal from LSTM. Model trained on normalized features."""
        if self._lstm_model is None:
            return 0.0
        try:
            X_arr = np.asarray(X)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(1, 1, -1)
            pred = self._lstm_model.predict(X_arr)
            raw = pred[0][0] if hasattr(pred, "__len__") else pred
            # Convert price prediction to return signal relative to base prediction
            return float(np.clip((raw - 1.12) / 1.12, -0.01, 0.01))
        except Exception as e:
            logger.debug(f"LSTM predict error: {e}")
            return 0.0

    def _lstm_confidence(self, X: np.ndarray) -> float:
        return 0.6

    def _rule_prediction(self, X, symbol: str = "EURUSD") -> float:
        return 0.0

    def on_features(self, update: FeatureUpdate) -> Optional[Signal]:
        if update is None or update.features is None:
            return None

        symbol = update.symbol
        df = update.ohlcv
        regime_str = None
        if df is not None and len(df) > 60:
            regime_str = self.regime_detector.detect_regime(df)
        else:
            regime_str = self._regimes.get(symbol, Regime.RANGING)
            if isinstance(regime_str, Regime):
                regime_str = regime_str.value

        try:
            regime = Regime(regime_str)
        except ValueError:
            regime = Regime.RANGING
        self._regimes[symbol] = regime

        if not self.regime_detector.should_trade(regime_str):
            return None

        pred = self.ensemble.predict(update.features, regime=regime_str)
        if pred is None or not hasattr(pred, "confidence") or pred.confidence is None:
            return None

        should_trade, direction_str, agreement = self.ensemble.should_trade(
            pred,
            update.price,
            min_confidence=0.40,
        )

        if not should_trade:
            return None

        if direction_str == "BUY":
            direction = SignalDirection.BUY
        elif direction_str == "SELL":
            direction = SignalDirection.SELL
        else:
            logger.warning(f"Unknown direction '{direction_str}', defaulting to HOLD")
            return None

        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=getattr(pred, "confidence", 0.5),
            regime=regime,
            price=update.price,
            metadata={
                "agreement": agreement,
                "expert_outputs": getattr(pred, "expert_outputs", {}),
                "ensemble_price": getattr(pred, "price", update.price),
            },
        )

    def on_trade_result(self, symbol: str, prediction: float, actual: float) -> bool:
        monitor = self.drift_monitors.get(symbol)
        if monitor is None:
            monitor = DriftMonitor()
            self.drift_monitors[symbol] = monitor
        return monitor.update(prediction, actual)
