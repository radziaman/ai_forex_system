"""Pure signal generation: features + OHLCV → Signal. No I/O, no risk."""
from typing import Optional, Dict
from loguru import logger

from infrastructure.service_base import TradingService
from infrastructure.config_v2 import AppConfig
from services import Signal, SignalDirection, Regime, FeatureUpdate
from rts_ai_fx.regime_detector import HMMRegimeDetector
from rts_ai_fx.ensemble import MoEEnsemble
from rts_ai_fx.drift_detector import DriftMonitor


class SignalEngine(TradingService):
    """Transforms features into trading signals. Pure — no side effects."""

    def __init__(self, config: AppConfig):
        super().__init__("signal_engine")
        self.config = config
        self.ensemble = MoEEnsemble()
        self.regime_detector = HMMRegimeDetector(n_regimes=4, lookback=60)
        self.drift_monitors: Dict[str, DriftMonitor] = {}
        self._regimes: Dict[str, Regime] = {}

    async def start(self) -> None:
        self.ensemble.add_expert(
            name="rule_based",
            predict_fn=self._rule_prediction,
            confidence_fn=lambda X: 0.7,
            regime="ranging",
        )
        self._running = True

    async def stop(self) -> None:
        self._running = False

    def on_features(self, update: FeatureUpdate) -> Optional[Signal]:
        """Process new features, return Signal or None."""
        if update is None or update.features is None:
            return None

        symbol = update.symbol

        # Regime detection (needs OHLCV DataFrame)
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

        # Ensemble inference
        pred = self.ensemble.predict(update.features, regime=regime_str)
        should_trade, direction_str, agreement = self.ensemble.should_trade(
            pred, update.price, min_confidence=0.40,
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
            confidence=pred.confidence,
            regime=regime,
            price=update.price,
            metadata={
                "agreement": agreement,
                "expert_outputs": pred.expert_outputs,
                "ensemble_price": pred.price,
            },
        )

    def on_trade_result(self, symbol: str, prediction: float, actual: float) -> bool:
        """Update drift monitors. Returns True if drift detected."""
        monitor = self.drift_monitors.get(symbol)
        if monitor is None:
            monitor = DriftMonitor()
            self.drift_monitors[symbol] = monitor
        return monitor.update(prediction, actual)

    def _rule_prediction(self, X, symbol: str = "EURUSD") -> float:
        return 1.12
