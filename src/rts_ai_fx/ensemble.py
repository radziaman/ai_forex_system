"""
Mixture-of-Experts Ensemble with HMM regime gating.
Each expert specializes in a market regime; gating network weights predictions.
"""
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field


@dataclass
class Expert:
    name: str
    predict: Callable
    confidence: Callable
    regime: str
    elo: float = 1200.0


@dataclass
class EnsemblePrediction:
    price: float = 0.0
    confidence: float = 0.0
    direction: str = "HOLD"
    expert_outputs: Dict[str, dict] = field(default_factory=dict)


class MoEEnsemble:
    """
    Mixture-of-Experts ensemble with HMM regime gating.
    - Expert models are registered per regime
    - Gating network (HMM posteriors) weights expert contributions
    - Elo ratings track individual expert performance
    """

    def __init__(self):
        self.experts: List[Expert] = []
        self.elo_ratings: Dict[str, float] = {}

    def add_expert(self, name: str, predict_fn: Callable, confidence_fn: Callable, regime: str = "ranging"):
        self.experts.append(Expert(name=name, predict=predict_fn, confidence=confidence_fn, regime=regime))
        self.elo_ratings.setdefault(name, 1200.0)

    def predict(self, X: np.ndarray, regime: str = "ranging", regime_posteriors: Optional[np.ndarray] = None) -> EnsemblePrediction:
        if not self.experts:
            return EnsemblePrediction()
        predictions = []
        confidences = []
        weights = []
        expert_outputs = {}
        for expert in self.experts:
            try:
                pred = float(np.array(expert.predict(X)).flatten()[0])
                conf = float(np.array(expert.confidence(X)).flatten()[0])
            except Exception:
                continue
            # Regime-based weighting: experts for current regime get boost
            regime_weight = 2.0 if expert.regime == regime else 0.5
            if regime_posteriors is not None:
                # Use HMM posterior probability for this regime
                regime_idx = ["trending", "ranging", "volatile", "crisis"].index(expert.regime) \
                    if expert.regime in ["trending", "ranging", "volatile", "crisis"] else 1
                if regime_idx < len(regime_posteriors):
                    regime_weight = regime_posteriors[regime_idx] * 2 + 0.5
            elo_weight = self.elo_ratings.get(expert.name, 1200.0) / 1200.0
            weight = regime_weight * elo_weight
            predictions.append(pred)
            confidences.append(conf)
            weights.append(weight)
            expert_outputs[expert.name] = {"prediction": pred, "confidence": conf, "weight": weight}
        if not predictions:
            return EnsemblePrediction()
        weights = np.array(weights)
        weights = weights / weights.sum()
        ensemble_price = float(np.average(predictions, weights=weights))
        ensemble_conf = float(np.average(confidences, weights=weights))
        return EnsemblePrediction(
            price=ensemble_price,
            confidence=ensemble_conf,
            direction="HOLD",
            expert_outputs=expert_outputs,
        )

    def should_trade(self, pred: EnsemblePrediction, current_price: float, min_confidence: float = 0.65) -> Tuple[bool, str, float]:
        """
        Determine if we should trade based on ensemble agreement and confidence.
        Uses weighted majority voting instead of requiring unanimity.
        """
        if not pred.expert_outputs:
            return False, "HOLD", 0.0
        buy_weight = 0.0
        sell_weight = 0.0
        total_weight = 0.0
        threshold = current_price * 0.0005
        for name, output in pred.expert_outputs.items():
            w = output["weight"]
            total_weight += w
            if output["prediction"] > current_price + threshold:
                buy_weight += w
            elif output["prediction"] < current_price - threshold:
                sell_weight += w
        if total_weight == 0:
            return False, "HOLD", 0.0
        buy_ratio = buy_weight / total_weight
        sell_ratio = sell_weight / total_weight
        if buy_ratio > 0.6 and buy_ratio > sell_ratio and pred.confidence >= min_confidence:
            return True, "BUY", buy_ratio
        elif sell_ratio > 0.6 and sell_ratio > buy_ratio and pred.confidence >= min_confidence:
            return True, "SELL", sell_ratio
        return False, "HOLD", 0.0

    def update_elo(self, name: str, was_correct: bool, k: float = 32.0):
        self.elo_ratings[name] = self.elo_ratings.get(name, 1200.0) + (k if was_correct else -k)
        self.elo_ratings[name] = max(100, min(3000, self.elo_ratings[name]))
