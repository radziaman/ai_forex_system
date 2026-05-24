"""
Mixture-of-Experts Ensemble with HMM regime gating.
Each expert specializes in a market regime; gating network weights predictions.
Enhanced with Sharpe-based dynamic weighting and MAML meta-learning (Enhancement #9).
"""

import numpy as np
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

try:
    import torch
except ImportError:
    torch: Any = None  # type: ignore[no-redef]


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
    """Mixture-of-Experts ensemble with HMM regime gating."""

    enabled: bool = True

    def __init__(self):
        self.experts: List[Expert] = []
        self.elo_ratings: Dict[str, float] = {}
        self.sharpe_ratios: Dict[str, float] = defaultdict(float)
        self.expert_returns: Dict[str, List[float]] = defaultdict(list)
        self.expert_win_rates: Dict[str, Dict] = defaultdict(
            lambda: {"wins": 0, "total": 0}
        )
        self.maml_agent = None  # MAML for meta-learning
        self.use_sharpe_weighting: bool = True
        self.use_maml_adaptation: bool = True
        self._tracker_weight_fn: Optional[Callable] = None

    def set_tracker_weight_fn(self, fn: Callable):
        """Set external weight function from StrategyTracker."""
        self._tracker_weight_fn = fn

    def add_expert(
        self,
        name: str,
        predict_fn: Callable,
        confidence_fn: Callable,
        regime: str = "ranging",
    ):
        self.experts.append(
            Expert(
                name=name, predict=predict_fn, confidence=confidence_fn, regime=regime
            )
        )
        self.elo_ratings.setdefault(name, 1200.0)

    def add_foundation_expert(self, registry=None):
        """Add foundation model (TimesFM/MOIRAI) as an ensemble expert."""
        try:
            from ai.foundation_models import FoundationModelRegistry

            self._foundation_registry = registry or FoundationModelRegistry()
            self.add_expert(
                name="foundation_model",
                predict_fn=self._foundation_predict,
                confidence_fn=lambda X: 0.55,
                regime="trending",
            )
        except Exception as e:
            logger = __import__("loguru").logger
            logger.warning(f"Foundation expert not registered: {e}")

    def add_mamba_expert(self):
        """Add Mamba state-space model as an ensemble expert."""
        try:
            from rts_ai_fx.mamba_model import MambaTimeSeriesModel

            self._mamba_model = MambaTimeSeriesModel(
                n_features=49,
                d_model=64,
                d_state=16,
                n_layers=2,
                max_seq_len=256,
            )
            self.add_expert(
                name="mamba_ssm",
                predict_fn=self._mamba_predict,
                confidence_fn=lambda X: 0.6,
                regime="ranging",
            )
        except Exception as e:
            logger = __import__("loguru").logger
            logger.warning(f"Mamba expert not registered: {e}")

    def _foundation_predict(self, X: np.ndarray) -> float:
        """Foundation model prediction wrapper."""
        try:
            if (
                hasattr(self, "_foundation_registry")
                and self._foundation_registry is not None
            ):
                # Simplified: use mean of last 5 predictions as crude estimate
                if X.ndim >= 2 and X.shape[-1] >= 1:
                    last_vals = X[-5:, 0] if X.ndim == 2 else X[-5:, -1, 0]
                    momentum = (
                        (last_vals[-1] / last_vals[0] - 1) if last_vals[0] != 0 else 0
                    )
                    return float(momentum)
            return 0.0
        except Exception:
            return 0.0

    def _mamba_predict(self, X: np.ndarray) -> float:
        """Mamba model prediction wrapper."""
        try:
            if hasattr(self, "_mamba_model") and self._mamba_model is not None:
                import torch

                # Ensure correct shape: (1, seq_len, n_features)
                if X.ndim == 2:
                    X_in = X.reshape(1, *X.shape)
                else:
                    X_in = X
                if X_in.shape[1] > 256:
                    X_in = X_in[:, -256:, :]
                with torch.no_grad():
                    pred = self._mamba_model(torch.from_numpy(X_in).float())
                    return float(pred.item())
            return 0.0
        except Exception:
            return 0.0

    def predict(
        self,
        X: np.ndarray,
        regime: str = "ranging",
        regime_posteriors: Optional[np.ndarray] = None,
    ) -> EnsemblePrediction:
        if not self.enabled or not self.experts:
            return EnsemblePrediction()

        # MAML meta-adaptation (Enhancement #9)
        if self.maml_agent and self.use_maml_adaptation:
            X_adapted = self._maml_adapt(X, regime)
        else:
            X_adapted = X

        predictions = []
        confidences = []
        weight_values: List[float] = []
        expert_outputs = {}

        for expert in self.experts:
            try:
                pred = float(np.array(expert.predict(X_adapted)).flatten()[0])
                conf = float(np.array(expert.confidence(X_adapted)).flatten()[0])
            except Exception:
                continue

            # Dynamic regime-based weighting
            regime_weight = self._calculate_regime_weight(
                expert, regime, regime_posteriors
            )

            # Elo rating weight (performance-based)
            elo_weight = self.elo_ratings.get(expert.name, 1200.0) / 1200.0

            # Sharpe ratio weight (Enhancement #9)
            if self.use_sharpe_weighting:
                sharpe_weight = max(
                    0.1, min(2.0, self.sharpe_ratios.get(expert.name, 1.0))
                )
            else:
                sharpe_weight = 1.0

            # Confidence-adjusted weight
            conf_weight = 0.5 + 0.5 * conf

            # Strategy tracker dynamic weight (Phase 4: MoE dynamic weighting)
            tracker_weight = 1.0
            if self._tracker_weight_fn:
                tracker_weight = self._tracker_weight_fn(expert.name, regime)

            # Combined weight
            weight = (
                regime_weight
                * elo_weight
                * sharpe_weight
                * conf_weight
                * tracker_weight
            )
            predictions.append(pred)
            confidences.append(conf)
            weight_values.append(weight)
            expert_outputs[expert.name] = {
                "prediction": pred,
                "confidence": conf,
                "weight": weight,
            }

        if not predictions:
            return EnsemblePrediction()

        weights = np.array(weight_values)
        total = weights.sum()
        if total == 0:
            return EnsemblePrediction()

        weights = weights / total
        ensemble_price = float(np.average(predictions, weights=weights))
        ensemble_conf = float(np.average(confidences, weights=weights))

        # Determine direction based on weighted ensemble
        direction = self._determine_direction(ensemble_price, confidences, weights)

        return EnsemblePrediction(
            price=ensemble_price,
            confidence=ensemble_conf,
            direction=direction,
            expert_outputs=expert_outputs,
        )

    def _maml_adapt(self, X: np.ndarray, regime: str) -> np.ndarray:
        """Adapt features using MAML for few-shot learning.
        Uses frozen base model features for feature transformation."""
        try:
            if self.maml_agent and hasattr(self.maml_agent, "base_model"):
                X_tensor = (
                    torch.FloatTensor(X)
                    if X.ndim > 1
                    else torch.FloatTensor(X).unsqueeze(0)
                )
                with torch.no_grad():
                    adapted = self.maml_agent.base_model(X_tensor)
                return np.asarray(adapted.cpu().numpy())
        except Exception:
            pass
        return X

    def update_sharpe(self, expert_name: str, returns: List[float]):
        """Update Sharpe ratio for an expert based on recent returns."""
        if not returns or len(returns) < 2:
            return
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return > 0:
            sharpe = mean_return / std_return * np.sqrt(252)  # Annualized
            self.sharpe_ratios[expert_name] = sharpe
            self.expert_returns[expert_name].extend(returns)
            # Keep only last 100 returns
            if len(self.expert_returns[expert_name]) > 100:
                self.expert_returns[expert_name] = self.expert_returns[expert_name][
                    -100:
                ]

    def update_expert_result(self, expert_name: str, pnl: float):
        """Track expert performance for win rate calculation."""
        # Ensure defaultdict entry exists
        _ = self.expert_win_rates[expert_name]
        self.expert_win_rates[expert_name]["total"] += 1
        if pnl > 0:
            self.expert_win_rates[expert_name]["wins"] += 1
        # Update Sharpe if we have enough data
        if expert_name in self.expert_returns:
            self.update_sharpe(expert_name, self.expert_returns[expert_name][-20:])

    def should_trade(
        self,
        pred: EnsemblePrediction,
        current_price: float,
        min_confidence: float = 0.65,
    ) -> Tuple[bool, str, float]:
        """
        Determine if we should trade based on ensemble agreement and confidence.
        Uses weighted majority voting on return-based signals (instrument-agnostic).
        """
        if not pred.expert_outputs:
            return False, "HOLD", 0.0
        buy_weight = 0.0
        sell_weight = 0.0
        total_weight = 0.0
        threshold = 0.0003
        for name, output in pred.expert_outputs.items():
            w = output["weight"]
            total_weight += w
            if output["prediction"] > threshold:
                buy_weight += w
            elif output["prediction"] < -threshold:
                sell_weight += w
        if total_weight == 0:
            return False, "HOLD", 0.0
        buy_ratio = buy_weight / total_weight
        sell_ratio = sell_weight / total_weight
        if (
            buy_ratio > 0.6
            and buy_ratio > sell_ratio
            and pred.confidence >= min_confidence
        ):
            return True, "BUY", buy_ratio
        elif (
            sell_ratio > 0.6
            and sell_ratio > buy_ratio
            and pred.confidence >= min_confidence
        ):
            return True, "SELL", sell_ratio
        return False, "HOLD", 0.0

    def _calculate_regime_weight(
        self, expert: Expert, regime: str, regime_posteriors: Optional[np.ndarray]
    ) -> float:
        """Calculate regime-based weight for expert."""
        base_weight = 1.0
        if expert.regime == regime:
            base_weight = 2.0
        elif expert.regime != "ranging":  # Non-rangng experts get penalty
            base_weight = 0.5

        if regime_posteriors is not None:
            regime_idx = 0
            try:
                regime_names = ["trending", "ranging", "volatile", "crisis"]
                regime_idx = (
                    regime_names.index(expert.regime)
                    if expert.regime in regime_names
                    else 1
                )
                if regime_idx < len(regime_posteriors):
                    base_weight = regime_posteriors[regime_idx] * 2 + 0.5
            except (ValueError, IndexError):
                pass

        return base_weight

    def _determine_direction(
        self, ensemble_price: float, confidences: list, weights: np.ndarray
    ) -> str:
        """Determine direction based on whether prediction is above/below input.
        Only used as fallback — should_trade() computes direction independently from expert outputs.  # noqa: E501
        """
        avg_conf = (
            np.average(confidences, weights=weights) if weights.sum() > 0 else 0.0
        )
        if avg_conf > 0.5:
            return "BUY" if ensemble_price > 0 else "SELL"
        return "HOLD"

    def update_elo(self, name: str, was_correct: bool, k: Optional[float] = None):
        """Update Elo rating with automatic decay of k-factor.
        Phase 5: k decays as expertise accumulates: k = 32 / (1 + trades/100).
        """
        if k is None:
            total = self.expert_win_rates.get(name, {}).get("total", 0)
            k = 32.0 / (1.0 + total / 100.0)
        self.elo_ratings[name] = self.elo_ratings.get(name, 1200.0) + (
            k if was_correct else -k
        )
        self.elo_ratings[name] = max(100, min(3000, self.elo_ratings[name]))
