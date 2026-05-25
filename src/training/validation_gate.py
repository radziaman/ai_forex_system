"""
Validation Gate — Mandatory checks before model promotion.

Ensures every deployed model passes walk-forward analysis and
crisis scenario stress tests before being promoted to production.
"""

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

from loguru import logger


class GateDecision(Enum):
    """Decision of the validation gate."""

    APPROVED = auto()
    CONDITIONAL_APPROVED = auto()
    REJECTED = auto()


@dataclass
class GateConfig:
    """Configuration thresholds for the validation gate."""

    min_sharpe: float = 1.0
    max_drawdown_pct: float = 0.10
    crisis_max_loss: float = 0.15
    min_folds: int = 4
    conditional_sharpe: float = 0.8


@dataclass
class ValidationResult:
    """Result of evaluating a model through the validation gate."""

    model_name: str
    decision: GateDecision
    reason: str
    sharpe: float
    drawdown: float
    crisis_loss: float
    timestamp: float = field(default_factory=time.time)
    details: Dict = field(default_factory=dict)


class ValidationGate:
    """
    Validates a model against mandatory walk-forward and stress-test criteria.

    Checks performed (in order):
      1. Minimum number of walk-forward folds.
      2. Sharpe ratio threshold.
      3. Maximum drawdown threshold.
      4. All crisis scenarios pass maximum loss threshold.
    """

    def __init__(self, config: Optional[GateConfig] = None):
        self.config = config or GateConfig()

    def evaluate(
        self,
        model_name: str,
        walk_forward_results: Dict,
        stress_test_results: Dict[str, Dict],
    ) -> ValidationResult:
        """
        Run a model through the validation gate.

        Parameters
        ----------
        model_name : str
            Identifier for the model being evaluated.
        walk_forward_results : Dict
            Must contain keys ``folds`` (list of per-fold dicts),
            ``sharpe`` (float), and ``max_drawdown`` (float).
        stress_test_results : Dict[str, Dict]
            Mapping of scenario name -> result dict, each containing
            at least a ``max_loss`` key (float).

        Returns
        -------
        ValidationResult
            The gate decision along with detailed metrics.
        """
        folds: List = walk_forward_results.get("folds", [])
        sharpe: float = walk_forward_results.get("sharpe", 0.0)
        drawdown: float = walk_forward_results.get("max_drawdown", 1.0)

        crisis_loss = 0.0
        for scenario, scenario_result in stress_test_results.items():
            loss = scenario_result.get("max_loss", 0.0)
            if loss > crisis_loss:
                crisis_loss = loss

        _check_folds: Optional[ValidationResult] = self._check_min_folds(folds)
        if _check_folds is not None:
            return _check_folds

        _check_sharpe: Optional[ValidationResult] = self._check_sharpe(sharpe)
        if _check_sharpe is not None:
            return _check_sharpe

        _check_dd: Optional[ValidationResult] = self._check_drawdown(drawdown)
        if _check_dd is not None:
            return _check_dd

        _check_stress: Optional[ValidationResult] = self._check_stress_tests(
            stress_test_results
        )
        if _check_stress is not None:
            return _check_stress

        # All checks passed — determine approval level
        if sharpe >= self.config.min_sharpe:
            decision = GateDecision.APPROVED
            reason = f"All checks passed (Sharpe {sharpe:.2f})"
        else:
            decision = GateDecision.CONDITIONAL_APPROVED
            reason = (
                f"Conditional approval — Sharpe {sharpe:.2f} "
                f"(min {self.config.min_sharpe}) but above conditional "
                f"threshold {self.config.conditional_sharpe}"
            )

        result = ValidationResult(
            model_name=model_name,
            decision=decision,
            reason=reason,
            sharpe=sharpe,
            drawdown=drawdown,
            crisis_loss=crisis_loss,
            details={
                "num_folds": len(folds),
                "scenarios_passed": len(stress_test_results),
            },
        )
        self._log_result(result)
        return result

    def _check_min_folds(self, folds: List) -> Optional[ValidationResult]:
        """Reject if the number of walk-forward folds is below the minimum."""
        if len(folds) < self.config.min_folds:
            result = ValidationResult(
                model_name="",
                decision=GateDecision.REJECTED,
                reason=(
                    f"Only {len(folds)} walk-forward folds "
                    f"(minimum {self.config.min_folds} required)"
                ),
                sharpe=0.0,
                drawdown=0.0,
                crisis_loss=0.0,
            )
            self._log_result(result)
            return result
        return None

    def _check_sharpe(self, sharpe: float) -> Optional[ValidationResult]:
        """Reject if Sharpe is below the conditional threshold."""
        if sharpe < self.config.conditional_sharpe:
            result = ValidationResult(
                model_name="",
                decision=GateDecision.REJECTED,
                reason=(
                    f"Sharpe {sharpe:.2f} below conditional threshold "
                    f"{self.config.conditional_sharpe}"
                ),
                sharpe=sharpe,
                drawdown=0.0,
                crisis_loss=0.0,
            )
            self._log_result(result)
            return result
        return None

    def _check_drawdown(self, drawdown: float) -> Optional[ValidationResult]:
        """Reject if max drawdown exceeds the allowed threshold."""
        if drawdown > self.config.max_drawdown_pct:
            result = ValidationResult(
                model_name="",
                decision=GateDecision.REJECTED,
                reason=(
                    f"Max drawdown {drawdown:.2%} exceeds "
                    f"threshold {self.config.max_drawdown_pct:.0%}"
                ),
                sharpe=0.0,
                drawdown=drawdown,
                crisis_loss=0.0,
            )
            self._log_result(result)
            return result
        return None

    def _check_stress_tests(
        self, stress_test_results: Dict[str, Dict]
    ) -> Optional[ValidationResult]:
        """Reject if any crisis scenario exceeds the max loss threshold."""
        for scenario, scenario_data in stress_test_results.items():
            loss = scenario_data.get("max_loss", 0.0)
            if loss > self.config.crisis_max_loss:
                result = ValidationResult(
                    model_name="",
                    decision=GateDecision.REJECTED,
                    reason=(
                        f"Crisis scenario '{scenario}' lost "
                        f"{loss:.2%} (max allowed {self.config.crisis_max_loss:.0%})"
                    ),
                    sharpe=0.0,
                    drawdown=0.0,
                    crisis_loss=loss,
                )
                self._log_result(result)
                return result
        return None

    @staticmethod
    def _log_result(result: ValidationResult) -> None:
        """Log the validation result at the appropriate level."""
        if result.decision == GateDecision.APPROVED:
            logger.success(f"Validation Gate: {result.model_name} — {result.reason}")
        elif result.decision == GateDecision.CONDITIONAL_APPROVED:
            logger.warning(f"Validation Gate: {result.model_name} — {result.reason}")
        else:
            logger.error(f"Validation Gate: {result.model_name} — {result.reason}")
