"""
Smart Walk-Forward Optimization (Enhancement #14).
Implements Combinatorial Purged Cross-Validation (CPCV), out-of-sample condition testing,
and degradation tracking for robust model validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class WalkForwardResult:
    """Results from a single walk-forward fold."""

    fold_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_sharpe: float = 0.0
    test_sharpe: float = 0.0
    degradation: float = 0.0  # test_sharpe - train_sharpe
    is_overfit: bool = False
    train_returns: List[float] = field(default_factory=list)
    test_returns: List[float] = field(default_factory=list)
    model_performance: Dict[str, float] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Overall optimization results."""

    total_folds: int
    avg_train_sharpe: float = 0.0
    avg_test_sharpe: float = 0.0
    avg_degradation: float = 0.0
    stability_score: float = 0.0  # Inverse of degradation variance
    passed: bool = False
    fold_results: List[WalkForwardResult] = field(default_factory=list)


class SmartWalkForward:
    """
    Smart Walk-Forward with CPCV and degradation tracking (Enhancement #14).
    Goes beyond standard purged walk-forward by testing on multiple
    combinatorial splits to assess model stability.
    """

    def __init__(
        self,
        n_folds: int = 6,
        test_window: int = 252,
        embargo: int = 10,
        min_train_window: int = 504,
        use_cpcv: bool = True,  # Combinatorial Purged Cross-Validation
        n_combinations: int = 10,  # Number of random combinations
    ):
        self.n_folds = n_folds
        self.test_window = test_window
        self.embargo = embargo
        self.min_train_window = min_train_window
        self.use_cpcv = use_cpcv
        self.n_combinations = n_combinations

    def run(
        self,
        prices: np.ndarray,
        features_fn: Callable,  # fn(prices, features) -> List[float]
        features: np.ndarray,
        regimes: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        """
        Run smart walk-forward optimization.
        """
        n = len(prices)
        if n < self.min_train_window + self.test_window:
            logger.warning(
                f"Insufficient data: {n} bars (need {self.min_train_window + self.test_window})"
            )
            return OptimizationResult(total_folds=0)

        fold_results = []

        if self.use_cpcv:
            fold_results = self._run_cpcv(prices, features_fn, features, regimes)
        else:
            fold_results = self._run_standard(prices, features_fn, features, regimes)

        if not fold_results:
            return OptimizationResult(total_folds=0)

        # Calculate summary statistics
        train_sharpe = np.mean([r.train_sharpe for r in fold_results])
        test_sharpe = np.mean([r.test_sharpe for r in fold_results])
        degradations = [r.degradation for r in fold_results]
        avg_degradation = np.mean(degradations)
        stability = 1.0 / (1.0 + np.std(degradations))

        # Pass if avg test Sharpe > 0.5 and degradation is not too negative
        passed = test_sharpe > 0.5 and avg_degradation > -0.5

        return OptimizationResult(
            total_folds=len(fold_results),
            avg_train_sharpe=train_sharpe,
            avg_test_sharpe=test_sharpe,
            avg_degradation=avg_degradation,
            stability_score=stability,
            passed=passed,
            fold_results=fold_results,
        )

    def _run_standard(
        self, prices, features_fn, features, regimes
    ) -> List[WalkForwardResult]:
        """Run standard purged walk-forward."""
        n = len(prices)
        fold_results = []
        fold_id = 0

        for start in range(0, n - self.test_window, self.test_window):
            train_start = start
            train_end = start + self.min_train_window
            test_start = train_end + self.embargo
            test_end = min(test_start + self.test_window, n)

            if test_end > n or train_end - train_start < self.min_train_window:
                continue

            result = self._evaluate_fold(
                fold_id,
                prices,
                features_fn,
                features,
                regimes,
                train_start,
                train_end,
                test_start,
                test_end,
            )
            fold_results.append(result)
            fold_id += 1

        return fold_results

    def _run_cpcv(
        self, prices, features_fn, features, regimes
    ) -> List[WalkForwardResult]:
        """
        Run Combinatorial Purged Cross-Validation (CPCV).
        Creates multiple train/test combinations to assess stability.
        """
        n = len(prices)
        fold_results = []
        fold_id = 0

        # Generate random train/test combinations
        for _ in range(self.n_combinations):
            # Random split point
            min_split = self.min_train_window
            max_split = n - self.test_window

            if min_split >= max_split:
                continue

            split = np.random.randint(min_split, max_split)
            train_start = 0
            train_end = split - self.embargo
            test_start = split
            test_end = min(split + self.test_window, n)

            result = self._evaluate_fold(
                fold_id,
                prices,
                features_fn,
                features,
                regimes,
                train_start,
                train_end,
                test_start,
                test_end,
            )
            fold_results.append(result)
            fold_id += 1

        return fold_results

    def _evaluate_fold(
        self,
        fold_id,
        prices,
        features_fn,
        features,
        regimes,
        train_start,
        train_end,
        test_start,
        test_end,
    ) -> WalkForwardResult:
        """Evaluate a single fold."""
        # Training period
        train_prices = prices[train_start:train_end]
        train_features = (
            features[train_start:train_end] if features is not None else None
        )
        train_regimes = regimes[train_start:train_end] if regimes is not None else None

        # Test period
        test_prices = prices[test_start:test_end]
        test_features = features[test_start:test_end] if features is not None else None
        test_regimes = regimes[test_start:test_end] if regimes is not None else None

        # Get returns from model predictions
        train_returns = (
            features_fn(train_prices, train_features, train_regimes)
            if callable(features_fn)
            else []
        )
        test_returns = (
            features_fn(test_prices, test_features, test_regimes)
            if callable(features_fn)
            else []
        )

        # Calculate Sharpe ratios
        train_sharpe = self._calculate_sharpe(train_returns)
        test_sharpe = self._calculate_sharpe(test_returns)

        # Degradation
        degradation = test_sharpe - train_sharpe
        is_overfit = degradation < -0.5  # Significant degradation

        return WalkForwardResult(
            fold_id=fold_id,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            train_sharpe=train_sharpe,
            test_sharpe=test_sharpe,
            degradation=degradation,
            is_overfit=is_overfit,
            train_returns=train_returns,
            test_returns=test_returns,
        )

    def _calculate_sharpe(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio from returns."""
        if not returns or len(returns) < 2:
            return 0.0
        returns = np.array(returns)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        if std_ret <= 0:
            return 0.0
        return mean_ret / std_ret * np.sqrt(252)  # Annualized

    def check_out_of_sample_conditions(
        self,
        returns: List[float],
        market_returns: Optional[List[float]] = None,
    ) -> Dict:
        """
        Test if model performs well under different market conditions (Enhancement #14).
        """
        if not returns or len(returns) < 10:
            return {"passed": False}

        returns = np.array(returns)

        # Test on uptrends, downtrends, high volatility, low volatility
        conditions = {
            "uptrend": returns[returns > 0],
            "downtrend": returns[returns < 0],
            "high_vol": returns[np.abs(returns) > np.percentile(np.abs(returns), 75)],
            "low_vol": returns[np.abs(returns) <= np.percentile(np.abs(returns), 25)],
        }

        results = {}
        for condition, subset in conditions.items():
            if len(subset) > 0:
                sharpe = self._calculate_sharpe(subset.tolist())
                results[condition] = {
                    "sharpe": sharpe,
                    "num_samples": len(subset),
                    "passed": sharpe > 0.3,
                }

        # Beta to market
        if market_returns and len(market_returns) == len(returns):
            covariance = np.cov(returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            beta = covariance / market_variance if market_variance > 0 else 1.0
            results["beta"] = beta

        return results

    def track_degradation(self, fold_results: List[WalkForwardResult]) -> Dict:
        """
        Track model degradation over time (Enhancement #14).
        Detect when model starts to decay and needs retraining.
        """
        if not fold_results:
            return {}

        degradations = [r.degradation for r in fold_results]
        cumulative_degradation = np.cumsum(degradations)

        # Detect significant degradation (3 consecutive negative degradations)
        alert = False
        for i in range(len(degradations) - 2):
            if all(d < -0.2 for d in degradations[i : i + 3]):
                alert = True
                break

        return {
            "degradations": degradations,
            "cumulative_degradation": cumulative_degradation.tolist(),
            "mean_degradation": np.mean(degradations),
            "std_degradation": np.std(degradations),
            "alert": alert,
            "num_overfit_folds": sum(1 for r in fold_results if r.is_overfit),
        }
