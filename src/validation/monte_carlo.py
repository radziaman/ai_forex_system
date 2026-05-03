"""
Monte Carlo Significance Tests for trading strategies.
Permutation testing to determine if strategy edge is statistically significant.
"""
import numpy as np
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from loguru import logger


@dataclass
class SigTestResult:
    actual_sharpe: float
    actual_return_pct: float
    p_value_sharpe: float
    p_value_return: float
    n_permutations: int
    is_significant_sharpe: bool
    is_significant_return: bool
    sharpe_percentile: float
    return_percentile: float


class MonteCarloSigTest:
    def __init__(self, n_permutations: int = 10000, alpha: float = 0.05):
        self.n_permutations = n_permutations
        self.alpha = alpha

    def test(
        self,
        actual_trades: List[Dict],
        return_col: str = "pnl",
    ) -> SigTestResult:
        pnls = np.array([t.get(return_col, 0) for t in actual_trades])
        if len(pnls) < 10:
            logger.warning("Too few trades for significance test")
            return SigTestResult(
                actual_sharpe=0.0,
                actual_return_pct=0.0,
                p_value_sharpe=1.0,
                p_value_return=1.0,
                n_permutations=0,
                is_significant_sharpe=False,
                is_significant_return=False,
                sharpe_percentile=0.5,
                return_percentile=0.5,
            )

        actual_sharpe = self._sharpe(pnls)
        actual_return = float(np.sum(pnls))

        shuffled_sharpes = np.zeros(self.n_permutations)
        shuffled_returns = np.zeros(self.n_permutations)

        for i in range(self.n_permutations):
            shuffled = np.random.permutation(pnls) * np.random.choice([1, -1], size=len(pnls))
            shuffled_sharpes[i] = self._sharpe(shuffled)
            shuffled_returns[i] = float(np.sum(shuffled))

        p_sharpe = float(np.mean(shuffled_sharpes >= actual_sharpe))
        p_return = float(np.mean(shuffled_returns >= actual_return))

        sharpe_pct = float(np.mean(shuffled_sharpes <= actual_sharpe))
        return_pct = float(np.mean(shuffled_returns <= actual_return))

        result = SigTestResult(
            actual_sharpe=actual_sharpe,
            actual_return_pct=actual_return,
            p_value_sharpe=p_sharpe,
            p_value_return=p_return,
            n_permutations=self.n_permutations,
            is_significant_sharpe=p_sharpe < self.alpha,
            is_significant_return=p_return < self.alpha,
            sharpe_percentile=sharpe_pct,
            return_percentile=return_pct,
        )

        logger.info("=" * 60)
        logger.info("MONTE CARLO SIGNIFICANCE TEST")
        logger.info(f"  Actual Sharpe: {actual_sharpe:.3f} (p={p_sharpe:.4f})")
        logger.info(f"  Actual Return: {actual_return:.2f} (p={p_return:.4f})")
        logger.info(
            f"  Sharpe percentile: {sharpe_pct:.1%} "
            f"{'(SIGNIFICANT)' if result.is_significant_sharpe else '(NOT significant)'}"
        )
        logger.info(
            f"  Return percentile: {return_pct:.1%} "
            f"{'(SIGNIFICANT)' if result.is_significant_return else '(NOT significant)'}"
        )
        logger.info(f"  Permutations: {self.n_permutations}")
        logger.info("=" * 60)

        return result

    @staticmethod
    def _sharpe(pnls: np.ndarray) -> float:
        if len(pnls) < 2 or np.std(pnls) == 0:
            return 0.0
        return float(np.mean(pnls) / np.std(pnls) * np.sqrt(252))

    @staticmethod
    def deflate_sharpe(
        sharpe: float, n_trades: int, n_permutations: int = 10000
    ) -> float:
        """Sharpe ratio adjusted for multiple testing / selection bias."""
        if n_trades < 2:
            return 0.0
        e_max = (
            np.sqrt(2 * np.log(n_permutations))
            - (np.log(np.log(n_permutations)) + np.log(4 * np.pi))
            / (2 * np.sqrt(2 * np.log(n_permutations)))
        )
        return (sharpe - e_max) / np.sqrt(1 / n_trades)

    def run_battery(
        self,
        trades_by_regime: Dict[str, List[Dict]],
        trades_overall: List[Dict],
    ) -> Dict:
        results = {}
        results["overall"] = self.test(trades_overall)
        for regime, trades in trades_by_regime.items():
            if len(trades) >= 10:
                label = f"regime_{regime}"
                results[label] = self.test(trades)
        return results
