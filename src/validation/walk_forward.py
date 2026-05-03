"""
Purged Walk-Forward Validation with embargo periods.
Prevents data leakage between train/test splits for time series.
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Callable, Dict
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class WFResult:
    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    sharpe: float = 0.0
    return_pct: float = 0.0
    max_dd: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    n_trades: int = 0
    trades: List[Dict] = field(default_factory=list)


class PurgedWalkForward:
    def __init__(
        self,
        n_folds: int = 6,
        test_window: int = 252,
        embargo: int = 10,
        min_train_window: int = 504,
    ):
        self.n_folds = n_folds
        self.test_window = test_window
        self.embargo = embargo
        self.min_train_window = min_train_window

    def split(
        self, prices: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        total = len(prices)
        folds = []
        for i in range(self.n_folds):
            test_end = total - (self.n_folds - 1 - i) * self.test_window
            test_start = test_end - self.test_window
            if test_start < self.min_train_window:
                continue
            train_end = test_start - self.embargo
            if train_end < self.min_train_window:
                continue
            folds.append((0, train_end, test_start, test_end))
        return folds

    def run(
        self,
        prices: np.ndarray,
        strategy_fn: Callable,
        features: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> List[WFResult]:
        folds = self.split(prices)
        results = []
        for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
            train_prices = prices[tr_s:tr_e]
            test_prices = prices[te_s:te_e]
            train_feat = features[tr_s:tr_e] if features is not None else None
            test_feat = features[te_s:te_e] if features is not None else None

            trades = strategy_fn(train_prices, test_prices, train_feat, test_feat)

            res = WFResult(
                fold=i + 1,
                train_start=tr_s,
                train_end=tr_e,
                test_start=te_s,
                test_end=te_e,
                trades=trades,
            )

            if trades:
                pnls = np.array([t.get("pnl", 0) for t in trades])
                res.return_pct = float(np.sum(pnls))
                res.n_trades = len(trades)
                wins = pnls > 0
                res.win_rate = float(np.mean(wins)) if len(wins) > 0 else 0.0
                gross_win = np.sum(pnls[wins]) if np.any(wins) else 0.0
                gross_loss = abs(np.sum(pnls[~wins])) if np.any(~wins) else 1.0
                res.profit_factor = float(gross_win / max(gross_loss, 1e-8))
                if len(pnls) > 1 and np.std(pnls) > 0:
                    res.sharpe = float(
                        np.mean(pnls) / np.std(pnls) * np.sqrt(252)
                    )
                cum = np.cumsum(pnls)
                peak = np.maximum.accumulate(cum)
                dd = peak - cum
                res.max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0

            results.append(res)
            if verbose:
                logger.info(
                    f"Fold {res.fold}: Return={res.return_pct:.2f}% | "
                    f"Sharpe={res.sharpe:.2f} | WinRate={res.win_rate:.1%} | "
                    f"PF={res.profit_factor:.2f} | MaxDD={res.max_dd:.2f}% | "
                    f"Trades={res.n_trades}"
                )

        if verbose and results:
            avg_sharpe = np.mean([r.sharpe for r in results])
            avg_return = np.mean([r.return_pct for r in results])
            avg_dd = np.mean([r.max_dd for r in results])
            std_sharpe = np.std([r.sharpe for r in results])
            logger.info("=" * 60)
            logger.info(
                f"WALK-FORWARD: Avg Sharpe={avg_sharpe:.2f}±{std_sharpe:.2f} | "
                f"Avg Return={avg_return:.2f}% | Avg MaxDD={avg_dd:.2f}%"
            )
            logger.info("=" * 60)

        return results

    @staticmethod
    def summary(results: List[WFResult]) -> Dict:
        if not results:
            return {}
        sharpes = [r.sharpe for r in results]
        returns = [r.return_pct for r in results]
        dds = [r.max_dd for r in results]
        wrs = [r.win_rate for r in results]
        pfs = [r.profit_factor for r in results]
        return {
            "avg_sharpe": float(np.mean(sharpes)),
            "std_sharpe": float(np.std(sharpes)),
            "min_sharpe": float(np.min(sharpes)),
            "max_sharpe": float(np.max(sharpes)),
            "avg_return_pct": float(np.mean(returns)),
            "avg_max_dd_pct": float(np.mean(dds)),
            "avg_win_rate": float(np.mean(wrs)),
            "avg_profit_factor": float(np.mean(pfs)),
            "total_folds": len(results),
            "avg_trades_per_fold": int(np.mean([r.n_trades for r in results])),
        }
