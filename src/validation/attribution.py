"""Trade attribution and alpha decay tracking."""

from dataclasses import dataclass
from typing import Any, Dict, List
from collections import defaultdict, deque

import numpy as np


@dataclass
class TradeAttribution:
    """Decomposition of a single trade's PnL into components."""

    alpha_signal: float = 0.0
    execution_quality: float = 0.0
    slippage: float = 0.0
    luck: float = 0.0
    total_pnl: float = 0.0

    @property
    def unexplained(self) -> float:
        """Residual PnL not captured by the decomposition."""
        return (
            self.total_pnl
            - self.alpha_signal
            - self.execution_quality
            - self.slippage
            - self.luck
        )


class StrategyAttributionEngine:
    """Tracks per-strategy PnL attribution and alpha decay over time.

    Determines whether a strategy should be disabled based on recent
    performance degradation.
    """

    def __init__(
        self,
        slippage_estimate: float = 0.0001,
        luck_window: int = 50,
    ):
        self.slippage_estimate = slippage_estimate
        self.luck_window = luck_window
        self._trade_history: Dict[str, List[TradeAttribution]] = defaultdict(list)
        self._returns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self._sharpe_history: Dict[str, List[float]] = defaultdict(list)

    def attribute_trade(self, trade_record: Dict[str, Any]) -> TradeAttribution:
        """Decompose a trade record into attribution components.

        Expected trade_record keys:
          - pnl: float (total realized PnL)
          - expected_pnl: float (PnL predicted by signal model)
          - fill_price: float
          - signal_price: float (price at signal generation)
          - slippage_bps: optional float (explicit slippage)
          - strategy: str (name of the strategy)
          - market_return: float (buy-and-hold return over trade duration)
        """
        total_pnl = float(trade_record.get("pnl", 0.0))
        expected_pnl = float(trade_record.get("expected_pnl", 0.0))
        fill_price = float(trade_record.get("fill_price", 0.0))
        signal_price = float(trade_record.get("signal_price", fill_price))
        market_return = float(trade_record.get("market_return", 0.0))
        strategy = str(trade_record.get("strategy", "unknown"))

        # Slippage
        explicit_slippage = trade_record.get("slippage_bps")
        if explicit_slippage is not None:
            slippage = -abs(float(explicit_slippage))
        else:
            slippage = -abs(fill_price - signal_price) * (1 if total_pnl >= 0 else -1)

        # Execution quality = difference between expected and total minus slippage
        execution_quality = total_pnl - expected_pnl - slippage

        # Luck = portion correlated with overall market return but not alpha
        luck = market_return * abs(total_pnl) * 0.5

        # Alpha signal = expected PnL (what the strategy 'earned' in theory)
        alpha_signal = expected_pnl

        attribution = TradeAttribution(
            alpha_signal=alpha_signal,
            execution_quality=execution_quality,
            slippage=slippage,
            luck=luck,
            total_pnl=total_pnl,
        )

        self._trade_history[strategy].append(attribution)
        self._returns[strategy].append(total_pnl)
        return attribution

    def calculate_alpha_decay(
        self, strategy_name: str, window: int = 20
    ) -> Dict[str, Any]:
        """Return Sharpe and other decay metrics for a strategy over a
        rolling window."""
        returns = list(self._returns.get(strategy_name, []))
        if len(returns) < window:
            return {
                "sharpe": 0.0,
                "mean_return": 0.0,
                "std_return": 0.0,
                "samples": len(returns),
                "window": window,
            }
        recent = np.array(list(returns)[-window:])
        mean_r = float(np.mean(recent))
        std_r = float(np.std(recent))
        sharpe = (mean_r / std_r * np.sqrt(252)) if std_r > 1e-12 else 0.0
        self._sharpe_history[strategy_name].append(sharpe)
        return {
            "sharpe": round(sharpe, 4),
            "mean_return": round(mean_r, 6),
            "std_return": round(std_r, 6),
            "samples": len(recent),
            "window": window,
        }

    def should_disable(self, strategy_name: str, min_trades: int = 20) -> bool:
        """Return True if the strategy's alpha has decayed below zero for min_trades."""
        trades = self._trade_history.get(strategy_name, [])
        if len(trades) < min_trades:
            return False
        recent = trades[-min_trades:]
        alpha_pnl = [t.alpha_signal for t in recent]
        return all(a <= 0 for a in alpha_pnl)

    def get_report(self) -> Dict[str, Any]:
        """Return a full attribution report for all tracked strategies."""
        report: Dict[str, Any] = {}
        for strategy, trades in self._trade_history.items():
            if not trades:
                continue
            total_pnl = sum(t.total_pnl for t in trades)
            total_alpha = sum(t.alpha_signal for t in trades)
            total_exec = sum(t.execution_quality for t in trades)
            total_slippage = sum(t.slippage for t in trades)
            total_luck = sum(t.luck for t in trades)
            n = len(trades)
            wins = sum(1 for t in trades if t.total_pnl > 0)
            win_rate = wins / n if n > 0 else 0.0
            recent_pnl = [round(t.total_pnl, 4) for t in trades[-10:]]
            decay = self.calculate_alpha_decay(strategy, window=20)
            report[strategy] = {
                "trades": n,
                "total_pnl": round(total_pnl, 4),
                "alpha_signal": round(total_alpha, 4),
                "execution_quality": round(total_exec, 4),
                "slippage": round(total_slippage, 4),
                "luck": round(total_luck, 4),
                "win_rate": round(win_rate, 4),
                "recent_pnl": recent_pnl,
                "alpha_decay": decay,
                "should_disable": self.should_disable(strategy),
            }
        return report
