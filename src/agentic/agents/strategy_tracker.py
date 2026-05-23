"""
Strategy Performance Tracker — tracks per-expert PnL, win rates,
and computes dynamic weights based on recent Sharpe ratio.
"""

from __future__ import annotations
import numpy as np
from collections import defaultdict, deque
from typing import Dict, Optional
from dataclasses import dataclass, field


@dataclass
class StrategyStats:
    """Per-strategy performance statistics."""

    name: str
    regime: str = ""
    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    recent_pnls: deque = field(default_factory=lambda: deque(maxlen=20))

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades > 0 else 0.0

    @property
    def recent_win_rate(self) -> float:
        if len(self.recent_pnls) == 0:
            return 0.0
        wins = sum(1 for p in self.recent_pnls if p > 0)
        return wins / len(self.recent_pnls)

    @property
    def recent_sharpe(self) -> float:
        if len(self.recent_pnls) < 3:
            return 0.0
        arr = np.array(self.recent_pnls)
        if np.std(arr) < 1e-10:
            return 0.0
        return float(np.mean(arr) / np.std(arr))

    def record_trade(self, pnl: float):
        self.trades += 1
        self.total_pnl += pnl
        self.recent_pnls.append(pnl)
        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "regime": self.regime,
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 3),
            "recent_win_rate": round(self.recent_win_rate, 3),
            "recent_sharpe": round(self.recent_sharpe, 3),
            "total_pnl": round(self.total_pnl, 2),
        }


class StrategyTracker:
    """
    Tracks performance of all strategies across regimes.
    Provides dynamic weights based on recent Sharpe + regime match.
    Auto-disables strategies with sustained poor performance.
    """

    def __init__(self, window_size: int = 20):
        self._strategies: Dict[str, StrategyStats] = {}
        self._by_regime: Dict[str, dict] = defaultdict(dict)

    def register_strategy(self, name: str, regime: str = ""):
        if name not in self._strategies:
            self._strategies[name] = StrategyStats(name=name, regime=regime)
            if regime:
                self._by_regime[regime][name] = self._strategies[name]

    def record_trade(self, strategy_name: str, pnl: float):
        stats = self._strategies.get(strategy_name)
        if stats is not None:
            stats.record_trade(pnl)

    def get_best_strategy(self, regime: str, min_trades: int = 5) -> Optional[str]:
        """Return the best-performing strategy for a given regime."""
        best_name = None
        best_sharpe = -float("inf")
        for name, stats in self._strategies.items():
            if stats.regime == regime and stats.trades >= min_trades:
                if stats.recent_sharpe > best_sharpe:
                    best_sharpe = stats.recent_sharpe
                    best_name = name
        return best_name

    def get_weight(self, strategy_name: str, regime: str) -> float:
        """
        Returns dynamic weight based on recent Sharpe + regime match.

        - < 3 trades: neutral weight 1.0
        - 10+ trades and Sharpe < -2.0: almost disabled (0.05)
        - 10+ trades and Sharpe < -1.0: heavily penalised (0.1)
        - Otherwise: 0.5 + 0.5 * clip(Sharpe, -1, 1), regime match gets 1.2x
        - Clamped to [0.1, 2.0]
        """
        stats = self._strategies.get(strategy_name)
        if stats is None or stats.trades < 3:
            return 1.0
        sharpe = stats.recent_sharpe
        # Auto-disable very bad strategies
        if stats.trades >= 10 and sharpe < -2.0:
            return 0.05
        if stats.trades >= 10 and sharpe < -1.0:
            return 0.1
        weight = 0.5 + 0.5 * max(-1.0, min(1.0, sharpe))
        if stats.regime == regime:
            weight *= 1.2
        return max(0.1, min(2.0, weight))

    def summary(self) -> dict:
        return {name: stats.to_dict() for name, stats in self._strategies.items()}


class PerSymbolStrategyTracker:
    """Tracks strategy performance per symbol, enabling auto-disable of losing pairs."""

    def __init__(self, min_learn_trades: int = 10, reeval_interval: int = 50):
        self._data: Dict[str, Dict[str, StrategyStats]] = defaultdict(dict)
        self._by_regime: Dict[str, dict] = defaultdict(dict)
        self.min_learn_trades = min_learn_trades
        self.reeval_interval = (
            reeval_interval  # Re-evaluate blocked symbols every N trades
        )
        self._total_trades_global: int = 0  # Global trade counter for re-evaluation
        self._blocked_at: Dict[str, int] = (
            {}
        )  # When each symbol was blocked (global trade count)

    def register(self, symbol: str, strategy: str, regime: str = ""):
        if strategy not in self._data[symbol]:
            self._data[symbol][strategy] = StrategyStats(name=strategy, regime=regime)

    def record_trade(self, symbol: str, strategy: str, pnl: float):
        stats = self._data.get(symbol, {}).get(strategy)
        if stats:
            stats.record_trade(pnl)
        self._total_trades_global += 1

        # If symbol was blocked, check if it's time to re-evaluate
        if symbol in self._blocked_at:
            trades_since_block = self._total_trades_global - self._blocked_at[symbol]
            if trades_since_block >= self.reeval_interval:
                del self._blocked_at[
                    symbol
                ]  # Remove block — will be re-checked on next signal

    def is_strategy_enabled(self, symbol: str, strategy: str) -> bool:
        """A strategy is enabled on a symbol if:
        - It hasn't been tried yet (learning phase) OR
        - It has >= min trades and Sharpe > -0.5
        """
        stats = self._data.get(symbol, {}).get(strategy)
        if stats is None or stats.trades < self.min_learn_trades:
            return True  # Learning phase — try it
        return stats.recent_sharpe > -0.5

    def is_symbol_tradeable(self, symbol: str) -> bool:
        """A symbol is tradeable if at least one strategy has positive expectancy.

        Blocked symbols are automatically re-evaluated every `reeval_interval` global trades.  # noqa: E501
        """
        strategies = self._data.get(symbol, {})
        if not strategies:
            return True  # No data yet — try it

        tradeable = any(
            s.trades >= self.min_learn_trades and s.recent_sharpe > 0
            for s in strategies.values()
        )

        if not tradeable:
            # Check if it's time to re-evaluate a blocked symbol
            if symbol not in self._blocked_at:
                self._blocked_at[symbol] = self._total_trades_global
                return False
            trades_since_block = self._total_trades_global - self._blocked_at.get(
                symbol, 0
            )
            if trades_since_block >= self.reeval_interval:
                # Re-evaluation: try this symbol again
                del self._blocked_at[symbol]
                return True
            return False

        # Symbol is tradeable — clear any pending block
        self._blocked_at.pop(symbol, None)
        return True

    def get_best_strategy(self, symbol: str, regime: str) -> Optional[str]:
        """Return the best-performing enabled strategy for a symbol."""
        candidates = self._data.get(symbol, {})
        best_name, best_sharpe = None, -float("inf")
        for name, stats in candidates.items():
            if stats.trades < self.min_learn_trades:
                continue
            if stats.recent_sharpe > best_sharpe and stats.recent_sharpe > -0.5:
                # Bonus for regime match
                sharpe = stats.recent_sharpe * (1.2 if stats.regime == regime else 1.0)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_name = name
        return best_name

    def summary(self) -> dict:
        result = {}
        for symbol, strategies in self._data.items():
            result[symbol] = {
                "tradeable": self.is_symbol_tradeable(symbol),
                "strategies": {
                    name: {
                        **stats.to_dict(),
                        "enabled": self.is_strategy_enabled(symbol, name),
                    }
                    for name, stats in strategies.items()
                },
            }
        return result
