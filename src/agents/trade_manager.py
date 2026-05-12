"""
Trade Manager — autonomous trade analytics, pattern detection, and performance tracking.
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class TradeAnalytics:
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    sharpe: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    win_rate: float = 0.0

    def update(self, pnl: float):
        self.total_trades += 1
        self.total_pnl += pnl
        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1
        self.win_rate = self.wins / max(self.total_trades, 1)


class TradeManager:
    """
    Tracks and analyzes all trades. Provides:
    - Per-symbol, per-regime, per-day-of-week performance breakdowns
    - Sharpe, profit factor, win rate tracking
    - Pattern detection (best/worst hours, regimes, symbols)
    """

    def __init__(self, max_history: int = 1000):
        self._trades: List[Dict] = []
        self._max_history = max_history
        self._by_symbol: Dict[str, TradeAnalytics] = defaultdict(TradeAnalytics)
        self._by_regime: Dict[str, TradeAnalytics] = defaultdict(TradeAnalytics)
        self._by_dow: Dict[int, TradeAnalytics] = defaultdict(TradeAnalytics)
        self._by_hour: Dict[int, TradeAnalytics] = defaultdict(TradeAnalytics)
        self._pnl_series: deque = deque(maxlen=500)

    def record_trade(self, trade_data: Dict):
        """Record a completed trade for analysis."""
        self._trades.append(trade_data)
        if len(self._trades) > self._max_history:
            self._trades = self._trades[-self._max_history:]

        pnl = trade_data.get("pnl", 0)
        symbol = trade_data.get("symbol", "UNKNOWN")
        regime = trade_data.get("regime", "unknown")
        timestamp = trade_data.get("timestamp", time.time())

        self._pnl_series.append(pnl)
        self._by_symbol[symbol].update(pnl)
        self._by_regime[regime].update(pnl)

        try:
            dt = time.gmtime(timestamp)
            self._by_dow[dt.tm_wday].update(pnl)
            self._by_hour[dt.tm_hour].update(pnl)
        except Exception:
            pass

    def get_analytics(self) -> Dict:
        """Get comprehensive analytics summary."""
        pnls = np.array(self._pnl_series) if self._pnl_series else np.array([0])
        sharpe = self._compute_sharpe(pnls)
        pf = self._compute_profit_factor(pnls)

        return {
            "overall": TradeAnalytics(
                total_trades=len(self._trades),
                wins=sum(1 for p in self._pnl_series if p > 0),
                losses=sum(1 for p in self._pnl_series if p < 0),
                total_pnl=float(np.sum(pnls)),
                sharpe=sharpe,
                profit_factor=pf,
                avg_win=float(np.mean(pnls[pnls > 0])) if np.any(pnls > 0) else 0.0,
                avg_loss=float(np.mean(pnls[pnls < 0])) if np.any(pnls < 0) else 0.0,
                max_consecutive_wins=self._max_consecutive(pnls > 0),
                max_consecutive_losses=self._max_consecutive(pnls < 0),
                win_rate=float(np.mean(pnls > 0)) if len(pnls) > 0 else 0.0,
            ),
            "by_symbol": {
                sym: {
                    "trades": a.total_trades,
                    "wins": a.wins,
                    "pnl": round(a.total_pnl, 2),
                    "win_rate": round(a.win_rate, 3),
                }
                for sym, a in sorted(
                    self._by_symbol.items(), key=lambda x: x[1].total_pnl, reverse=True
                )[:10]
            },
            "by_regime": {
                reg: {
                    "trades": a.total_trades,
                    "pnl": round(a.total_pnl, 2),
                    "win_rate": round(a.win_rate, 3),
                }
                for reg, a in sorted(
                    self._by_regime.items(), key=lambda x: x[1].total_pnl, reverse=True
                )
            },
            "best_symbol": self._best_key(self._by_symbol),
            "worst_symbol": self._worst_key(self._by_symbol),
            "best_regime": self._best_key(self._by_regime),
            "worst_regime": self._worst_key(self._by_regime),
        }

    def get_trade_summary(self) -> str:
        """Get a human-readable summary."""
        a = self.get_analytics()
        o = a["overall"]
        lines = [
            f"Trades: {o.total_trades}  |  "
            f"Win: {o.win_rate:.1%}  |  "
            f"PnL: ${o.total_pnl:+.2f}  |  "
            f"Sharpe: {o.sharpe:.2f}  |  "
            f"PF: {o.profit_factor:.2f}",
        ]
        if a["best_symbol"]:
            lines.append(f"Best: {a['best_symbol']}")
        if a["worst_symbol"]:
            lines.append(f"Worst: {a['worst_symbol']}")
        return "  |  ".join(lines)

    def _best_key(self, data: Dict) -> str:
        if not data:
            return ""
        return max(data, key=lambda k: data[k].total_pnl)

    def _worst_key(self, data: Dict) -> str:
        if not data:
            return ""
        return min(data, key=lambda k: data[k].total_pnl)

    @staticmethod
    def _compute_sharpe(pnls: np.ndarray) -> float:
        if len(pnls) < 5 or np.std(pnls) == 0:
            return 0.0
        return float(np.mean(pnls) / np.std(pnls)) if any(p != 0 for p in pnls) else 0.0

    @staticmethod
    def _compute_profit_factor(pnls: np.ndarray) -> float:
        wins = pnls[pnls > 0].sum()
        losses = abs(pnls[pnls < 0].sum())
        return float(wins / max(losses, 1e-10))

    @staticmethod
    def _max_consecutive(condition: np.ndarray) -> int:
        if len(condition) == 0:
            return 0
        counts = np.diff(np.concatenate(([0], condition, [0]))).cumsum()
        peaks = np.where(np.diff(counts) < 0)[0]
        return int(np.max(np.diff(peaks))) if len(peaks) > 0 else int(condition.sum())
