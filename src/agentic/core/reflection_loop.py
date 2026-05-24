"""
Reflection Loop — periodic review of recent trading performance.

Runs on a configurable schedule (default: every 50 trades) to:
  1. Analyze recent trade outcomes
  2. Identify patterns in wins and losses
  3. Suggest strategy adjustments
  4. Detect regime changes that may require adaptation

Output feeds into LearningAgent for model retraining decisions.
"""

import time
from typing import Dict, List, Optional, Callable
import numpy as np
from loguru import logger

from .llm_brain import LLMBrain, LLMReflection


class ReflectionLoop:
    """Periodic performance review and strategy adjustment.

    Trades are batched into reflection windows. After each window,
    the reflection engine analyzes performance and outputs suggestions.
    """

    def __init__(
        self,
        llm_brain: Optional[LLMBrain] = None,
        trade_window: int = 50,
        min_trades_for_reflection: int = 10,
        on_reflection: Optional[Callable[[LLMReflection], None]] = None,
    ):
        self.brain = llm_brain or LLMBrain()
        self.trade_window = trade_window
        self.min_trades = min_trades_for_reflection
        self._on_reflection = on_reflection
        self._trade_buffer: List[Dict] = []
        self._reflection_count = 0
        self._last_reflection_time = time.time()

    def record_trade(self, trade: Dict):
        """Record a completed trade for batch analysis."""
        self._trade_buffer.append(trade)
        if len(self._trade_buffer) >= self.trade_window:
            self.run_reflection()

    def run_reflection(self) -> Optional[LLMReflection]:
        """Run reflection on buffered trades."""
        if len(self._trade_buffer) < self.min_trades:
            return None

        wins = sum(1 for t in self._trade_buffer if t.get("pnl", 0) > 0)
        total = len(self._trade_buffer)

        pnl_vals = [t.get("pnl", 0) for t in self._trade_buffer]
        sharpe = (
            (float(np.mean(pnl_vals)) / max(float(np.std(pnl_vals)), 1e-10))
            * (252**0.5)
            if len(pnl_vals) > 1
            else 0.0
        )

        performance = {"total_trades": total, "wins": wins, "sharpe": sharpe}
        reflection = self.brain.reflect_on_trades(self._trade_buffer, performance)

        self._reflection_count += 1
        self._last_reflection_time = time.time()
        self._trade_buffer = []

        if self._on_reflection:
            self._on_reflection(reflection)

        logger.info(
            f"Reflection #{self._reflection_count}: {total} trades, "
            f"WR={wins/total:.0%}, Sharpe={sharpe:.2f}"
        )
        return reflection

    def get_stats(self) -> Dict:
        return {
            "reflection_count": self._reflection_count,
            "buffered_trades": len(self._trade_buffer),
            "last_reflection": self._last_reflection_time,
            "trade_window": self.trade_window,
        }
