"""
Performance Agent — autonomous trade analytics and performance tracking.

Identity: I am the scorekeeper. I track every trade and measure system performance.
Purpose: I answer: are we making money? What works? What doesn't?
Autonomy: I independently compute Sharpe, profit factor, win rate, and per-symbol/regime breakdowns.
"""

from __future__ import annotations
import time
import numpy as np
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict, deque
from loguru import logger

from agentic.core.base_agent import BaseAgent
from agentic.core.agent_message import (
    AgentMessage,
    MessageType,
    MessagePriority,
    AgentIntention,
)
from agentic.core.agent_consciousness import ConsciousnessLevel


class PerformanceAgent(BaseAgent):
    """
    Autonomous performance tracking and analytics.

    Responsibilities:
    - Track all trades with per-symbol, per-regime, per-day-of-week breakdowns
    - Compute Sharpe ratio, profit factor, win rate in real-time
    - Detect performance degradation trends
    - Identify best/worst symbols, regimes, hours
    - Maintain rolling performance windows (20, 50, 100 trades)
    """

    def __init__(self):
        super().__init__(
            name="performance_agent",
            role="Performance Analytics Engine",
            purpose="Track, analyze, and report system trading performance across all dimensions",
            domain="performance",
            capabilities={
                "trade_tracking",
                "sharpe_ratio",
                "profit_factor",
                "win_rate_analysis",
                "per_symbol_analytics",
                "per_regime_analytics",
                "performance_degradation_detection",
                "rolling_statistics",
            },
            tick_interval=5.0,
            consciousness_level=ConsciousnessLevel.REFLECTIVE,
        )
        self._trades: List[Dict] = []
        self._max_trades = 1000
        self._pnl_series: deque = deque(maxlen=500)
        self._by_symbol: Dict[str, Dict] = defaultdict(
            lambda: {"trades": 0, "wins": 0, "pnl": 0.0}
        )
        self._by_regime: Dict[str, Dict] = defaultdict(
            lambda: {"trades": 0, "wins": 0, "pnl": 0.0}
        )
        self._by_dow: Dict[int, Dict] = defaultdict(
            lambda: {"trades": 0, "wins": 0, "pnl": 0.0}
        )
        self._by_hour: Dict[int, Dict] = defaultdict(
            lambda: {"trades": 0, "wins": 0, "pnl": 0.0}
        )

        self.subscribe(MessageType.EXECUTION_RESULT)
        self.subscribe(MessageType.POSITION_CLOSED)
        self.subscribe(MessageType.DIAGNOSTIC_REQUEST)

    async def _on_start(self):
        self.set_world("performance.status", "ready")
        self.log_state("Performance tracker ready")

    async def perceive(self) -> Dict[str, Any]:
        return {"trade_count": len(self._trades)}

    async def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        needs_report = (
            self.consciousness.cycle_count % 20 == 0 and len(self._trades) > 0
        )
        return {"compute_report": needs_report}

    async def act(self, decision: Dict[str, Any]):
        if decision.get("compute_report"):
            stats = self._compute_stats()
            self.set_world("performance.stats", stats)
            self.memory.know("performance.sharpe", stats.get("sharpe", 0), ttl=600)
            self.memory.know("performance.win_rate", stats.get("win_rate", 0), ttl=600)
            self.memory.know("performance.pnl", stats.get("total_pnl", 0), ttl=600)

    async def reflect(self, outcome: Dict[str, Any]):
        if self.consciousness.cycle_count % 100 == 0 and len(self._trades) >= 10:
            stats = self._compute_stats()
            if stats.get("sharpe", 0) < 0:
                self.memory.remember(
                    event_type="performance_warning",
                    description=f"Negative Sharpe: {stats['sharpe']:.2f}",
                    importance=0.7,
                    emotion="warning",
                    data=stats,
                )
            degradation = self._detect_degradation()
            if degradation:
                self.memory.remember(
                    event_type="performance_degradation",
                    description=f"Win rate dropped {degradation['drop']:.1%} over last {degradation['window']} trades",
                    importance=0.8,
                    emotion="warning",
                )

    async def on_message(self, message: AgentMessage):
        if message.msg_type == MessageType.EXECUTION_RESULT:
            payload = message.payload if isinstance(message.payload, dict) else {}
            if payload.get("success"):
                self.memory.focus("last_trade", payload)
        elif message.msg_type == MessageType.POSITION_CLOSED:
            payload = message.payload if isinstance(message.payload, dict) else {}
            self.record_trade(payload)
        elif message.msg_type == MessageType.DIAGNOSTIC_REQUEST:
            await self.send(
                MessageType.DIAGNOSTIC_RESULT,
                payload={"agent": self.name, **self._compute_stats()},
                target=message.source_agent,
            )

    def record_trade(self, trade_data: Dict):
        pnl = trade_data.get("pnl", 0)
        symbol = trade_data.get("symbol", "UNKNOWN")
        regime = trade_data.get("regime", "unknown")
        timestamp = trade_data.get("timestamp", time.time())

        self._pnl_series.append(pnl)
        self._trades.append(trade_data)
        if len(self._trades) > self._max_trades:
            self._trades = self._trades[-self._max_trades :]

        self._by_symbol[symbol]["trades"] += 1
        self._by_symbol[symbol]["pnl"] += pnl
        if pnl > 0:
            self._by_symbol[symbol]["wins"] += 1

        self._by_regime[regime]["trades"] += 1
        self._by_regime[regime]["pnl"] += pnl
        if pnl > 0:
            self._by_regime[regime]["wins"] += 1

        try:
            dt = time.gmtime(timestamp)
            self._by_dow[dt.tm_wday]["trades"] += 1
            self._by_dow[dt.tm_wday]["pnl"] += pnl
            self._by_hour[dt.tm_hour]["trades"] += 1
            self._by_hour[dt.tm_hour]["pnl"] += pnl
        except Exception:
            pass

    def _compute_stats(self) -> Dict:
        pnls = np.array(self._pnl_series) if self._pnl_series else np.array([0])
        if len(pnls) < 2:
            return {
                "total_trades": len(self._trades),
                "total_pnl": 0,
                "sharpe": 0,
                "win_rate": 0,
                "profit_factor": 0,
            }

        sharpe = self._sharpe(pnls)
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        win_rate = len(wins) / max(len(pnls), 1)
        profit_factor = float(np.sum(wins) / max(abs(np.sum(losses)), 1e-10))

        return {
            "total_trades": len(self._trades),
            "total_pnl": round(float(np.sum(pnls)), 2),
            "sharpe": round(float(sharpe), 3),
            "win_rate": round(float(win_rate), 3),
            "profit_factor": round(float(profit_factor), 3),
            "avg_win": round(float(np.mean(wins)), 2) if len(wins) > 0 else 0,
            "avg_loss": round(float(np.mean(losses)), 2) if len(losses) > 0 else 0,
            "max_consecutive_wins": self._max_consecutive(pnls > 0),
            "max_consecutive_losses": self._max_consecutive(pnls < 0),
            "best_symbol": self._best_key(self._by_symbol),
            "worst_symbol": self._worst_key(self._by_symbol),
            "best_regime": self._best_key(self._by_regime),
            "worst_regime": self._worst_key(self._by_regime),
            "per_symbol": {
                s: {
                    "trades": d["trades"],
                    "pnl": round(d["pnl"], 2),
                    "win_rate": round(d["wins"] / max(d["trades"], 1), 3),
                }
                for s, d in sorted(
                    self._by_symbol.items(), key=lambda x: x[1]["pnl"], reverse=True
                )[:5]
            },
            "per_regime": {
                r: {"trades": d["trades"], "pnl": round(d["pnl"], 2)}
                for r, d in self._by_regime.items()
            },
        }

    def _detect_degradation(self) -> Optional[Dict]:
        if len(self._trades) < 30:
            return None
        recent = self._trades[-20:]
        older = self._trades[-40:-20]
        if len(recent) < 10 or len(older) < 10:
            return None
        recent_wr = sum(1 for t in recent if t.get("pnl", 0) > 0) / max(len(recent), 1)
        older_wr = sum(1 for t in older if t.get("pnl", 0) > 0) / max(len(older), 1)
        drop = older_wr - recent_wr
        if drop > 0.15:
            return {
                "window": 20,
                "drop": drop,
                "recent_wr": recent_wr,
                "older_wr": older_wr,
            }
        return None

    @staticmethod
    def _sharpe(pnls: np.ndarray) -> float:
        if len(pnls) < 2 or np.std(pnls) == 0:
            return 0.0
        return float(np.mean(pnls) / np.std(pnls)) if np.any(pnls != 0) else 0.0

    @staticmethod
    def _max_consecutive(condition: np.ndarray) -> int:
        if len(condition) == 0:
            return 0
        counts = np.diff(np.concatenate(([0], condition, [0]))).cumsum()
        peaks = np.where(np.diff(counts) < 0)[0]
        return int(np.max(np.diff(peaks))) if len(peaks) > 0 else int(condition.sum())

    @staticmethod
    def _best_key(data: Dict) -> str:
        if not data:
            return ""
        return max(data, key=lambda k: data[k]["pnl"])

    @staticmethod
    def _worst_key(data: Dict) -> str:
        if not data:
            return ""
        return min(data, key=lambda k: data[k]["pnl"])
