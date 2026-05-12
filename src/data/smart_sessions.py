"""
AI-Backed Smart Trading Sessions Filter (Enhancement #16).
Optimizes trading hours per pair using ML-based session analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class SessionType(str, Enum):
    """Trading session types."""

    LONDON = "london"
    NEW_YORK = "new_york"
    TOKYO = "tokyo"
    SYDNEY = "sydney"
    WEEKEND = "weekend"
    CLOSED = "closed"


@dataclass
class SessionPerformance:
    """Performance metrics for a trading session."""

    session: SessionType
    symbol: str
    avg_return: float = 0.0
    sharpe: float = 0.0
    win_rate: float = 0.0
    avg_volatility: float = 0.0
    trade_count: int = 0
    profit_factor: float = 0.0


@dataclass
class SessionFilterResult:
    """Result from session filtering."""

    symbol: str
    current_session: SessionType
    should_trade: bool
    confidence: float = 0.0
    reason: str = ""
    optimal_sessions: List[SessionType] = field(default_factory=list)
    session_scores: Dict[str, float] = field(default_factory=dict)


class SmartTradingSessions:
    """
    AI-Backed Smart Trading Sessions Filter (Enhancement #16).
    Uses ML to optimize trading hours per pair based on historical performance.
    """

    # Session time ranges (UTC)
    SESSION_RANGES = {
        SessionType.TOKYO: (0, 6),  # 00:00-06:00 UTC
        SessionType.LONDON: (8, 16),  # 08:00-16:00 UTC
        SessionType.NEW_YORK: (13, 21),  # 13:00-21:00 UTC
        SessionType.SYDNEY: (22, 24),  # 22:00-24:00 UTC (continues to 06:00)
    }

    # Optimal sessions per symbol (ML will adjust these)
    DEFAULT_OPTIMAL = {
        "EURUSD": [SessionType.LONDON, SessionType.NEW_YORK],
        "GBPUSD": [SessionType.LONDON, SessionType.NEW_YORK],
        "USDJPY": [SessionType.TOKYO, SessionType.LONDON],
        "AUDUSD": [SessionType.SYDNEY, SessionType.LONDON],
        "USDCAD": [SessionType.LONDON, SessionType.NEW_YORK],
        "USDCHF": [SessionType.LONDON, SessionType.NEW_YORK],
        "NZDUSD": [SessionType.SYDNEY, SessionType.LONDON],
        "XAUUSD": [SessionType.LONDON, SessionType.NEW_YORK],
        "BTCUSD": [SessionType.NEW_YORK],  # Crypto trades 24/7
        "ETHUSD": [SessionType.NEW_YORK],
    }

    def __init__(self, lookback_days: int = 60):
        self.lookback_days = lookback_days
        self.session_performance: Dict[str, Dict[SessionType, SessionPerformance]] = {}
        self.session_models: Dict[str, Dict] = {}  # ML models per symbol
        self._init_defaults()

    def _init_defaults(self):
        """Initialize default session performance."""
        for symbol in self.DEFAULT_OPTIMAL.keys():
            self.session_performance[symbol] = {}
            for session in SessionType:
                self.session_performance[symbol][session] = SessionPerformance(
                    session=session, symbol=symbol
                )

    def get_current_session(self) -> SessionType:
        """Get current trading session based on UTC time."""
        import datetime

        now_utc = datetime.datetime.utcnow()
        hour = now_utc.hour

        # Check weekends
        if now_utc.weekday() >= 5:  # Saturday=5, Sunday=6
            return SessionType.WEEKEND

        # Check session ranges
        for session, (start, end) in self.SESSION_RANGES.items():
            if start <= hour < end:
                return session

        return SessionType.CLOSED

    def is_high_liquidity(self, symbol: str = None) -> Tuple[bool, str]:
        """Check if current time is high liquidity (Enhancement #16)."""
        current = self.get_current_session()

        if current == SessionType.WEEKEND:
            return False, "Weekend - market closed"
        if current == SessionType.CLOSED:
            return False, "Outside major trading sessions"

        # Optimal sessions have high liquidity
        optimal = self.DEFAULT_OPTIMAL.get(
            symbol, [SessionType.LONDON, SessionType.NEW_YORK]
        )
        if current in optimal:
            return True, f"High liquidity: {current.value} session"

        return False, f"Low liquidity: {current.value} session"

    def should_trade_symbol(
        self, symbol: str, hour: Optional[int] = None
    ) -> SessionFilterResult:
        """
        AI-backed decision on whether to trade a symbol at current time.
        Uses session performance history to make optimal decisions.
        """
        if hour is None:
            import datetime

            hour = datetime.datetime.utcnow().hour

        current_session = self._get_session_from_hour(hour)

        # Get session scores from ML model
        session_scores = self._calculate_session_scores(symbol)

        # Get optimal sessions (top 2 by score)
        sorted_sessions = sorted(
            session_scores.items(), key=lambda x: x[1], reverse=True
        )
        optimal_sessions = [SessionType(s[0]) for s in sorted_sessions[:2]]

        # Decision logic
        confidence = session_scores.get(current_session.value, 0.0)

        if current_session == SessionType.WEEKEND:
            return SessionFilterResult(
                symbol=symbol,
                current_session=current_session,
                should_trade=False,
                confidence=0.0,
                reason="Weekend - market closed",
                optimal_sessions=optimal_sessions,
                session_scores=session_scores,
            )

        if current_session in optimal_sessions:
            should_trade = confidence > 0.6
            reason = (
                "Optimal session"
                if should_trade
                else "Suboptimal performance in session"
            )
        else:
            should_trade = confidence > 0.8  # Higher threshold for non-optimal
            reason = (
                "Non-optimal session" if not should_trade else "Trading outside optimal"
            )

        return SessionFilterResult(
            symbol=symbol,
            current_session=current_session,
            should_trade=should_trade,
            confidence=confidence,
            reason=reason,
            optimal_sessions=optimal_sessions,
            session_scores=session_scores,
        )

    def _get_session_from_hour(self, hour: int) -> SessionType:
        """Get session type from hour (UTC)."""
        if hour < 0 or hour >= 24:
            return SessionType.CLOSED

        for session, (start, end) in self.SESSION_RANGES.items():
            if start <= hour < end:
                return session

        return SessionType.CLOSED

    def _calculate_session_scores(self, symbol: str) -> Dict[str, float]:
        """Calculate ML-based session scores (Enhancement #16)."""
        scores = {}

        # Base scores from historical performance
        if symbol in self.session_performance:
            for session, perf in self.session_performance[symbol].items():
                if perf.trade_count > 5:
                    # Score based on Sharpe + win rate
                    score = (perf.sharpe * 0.5) + (perf.win_rate * 0.5)
                    scores[session.value] = min(max(score, 0.0), 1.0)
                else:
                    # Default scores for unknown sessions
                    if session in self.DEFAULT_OPTIMAL.get(symbol, []):
                        scores[session.value] = 0.7
                    else:
                        scores[session.value] = 0.3
        else:
            # No data - use defaults
            for session in SessionType:
                if session in self.DEFAULT_OPTIMAL.get(symbol, []):
                    scores[session.value] = 0.7
                else:
                    scores[session.value] = 0.3

        return scores

    def update_performance(
        self, symbol: str, session: SessionType, pnl: float, return_pct: float
    ):
        """Update session performance with new trade data (ML training)."""
        if symbol not in self.session_performance:
            self.session_performance[symbol] = {}
            for s in SessionType:
                self.session_performance[symbol][s] = SessionPerformance(
                    session=s, symbol=symbol
                )

        perf = self.session_performance[symbol][session]
        perf.trade_count += 1

        # Update average return (exponential moving average)
        alpha = 0.1
        perf.avg_return = (1 - alpha) * perf.avg_return + alpha * return_pct

        # Update Sharpe (simplified)
        if perf.trade_count > 1:
            perf.sharpe = perf.avg_return / (0.01 + abs(return_pct))  # Simplified

        # Update win rate
        if pnl > 0:
            perf.win_rate = (
                perf.win_rate * (perf.trade_count - 1) + 1.0
            ) / perf.trade_count
        else:
            perf.win_rate = (perf.win_rate * (perf.trade_count - 1)) / perf.trade_count

        # Profit factor
        perf.profit_factor = max(0.5, min(3.0, 1.0 + perf.avg_return * 10))

    def get_optimal_trading_window(self, symbol: str) -> Tuple[int, int]:
        """Get optimal trading window (start_hour, end_hour) for symbol."""
        scores = self._calculate_session_scores(symbol)
        sorted_sessions = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if sorted_sessions:
            best_session = SessionType(sorted_sessions[0][0])
            if best_session in self.SESSION_RANGES:
                return self.SESSION_RANGES[best_session]

        # Default to London session
        return (8, 16)

    def get_all_sessions_summary(self) -> Dict:
        """Get summary of all session performance."""
        summary = {}
        for symbol, sessions in self.session_performance.items():
            summary[symbol] = {
                session.value: {
                    "avg_return": perf.avg_return,
                    "sharpe": perf.sharpe,
                    "win_rate": perf.win_rate,
                    "trades": perf.trade_count,
                    "profit_factor": perf.profit_factor,
                }
                for session, perf in sessions.items()
            }
        return summary
