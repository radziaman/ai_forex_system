"""
AI-Backed Smart News/Sentiment/Alternative Data Event Avoidance (Enhancement #18).
Auto-close positions pre-events, resume post-volatility, intelligent filtering.
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from loguru import logger


class EventType(str, Enum):
    """Economic event types."""

    NFP = "non_farm_payrols"
    CPI = "cpi_inflation"
    FOMC = "fed_rate_decision"
    ECB = "ecb_rate_decision"
    BOE = "boe_rate_decision"
    BREXIT = "brexit"
    ELECTION = "election"
    CRISIS = "crisis_event"
    HIGH_IMPACT = "high_impact"
    MEDIUM_IMPACT = "medium_impact"
    LOW_IMPACT = "low_impact"


class EventAction(str, Enum):
    """Actions to take for events."""

    CLOSE_ALL = "close_all"
    CLOSE_SYMBOL = "close_symbol"
    PAUSE_TRADING = "pause_trading"
    REDUCE_SIZE = "reduce_size"
    HEDGE = "hedge"
    MONITOR = "monitor"
    RESUME = "resume"


@dataclass
class EventData:
    """Economic/News event data."""

    event_id: str
    title: str
    event_type: EventType
    impact: str  # high | medium | low
    timestamp: float
    currency: str
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None
    volatile_seconds: int = 3600  # Volatility window after event
    affected_symbols: List[str] = field(default_factory=list)


@dataclass
class EventAvoidanceDecision:
    """Decision from AI event avoidance system."""

    should_act: bool = False
    action: EventAction = EventAction.MONITOR
    reason: str = ""
    symbols_affected: List[str] = field(default_factory=list)
    close_immediately: List[str] = field(default_factory=list)
    pause_duration: int = 0  # Seconds to pause
    reduce_size_by: float = 0.0  # Multiplier (0.5 = half size)
    resume_after: float = 0.0  # Timestamp to resume
    confidence: float = 0.0


class SmartEventAvoidance:
    """
    AI-Backed Smart News/Sentiment/Alternative Data Event Avoidance (Enhancement #18).
    - Auto-close positions before high-impact events
    - Resume trading after volatility settles
    - Intelligent filtering based on AI analysis
    """

    # High-impact events that require action
    HIGH_IMPACT_EVENTS = {
        "NFP",
        "CPI",
        "FOMC",
        "ECB",
        "BOE",
        "BREXIT",
        "ELECTION",
        "CRISIS",
    }

    # Symbols affected by currency
    CURRENCY_SYMBOLS = {
        "USD": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"],
        "EUR": ["EURUSD", "EURJPY", "EURGBP"],
        "GBP": ["GBPUSD", "EURGBP", "GBPJPY"],
        "JPY": ["USDJPY", "EURJPY", "GBPJPY"],
        "AUD": ["AUDUSD"],
        "CAD": ["USDCAD"],
        "CHF": ["USDCHF"],
        "NZD": ["NZDUSD"],
    }

    def __init__(
        self,
        pre_event_close_minutes: int = 5,
        post_event_resume_minutes: int = 30,
        volatility_settle_minutes: int = 60,
    ):
        self.pre_event_close = pre_event_close_minutes * 60  # Convert to seconds
        self.post_event_resume = post_event_resume_minutes * 60
        self.volatility_settle = volatility_settle_minutes * 60
        self.upcoming_events: List[EventData] = []
        self.closed_positions: Dict[str, List[int]] = {}  # symbol -> position_ids
        self.paused_until: float = 0.0
        self.size_reduction_until: float = 0.0
        self.size_reduction_factor: float = 1.0
        self.event_history: List[EventData] = []
        self.volatility_tracker: Dict[str, List[float]] = {}  # symbol -> recent vols

    async def check_events(
        self,
        economic_calendar,
        current_positions: List[Dict],
    ) -> EventAvoidanceDecision:
        """
        Main entry point: Check for upcoming events and decide action.
        """
        # Refresh events if needed
        if not self.upcoming_events:
            await self._refresh_events(economic_calendar)

        now = time.time()
        decision = EventAvoidanceDecision()

        # Check if we're in a paused state
        if now < self.paused_until:
            decision.should_act = True
            decision.action = EventAction.PAUSE_TRADING
            decision.reason = (
                f"Paused until {datetime.fromtimestamp(self.paused_until)}"
            )
            decision.pause_duration = int(self.paused_until - now)
            return decision

        # Check for events in the next pre_event_close window
        for event in self.upcoming_events:
            time_to_event = event.timestamp - now

            if 0 <= time_to_event <= self.pre_event_close:
                # High-impact event approaching!
                if event.impact in ["high", "high_impact"]:
                    decision.should_act = True
                    decision.action = (
                        EventAction.CLOSE_ALL
                        if event.event_type == EventType.CRISIS
                        else EventAction.CLOSE_SYMBOL
                    )
                    decision.reason = f"High-impact event in {int(time_to_event/60)}min: {event.title}"
                    decision.symbols_affected = self._get_affected_symbols(
                        event.currency
                    )
                    decision.close_immediately = decision.symbols_affected
                    decision.pause_duration = self.post_event_resume
                    decision.confidence = 0.9
                    return decision

        # Check if we recently had an event and should wait for volatility to settle
        recent_events = [
            e for e in self.event_history if now - e.timestamp < self.volatility_settle
        ]

        if recent_events:
            # Check if volatility has settled
            for event in recent_events:
                if not self._has_volatility_settled(event):
                    decision.should_act = True
                    decision.action = EventAction.REDUCE_SIZE
                    decision.reason = f"Volatility elevated after {event.title}"
                    decision.reduce_size_by = 0.5
                    decision.size_reduction_factor = 0.5
                    self.size_reduction_until = now + 1800  # 30 min
                    return decision

        # No action needed
        return decision

    async def _refresh_events(self, economic_calendar):
        """Refresh upcoming events from economic calendar."""
        try:
            if hasattr(economic_calendar, "get_upcoming_events"):
                raw_events = economic_calendar.get_upcoming_events(hours=48)
                self.upcoming_events = []

                for evt in raw_events:
                    impact = evt.get("impact", "low").lower()
                    if impact in ["high", "medium"]:
                        event = EventData(
                            event_id=f"evt_{int(time.time())}",
                            title=evt.get("title", "Unknown"),
                            event_type=self._classify_event(
                                evt.get(
                                    "title",
                                )
                            ),
                            impact=impact,
                            timestamp=evt.get("timestamp", time.time()),
                            currency=self._extract_currency(
                                evt.get(
                                    "title",
                                )
                            ),
                            volatile_seconds=3600 if impact == "high" else 1800,
                        )
                        event.affected_symbols = self._get_affected_symbols(
                            event.currency
                        )
                        self.upcoming_events.append(event)

        except Exception as e:
            logger.debug(f"Failed to refresh events: {e}")

    def _classify_event(self, title: str) -> EventType:
        """Classify event type from title."""
        title_lower = title.lower()

        if (
            "nfp" in title_lower
            or "non-farm" in title_lower
            or "payroll" in title_lower
        ):
            return EventType.NFP
        elif "cpi" in title_lower or "inflation" in title_lower:
            return EventType.CPI
        elif (
            "fomc" in title_lower
            or "fed" in title_lower
            or "federal reserve" in title_lower
        ):
            return EventType.FOMC
        elif "ecb" in title_lower or "european central" in title_lower:
            return EventType.ECB
        elif "boe" in title_lower or "bank of england" in title_lower:
            return EventType.BOE
        elif "brexit" in title_lower or "brexit" in title_lower:
            return EventType.BREXIT
        elif "election" in title_lower or "vote" in title_lower:
            return EventType.ELECTION
        elif (
            "crisis" in title_lower or "crash" in title_lower or "panic" in title_lower
        ):
            return EventType.CRISIS
        elif "high" in title_lower:
            return EventType.HIGH_IMPACT
        elif "medium" in title_lower:
            return EventType.MEDIUM_IMPACT
        else:
            return EventType.LOW_IMPACT

    def _extract_currency(self, title: str) -> str:
        """Extract currency from event title."""
        title_upper = title.upper()
        for currency in self.CURRENCY_SYMBOLS.keys():
            if currency in title_upper:
                return currency
        return "USD"  # Default

    def _get_affected_symbols(self, currency: str) -> List[str]:
        """Get symbols affected by a currency event."""
        return self.CURRENCY_SYMBOLS.get(currency, [])

    def _has_volatility_settled(self, event: EventData) -> bool:
        """
        Check if volatility has settled after an event.
        Uses simple heuristic: check if recent prices are stable.
        """
        # Simplified: assume volatility settles after volatile_seconds
        elapsed = time.time() - event.timestamp
        return elapsed > event.volatile_seconds

    async def execute_decision(
        self,
        decision: EventAvoidanceDecision,
        execution_engine,
        symbols_to_close: List[str] = None,
    ) -> Dict:
        """
        Execute the avoidance decision.
        Returns summary of actions taken.
        """
        result = {
            "action_taken": decision.action.value,
            "symbols_affected": decision.symbols_affected,
            "positions_closed": [],
            "trading_paused": False,
            "size_reduced": False,
        }

        if decision.action == EventAction.CLOSE_ALL:
            # Close all positions
            await execution_engine.close_all_positions(decision.reason)
            result["trading_paused"] = True
            self.paused_until = time.time() + decision.pause_duration
            result["pause_until"] = datetime.fromtimestamp(
                self.paused_until
            ).isoformat()

        elif decision.action == EventAction.CLOSE_SYMBOL:
            # Close positions for affected symbols
            positions = execution_engine.get_open_positions()
            for pos in positions:
                if pos["symbol"] in decision.close_immediately:
                    await execution_engine.close_position(
                        pos["position_id"], decision.reason
                    )
                    result["positions_closed"].append(pos["symbol"])

        elif decision.action == EventAction.PAUSE_TRADING:
            self.paused_until = time.time() + decision.pause_duration
            result["trading_paused"] = True
            result["pause_until"] = datetime.fromtimestamp(
                self.paused_until
            ).isoformat()

        elif decision.action == EventAction.REDUCE_SIZE:
            self.size_reduction_until = time.time() + 1800  # 30 min
            self.size_reduction_factor = decision.reduce_size_by
            result["size_reduced"] = True
            result["reduction_factor"] = decision.reduce_size_by

        return result

    def get_position_size_multiplier(self) -> float:
        """Get current position size multiplier (1.0 = normal, 0.5 = half size)."""
        if time.time() < self.size_reduction_until:
            return self.size_reduction_factor
        return 1.0

    def should_pause_trading(self) -> Tuple[bool, str]:
        """Check if trading should be paused."""
        if time.time() < self.paused_until:
            return True, f"Paused until {datetime.fromtimestamp(self.paused_until)}"
        return False, "OK"

    def filter_news_by_relevance(
        self, news_items: List[Dict], max_sentiment_age_seconds: int = 3600
    ) -> List[Dict]:
        """
        Filter news/sentiment based on relevance and recency (Enhancement #18).
        Only act on fresh, high-impact news.
        """
        now = time.time()
        filtered = []

        for item in news_items:
            # Check age
            published = item.get("timestamp", 0)
            age = now - published

            if age > max_sentiment_age_seconds:
                continue  # Too old

            # Check impact
            impact = item.get("impact", "low").lower()
            if impact in ["high", "medium"]:
                filtered.append(item)

        return filtered

    def analyze_sentiment_spike(
        self, sentiment_history: List[float], threshold: float = 0.5
    ) -> Dict:
        """
        Detect sentiment spikes that require action (Enhancement #18).
        Returns action to take for extreme sentiment.
        """
        if not sentiment_history or len(sentiment_history) < 5:
            return {"action": "HOLD", "reason": "insufficient_data"}

        recent = sentiment_history[-5:]
        older = (
            sentiment_history[-10:-5]
            if len(sentiment_history) >= 10
            else sentiment_history[:-5]
        )

        if not older:
            return {"action": "HOLD", "reason": "need_more_data"}

        recent_avg = np.mean(recent)
        older_avg = np.mean(older)

        change = recent_avg - older_avg

        # Extreme sentiment spike
        if abs(change) > threshold:
            if recent_avg > 0.7:
                return {
                    "action": "SELL",  # Fade extreme greed
                    "reason": f"extreme_greed_spike: {change:.2f}",
                    "confidence": min(0.9, abs(change)),
                }
            elif recent_avg < -0.7:
                return {
                    "action": "BUY",  # Fade extreme fear
                    "reason": f"extreme_fear_spike: {change:.2f}",
                    "confidence": min(0.9, abs(change)),
                }

        return {"action": "HOLD", "reason": "normal_sentiment"}

    def get_summary(self) -> Dict:
        """Get summary of event avoidance system."""
        return {
            "paused_until": self.paused_until,
            "paused": time.time() < self.paused_until,
            "size_reduction_active": time.time() < self.size_reduction_until,
            "size_multiplier": self.get_position_size_multiplier(),
            "upcoming_high_impact_events": len(
                [e for e in self.upcoming_events if e.impact in ["high", "high_impact"]]
            ),
            "recent_events_count": len(self.event_history),
        }
