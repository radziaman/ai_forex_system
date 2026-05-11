"""
Market session utilities — handles forex market hours, liquidity checks.
Forex trades 24/5: Sunday 22:00 GMT to Friday 22:00 GMT.
"""
import pandas as pd
from typing import Optional, Tuple
from loguru import logger


# Major trading sessions (GMT/UTC)
SESSION_TIMES = {
    "sydney": (21, 6),   # 21:00-06:00 GMT
    "tokyo": (23, 8),     # 23:00-08:00 GMT  
    "london": (7, 16),    # 07:00-16:00 GMT
    "new_york": (12, 21), # 12:00-21:00 GMT
}

# High-liquidity overlap periods
OVERLAPS = [
    ("london", "new_york"),   # 12:00-16:00 GMT (best liquidity)
    ("sydney", "tokyo"),     # 23:00-06:00 GMT
]

LOW_LIQUIDITY_HOURS = [
    (5, 6),   # Just before London open
    (20, 21), # Just before Sydney/Tokyo
]

WEEKEND_DAYS = [5, 6]  # Saturday=5, Sunday=6


class MarketSession:
    """Manages market session awareness and liquidity checks."""

    @staticmethod
    def current_gmt_hour() -> int:
        """Get current hour in GMT."""
        return pd.Timestamp.now('GMT').hour

    @staticmethod
    def current_day_of_week() -> int:
        """Get current day of week (0=Monday, 6=Sunday)."""
        return pd.Timestamp.now('GMT').dayofweek

    @staticmethod
    def is_weekend() -> bool:
        """Check if currently weekend (Saturday or Sunday)."""
        return MarketSession.current_day_of_week() in WEEKEND_DAYS

    @staticmethod
    def is_market_open() -> bool:
        """Check if forex market is open (24/5: Sunday 22:00 to Friday 22:00)."""
        now = pd.Timestamp.now('GMT')
        day = now.dayofweek
        hour = now.hour

        # Weekend: Friday 22:00 GMT to Sunday 22:00 GMT
        if day == 5 and hour >= 22:   # Friday 22:00+ = closed
            return False
        if day == 6 and hour < 22:    # Sunday before 22:00 = closed
            return False
        # Sunday 22:00+ and all Monday-Friday before 22:00 = open
        return True

    @staticmethod
    def get_active_sessions() -> list:
        """Get list of currently active trading sessions."""
        hour = MarketSession.current_gmt_hour()
        active = []
        for name, (start, end) in SESSION_TIMES.items():
            if start > end:  # Crosses midnight
                if hour >= start or hour < end:
                    active.append(name)
            else:
                if start <= hour < end:
                    active.append(name)
        return active

    @staticmethod
    def is_high_liquidity() -> Tuple[bool, str]:
        """
        Check if currently in high-liquidity period.
        Returns (is_high, reason).
        """
        active = MarketSession.get_active_sessions()
        if len(active) >= 2:
            return True, "Overlap: " + ", ".join(active)
        if "london" in active or "new_york" in active:
            return True, "Major session: " + active[0]
        none_str = "none"
        return False, "Low-liquidity session: " + (active[0] if active else none_str)

    @staticmethod
    def is_low_liquidity_period() -> bool:
        """Check if currently in known low-liquidity hours."""
        hour = MarketSession.current_gmt_hour()
        for start, end in LOW_LIQUIDITY_HOURS:
            if start <= hour < end:
                return True
        return False

    @staticmethod
    def seconds_since_market_open() -> Optional[float]:
        """Get seconds since market opened (for gap detection)."""
        now = pd.Timestamp.now('GMT')
        day = now.dayofweek
        hour = now.hour

        if MarketSession.is_weekend():
            return None

        # Find last Sunday 22:00
        days_back = 0
        while days_back < 7:
            check = now - pd.Timedelta(days=days_back)
            if check.dayofweek == 6 and check.hour >= 22:
                open_time = check.replace(hour=22, minute=0, second=0)
                return (now - open_time).total_seconds()
            days_back += 1
        return None

    @staticmethod
    def should_pause_trading(reason: str = "") -> Tuple[bool, str]:
        """
        Comprehensive check if trading should be paused.
        Returns (should_pause, reason).
        """
        if not MarketSession.is_market_open():
            return True, "Market closed (outside 24/5 window)"

        if MarketSession.is_weekend():
            return True, "Weekend - market closed"

        if MarketSession.is_low_liquidity_period():
            return True, "Low-liquidity period"

        return False, ""
