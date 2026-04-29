"""Risk management module - SL/TP, position sizing, correlation filtering"""

import pandas as pd
from typing import List, Dict, Optional, Tuple


class RiskManager:
    """Manages risk per trade (Zenox-style disciplined approach)"""

    def __init__(
        self,
        account_balance: float,
        max_risk_pct: float = 0.02,
        max_daily_drawdown: float = 0.05,
        max_open_positions: int = 5,
    ):
        self.account_balance = account_balance
        self.max_risk_pct = max_risk_pct
        self.max_daily_drawdown = max_daily_drawdown
        self.max_open_positions = max_open_positions
        self.daily_pnl = 0.0
        self.open_positions: List[Dict] = []

    def calculate_position_size(
        self, entry: float, stop_loss: float, method: str = "fixed"
    ) -> float:
        """Calculate position size using Kelly Criterion or volatility-based"""
        risk_amount = self.account_balance * self.max_risk_pct
        price_risk = abs(entry - stop_loss) / entry

        if method == "kelly":
            win_rate = 0.65
            avg_win = 0.02
            avg_loss = 0.01
            kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly = min(kelly, 0.25)
            risk_amount = self.account_balance * kelly

        lot_size = risk_amount / (price_risk * entry)
        return min(lot_size, self.account_balance * 0.1)

    def calculate_dynamic_sl_tp(
        self, entry: float, atr: float, direction: str = "long"
    ) -> Tuple[float, float]:
        """ATR-based SL/TP (Aurum AI approach)"""
        sl_multiplier = 2.0
        tp_multiplier = 3.0

        if direction == "long":
            stop_loss = entry - (atr * sl_multiplier)
            take_profit = entry + (atr * tp_multiplier)
        else:
            stop_loss = entry + (atr * sl_multiplier)
            take_profit = entry - (atr * tp_multiplier)

        return stop_loss, take_profit

    def check_correlation(
        self,
        new_pair: str,
        open_pairs: List[str],
        correlation_matrix: Optional[pd.DataFrame] = None,
    ) -> bool:
        """Correlation filtering (Zenox EA logic)"""
        if correlation_matrix is not None and new_pair in correlation_matrix.columns:
            for pair in open_pairs:
                if pair in correlation_matrix.columns:
                    corr = abs(correlation_matrix.loc[new_pair, pair])
                    if corr > 0.80:
                        return False
        return len(open_pairs) < self.max_open_positions

    def check_daily_drawdown(self, current_pnl: float) -> bool:
        """Daily drawdown protection"""
        self.daily_pnl = current_pnl
        max_loss = self.account_balance * self.max_daily_drawdown
        return current_pnl > -max_loss

    def can_open_position(self, pair: str, current_pnl: float) -> bool:
        """Master risk check before opening position"""
        if not self.check_daily_drawdown(current_pnl):
            return False
        open_pairs = [p["pair"] for p in self.open_positions]
        if not self.check_correlation(pair, open_pairs):
            return False
        return True


class TrailingStopManager:
    """Multi-tier trailing stop (Zenox EA 3-tier system)"""

    def __init__(
        self,
        tp1_pct: float = 0.01,
        tp2_pct: float = 0.02,
        tp3_pct: float = 0.03,
        trail_atr_multiplier: float = 2.0,
    ):
        self.tp_levels = [tp1_pct, tp2_pct, tp3_pct]
        self.trail_multiplier = trail_atr_multiplier
        self.tp_hit = [False, False, False]

    def update_trailing_stop(
        self, entry: float, current_price: float, atr: float, direction: str = "long"
    ) -> Optional[float]:
        """Update trailing stop based on TP levels"""
        pnl_pct = (
            (current_price - entry) / entry
            if direction == "long"
            else (entry - current_price) / entry
        )

        if pnl_pct >= self.tp_levels[0] and not self.tp_hit[0]:
            self.tp_hit[0] = True
            return entry

        if pnl_pct >= self.tp_levels[1] and not self.tp_hit[1]:
            self.tp_hit[1] = True
            if direction == "long":
                return entry + (atr * self.trail_multiplier)
            else:
                return entry - (atr * self.trail_multiplier)

        if pnl_pct >= self.tp_levels[2] and not self.tp_hit[2]:
            self.tp_hit[2] = True
            if direction == "long":
                return current_price - (atr * self.trail_multiplier)
            else:
                return current_price + (atr * self.trail_multiplier)

        return None

    def partial_close_sizes(self) -> List[float]:
        """Return position sizes for each TP level (30%/30%/40%)"""
        return [0.3, 0.3, 0.4]
