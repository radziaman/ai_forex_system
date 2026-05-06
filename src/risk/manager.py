"""
Risk Management — consolidated from rts_ai_fx/risk.py + dynamic_position_sizing.py.
Adds VaR (Historical Simulation), stress testing, pre-trade checks.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum


# Use plain strings to avoid enum comparison issues
TRADE_MODE_PAPER = "PAPER"
TRADE_MODE_LIVE = "LIVE"


@dataclass
class RiskParameters:
    max_risk_per_trade: float = 0.02
    max_drawdown: float = 0.10
    max_margin_usage: float = 0.80
    min_win_rate_live: float = 0.55
    min_win_rate_paper: float = 0.45
    kelly_fraction: float = 0.25
    sl_atr_multiplier: float = 2.0
    tp_atr_multiplier: float = 4.0
    trailing_activation_rr: float = 1.0
    trailing_distance_rr: float = 0.5
    max_consecutive_losses: int = 5
    max_daily_loss: float = 0.05
    max_positions: int = 5


@dataclass
class TradeRecord:
    timestamp: float = 0.0
    symbol: str = ""
    direction: str = ""
    volume: float = 0.0
    entry_price: float = 0.0
    exit_price: float = 0.0
    pnl: float = 0.0
    status: str = "OPEN"
    reason: str = ""


class RiskManager:
    """Full risk management suite — Kelly sizing, VaR, drawdown, correlation, stress tests."""

    def __init__(self, params: RiskParameters, initial_balance: float = 100_000.0):
        self.params = params
        self.initial_balance = initial_balance
        self.peak_balance = initial_balance  # Track peak for accurate drawdown
        self.mode: str = TRADE_MODE_PAPER
        self.kill_switch_triggered = False
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.trade_history: List[TradeRecord] = []
        self._price_history: List[float] = []
        self._var_lookback = 60
        self._on_trade_close = None

    def calculate_kelly_size(
        self, balance: float, price: float, atr: float, confidence: float
    ) -> float:
        """Kelly Criterion position sizing with adaptive win-rate tracking."""
        if atr <= 0:
            atr = price * 0.001
        win_rate = self.get_win_rate() or 0.55
        # Compute actual payoff ratio from trade history
        wins = [t.pnl for t in self.trade_history if t.status == "CLOSED" and t.pnl > 0]
        losses = [t.pnl for t in self.trade_history if t.status == "CLOSED" and t.pnl < 0]
        avg_win = np.mean(wins) if wins else 0.02
        avg_loss = abs(np.mean(losses)) if losses else 0.01
        b = avg_win / avg_loss if avg_loss > 0 else 2.0
        kelly = (win_rate * b - (1 - win_rate)) / b
        kelly = np.clip(kelly * self.params.kelly_fraction, 0.01, 0.25)
        risk_amount = balance * kelly * confidence
        sl_distance = atr * self.params.sl_atr_multiplier
        if sl_distance / price <= 0:
            sl_distance = price * 0.015
        size = risk_amount / (sl_distance / price * price)
        return float(np.clip(size, 0, balance * self.params.max_margin_usage / price))

    def calculate_atr_sl_tp(self, entry: float, atr: float) -> Tuple[float, float]:
        sl = entry - (atr * self.params.sl_atr_multiplier)
        tp = entry + (atr * self.params.tp_atr_multiplier)
        return sl, tp

    def var(self, confidence: float = 0.95) -> float:
        """Historical VaR — returns the max loss not exceeded at given confidence."""
        if len(self._price_history) < 20:
            return self.initial_balance * 0.02
        returns = np.diff(self._price_history) / self._price_history[:-1]
        if len(returns) < 10:
            return self.initial_balance * 0.02
        var_val = float(np.percentile(returns, (1 - confidence) * 100)) * self.initial_balance
        return max(var_val, -self.initial_balance * 0.10)  # Cap at 10% loss

    def cvar(self, confidence: float = 0.95) -> float:
        """Conditional VaR (expected shortfall) — average loss beyond VaR."""
        if len(self._price_history) < 20:
            return self.initial_balance * 0.03
        returns = np.diff(self._price_history) / self._price_history[:-1]
        var_val = self.var(confidence)
        tail = returns[returns <= var_val / self.initial_balance]
        if len(tail) == 0:
            return var_val
        cvar_val = float(np.mean(tail)) * self.initial_balance
        return max(cvar_val, -self.initial_balance * 0.15)  # Cap at 15% loss

    def stress_test(
        self, scenario_returns: List[float]
    ) -> dict:
        """Run stress test against historical crisis scenarios."""
        if not scenario_returns:
            return {"max_loss": 0, "impact": 0}
        current_balance = self.initial_balance
        max_loss = 0
        for ret in scenario_returns:
            loss = current_balance * abs(ret)
            if loss > max_loss:
                max_loss = loss
        return {
            "max_loss": max_loss,
            "max_loss_pct": max_loss / self.initial_balance,
            "scenarios_tested": len(scenario_returns),
        }

    def get_win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    def pre_trade_checks(
        self, balance: float, equity: float, margin: float, current_pnl: float
    ) -> Tuple[bool, str]:
        """Run all pre-trade risk checks. Returns (approved, reason)."""
        if self.kill_switch_triggered:
            return False, "Kill switch active"
        # Track peak balance for accurate drawdown
        if balance > self.peak_balance:
            self.peak_balance = balance
        drawdown = (self.peak_balance - balance) / self.peak_balance if self.peak_balance > 0 else 0.0
        if drawdown > self.params.max_drawdown:
            self.kill_switch_triggered = True
            return False, f"Max drawdown exceeded: {drawdown:.1%}"
        daily_loss = current_pnl / self.initial_balance
        if daily_loss < -self.params.max_daily_loss:
            return False, f"Daily loss limit hit: {daily_loss:.1%}"
        if self.consecutive_losses >= self.params.max_consecutive_losses:
            return False, f"Max consecutive losses: {self.consecutive_losses}"
        margin_used = margin / equity if equity > 0 else 0
        if margin_used > self.params.max_margin_usage:
            return False, f"Margin usage too high: {margin_used:.1%}"
        return True, "OK"

    def check_correlation(
        self, new_pair: str, open_pairs: List[str], corr_matrix: Optional[pd.DataFrame] = None
    ) -> bool:
        if corr_matrix is not None and new_pair in corr_matrix.columns:
            for pair in open_pairs:
                if pair in corr_matrix.columns:
                    if abs(corr_matrix.loc[new_pair, pair]) > 0.80:
                        return False
        return len(open_pairs) < self.params.max_positions

    def record_trade(self, trade: TradeRecord, exit_price: float, pnl: float):
        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.status = "CLOSED"
        self.trade_history.append(trade)
        self.total_trades += 1
        self.daily_pnl += pnl  # Track daily PnL for daily loss limit
        if pnl > 0:
            self.wins += 1
            self.consecutive_losses = 0
        else:
            self.losses += 1
            self.consecutive_losses += 1
        if self._on_trade_close:
            try:
                self._on_trade_close(trade)
            except Exception:
                pass

    def update_price_history(self, price: float):
        self._price_history.append(price)
        if len(self._price_history) > self._var_lookback * 24:
            self._price_history = self._price_history[-self._var_lookback * 24:]

    def update_trailing_stops(self, prices: dict):
        """Stub for engine compatibility — trailing stops managed by TrailingStopManager."""
        pass

    def reset_daily_stats(self, balance: float):
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0


class TrailingStopManager:
    """Multi-tier trailing stop (Zenox EA 3-tier system)."""

    def __init__(self, tp_pcts: Optional[List[float]] = None, trail_atr_mult: float = 2.0):
        self.tp_levels = tp_pcts or [0.01, 0.02, 0.03]
        self.trail_mult = trail_atr_mult
        self.tp_hit = [False, False, False]

    def update(
        self, entry: float, current: float, atr: float, direction: str = "long"
    ) -> Optional[float]:
        pnl_pct = (current - entry) / entry if direction == "long" else (entry - current) / entry
        for i, level in enumerate(self.tp_levels):
            if pnl_pct >= level and not self.tp_hit[i]:
                self.tp_hit[i] = True
                if i == 0:
                    return entry
                elif i == 1:
                    return entry + (atr * self.trail_mult) if direction == "long" else entry - (atr * self.trail_mult)
                else:
                    return current - (atr * self.trail_mult) if direction == "long" else current + (atr * self.trail_mult)
        return None

    def partial_close_sizes(self) -> List[float]:
        return [0.3, 0.3, 0.4]
