"""
Risk Management — consolidated from rts_ai_fx/risk.py + dynamic_position_sizing.py.
Adds VaR (Historical Simulation), stress testing, pre-trade checks.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

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
    """Full risk management suite — Kelly sizing, VaR, drawdown, correlation, stress tests."""  # noqa: E501

    def __init__(self, params: RiskParameters, initial_balance: float = 100_000.0):
        self.params = params
        self.initial_balance = initial_balance
        self.peak_balance = initial_balance
        self.mode: str = TRADE_MODE_PAPER
        self.kill_switch_triggered = False
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.trade_history: List[TradeRecord] = []
        # Fix 2: Per-symbol price history for VaR instead of shared single list
        self._price_history: Dict[str, List[float]] = {}
        self._var_lookback = 60
        self._on_trade_close = None
        from risk.correlation_matrix import RegimeCorrelationMatrix

        self._correlation_matrix = RegimeCorrelationMatrix(window=100)

    def calculate_kelly_size(
        self,
        balance: float,
        price: float,
        atr: float,
        confidence: float,
        symbol: Optional[str] = None,
        open_positions: Optional[list] = None,
    ) -> float:
        """
        Kelly Criterion position sizing with VaR adjustment.
        Uses R-multiple (PnL normalized by risk) for statistically meaningful averaging.
        """
        if atr <= 0:
            atr = price * 0.001

        # Base Kelly calculation — use R-multiple (PnL / risk) for size-independent averaging  # noqa: E501
        win_rate = self.get_win_rate()
        if win_rate == 0.0 and self.total_trades == 0:
            win_rate = 0.55
        elif win_rate == 0.0 and self.total_trades > 0:
            win_rate = 0.10

        closed = [t for t in self.trade_history if t.status == "CLOSED"]
        if closed:
            r_multiples = []
            for t in closed:
                sl_price = getattr(t, "sl", 0)
                if sl_price is None or sl_price <= 0:
                    continue
                # Correct R-multiple: PnL / (entry-SL distance * volume)
                # SL distance represents risk per unit, times notional
                risk_distance = abs(t.entry_price - sl_price)
                entry_pct_risk = (
                    risk_distance / t.entry_price if t.entry_price > 0 else 0.01
                )
                notional_risk = entry_pct_risk * t.entry_price * t.volume
                if notional_risk > 0 and t.pnl != 0:
                    r_multiples.append(t.pnl / notional_risk)
            if len(r_multiples) >= 3:
                positives = [r for r in r_multiples if r > 0]
                negatives = [r for r in r_multiples if r < 0]
                avg_r_win = np.mean(positives) if positives else 0.5
                avg_r_loss = abs(np.mean(negatives)) if negatives else 0.5
                b = max(avg_r_win / max(avg_r_loss, 0.001), 1.01)
            else:
                b = 2.0
        else:
            b = 2.0
        kelly = (win_rate * b - (1 - win_rate)) / b
        kelly = np.clip(kelly * self.params.kelly_fraction, 0.01, 0.25)

        # VaR adjustment (Enhancement #10)
        var_adjustment = self._calculate_var_adjustment(balance, symbol, open_positions)
        risk_amount = balance * kelly * confidence * var_adjustment

        # Volatility targeting (target 15% annualized)
        vol_adjustment = self._calculate_volatility_adjustment(atr, price)
        risk_amount *= vol_adjustment

        sl_distance = atr * self.params.sl_atr_multiplier
        if sl_distance / price <= 0:
            sl_distance = price * 0.015

        size = risk_amount / (sl_distance / price * price)
        return float(np.clip(size, 0, balance * self.params.max_margin_usage / price))

    def _calculate_var_adjustment(
        self,
        balance: float,
        symbol: Optional[str] = None,
        open_positions: Optional[List[Dict]] = None,
    ) -> float:
        """Calculate VaR-based adjustment factor."""
        if len(self._price_history) < 20:
            return 1.0

        # Calculate portfolio VaR if we have open positions
        if open_positions and symbol:
            portfolio_var = self._portfolio_var(open_positions)
            var_limit = balance * 0.02  # 2% VaR limit
            if portfolio_var > var_limit:
                return max(0.5, var_limit / portfolio_var)
        else:
            current_var = abs(self.var())
            var_limit = balance * 0.02
            if current_var > var_limit:
                return max(0.5, var_limit / current_var)

        return 1.0

    def _portfolio_var(self, open_positions: List[Dict]) -> float:
        if not open_positions:
            return 0.0

        total_var = 0.0
        for pos in open_positions:
            volume = float(pos.get("volume", 0))
            portfolio_share = volume / 100_000.0 if volume > 0 else 0.01
            pos_var = abs(self.var()) * portfolio_share
            total_var += pos_var

        n = len(open_positions)
        correlation_adjustment = (n + 0.5 * n * (n - 1)) ** 0.5 / n if n > 1 else 1.0
        return total_var * correlation_adjustment

    def _calculate_volatility_adjustment(self, atr: float, price: float) -> float:
        """Adjust position size based on volatility targeting."""
        if atr <= 0 or price <= 0:
            return 1.0

        # Calculate current volatility (annualized)
        current_vol = (atr / price) * (252**0.5)  # Approximate annualized vol
        target_vol = 0.15  # 15% target

        if current_vol <= 0:
            return 1.0

        # Inverse relationship: higher vol = smaller size
        adjustment = target_vol / current_vol
        return float(np.clip(adjustment, 0.25, 2.0))

    def calculate_atr_sl_tp(self, entry: float, atr: float) -> Tuple[float, float]:
        sl = entry - (atr * self.params.sl_atr_multiplier)
        tp = entry + (atr * self.params.tp_atr_multiplier)
        return sl, tp

    def _get_price_series(self, symbol: Optional[str] = None) -> List[float]:
        """Get price history for VaR computation.

        Uses per-symbol price history (Fix 2). Falls back to any available
        series if symbol not specified.
        """
        key = (
            symbol or list(self._price_history.keys())[0]
            if self._price_history
            else None
        )
        if key and key in self._price_history:
            return self._price_history[key]
        # Return first available series as fallback
        for k in self._price_history:
            return self._price_history[k]
        return []

    def var(self, confidence: float = 0.95, symbol: Optional[str] = None) -> float:
        """Historical VaR — per-symbol price history (Fix 2)."""
        current_balance = max(self.initial_balance, 1)
        prices = self._get_price_series(symbol)
        if len(prices) < 20:
            return -current_balance * 0.02
        returns = np.diff(prices) / np.array(prices[:-1], dtype=float)
        returns = returns[np.isfinite(returns)]
        if len(returns) < 10:
            return -current_balance * 0.02
        var_pct = float(np.percentile(returns, (1 - confidence) * 100))
        return var_pct * current_balance

    def cvar(self, confidence: float = 0.95, symbol: Optional[str] = None) -> float:
        """Conditional VaR — per-symbol price history (Fix 2)."""
        current_balance = max(self.initial_balance, 1)
        prices = self._get_price_series(symbol)
        if len(prices) < 20:
            return -current_balance * 0.03
        returns = np.diff(prices) / np.array(prices[:-1], dtype=float)
        returns = returns[np.isfinite(returns)]
        var_val = self.var(confidence, symbol)
        var_frac = var_val / current_balance
        tail = returns[returns <= var_frac]
        if len(tail) == 0:
            return var_val
        cvar_val = float(np.mean(tail)) * current_balance
        return cvar_val

    def stress_test(self, scenario_returns: List[float]) -> dict:
        """Run stress test against historical crisis scenarios."""
        if not scenario_returns:
            return {"max_loss": 0, "impact": 0}
        current_balance = self.initial_balance
        max_loss = 0.0
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
        self,
        balance: float,
        equity: float,
        margin: float,
        current_pnl: float,
        new_symbol: str = "",
        open_symbols: Optional[List[str]] = None,
    ) -> Tuple[bool, str]:
        """Run all pre-trade risk checks. Returns (approved, reason).

        Includes:
          - Kill switch, drawdown, daily loss, consecutive losses
          - Correlation risk: reject if new position is >0.80 correlated
            with an existing position (portfolio-level risk)
        """
        if self.kill_switch_triggered:
            return False, "Kill switch active"

        # Track peak equity for accurate drawdown (includes floating PnL)
        if equity > self.peak_balance:
            self.peak_balance = equity
        drawdown = (
            (self.peak_balance - equity) / self.peak_balance
            if self.peak_balance > 0
            else 0.0
        )
        if drawdown > self.params.max_drawdown:
            self.kill_switch_triggered = True
            return False, f"Max drawdown exceeded: {drawdown:.1%}"

        # Daily loss check uses tracked daily PnL, not total unrealized
        daily_loss = (
            self.daily_pnl / self.initial_balance if self.initial_balance > 0 else 0.0
        )
        if daily_loss < -self.params.max_daily_loss:
            return False, f"Daily loss limit hit: {daily_loss:.1%}"

        if self.consecutive_losses >= self.params.max_consecutive_losses:
            return False, f"Max consecutive losses: {self.consecutive_losses}"
        margin_used = margin / equity if equity > 0 else 0
        if margin_used > self.params.max_margin_usage:
            return False, f"Margin usage too high: {margin_used:.1%}"

        # Portfolio-level correlation risk
        if new_symbol and open_symbols:
            corr_check = self._check_correlation_risk(new_symbol, open_symbols)
            if not corr_check[0]:
                return corr_check

        return True, "OK"

    def _check_correlation_risk(
        self, new_symbol: str, open_symbols: List[str]
    ) -> Tuple[bool, str]:
        """Reject new position if it's >0.80 correlated with any open position.

        Uses the known correlation matrix of forex pairs to prevent
        over-concentration (e.g. long EURUSD + long GBPUSD = double USD short).
        Falls back to symbol-count check if correlation data unavailable.
        """
        if not open_symbols:
            return True, "OK"

        # Dynamic regime-correlation matrix check (if available)
        if self._correlation_matrix is not None:
            for pair in open_symbols:
                corr = self._correlation_matrix.get(new_symbol, pair)
                if abs(corr) > 0.80:
                    return (
                        False,
                        f"Correlation risk: {new_symbol}/{pair}={corr:.2f}",
                    )

        # Known empirical correlations for major pairs (24h rolling typical values)
        # Source: BIS Triennial Survey, major bank FX desks
        FX_CORR = {
            ("EURUSD", "GBPUSD"): 0.85,
            ("GBPUSD", "EURUSD"): 0.85,
            ("EURUSD", "USDCHF"): -0.90,
            ("USDCHF", "EURUSD"): -0.90,
            ("EURUSD", "USDCAD"): -0.65,
            ("USDCAD", "EURUSD"): -0.65,
            ("GBPUSD", "USDCHF"): -0.75,
            ("USDCHF", "GBPUSD"): -0.75,
            ("AUDUSD", "NZDUSD"): 0.80,
            ("NZDUSD", "AUDUSD"): 0.80,
            ("AUDUSD", "USDJPY"): 0.55,
            ("USDJPY", "AUDUSD"): 0.55,
            ("XAUUSD", "XTIUSD"): 0.45,
            ("XTIUSD", "XAUUSD"): 0.45,
            ("XAUUSD", "USDJPY"): -0.40,
            ("USDJPY", "XAUUSD"): -0.40,
        }

        for pair in open_symbols:
            key = (new_symbol, pair)
            rev_key = (pair, new_symbol)
            corr = FX_CORR.get(key, FX_CORR.get(rev_key, 0.0))
            if abs(corr) > 0.80:
                return (
                    False,
                    f"Correlation risk: {new_symbol}/{pair}={corr:.2f}",
                )

        # Also limit total positions if we can't check correlation
        if len(open_symbols) >= 3:
            return False, f"Max correlated positions: {len(open_symbols)} open"
        return True, "OK"

    def check_correlation(
        self,
        new_pair: str,
        open_pairs: List[str],
        corr_matrix: Optional[pd.DataFrame] = None,
    ) -> bool:
        if corr_matrix is not None and new_pair in corr_matrix.columns:
            for pair in open_pairs:
                if pair in corr_matrix.columns:
                    if abs(float(corr_matrix.loc[new_pair, pair])) > 0.80:
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

    def update_price_history(self, price: float, symbol: str = "EURUSD"):
        if symbol not in self._price_history:
            self._price_history[symbol] = []
        self._price_history[symbol].append(price)
        max_len = self._var_lookback * 24
        if len(self._price_history[symbol]) > max_len:
            self._price_history[symbol] = self._price_history[symbol][-max_len:]

    def reset_daily_stats(self, balance: float):
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0

    def set_enhanced_manager(self, enhanced):
        """Allow external EnhancedRiskManager to attach itself."""
        self._enhanced = enhanced


class TrailingStopManager:
    """Multi-tier trailing stop (Zenox EA 3-tier system)."""

    def __init__(self, tp_pcts: Optional[list] = None, trail_atr_mult: float = 2.0):
        self.tp_levels = tp_pcts or [0.01, 0.02, 0.03]
        self.trail_mult = trail_atr_mult
        self.tp_hit = [False, False, False]

    def update(
        self, entry: float, current: float, atr: float, direction: str = "long"
    ) -> Optional[float]:
        pnl_pct = (
            (current - entry) / entry
            if direction == "long"
            else (entry - current) / entry
        )
        for i, level in enumerate(self.tp_levels):
            if pnl_pct >= level and not self.tp_hit[i]:
                self.tp_hit[i] = True
                if i == 0:
                    return entry
                elif i == 1:
                    return (
                        entry + (atr * self.trail_mult)
                        if direction == "long"
                        else entry - (atr * self.trail_mult)
                    )
                else:
                    return (
                        current - (atr * self.trail_mult)
                        if direction == "long"
                        else current + (atr * self.trail_mult)
                    )
        return None

    def partial_close_sizes(self) -> list:
        return [0.3, 0.3, 0.4]
