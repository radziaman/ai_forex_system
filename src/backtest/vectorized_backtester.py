"""
Vectorized Backtester for rapid strategy evaluation.
Processes bars at array speed (no 1-second loop) with realistic costs.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class BacktestResult:
    total_return_pct: float
    annual_return_pct: float
    sharpe: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_hold_bars: float
    avg_win_pct: float
    avg_loss_pct: float
    calmar_ratio: float
    sortino: float
    trade_pnls: np.ndarray = field(default_factory=lambda: np.array([]))
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    regime_returns: Dict[str, float] = field(default_factory=dict)
    monthly_returns: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_dict(self) -> Dict:
        return {
            "total_return_pct": round(self.total_return_pct, 2),
            "annual_return_pct": round(self.annual_return_pct, 2),
            "sharpe": round(self.sharpe, 3),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "win_rate": round(self.win_rate, 3),
            "profit_factor": round(self.profit_factor, 3),
            "total_trades": self.total_trades,
            "avg_hold_bars": round(self.avg_hold_bars, 1),
            "avg_win_pct": round(self.avg_win_pct, 3),
            "avg_loss_pct": round(self.avg_loss_pct, 3),
            "calmar_ratio": round(self.calmar_ratio, 3),
            "sortino": round(self.sortino, 3),
        }


class VectorizedBacktester:
    def __init__(
        self,
        spread_pips: float = 0.5,
        commission_per_lot: float = 7.0,
        slippage_model: str = "moderate",
        pip_size: float = 0.0001,
        lot_size: float = 100_000,
    ):
        self.spread_pips = spread_pips
        self.commission_per_lot = commission_per_lot
        self.slippage_model = slippage_model
        self.pip_size = pip_size
        self.lot_size = lot_size

    def run(
        self,
        prices: np.ndarray,
        signal_fn: Callable,
        features: Optional[np.ndarray] = None,
        atr: Optional[np.ndarray] = None,
        regimes: Optional[np.ndarray] = None,
        sl_atr: float = 2.0,
        tp_atr: float = 4.0,
        cost_multiplier: float = 1.0,
    ) -> BacktestResult:
        signals = signal_fn(prices, features)
        n = len(prices)

        position = 0
        trades_pnls = []
        trade_entry_idx = []
        trade_exit_idx = []
        trade_regimes = []
        equity = [1.0]
        entry_price = 0.0
        entry_idx = 0

        for i in range(1, n):
            sig = signals[i] if i < len(signals) else 0

            if position == 0 and sig != 0:
                position = sig
                entry_price = prices[i] + self._slippage(prices[i], sig)
                entry_idx = i
                continue

            if position != 0:
                sl_hit = tp_hit = False
                sl_price = entry_price - position * atr[i] * sl_atr * self.pip_size if atr is not None else None
                tp_price = entry_price + position * atr[i] * tp_atr * self.pip_size if atr is not None else None

                if sl_price is not None:
                    if position == 1 and prices[i] <= sl_price:
                        sl_hit = True
                    elif position == -1 and prices[i] >= sl_price:
                        sl_hit = True

                if tp_price is not None:
                    if position == 1 and prices[i] >= tp_price:
                        tp_hit = True
                    elif position == -1 and prices[i] <= tp_price:
                        tp_hit = True

                close_signal = sl_hit or tp_hit or (sig == -position)
                exit_sig = sig != 0 and sig != position
                should_close = close_signal or exit_sig

                if should_close or i == n - 1:
                    exit_price = prices[i] - self._slippage(prices[i], position)
                    raw_pnl = position * (exit_price - entry_price) / self.pip_size
                    cost = self._transaction_cost(abs(entry_price - exit_price), entry_price)
                    net_pnl = raw_pnl - cost * cost_multiplier
                    trades_pnls.append(net_pnl)
                    trade_entry_idx.append(entry_idx)
                    trade_exit_idx.append(i)
                    if regimes is not None:
                        trade_regimes.append(regimes[entry_idx])
                    position = 0

            equity.append(equity[-1] * (1 + (trades_pnls[-1] / 100) if trades_pnls else 0))

        trade_pnls = np.array(trades_pnls)
        equity_curve = np.array(equity)

        if len(trade_pnls) == 0:
            return BacktestResult(
                total_return_pct=0.0, annual_return_pct=0.0, sharpe=0.0,
                max_drawdown_pct=0.0, win_rate=0.0, profit_factor=0.0,
                total_trades=0, avg_hold_bars=0.0, avg_win_pct=0.0,
                avg_loss_pct=0.0, calmar_ratio=0.0, sortino=0.0,
                trade_pnls=trade_pnls, equity_curve=equity_curve,
            )

        total_return = float(equity_curve[-1] - 1.0) * 100
        years = n / 252
        annual_return = (1 + total_return / 100) ** (1 / max(years, 0.01)) - 1
        annual_return_pct = annual_return * 100

        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe = float(np.mean(returns) / max(np.std(returns), 1e-10) * np.sqrt(252)) if len(returns) > 1 else 0.0

        cum = np.maximum.accumulate(equity_curve)
        dd = (cum - equity_curve) / cum
        max_dd = float(np.max(dd)) * 100

        wins = trade_pnls[trade_pnls > 0]
        losses = trade_pnls[trade_pnls < 0]
        win_rate = len(wins) / max(len(trade_pnls), 1)
        profit_factor = float(np.sum(wins) / max(abs(np.sum(losses)), 1e-10))

        avg_hold = float(np.mean(np.array(trade_exit_idx) - np.array(trade_entry_idx))) if trade_exit_idx else 0.0
        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0

        calmar = annual_return_pct / max(max_dd, 0.01)
        downside = returns[returns < 0]
        sortino = float(np.mean(returns) / max(np.std(downside), 1e-10) * np.sqrt(252)) if len(downside) > 1 else 0.0

        regime_returns = {}
        if regimes is not None and trade_regimes:
            unique_regimes = set(trade_regimes)
            for r in unique_regimes:
                mask = np.array([tr == r for tr in trade_regimes])
                regime_returns[r] = float(np.sum(trade_pnls[mask]))

        monthly = self._compute_monthly_returns(equity_curve, n)

        return BacktestResult(
            total_return_pct=total_return,
            annual_return_pct=annual_return_pct,
            sharpe=sharpe,
            max_drawdown_pct=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trade_pnls),
            avg_hold_bars=avg_hold,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            calmar_ratio=calmar,
            sortino=sortino,
            trade_pnls=trade_pnls,
            equity_curve=equity_curve,
            regime_returns=regime_returns,
            monthly_returns=monthly,
        )

    def _slippage(self, price: float, direction: int) -> float:
        if self.slippage_model == "none":
            return 0.0
        elif self.slippage_model == "conservative":
            return self.spread_pips * self.pip_size * 0.5
        elif self.slippage_model == "moderate":
            return self.spread_pips * self.pip_size * 1.0
        elif self.slippage_model == "aggressive":
            return self.spread_pips * self.pip_size * 2.0
        return self.spread_pips * self.pip_size * 0.5

    def _transaction_cost(self, pips_moved: float, price: float) -> float:
        spread_cost = self.spread_pips
        lots = 1.0
        commission = lots * self.commission_per_lot / price
        return spread_cost + commission

    @staticmethod
    def _compute_monthly_returns(equity_curve: np.ndarray, n_bars: int) -> np.ndarray:
        monthly = []
        days_per_bar = 252 / max(n_bars, 1)
        bars_per_month = int(21 / max(days_per_bar, 1))
        for i in range(bars_per_month, len(equity_curve), bars_per_month):
            ret = equity_curve[i] / equity_curve[i - bars_per_month] - 1
            monthly.append(ret)
        return np.array(monthly) if monthly else np.array([])

    @staticmethod
    def monte_carlo_equity_curves(
        trade_pnls: np.ndarray,
        n_simulations: int = 1000,
        n_trades: Optional[int] = None,
    ) -> np.ndarray:
        if len(trade_pnls) == 0:
            return np.zeros((n_simulations, 100))
        n = n_trades or len(trade_pnls)
        curves = np.zeros((n_simulations, n + 1))
        curves[:, 0] = 1.0
        for i in range(n_simulations):
            sampled = np.random.choice(trade_pnls, size=n, replace=True)
            curves[i, 1:] = np.cumprod(1 + sampled / 100)
        return curves

    @staticmethod
    def compute_confidence_intervals(
        curves: np.ndarray, confidence: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lower = (1 - confidence) / 2 * 100
        upper = (1 + confidence) / 2 * 100
        median = np.percentile(curves, 50, axis=0)
        lb = np.percentile(curves, lower, axis=0)
        ub = np.percentile(curves, upper, axis=0)
        return median, lb, ub

    def run_with_sensitivity(
        self,
        prices: np.ndarray,
        signal_fn: Callable,
        features: Optional[np.ndarray] = None,
        atr: Optional[np.ndarray] = None,
        regimes: Optional[np.ndarray] = None,
    ) -> Dict:
        results = {}
        for slip in ["conservative", "moderate", "aggressive"]:
            self.slippage_model = slip
            for mult in [0.5, 1.0, 2.0]:
                r = self.run(
                    prices, signal_fn, features, atr, regimes,
                    cost_multiplier=mult,
                )
                key = f"{slip}_cost{mult}x"
                results[key] = r
        return results
