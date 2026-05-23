"""
Enhanced Risk Manager — MAE/MFE tracking, stress testing, CVaR-Kelly sizing,
rolling correlation, drawdown circuit breaker.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger

from risk.manager import RiskManager, RiskParameters, TradeRecord


@dataclass
class MAEMFETracker:
    mae: float = 0.0
    mfe: float = 0.0
    mae_ratio: float = 0.0
    mfe_ratio: float = 0.0


@dataclass
class StressScenario:
    name: str = ""
    shocks: List[float] = field(default_factory=list)


class EnhancedRiskManager:
    """Wraps RiskManager with advanced risk analytics."""

    def __init__(
        self,
        params: RiskParameters,
        initial_balance: float = 100_000.0,
        base_manager: Optional[RiskManager] = None,
    ):
        self._base = base_manager or RiskManager(params, initial_balance)
        self._mae_mfe: Dict[int, Dict[str, Any]] = {}
        self._returns: Dict[str, List[float]] = {}
        self._price_history_for_returns: Dict[str, float] = {}

    def __getattr__(self, name: str):
        return getattr(self._base, name)

    def update_price_history(self, price: float, symbol: str = "EURUSD"):
        """Update base price history and track returns for correlation."""
        self._base.update_price_history(price, symbol)
        prev = self._price_history_for_returns.get(symbol)
        if prev is not None and prev > 0:
            ret = (price - prev) / prev
            if symbol not in self._returns:
                self._returns[symbol] = []
            self._returns[symbol].append(ret)
            max_len = 1440  # 24h of 1-min bars approx
            if len(self._returns[symbol]) > max_len:
                self._returns[symbol] = self._returns[symbol][-max_len:]
        self._price_history_for_returns[symbol] = price

    def _rolling_correlation(self) -> pd.DataFrame:
        """Compute 24h exponential decay correlation matrix."""
        if len(self._returns) < 2:
            return pd.DataFrame()
        min_len = min(len(v) for v in self._returns.values())
        if min_len < 5:
            return pd.DataFrame()
        data = {sym: vals[-min_len:] for sym, vals in self._returns.items()}
        df = pd.DataFrame(data)
        symbols = list(df.columns)
        n = len(symbols)
        if n < 2:
            return pd.DataFrame()
        weights = np.exp(np.linspace(-1.0, 0.0, min_len))
        weights /= weights.sum()

        def weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
            mean_x = np.average(x, weights=w)
            mean_y = np.average(y, weights=w)
            cov = np.average((x - mean_x) * (y - mean_y), weights=w)
            std_x = np.sqrt(np.average((x - mean_x) ** 2, weights=w))
            std_y = np.sqrt(np.average((y - mean_y) ** 2, weights=w))
            if std_x == 0 or std_y == 0:
                return 0.0
            return float(cov / (std_x * std_y))

        corr = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                c = weighted_corr(df.iloc[:, i].values, df.iloc[:, j].values, weights)
                corr[i, j] = c
                corr[j, i] = c
        return pd.DataFrame(corr, index=symbols, columns=symbols)

    def calculate_cvar_kelly_size(
        self,
        balance: float,
        price: float,
        atr: float,
        confidence: float,
        symbol: Optional[str] = None,
        open_positions: Optional[list] = None,
    ) -> float:
        """CVaR-adjusted Kelly position sizing."""
        size = self._base.calculate_kelly_size(
            balance, price, atr, confidence, symbol, open_positions
        )
        cvar_val = self._base.cvar(confidence, symbol)
        cvar_pct = abs(cvar_val) / balance if balance > 0 else 0.0
        # Reduce size if CVaR > 3% of balance
        if cvar_pct > 0.03:
            adjustment = max(0.1, 1.0 - (cvar_pct - 0.03) / 0.07)
            size *= adjustment
        return float(np.clip(size, 0, balance / max(price, 1e-9)))

    def record_trade_open(self, trade: TradeRecord):
        """Start tracking MAE/MFE for a new trade."""
        self._mae_mfe[trade.position_id] = {
            "entry_price": trade.entry_price,
            "direction": trade.direction,
            "high_price": trade.entry_price,
            "low_price": trade.entry_price,
            "tracker": MAEMFETracker(),
        }

    def update_trade_mae_mfe(self, position_id: int, current_price: float):
        """Update high/low tracking for an open trade."""
        info = self._mae_mfe.get(position_id)
        if info is None:
            return
        info["high_price"] = max(info["high_price"], current_price)
        info["low_price"] = min(info["low_price"], current_price)
        entry = info["entry_price"]
        direction = info["direction"]
        if entry <= 0:
            return
        if direction == "BUY":
            mae = entry - info["low_price"]
            mfe = info["high_price"] - entry
        else:
            mae = info["high_price"] - entry
            mfe = entry - info["low_price"]
        risk_distance = abs(current_price - entry)
        tracker = info["tracker"]
        tracker.mae = mae
        tracker.mfe = mfe
        tracker.mae_ratio = mae / risk_distance if risk_distance > 0 else 0.0
        tracker.mfe_ratio = mfe / risk_distance if risk_distance > 0 else 0.0

    def record_trade_close(self, trade: TradeRecord, exit_price: float, pnl: float):
        """Finalize MAE/MFE and warn if mae_ratio > 3.0."""
        info = self._mae_mfe.pop(trade.position_id, None)
        if info is not None:
            tracker = info["tracker"]
            entry = info["entry_price"]
            direction = info["direction"]
            if entry > 0:
                if direction == "BUY":
                    mae = entry - info["low_price"]
                    mfe = info["high_price"] - entry
                else:
                    mae = info["high_price"] - entry
                    mfe = entry - info["low_price"]
                sl_distance = abs(trade.sl - entry) if trade.sl else entry * 0.01
                tracker.mae = mae
                tracker.mfe = mfe
                tracker.mae_ratio = mae / sl_distance if sl_distance > 0 else 0.0
                tracker.mfe_ratio = mfe / sl_distance if sl_distance > 0 else 0.0
                if tracker.mae_ratio > 3.0:
                    logger.warning(
                        f"Trade {trade.position_id} adverse excursion ratio "
                        f"{tracker.mae_ratio:.2f}x risk — review stop placement"
                    )
        self._base.record_trade(trade, exit_price, pnl)

    def pre_trade_checks(
        self,
        balance: float,
        equity: float,
        margin: float,
        current_pnl: float,
        new_symbol: str = "",
        open_symbols: Optional[List[str]] = None,
    ) -> Tuple[bool, str]:
        """Run pre-trade checks with enhanced drawdown circuit breaker."""
        approved, reason = self._base.pre_trade_checks(
            balance,
            equity,
            margin,
            current_pnl,
            new_symbol,
            open_symbols,
        )
        if not approved and "drawdown" in reason.lower():
            self._base.kill_switch_triggered = True
        return approved, reason

    def stress_test_flash_crash(
        self, symbol: str, current_price: float
    ) -> Dict[str, Any]:
        """Simulate flash crash scenario."""
        scenario = StressScenario(name="flash_crash", shocks=[-0.10, -0.08, -0.05])
        notional = current_price * 100_000  # 1 std lot approx
        max_loss = 0.0
        for shock in scenario.shocks:
            loss = notional * abs(shock)
            if loss > max_loss:
                max_loss = loss
        return {
            "scenario": scenario.name,
            "symbol": symbol,
            "max_loss": max_loss,
            "max_loss_pct": max_loss / self._base.initial_balance,
            "shocks": scenario.shocks,
        }

    def get_mae_mfe_summary(self) -> Dict[int, Dict[str, float]]:
        """Return MAE/MFE summary for all open trades."""
        summary = {}
        for pos_id, info in self._mae_mfe.items():
            tracker = info["tracker"]
            summary[pos_id] = {
                "mae": tracker.mae,
                "mfe": tracker.mfe,
                "mae_ratio": tracker.mae_ratio,
                "mfe_ratio": tracker.mfe_ratio,
            }
        return summary
