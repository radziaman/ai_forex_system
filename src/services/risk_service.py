"""
RiskService — centralized risk management for the trading system.
"""
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from loguru import logger

from risk.manager import RiskManager, RiskParameters, TrailingStopManager
from risk.circuit_breaker import CircuitBreaker, MarketStressSnapshot
from execution.cost_model import CostModel
from data.data_manager import SYMBOLS


class RiskService:
    """Centralized risk management — position sizing, circuit breakers, cost modeling."""

    def __init__(self, config, mode: str, initial_balance: float):
        self.config = config
        self.mode = mode

        # Risk manager
        params = RiskParameters(
            max_risk_per_trade=config.trading.max_risk_per_trade,
            max_drawdown=config.trading.max_drawdown,
            max_margin_usage=config.trading.max_margin_usage,
        )
        self.manager = RiskManager(params, initial_balance=initial_balance)
        self.manager.mode = "PAPER" if mode == "paper" else "LIVE"

        # Trailing stop
        self.trailing_stop = TrailingStopManager(
            tp_pcts=[0.01, 0.02, 0.03],
            trail_atr_mult=config.trading.sl_atr_multiplier,
        )

        # Cost model
        self.cost_model = CostModel(commission_per_lot=config.trading.commission_per_lot)

        # Circuit breakers per symbol
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        for sym in SYMBOLS:
            self.circuit_breakers[sym] = CircuitBreaker(
                price_velocity_threshold=0.005,
                spread_multiplier_threshold=5.0,
                volume_spike_multiplier=10.0,
            )

        self._last_daily_reset_day = -1

    def check_circuit_breakers(self, symbol: str, tick) -> Tuple[bool, str, Optional[MarketStressSnapshot]]:
        cb = self.circuit_breakers.get(symbol)
        if not cb:
            return False, "ok", None
        tick_dict = {
            "bid": getattr(tick, 'bid', 0),
            "ask": getattr(tick, 'ask', 0),
            "price": getattr(tick, 'bid', 0),
            "volume": getattr(tick, 'volume', 0),
        }
        return cb.check_market_health(symbol, tick_dict)

    def check_daily_reset(self):
        """Reset daily stats when day changes."""
        today = time.localtime().tm_yday
        if today != self._last_daily_reset_day:
            self.manager.reset_daily_stats(self.manager.initial_balance)
            self._last_daily_reset_day = today
            return True
        return False

    def pre_trade_check(self, account_info: Dict) -> Tuple[bool, str]:
        balance = account_info.get("balance", 100_000)
        equity = account_info.get("equity", 100_000)
        margin = account_info.get("margin", 0)
        daily_pnl = self.manager.daily_pnl
        return self.manager.pre_trade_checks(balance, equity, margin, daily_pnl)

    def calculate_kelly_size(self, balance: float, price: float, atr: float, confidence: float,
                             symbol: str = None) -> float:
        return self.manager.calculate_kelly_size(balance, price, atr, confidence, symbol)

    def calculate_sl_tp(self, direction: str, entry: float, atr: float, regime_params: Dict) -> Tuple[float, float]:
        sl_atr = regime_params.get("sl_atr", 1.5)
        tp_atr = regime_params.get("tp_atr", 3.0)
        if direction == "BUY":
            sl = entry - atr * sl_atr
            tp = entry + atr * tp_atr
        else:
            sl = entry + atr * sl_atr
            tp = entry - atr * tp_atr
        return sl, tp

    def check_cost(self, symbol: str, direction: str, volume: float, price: float,
                   atr: float, spread_pips: float) -> bool:
        cost = self.cost_model.calculate(
            symbol=symbol, direction=direction, volume=volume,
            price=price, atr=atr, actual_spread_pips=spread_pips,
        )
        return cost.is_acceptable

    def get_spread_warning(self, symbol: str, spread_pips: float) -> str:
        return self.cost_model.get_spread_warning_level(symbol, spread_pips)

    def update_price_history(self, price: float):
        self.manager.update_price_history(price)

    def get_status(self) -> Dict:
        return {
            "daily_pnl": self.manager.daily_pnl,
            "daily_trades": self.manager.daily_trades,
            "consecutive_losses": self.manager.consecutive_losses,
            "total_trades": self.manager.total_trades,
            "wins": self.manager.wins,
            "losses": self.manager.losses,
            "kill_switch": self.manager.kill_switch_triggered,
            "var": self.manager.var(),
            "cvar": self.manager.cvar(),
        }
