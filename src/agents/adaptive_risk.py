"""
Adaptive Risk Manager — autonomous risk parameter tuning with self-preservation.
Monitors volatility, regime, equity; adjusts sizing in real-time.
"""

import numpy as np
import time
from typing import Dict, Optional, Tuple
from collections import deque
from loguru import logger


class AdaptiveRiskManager:
    """
    Wraps RiskManager with dynamic parameter adjustment based on:
    - Market volatility (ATR regime)
    - Equity drawdown
    - Consecutive losses
    - Win rate trends
    - Regime (trending/ranging/volatile/crisis)

    Self-preservation: auto-reduces risk during stress, halts at critical levels.
    """

    def __init__(
        self,
        risk_manager,
        config,
        initial_kelly: float = 0.25,
        max_risk_per_trade: float = 0.02,
    ):
        self._rm = risk_manager
        self._config = config
        self.base_kelly = initial_kelly
        self.base_risk = max_risk_per_trade
        self.effective_kelly = initial_kelly
        self.effective_risk = max_risk_per_trade

        self._equity_high_water = risk_manager.initial_balance
        self._consecutive_losses = 0
        self._total_trades = 0
        self._regime_history: deque = deque(maxlen=20)
        self._volatility_regime: str = "normal"
        self._drawdown_regime: str = "safe"
        self._trade_pnls: deque = deque(maxlen=50)
        self._kelly_adjustments: deque = deque(maxlen=20)

        self.halted = False
        self.halt_reason = ""

    @property
    def risk_manager(self):
        return self._rm

    def update(
        self,
        balance: float,
        equity: float,
        atr: float,
        price: float,
        regime: str = "ranging",
    ) -> Dict:
        """Update agent state. Returns adjustment summary."""
        self._update_equity_high_water(equity)
        self._update_drawdown_regime(equity)
        self._update_volatility_regime(atr, price)
        self._regime_history.append(regime)
        self._adjust_params(regime)
        return self.get_status()

    def on_trade_result(self, pnl: float):
        """Feed trade outcome. Agent learns from results."""
        self._trade_pnls.append(pnl)
        self._total_trades += 1
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0
        self._adjust_kelly()

    def compute_adjusted_confidence(self, base_confidence: float) -> float:
        """
        Adjust signal confidence based on current risk state.
        Returns adjusted confidence in [0, 1].
        """
        penalty = 1.0
        if self._drawdown_regime == "warning":
            penalty *= 0.85
        elif self._drawdown_regime == "critical":
            penalty *= 0.5
            if self._consecutive_losses >= 3:
                return 0.0

        if self._volatility_regime == "high":
            penalty *= 0.8
        elif self._volatility_regime == "extreme":
            penalty *= 0.4

        if self._consecutive_losses >= 4:
            penalty *= 0.5
        if self._consecutive_losses >= 6:
            self._halt("max_consecutive_losses")

        return base_confidence * penalty

    def compute_adjusted_kelly(self, base_kelly: float) -> float:
        """Apply dynamic adjustments to raw Kelly value."""
        adjusted = base_kelly * self._kelly_multiplier()
        return float(np.clip(adjusted, 0.0, 0.5))

    def _kelly_multiplier(self) -> float:
        mult = 1.0
        if self._drawdown_regime == "warning":
            mult *= 0.6
        elif self._drawdown_regime == "critical":
            mult *= 0.2
        if self._volatility_regime == "high":
            mult *= 0.7
        elif self._volatility_regime == "extreme":
            mult *= 0.3
        recent_wr = self._recent_win_rate()
        if recent_wr is not None and recent_wr < 0.3:
            mult *= 0.4
        elif recent_wr is not None and recent_wr > 0.7:
            mult *= 1.1
        return max(mult, 0.05)

    def get_status(self) -> Dict:
        return {
            "effective_kelly": round(self.effective_kelly, 3),
            "effective_risk": round(self.effective_risk, 4),
            "drawdown_regime": self._drawdown_regime,
            "volatility_regime": self._volatility_regime,
            "consecutive_losses": self._consecutive_losses,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "equity_high_water": round(self._equity_high_water, 2),
        }

    def _update_equity_high_water(self, equity: float):
        if equity > self._equity_high_water:
            self._equity_high_water = equity

    def _update_drawdown_regime(self, equity: float):
        dd = (
            (self._equity_high_water - equity) / max(self._equity_high_water, 1)
            if self._equity_high_water > 0
            else 0.0
        )
        if dd > self._config.trading.max_drawdown:
            self._drawdown_regime = "critical"
            self._halt("drawdown_exceeded")
        elif dd > self._config.trading.max_drawdown * 0.6:
            self._drawdown_regime = "warning"
        else:
            self._drawdown_regime = "safe"

    def _update_volatility_regime(self, atr: float, price: float):
        if price <= 0 or atr <= 0:
            return
        vol = atr / price
        if vol > 0.02:
            self._volatility_regime = "extreme"
        elif vol > 0.01:
            self._volatility_regime = "high"
        else:
            self._volatility_regime = "normal"

    def _adjust_params(self, regime: str):
        self.effective_kelly = self.base_kelly * self._kelly_multiplier()
        self.effective_risk = self.base_risk * self._kelly_multiplier()
        if regime == "crisis":
            self.effective_kelly *= 0.3
            self.effective_risk *= 0.3

    def _adjust_kelly(self):
        if len(self._trade_pnls) < 10:
            return
        recent = list(self._trade_pnls)[-20:]
        wins = sum(1 for p in recent if p > 0)
        total = len(recent)
        wr = wins / total if total > 0 else 0.5
        if wr < 0.3 and self.base_kelly > 0.1:
            self.base_kelly = max(0.05, self.base_kelly * 0.85)
            self._kelly_adjustments.append(self.base_kelly)
        elif wr > 0.65 and self.base_kelly < 0.3:
            self.base_kelly = min(0.3, self.base_kelly * 1.1)
            self._kelly_adjustments.append(self.base_kelly)

    def _recent_win_rate(self) -> Optional[float]:
        recent = list(self._trade_pnls)[-20:]
        if len(recent) < 5:
            return None
        wins = sum(1 for p in recent if p > 0)
        return wins / len(recent)

    def _halt(self, reason: str):
        self.halted = True
        self.halt_reason = reason
        self._rm.kill_switch_triggered = True
        logger.error(f"[AdaptiveRisk] HALTED: {reason}")

    def try_recover(self) -> bool:
        """Attempt recovery from halted state. Returns True if recovered."""
        if not self.halted:
            return True
        recent = list(self._trade_pnls)[-5:] if len(self._trade_pnls) >= 5 else []
        if len(recent) >= 3 and all(p > 0 for p in recent[-3:]):
            self.halted = False
            self.halt_reason = ""
            self._rm.kill_switch_triggered = False
            self._consecutive_losses = 0
            logger.success("[AdaptiveRisk] Recovered from halted state")
            return True
        return False
