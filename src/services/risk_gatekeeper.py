"""Risk Gatekeeper: evaluates Signal → TradeDecision or rejects."""

from typing import Optional
from loguru import logger

from infrastructure.service_base import TradingService
from infrastructure.config_v2 import AppConfig
from services import Signal, TradeDecision, SignalDirection
from risk.manager import RiskManager, RiskParameters
from execution.cost_model import CostModel


class RiskGatekeeper(TradingService):
    """Gatekeeper: receives Signal, returns TradeDecision or None."""

    def __init__(self, config: AppConfig, initial_balance: float = 100_000.0):
        super().__init__("risk_gatekeeper")
        self.config = config
        params = RiskParameters(
            max_risk_per_trade=config.trading.max_risk_per_trade,
            max_drawdown=config.trading.max_drawdown,
            max_margin_usage=config.trading.max_margin_usage,
        )
        self.risk_manager = RiskManager(params, initial_balance)
        self.cost_model = CostModel(
            commission_per_lot=config.trading.commission_per_lot
        )

    async def start(self) -> None:
        self._running = True
        logger.info("RiskGatekeeper: Kelly sizing, VaR/CVaR, drawdown limits")

    async def stop(self) -> None:
        self._running = False

    def evaluate(
        self,
        signal: Signal,
        balance: float,
        equity: float,
        margin: float,
        atr: float,
        open_positions_count: int,
    ) -> Optional[TradeDecision]:
        """Gate: returns TradeDecision or None (rejected)."""
        if atr <= 0:
            atr = signal.price * 0.001

        # Max positions check
        if open_positions_count >= self.config.trading.max_positions:
            logger.info(
                f"Risk reject {signal.symbol}: max positions reached ({open_positions_count})"
            )
            return None

        # Pre-trade risk checks (drawdown, daily loss, consecutive losses, margin)
        approved, reason = self.risk_manager.pre_trade_checks(
            balance, equity, margin, equity - balance
        )
        if not approved:
            logger.info(f"Risk reject {signal.symbol}: {reason}")
            return None

        volume = self.risk_manager.calculate_kelly_size(
            balance,
            signal.price,
            atr,
            signal.confidence,
            symbol=signal.symbol,
        )
        lot_min = 0.01
        volume = max(round(volume / lot_min) * lot_min, lot_min)
        volume = min(volume, balance * 0.5 / signal.price)

        cost = self.cost_model.calculate(
            symbol=signal.symbol,
            direction=signal.direction.value,
            volume=volume,
            price=signal.price,
            atr=atr,
        )
        if not cost.is_acceptable:
            logger.info(f"Cost reject {signal.symbol}: {cost.rejection_reason}")
            return None

        if signal.direction == SignalDirection.BUY:
            sl_price = signal.price - atr * self.config.trading.sl_atr_multiplier
            tp_price = signal.price + atr * self.config.trading.tp_atr_multiplier
        else:
            sl_price = signal.price + atr * self.config.trading.sl_atr_multiplier
            tp_price = signal.price - atr * self.config.trading.tp_atr_multiplier

        return TradeDecision(
            signal=signal,
            volume=volume,
            sl_price=sl_price,
            tp_price=tp_price,
        )
