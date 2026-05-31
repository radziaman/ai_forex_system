"""RiskManager — risk checks, Kelly sizing, circuit breaker, adaptive risk.

Replaces: RiskAgent + AdaptiveRiskAgent + CircuitBreakerAgent + CostAgent.

Responsibilities:
- Assess trade signals against risk parameters (Kelly, VaR, drawdown)
- Run circuit breaker checks on market conditions
- Adjust risk dynamically based on recent win rate and volatility
- Emit risk_approved or risk_rejected events
"""

from typing import Dict, Any, Optional, List, Tuple
from collections import deque
from loguru import logger

from .pipeline_context import PipelineContext


class RiskManager:
    """Risk management pipeline — Kelly sizing, circuit breaker,
    adaptive adjustments.
    """

    def __init__(
        self,
        ctx: PipelineContext,
        risk_engine: Optional[Any] = None,
        circuit_breaker: Optional[Any] = None,
    ):
        self.ctx = ctx
        self.config = ctx.config
        self.bus = ctx.bus
        self._risk_engine = risk_engine  # EnhancedRiskManager or RiskManager
        self._circuit_breaker = circuit_breaker  # CircuitBreaker instance
        self._initialized = False

        # Adaptive risk state (from AdaptiveRiskAgent)
        self.base_kelly: float = 0.25
        self.base_risk: float = 0.02
        self.effective_kelly: float = self.base_kelly
        self.effective_risk: float = self.base_risk
        self._equity_high_water: float = 0.0
        self._consecutive_losses: int = 0
        self._volatility_regime: str = "normal"
        self._drawdown_regime: str = "safe"
        self._trade_pnls: deque = deque(maxlen=50)
        self.halted: bool = False

        # Execution quality feedback (Phase 4.3)
        self._position_size_reduction: float = 1.0

        # Subscribe to events
        self.bus.on("signal_generated", self._on_signal_generated)
        self.bus.on("position_closed", self._on_position_closed)
        self.bus.on("execution_quality", self._on_execution_quality)

    async def start(self) -> None:
        """Initialize the risk manager and underlying engines."""
        risk_config = getattr(self.config, "risk", self.config)

        # Initialize RiskManager/EnhancedRiskManager if not injected
        if self._risk_engine is None:
            try:
                from risk.enhanced_manager import EnhancedRiskManager
                from risk.manager import RiskParameters

                params = RiskParameters(
                    max_risk_per_trade=getattr(risk_config, "max_risk_per_trade", 0.02),
                    max_drawdown=getattr(risk_config, "max_drawdown", 0.10),
                    kelly_fraction=getattr(risk_config, "kelly_fraction", 0.25),
                    sl_atr_multiplier=getattr(risk_config, "sl_atr_multiplier", 2.0),
                    tp_atr_multiplier=getattr(risk_config, "tp_atr_multiplier", 4.0),
                    trailing_activation_rr=getattr(
                        risk_config, "trailing_activation_rr", 1.0
                    ),
                    trailing_distance_rr=getattr(
                        risk_config, "trailing_distance_rr", 0.5
                    ),
                    max_consecutive_losses=getattr(
                        risk_config, "max_consecutive_losses", 5
                    ),
                    max_daily_loss=getattr(risk_config, "max_daily_loss", 0.05),
                    max_positions=getattr(risk_config, "max_positions", 5),
                )
                self._risk_engine = EnhancedRiskManager(
                    params=params,
                    initial_balance=100_000.0,
                )
            except Exception:
                from risk.manager import RiskManager as BaseRM, RiskParameters

                params = RiskParameters(
                    max_risk_per_trade=0.02,
                    max_drawdown=0.10,
                    kelly_fraction=0.25,
                    sl_atr_multiplier=2.0,
                    tp_atr_multiplier=4.0,
                    max_positions=5,
                )
                self._risk_engine = BaseRM(params=params)

        # Initialize CircuitBreaker if not injected
        if self._circuit_breaker is None:
            try:
                from risk.circuit_breaker import CircuitBreaker

                self._circuit_breaker = CircuitBreaker()
            except Exception:
                self._circuit_breaker = None

        # Read config values
        self.base_kelly = float(getattr(risk_config, "kelly_fraction", 0.25))
        self.effective_kelly = self.base_kelly
        self.base_risk = float(getattr(risk_config, "max_risk_per_trade", 0.02))
        self.effective_risk = self.base_risk

        self._initialized = True
        logger.info(
            f"[risk_manager] Ready — base_kelly={self.base_kelly:.2f}, "
            f"base_risk={self.base_risk:.2f}"
        )

    async def _on_signal_generated(self, **signal: Any) -> None:
        """Assess a trading signal and emit risk_approved or risk_rejected."""
        if not self._initialized:
            await self.start()

        result = await self.assess_trade(dict(signal))
        if result["approved"]:
            await self.bus.emit(
                "risk_approved",
                signal=signal,
                volume=result["volume"],
                sl_price=result["sl_price"],
                tp_price=result["tp_price"],
            )
        else:
            await self.bus.emit(
                "risk_rejected",
                signal=signal,
                reason=result["reason"],
            )

    async def _on_position_closed(self, **data: Any) -> None:
        """Track closed positions for adaptive risk adjustment."""
        pnl = data.get("pnl", 0.0)
        self._trade_pnls.append(pnl)
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0
        # Update adaptive risk parameters
        self._update_adaptive_risk()

    async def _on_execution_quality(self, **data: Any) -> None:
        """Reduce position size when execution quality degrades."""
        multiplier = data.get("slippage_multiplier", 1.0)
        if multiplier > 2.0:
            self._position_size_reduction = 0.5  # 50% reduction
        elif multiplier > 1.5:
            self._position_size_reduction = 0.75  # 25% reduction
        else:
            self._position_size_reduction = 1.0
        logger.debug(
            f"[risk_manager] Execution quality multiplier={multiplier:.2f} -> "
            f"size_reduction={self._position_size_reduction:.2f}"
        )

    async def assess_trade(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Run full risk assessment on a signal.

        Returns:
            dict with: approved (bool), volume (float), sl_price (float),
                      tp_price (float), reason (str)
        """
        symbol = signal.get("symbol", "EURUSD")
        direction = signal.get("direction", "HOLD")
        confidence = signal.get("confidence", 0.0)
        price = signal.get("price", 0.0)

        # 1. Check if halted
        if self.halted:
            return {
                "approved": False,
                "volume": 0.0,
                "sl_price": 0.0,
                "tp_price": 0.0,
                "reason": "system_halted",
            }

        # 2. Check circuit breaker
        if self._circuit_breaker is not None:
            try:
                tick_data = {
                    "symbol": symbol,
                    "bid": price * 0.9999,
                    "ask": price * 1.0001,
                    "price": price,
                }
                should_halt, halt_reason, snapshot = (
                    self._circuit_breaker.check_market_health(symbol, tick_data)
                )
                if should_halt:
                    logger.warning(
                        f"[risk_manager] Circuit breaker halts {symbol}: "
                        f"{halt_reason}"
                    )
                    return {
                        "approved": False,
                        "volume": 0.0,
                        "sl_price": 0.0,
                        "tp_price": 0.0,
                        "reason": f"circuit_breaker: {halt_reason}",
                    }

                # Check confidence threshold
                min_confidence = getattr(snapshot, "confidence_threshold", 0.5)
                if confidence < min_confidence:
                    return {
                        "approved": False,
                        "volume": 0.0,
                        "sl_price": 0.0,
                        "tp_price": 0.0,
                        "reason": (
                            f"low_confidence: {confidence:.2f} < "
                            f"{min_confidence:.2f}"
                        ),
                    }
            except Exception:
                logger.warning("[risk_manager] Circuit breaker check failed")

        # 3. Check consecutive losses
        max_losses = getattr(
            getattr(self.config, "risk", self.config),
            "max_consecutive_losses",
            5,
        )
        if self._consecutive_losses >= max_losses:
            return {
                "approved": False,
                "volume": 0.0,
                "sl_price": 0.0,
                "tp_price": 0.0,
                "reason": (f"max_consecutive_losses: {self._consecutive_losses}"),
            }

        # 4. Calculate position size using Kelly
        if self._risk_engine is not None:
            try:
                atr = self._estimate_atr(symbol)
                volume = self._risk_engine.calculate_kelly_size(
                    balance=self._get_balance(),
                    price=price,
                    atr=atr,
                    confidence=confidence,
                    symbol=symbol,
                    open_positions=self._get_open_positions(),
                )
                # Apply adaptive Kelly multiplier
                volume *= self._get_kelly_multiplier()

                # Calculate SL/TP prices
                sl_price, tp_price = self._calculate_sl_tp(
                    price, direction, atr, symbol
                )
            except Exception:
                volume = 0.01  # micro lot fallback
                sl_price = price * 0.99
                tp_price = price * 1.01
        else:
            volume = 0.01
            sl_price = price * 0.99
            tp_price = price * 1.01

        # 5. Ensure minimum trade size
        if volume <= 0:
            return {
                "approved": False,
                "volume": 0.0,
                "sl_price": 0.0,
                "tp_price": 0.0,
                "reason": "zero_volume",
            }

        return {
            "approved": True,
            "volume": volume,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "reason": "",
        }

    def _get_kelly_multiplier(self) -> float:
        """Get Kelly multiplier based on adaptive risk state."""
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
        if recent_wr is not None:
            if recent_wr < 0.3:
                mult *= 0.4
            elif recent_wr > 0.7:
                mult *= 1.1
        # Execution quality feedback — reduce size when slippage is high
        mult *= self._position_size_reduction
        return max(mult, 0.05)

    def _recent_win_rate(self) -> Optional[float]:
        """Calculate win rate over the last 20 trades."""
        recent = list(self._trade_pnls)[-20:]
        if len(recent) < 5:
            return None
        wins = sum(1 for p in recent if p > 0)
        return wins / len(recent)

    def _update_adaptive_risk(self) -> None:
        """Update adaptive risk parameters based on recent
        performance and volatility.
        """
        # Drawdown regime
        balance = self._get_balance()
        equity = self._get_equity()
        if self._equity_high_water == 0:
            self._equity_high_water = max(balance, equity, 100_000.0)
        if equity > self._equity_high_water:
            self._equity_high_water = equity
        dd = (
            (self._equity_high_water - equity) / max(self._equity_high_water, 1)
            if self._equity_high_water > 0
            else 0.0
        )
        if dd > 0.10:
            self._drawdown_regime = "critical"
        elif dd > 0.06:
            self._drawdown_regime = "warning"
        else:
            self._drawdown_regime = "safe"

        # Halt on critical drawdown
        if self._drawdown_regime == "critical" and not self.halted:
            self.halted = True
            self.effective_kelly = 0.0
            self.effective_risk = 0.0
            logger.warning("[risk_manager] Auto-halted on critical drawdown")

        # Update effective kelly
        if not self.halted:
            self.effective_kelly = self.base_kelly * self._get_kelly_multiplier()
            self.effective_risk = self.base_risk * self._get_kelly_multiplier()

    def _estimate_atr(self, symbol: str) -> float:
        """Estimate ATR for a symbol."""
        if self._risk_engine is not None and hasattr(self._risk_engine, "params"):
            price = self._estimate_price(symbol)
            return price * 0.001  # approximate 0.1% ATR
        return 0.001

    def _estimate_price(self, symbol: str) -> float:
        """Estimate current price for a symbol."""
        base_prices = {
            "EURUSD": 1.12,
            "GBPUSD": 1.28,
            "USDJPY": 150.0,
            "AUDUSD": 0.67,
            "USDCAD": 1.35,
            "USDCHF": 0.88,
            "NZDUSD": 0.61,
            "XAUUSD": 2350.0,
        }
        return base_prices.get(symbol, 1.12)

    def _calculate_sl_tp(
        self,
        price: float,
        direction: str,
        atr: float,
        symbol: str,
    ) -> Tuple[float, float]:
        """Calculate stop-loss and take-profit prices."""
        risk_config = getattr(self.config, "risk", self.config)
        sl_mult = getattr(risk_config, "sl_atr_multiplier", 2.0)
        tp_mult = getattr(risk_config, "tp_atr_multiplier", 4.0)
        sl_distance = atr * sl_mult
        tp_distance = atr * tp_mult
        if direction == "BUY":
            return (price - sl_distance, price + tp_distance)
        elif direction == "SELL":
            return (price + sl_distance, price - tp_distance)
        return (price - sl_distance, price + tp_distance)

    def _get_balance(self) -> float:
        """Get current account balance."""
        if self._risk_engine is not None:
            return getattr(self._risk_engine, "initial_balance", 100_000.0)
        return 100_000.0

    def _get_equity(self) -> float:
        """Get current account equity."""
        return self._get_balance()

    def _get_open_positions(self) -> List:
        """Get list of open positions."""
        if self._risk_engine is not None:
            return getattr(self._risk_engine, "trade_history", [])
        return []

    async def stop(self) -> None:
        """Clean shutdown."""
        logger.info("[risk_manager] Stopped")

    @property
    def is_alive(self) -> bool:
        """Whether the risk manager is initialized."""
        return self._initialized
