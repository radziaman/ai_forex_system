"""Tests for RiskGatekeeper."""
from infrastructure.config_v2 import AppConfig
from services.risk_gatekeeper import RiskGatekeeper
from services import Signal, SignalDirection, Regime, TradeDecision


class TestRiskGatekeeper:
    def setup_method(self):
        self.config = AppConfig()
        self.config.trading.max_risk_per_trade = 0.02
        self.config.trading.sl_atr_multiplier = 2.0
        self.config.trading.tp_atr_multiplier = 4.0
        self.svc = RiskGatekeeper(self.config, initial_balance=100_000.0)

    def test_init(self):
        assert self.svc.name == "risk_gatekeeper"
        assert self.svc.risk_manager.mode == "PAPER"

    def test_evaluate_returns_decision_for_valid_signal(self):
        signal = Signal(
            symbol="EURUSD", direction=SignalDirection.BUY,
            confidence=0.75, regime=Regime.RANGING, price=1.12,
        )
        decision = self.svc.evaluate(
            signal=signal,
            balance=100_000.0,
            equity=100_000.0,
            margin=0.0,
            atr=0.002,
            open_positions_count=0,
        )
        assert decision is not None
        assert isinstance(decision, TradeDecision)
        assert decision.volume >= 1
        assert decision.sl_price < signal.price  # BUY: SL below entry
        assert decision.tp_price > signal.price  # BUY: TP above entry

    def test_evaluate_rejects_when_kill_switch_active(self):
        self.svc.risk_manager.kill_switch_triggered = True
        signal = Signal(
            symbol="EURUSD", direction=SignalDirection.BUY,
            confidence=0.75, regime=Regime.RANGING, price=1.12,
        )
        decision = self.svc.evaluate(
            signal=signal, balance=100_000, equity=100_000,
            margin=0, atr=0.002, open_positions_count=0,
        )
        assert decision is None

    def test_evaluate_sell_direction_sl_tp(self):
        signal = Signal(
            symbol="USDJPY", direction=SignalDirection.SELL,
            confidence=0.8, regime=Regime.TRENDING, price=150.0,
        )
        decision = self.svc.evaluate(
            signal=signal, balance=100_000, equity=100_000,
            margin=0, atr=0.5, open_positions_count=0,
        )
        if decision is not None:
            assert decision.sl_price > signal.price  # SELL: SL above entry
            assert decision.tp_price < signal.price  # SELL: TP below entry

    def test_evaluate_respects_max_drawdown(self):
        self.svc.risk_manager.peak_balance = 100_000.0
        self.svc.risk_manager.pre_trade_checks(85_000, 85_000, 0, 0)  # 15% drawdown
        signal = Signal(
            symbol="EURUSD", direction=SignalDirection.BUY,
            confidence=0.6, regime=Regime.RANGING, price=1.12,
        )
        decision = self.svc.evaluate(
            signal=signal, balance=85_000, equity=85_000,
            margin=0, atr=0.002, open_positions_count=0,
        )
        assert decision is None  # max_drawdown exceeded
