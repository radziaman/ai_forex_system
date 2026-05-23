"""Tests for strategy attribution engine."""

import pytest

from validation.attribution import TradeAttribution, StrategyAttributionEngine


class TestTradeAttribution:
    def test_dataclass_defaults(self):
        ta = TradeAttribution()
        assert ta.alpha_signal == 0.0
        assert ta.execution_quality == 0.0
        assert ta.slippage == 0.0
        assert ta.luck == 0.0
        assert ta.total_pnl == 0.0

    def test_unexplained(self):
        ta = TradeAttribution(
            alpha_signal=10.0,
            execution_quality=5.0,
            slippage=-2.0,
            luck=1.0,
            total_pnl=20.0,
        )
        assert ta.unexplained == pytest.approx(6.0)


class TestStrategyAttributionEngine:
    def test_attribute_trade_basic(self):
        engine = StrategyAttributionEngine()
        trade = {
            "pnl": 100.0,
            "expected_pnl": 80.0,
            "fill_price": 1.1000,
            "signal_price": 1.0998,
            "strategy": "momentum",
            "market_return": 0.001,
        }
        attr = engine.attribute_trade(trade)
        assert attr.total_pnl == 100.0
        assert attr.alpha_signal == 80.0
        assert attr.slippage < 0
        assert attr.execution_quality != 0.0

    def test_attribute_trade_explicit_slippage(self):
        engine = StrategyAttributionEngine()
        trade = {
            "pnl": 50.0,
            "expected_pnl": 40.0,
            "fill_price": 1.1000,
            "signal_price": 1.0998,
            "slippage_bps": -0.0001,
            "strategy": "mean_rev",
            "market_return": 0.0,
        }
        attr = engine.attribute_trade(trade)
        assert attr.slippage == pytest.approx(-0.0001)

    def test_alpha_decay_calculation(self):
        engine = StrategyAttributionEngine()
        strat = "test_strat"
        for i in range(25):
            engine.attribute_trade(
                {
                    "pnl": 0.01 if i % 2 == 0 else -0.005,
                    "expected_pnl": 0.005,
                    "fill_price": 1.1,
                    "signal_price": 1.1,
                    "strategy": strat,
                    "market_return": 0.0,
                }
            )
        decay = engine.calculate_alpha_decay(strat, window=20)
        assert "sharpe" in decay
        assert "mean_return" in decay
        assert "std_return" in decay
        assert decay["samples"] == 20

    def test_should_disable_positive(self):
        engine = StrategyAttributionEngine()
        strat = "bad_strat"
        for _ in range(25):
            engine.attribute_trade(
                {
                    "pnl": -1.0,
                    "expected_pnl": -0.5,
                    "fill_price": 1.1,
                    "signal_price": 1.1,
                    "strategy": strat,
                    "market_return": 0.0,
                }
            )
        assert engine.should_disable(strat, min_trades=20)

    def test_should_disable_negative(self):
        engine = StrategyAttributionEngine()
        strat = "good_strat"
        for i in range(25):
            engine.attribute_trade(
                {
                    "pnl": 1.0 if i % 2 == 0 else -0.5,
                    "expected_pnl": 0.3,
                    "fill_price": 1.1,
                    "signal_price": 1.1,
                    "strategy": strat,
                    "market_return": 0.0,
                }
            )
        assert not engine.should_disable(strat, min_trades=20)

    def test_should_disable_insufficient_trades(self):
        engine = StrategyAttributionEngine()
        strat = "new_strat"
        engine.attribute_trade(
            {
                "pnl": -1.0,
                "expected_pnl": -0.5,
                "fill_price": 1.1,
                "signal_price": 1.1,
                "strategy": strat,
                "market_return": 0.0,
            }
        )
        assert not engine.should_disable(strat, min_trades=20)

    def test_get_report(self):
        engine = StrategyAttributionEngine()
        for strat in ["A", "B"]:
            for _ in range(10):
                engine.attribute_trade(
                    {
                        "pnl": 0.5,
                        "expected_pnl": 0.3,
                        "fill_price": 1.1,
                        "signal_price": 1.1,
                        "strategy": strat,
                        "market_return": 0.0,
                    }
                )
        report = engine.get_report()
        assert "A" in report
        assert "B" in report
        assert report["A"]["trades"] == 10
        assert "alpha_decay" in report["A"]
