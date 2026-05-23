"""Tests for alpha strategies."""

from typing import Optional

import numpy as np
import pandas as pd

from ai.alpha_strategies.stat_arb import StatArbStrategy
from ai.alpha_strategies.carry_trade import CarryTradeStrategy
from ai.alpha_strategies.event_driven import EventDrivenStrategy
from ai.alpha_strategies.vol_expansion import VolExpansionStrategy
from ai.alpha_strategies.order_flow_momentum import OrderFlowMomentumStrategy


def _make_df(rows: int = 100, extra_cols: Optional[dict] = None) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame."""
    np.random.seed(42)
    base = 1.1000
    closes = base + np.cumsum(np.random.randn(rows) * 0.0005)
    df = pd.DataFrame(
        {
            "open": closes - 0.0002,
            "high": closes + 0.0004,
            "low": closes - 0.0004,
            "close": closes,
            "volume": np.random.randint(100, 1000, size=rows).astype(float),
        }
    )
    # ATR-ish proxy
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    if extra_cols:
        for k, v in extra_cols.items():
            df[k] = v
    return df


class TestStatArbStrategy:
    def test_generates_signal(self):
        df = _make_df(80)
        df["close_pair_GBPUSD"] = df["close"] * 0.88 + np.random.randn(80) * 0.0001
        strat = StatArbStrategy("EURUSD", {}, pair_symbol="GBPUSD", lookback=60)
        sig = strat.generate_signal(df)
        assert "direction" in sig
        assert "confidence" in sig
        assert "meta" in sig
        assert sig["direction"] in ("BUY", "SELL", "HOLD")

    def test_z_score_extreme(self):
        df = _make_df(80)
        # Make pair diverge strongly at the end
        df["close_pair_GBPUSD"] = df["close"].copy()
        df.loc[df.index[-5:], "close_pair_GBPUSD"] += 0.005
        strat = StatArbStrategy("EURUSD", {}, pair_symbol="GBPUSD", lookback=60)
        sig = strat.generate_signal(df)
        # High divergence should trigger a signal
        assert sig["direction"] in ("BUY", "SELL")
        assert sig["confidence"] > 0.0

    def test_missing_pair_column(self):
        df = _make_df(80)
        strat = StatArbStrategy("EURUSD", {}, pair_symbol="GBPUSD")
        sig = strat.generate_signal(df)
        assert sig["direction"] == "HOLD"


class TestCarryTradeStrategy:
    def test_generates_signal(self):
        df = _make_df(30)
        strat = CarryTradeStrategy("AUDJPY", {"atr_threshold": 0.01})
        sig = strat.generate_signal(df)
        assert "direction" in sig
        assert sig["direction"] in ("BUY", "SELL", "HOLD")

    def test_audjpy_long_bias(self):
        """AUD (4.35%) vs JPY (0.1%) should yield a positive rate diff."""
        df = _make_df(30)
        # Force low ATR by making high/low very close to close
        df["high"] = df["close"] + 0.00005
        df["low"] = df["close"] - 0.00005
        strat = CarryTradeStrategy("AUDJPY", {"atr_threshold": 0.01})
        sig = strat.generate_signal(df)
        assert sig["direction"] == "BUY"
        assert sig["confidence"] > 0.0

    def test_high_vol_filter(self):
        df = _make_df(30)
        # Force high ATR
        df["high"] = df["close"] + 0.01
        df["low"] = df["close"] - 0.01
        strat = CarryTradeStrategy("AUDJPY", {"atr_threshold": 0.001})
        sig = strat.generate_signal(df)
        assert sig["direction"] == "HOLD"
        assert sig["meta"].get("reason") == "vol_too_high"


class TestEventDrivenStrategy:
    def test_generates_signal(self):
        df = _make_df(30)
        df["bb_width"] = np.linspace(0.001, 0.003, 30)
        strat = EventDrivenStrategy("EURUSD", {})
        sig = strat.generate_signal(df)
        assert "direction" in sig

    def test_low_vol_confirmed(self):
        df = _make_df(30)
        df["bb_width"] = np.full(30, 0.001)
        strat = EventDrivenStrategy("EURUSD", {})
        sig = strat.generate_signal(df)
        # If there is an upcoming event and low vol, may signal
        assert isinstance(sig["meta"], dict)


class TestVolExpansionStrategy:
    def test_generates_signal(self):
        df = _make_df(110)
        df["bb_width"] = np.concatenate([np.full(100, 0.001), np.full(10, 0.005)])
        df["volume"] = np.concatenate([np.full(100, 500.0), np.full(10, 1500.0)])
        strat = VolExpansionStrategy("EURUSD", {})
        sig = strat.generate_signal(df)
        assert "direction" in sig
        assert sig["direction"] in ("BUY", "SELL", "HOLD")

    def test_volume_spike_breakout(self):
        df = _make_df(110)
        df["bb_width"] = np.concatenate([np.full(100, 0.001), np.full(10, 0.005)])
        df["volume"] = np.concatenate([np.full(100, 500.0), np.full(10, 2000.0)])
        # Force positive momentum: set last close above third-last
        df.loc[df.index[-1], "close"] = float(df["close"].iloc[-3]) + 0.005
        # Use lower volume threshold so 2000/1250=1.6x counts as spike
        strat = VolExpansionStrategy("EURUSD", {"volume_spike_threshold": 1.5})
        sig = strat.generate_signal(df)
        assert sig["direction"] == "BUY"
        assert sig["confidence"] > 0.5


class TestOrderFlowMomentumStrategy:
    def test_generates_signal(self):
        df = _make_df(50)
        strat = OrderFlowMomentumStrategy("EURUSD", {})
        sig = strat.generate_signal(df)
        assert "direction" in sig
        assert sig["direction"] in ("BUY", "SELL", "HOLD")

    def test_cvd_divergence(self):
        df = _make_df(50)
        # Flat price, should not trigger divergence unless engine populated
        strat = OrderFlowMomentumStrategy("EURUSD", {"cvd_slope_threshold": 0.0})
        # Feed enough ticks to build cvd slope
        for _ in range(25):
            sig = strat.generate_signal(df)
        assert "direction" in sig
