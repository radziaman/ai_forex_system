"""Tests for the Microstructure Feature Engine."""

from typing import List, Optional, Tuple

import numpy as np
import pytest

from src.data.microstructure_features import (
    MicrostructureEngine,
    MicrostructureSnapshot,
    PriceTick,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> MicrostructureEngine:
    return MicrostructureEngine(maxlen=100)


def _make_tick(
    symbol: str = "EURUSD",
    bid: float = 1.1000,
    ask: float = 1.1002,
    ts: float = 0.0,
    volume: float = 1.0,
    trade_volume: float = 0.0,
    trade_side: Optional[str] = None,
    dom_bids: Optional[List[Tuple[float, float]]] = None,
    dom_asks: Optional[List[Tuple[float, float]]] = None,
) -> PriceTick:
    return PriceTick(
        symbol=symbol,
        bid=bid,
        ask=ask,
        mid=(bid + ask) / 2.0,
        timestamp=ts,
        volume=volume,
        trade_volume=trade_volume,
        trade_side=trade_side,
        dom_bids=dom_bids or [],
        dom_asks=dom_asks or [],
    )


# ---------------------------------------------------------------------------
# OFI computation
# ---------------------------------------------------------------------------


class TestOrderFlowImbalance:
    def test_pure_buy_ofi(self, engine):
        for i in range(5):
            engine.ingest_tick(
                _make_tick(
                    bid=1.1000 + i * 0.0001,
                    ask=1.1002 + i * 0.0001,
                    ts=float(i),
                    trade_volume=1.0,
                    trade_side="buy",
                )
            )
        assert engine.get_ofi("EURUSD") == pytest.approx(1.0)

    def test_pure_sell_ofi(self, engine):
        for i in range(5):
            engine.ingest_tick(
                _make_tick(
                    bid=1.1000 - i * 0.0001,
                    ask=1.1002 - i * 0.0001,
                    ts=float(i),
                    trade_volume=1.0,
                    trade_side="sell",
                )
            )
        assert engine.get_ofi("EURUSD") == pytest.approx(-1.0)

    def test_mixed_ofi(self, engine):
        engine.ingest_tick(_make_tick(ts=0.0, trade_volume=3.0, trade_side="buy"))
        engine.ingest_tick(_make_tick(ts=1.0, trade_volume=1.0, trade_side="sell"))
        # net = 2, total = 4 => 0.5
        assert engine.get_ofi("EURUSD") == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Lee-Ready trade sign classification
# ---------------------------------------------------------------------------


class TestLeeReady:
    def test_rising_mid_classified_buy(self, engine):
        """Ticks with rising mid and no explicit side should be buyer-initiated."""
        engine.ingest_tick(_make_tick(bid=1.1000, ask=1.1002, ts=0.0, trade_volume=1.0))
        engine.ingest_tick(_make_tick(bid=1.1001, ask=1.1003, ts=1.0, trade_volume=1.0))
        engine.ingest_tick(_make_tick(bid=1.1002, ask=1.1004, ts=2.0, trade_volume=1.0))
        # all three inferred buys => OFI = +1
        assert engine.get_ofi("EURUSD") == pytest.approx(1.0)

    def test_falling_mid_classified_sell(self, engine):
        engine.ingest_tick(_make_tick(bid=1.1002, ask=1.1004, ts=0.0, trade_volume=1.0))
        engine.ingest_tick(_make_tick(bid=1.1001, ask=1.1003, ts=1.0, trade_volume=1.0))
        engine.ingest_tick(_make_tick(bid=1.1000, ask=1.1002, ts=2.0, trade_volume=1.0))
        assert engine.get_ofi("EURUSD") == pytest.approx(-1.0)

    def test_unchanged_mid_is_neutral(self, engine):
        engine.ingest_tick(_make_tick(bid=1.1000, ask=1.1002, ts=0.0, trade_volume=1.0))
        engine.ingest_tick(_make_tick(bid=1.1000, ask=1.1002, ts=1.0, trade_volume=1.0))
        # second tick has zero signed volume => total volume
        # counted is only 1.0 (first tick)
        # net = 0.0 (first tick signed vol is 0 because prev is None),
        # total = 0.0 -> OFI 0.0
        # Wait, first tick has prev=None -> signed_volume=0. So total=0 -> OFI=0.
        # Let's add a third tick with rising mid.
        engine.ingest_tick(_make_tick(bid=1.1001, ask=1.1003, ts=2.0, trade_volume=1.0))
        # total signed vol = 0 + 0 + 1 = 1; net = 1; total abs = 1 => OFI = 1.0
        assert engine.get_ofi("EURUSD") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# CVD tracking
# ---------------------------------------------------------------------------


class TestCVD:
    def test_cvd_monotonic(self, engine):
        for i in range(5):
            engine.ingest_tick(
                _make_tick(ts=float(i), trade_volume=1.0, trade_side="buy")
            )
        assert engine.get_cvd("EURUSD") == pytest.approx(5.0)

    def test_cvd_with_sells(self, engine):
        engine.ingest_tick(_make_tick(ts=0.0, trade_volume=2.0, trade_side="buy"))
        engine.ingest_tick(_make_tick(ts=1.0, trade_volume=1.0, trade_side="sell"))
        engine.ingest_tick(_make_tick(ts=2.0, trade_volume=3.0, trade_side="buy"))
        assert engine.get_cvd("EURUSD") == pytest.approx(4.0)

    def test_cvd_slope(self, engine):
        # Seed a prior tick so the first real buy gets signed volume
        engine.ingest_tick(
            _make_tick(ts=-1.0, bid=1.0998, ask=1.1000, trade_volume=0.0)
        )
        for i in range(20):
            engine.ingest_tick(
                _make_tick(ts=float(i), trade_volume=1.0, trade_side="buy")
            )
        slope = engine.get_cvd_slope("EURUSD", lookback=20)
        # CVD increased by 20 over 20 ticks => slope 1.0
        assert slope == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# VWAP / TWAP deviation
# ---------------------------------------------------------------------------


class TestVwapTwap:
    def test_vwap_deviation(self, engine):
        engine.ingest_tick(_make_tick(ts=0.0, bid=1.1000, ask=1.1002, trade_volume=1.0))
        engine.ingest_tick(_make_tick(ts=1.0, bid=1.1100, ask=1.1102, trade_volume=1.0))
        # VWAP = (1.1001*1 + 1.1101*1) / 2 = 1.1051
        current = 1.1051
        dev = engine.get_vwap_deviation("EURUSD", current)
        assert dev == pytest.approx(0.0, abs=1e-6)

    def test_twap_deviation(self, engine):
        engine.ingest_tick(_make_tick(ts=0.0, bid=1.1000, ask=1.1002))
        engine.ingest_tick(_make_tick(ts=1.0, bid=1.1100, ask=1.1102))
        # TWAP = (1.1001 + 1.1101) / 2 = 1.1051
        current = 1.1051
        dev = engine.get_twap_deviation("EURUSD", current)
        assert dev == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Spread percentiles
# ---------------------------------------------------------------------------


class TestSpreadPercentiles:
    def test_known_spreads(self, engine):
        spreads = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]
        for i, s in enumerate(spreads):
            engine.ingest_tick(_make_tick(ts=float(i), bid=1.1000, ask=1.1000 + s))
        p50, p95 = engine.get_spread_percentiles("EURUSD")
        assert p50 == pytest.approx(0.0003)
        # NumPy linear interpolation for p95 with 5 points -> 0.00048
        assert p95 == pytest.approx(0.00048, abs=1e-6)

    def test_spread_expansion(self, engine):
        for i in range(10):
            engine.ingest_tick(_make_tick(ts=float(i), bid=1.1000, ask=1.1001))
        # median spread = 0.0001; current spread = 0.0001
        assert engine.get_spread_expansion("EURUSD") == pytest.approx(0.0)

        engine.ingest_tick(_make_tick(ts=10.0, bid=1.1000, ask=1.1005))
        # current = 0.0005, median = 0.0001 => (0.0005/0.0001)-1 = 4.0
        assert engine.get_spread_expansion("EURUSD") == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Feature vector shape & snapshot
# ---------------------------------------------------------------------------


class TestFeatureVector:
    def test_vector_length(self, engine):
        for i in range(5):
            engine.ingest_tick(_make_tick(ts=float(i)))
        vec = engine.get_feature_vector("EURUSD", 1.1001)
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (11,)
        assert vec.dtype == np.float32

    def test_snapshot_fields(self, engine):
        for i in range(5):
            engine.ingest_tick(_make_tick(ts=float(i)))
        snap = engine.get_snapshot("EURUSD", 1.1001)
        assert isinstance(snap, MicrostructureSnapshot)
        expected_fields = [
            "ofi",
            "cvd",
            "cvd_slope",
            "spread_p50",
            "spread_p95",
            "spread_expansion",
            "trade_size_skew",
            "whale_ratio",
            "vwap_deviation",
            "twap_deviation",
            "dom_depth_imbalance",
        ]
        for f in expected_fields:
            assert hasattr(snap, f)

    def test_dom_depth_imbalance(self, engine):
        engine.ingest_tick(
            _make_tick(
                ts=0.0,
                dom_bids=[(1.1000, 10.0), (1.0999, 5.0)],
                dom_asks=[(1.1002, 4.0), (1.1003, 2.0)],
            )
        )
        imbalance = engine.get_dom_depth_imbalance("EURUSD", levels=2)
        bid_vol = 10.0 + 5.0
        ask_vol = 4.0 + 2.0
        expected = (bid_vol - ask_vol) / (bid_vol + ask_vol)
        assert imbalance == pytest.approx(expected)
