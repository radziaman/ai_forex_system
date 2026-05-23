"""Tests for execution quality tracking and slice planning."""

import pytest

from execution.execution_quality import (
    ExecutionQualityTracker,
    SlicePlanner,
    ExecutionQualityReport,
)


def test_slippage_estimation_from_dom():
    tracker = ExecutionQualityTracker()
    depth = {
        "bids": [
            (1.1000, 10000),
            (1.0998, 20000),
            (1.0995, 50000),
        ],
        "asks": [
            (1.1002, 10000),
            (1.1004, 20000),
            (1.1007, 50000),
        ],
    }
    # BUY 15_000 -> consumes 10_000 @ 1.1002 + 5_000 @ 1.1004
    # avg = (10_000*1.1002 + 5_000*1.1004) / 15_000 = 1.1002667
    # slippage_price = 0.0000667 -> 0.667 pips
    slippage = tracker.estimate_slippage("EURUSD", "BUY", 15000, depth)
    assert slippage == pytest.approx(0.667, rel=1e-2)

    # SELL 25_000 -> consumes 10_000 @ 1.1000 + 15_000 @ 1.0998
    # avg = (10_000*1.1000 + 15_000*1.0998) / 25_000 = 1.09988
    # slippage_price = 0.00012 -> 1.2 pips
    slippage_sell = tracker.estimate_slippage("EURUSD", "SELL", 25000, depth)
    assert slippage_sell == pytest.approx(1.2, rel=1e-2)


def test_twap_slice_plan_sum_equals_total():
    total = 100_000
    slices = SlicePlanner.plan_twap(total, 5, 300)
    assert len(slices) == 5
    assert sum(slices) == pytest.approx(total, rel=1e-9)
    for s in slices:
        assert s >= 0


def test_vwap_slice_plan():
    profile = [0.2, 0.3, 0.5]
    total = 100_000
    slices = SlicePlanner.plan_vwap(total, profile)
    assert len(slices) == 3
    assert sum(slices) == pytest.approx(total, rel=1e-9)
    assert slices[0] == pytest.approx(20_000, rel=1e-9)
    assert slices[1] == pytest.approx(30_000, rel=1e-9)
    assert slices[2] == pytest.approx(50_000, rel=1e-9)


def test_fill_quality_recording():
    tracker = ExecutionQualityTracker()
    sym = "EURUSD"
    tracker.record_order_attempt(sym)
    tracker.record_fill(sym, 1.1000, 1.1001, "BUY", 10000)
    tracker.record_order_attempt(sym)
    tracker.record_fill(sym, 1.1000, 1.1002, "BUY", 10000)
    # one rejected attempt
    tracker.record_order_attempt(sym)

    dist = tracker.get_slippage_distribution(sym)
    assert dist["mean"] == pytest.approx(1.5, rel=1e-2)
    assert dist["p50"] == pytest.approx(1.5, rel=1e-2)
    assert dist["p95"] == pytest.approx(1.95, rel=1e-2)

    report = tracker.get_fill_quality_report(sym)
    assert isinstance(report, ExecutionQualityReport)
    assert report.avg_slippage_pips == pytest.approx(1.5, rel=1e-2)
    assert report.p95_slippage_pips == pytest.approx(1.95, rel=1e-2)
    assert report.fill_rate == pytest.approx(2 / 3, rel=1e-9)


def test_market_impact_calculation():
    tracker = ExecutionQualityTracker()
    sym = "EURUSD"
    tracker.set_adv(sym, 1_000_000)
    impact = tracker.calculate_market_impact(sym, 100_000, 0.0010, 1.1000)
    # sqrt(100_000 / 1_000_000) = 0.31622777
    # atr/price = 0.0010 / 1.1000 = 0.00090909
    # fraction = 0.000287348
    # bps = 2.87348
    assert impact > 0
    assert impact == pytest.approx(2.8748, rel=1e-2)


def test_should_slice():
    tracker = ExecutionQualityTracker()
    assert tracker.should_slice(5001) is True
    assert tracker.should_slice(5000) is False
    assert tracker.should_slice(1000) is False


def test_plan_slices():
    tracker = ExecutionQualityTracker()
    slices = tracker.plan_slices(100_000, "twap", n_slices=5, duration_sec=300)
    assert len(slices) == 5
    assert sum(slices) == pytest.approx(100_000, rel=1e-9)

    slices_vwap = tracker.plan_slices(100_000, "vwap", volume_profile=[0.2, 0.3, 0.5])
    assert len(slices_vwap) == 3
    assert sum(slices_vwap) == pytest.approx(100_000, rel=1e-9)
