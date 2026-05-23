"""Tests for the High-Frequency DOM Capture & Analytics."""

import pytest

from src.data.dom_analyzer import DOMAnalyzer


class TestDOMAnalyzer:
    def test_ingest_dom_creates_snapshot(self):
        dom = DOMAnalyzer("EURUSD", depth_levels=5)
        bids = [(1.1000, 10.0), (1.0999, 5.0), (1.0998, 2.0)]
        asks = [(1.1002, 4.0), (1.1003, 2.0), (1.1004, 1.0)]
        dom.ingest_dom(bids, asks)
        snap = dom.latest_snapshot()
        assert snap is not None
        assert snap.best_bid == pytest.approx(1.1000)
        assert snap.best_ask == pytest.approx(1.1002)

    def test_get_depth_imbalance(self):
        dom = DOMAnalyzer("EURUSD", depth_levels=5)
        bids = [(1.1000, 10.0), (1.0999, 5.0)]
        asks = [(1.1002, 4.0), (1.1003, 2.0)]
        dom.ingest_dom(bids, asks)
        imbalance = dom.get_depth_imbalance()
        total_vol = 10 + 5 + 4 + 2
        expected = (10 + 5) / total_vol
        assert imbalance == pytest.approx(expected, abs=1e-6)

    def test_get_liquidity_score(self):
        dom = DOMAnalyzer("EURUSD", depth_levels=5)
        bids = [(1.1000, 10.0), (1.0999, 5.0)]
        asks = [(1.1002, 4.0), (1.1003, 2.0)]
        dom.ingest_dom(bids, asks)
        score = dom.get_liquidity_score()
        assert score == pytest.approx(14.0)

    def test_detect_spoofing(self):
        dom = DOMAnalyzer("EURUSD", depth_levels=5)
        # Build history with large size then sudden drop
        for _ in range(5):
            dom.ingest_dom([(1.1000, 100.0)], [(1.1002, 4.0)])
        # Now size drops dramatically
        dom.ingest_dom([(1.1000, 10.0)], [(1.1002, 4.0)])
        assert dom.detect_spoofing(threshold_ratio=3.0)

    def test_detect_iceberg(self):
        dom = DOMAnalyzer("EURUSD", depth_levels=5)
        # Repeated similar-size changes at same level
        for size in [50.0, 45.0, 40.0, 35.0, 30.0]:
            dom.ingest_dom([(1.1000, size)], [(1.1002, 4.0)])
        assert dom.detect_iceberg(min_fills=3, size_tolerance=0.15)

    def test_get_support_resistance(self):
        dom = DOMAnalyzer("EURUSD", depth_levels=5)
        bids = [(1.1000, 100.0), (1.0995, 10.0)]
        asks = [(1.1002, 50.0), (1.1005, 5.0)]
        dom.ingest_dom(bids, asks)
        support, resistance = dom.get_support_resistance()
        assert support == pytest.approx(1.1000)
        assert resistance == pytest.approx(1.1002)

    def test_get_book_pressure(self):
        dom = DOMAnalyzer("EURUSD", depth_levels=5)
        bids = [(1.1000, 10.0), (1.0999, 5.0)]
        asks = [(1.1002, 4.0), (1.1003, 2.0)]
        dom.ingest_dom(bids, asks)
        pressure = dom.get_book_pressure()
        total = 10 + 5 + 4 + 2
        expected = (10 + 5 - 4 - 2) / total
        assert pressure == pytest.approx(expected, abs=1e-6)

    def test_empty_dom_returns_defaults(self):
        dom = DOMAnalyzer("EURUSD", depth_levels=5)
        assert dom.get_depth_imbalance() == pytest.approx(0.5)
        assert dom.get_liquidity_score() == pytest.approx(0.0)
        assert not dom.detect_spoofing()
        assert not dom.detect_iceberg()
        assert dom.get_book_pressure() == pytest.approx(0.0)
        assert dom.latest_snapshot() is None
