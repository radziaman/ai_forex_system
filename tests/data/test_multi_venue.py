"""Tests for the Multi-Venue Data Framework."""

import asyncio

from src.data.multi_venue_provider import MultiVenueProvider, VenueQuote


class TestMultiVenueProvider:
    def test_default_venues_registered(self):
        provider = MultiVenueProvider()
        assert "ctrader" in provider._venues
        assert "dukascopy" in provider._venues

    def test_register_venue(self):
        provider = MultiVenueProvider(venues=[])

        async def mock_fetcher(symbol: str):
            return VenueQuote(
                venue_name="lmax",
                bid=1.1000,
                ask=1.1002,
                mid=1.1001,
                timestamp=0.0,
            )

        provider.register_venue("lmax", mock_fetcher)
        assert "lmax" in provider._venues

    def test_get_all_quotes(self):
        provider = MultiVenueProvider()
        quotes = asyncio.run(provider.get_all_quotes("EURUSD"))
        assert len(quotes) == 2
        venues = {q.venue_name for q in quotes}
        assert venues == {"ctrader", "dukascopy"}

    def test_get_best_quote(self):
        provider = MultiVenueProvider()
        asyncio.run(provider.get_all_quotes("EURUSD"))
        best = provider.get_best_quote("EURUSD")
        assert best is not None
        # cTrader should have tighter spread than Dukascopy in mock data
        assert best.venue_name == "ctrader"

    def test_detect_arbitrage_false_when_similar(self):
        provider = MultiVenueProvider()
        asyncio.run(provider.get_all_quotes("EURUSD"))
        # Mock data spreads are within threshold
        assert not provider.detect_arbitrage("EURUSD")

    def test_detect_arbitrage_true_when_large_spread(self):
        provider = MultiVenueProvider()
        asyncio.run(provider.get_all_quotes("EURUSD"))
        # Force arbitrage by lowering threshold below any realistic spread
        assert provider.detect_arbitrage("EURUSD", threshold=0.00001)

    def test_get_quote_latency_report(self):
        provider = MultiVenueProvider()
        asyncio.run(provider.get_all_quotes("EURUSD"))
        report = provider.get_quote_latency_report()
        assert "ctrader" in report
        assert "dukascopy" in report
        for name, stats in report.items():
            assert "mean" in stats
            assert "min" in stats
            assert "max" in stats
            assert "count" in stats
            assert stats["count"] > 0
