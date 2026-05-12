"""
Tests for Dukascopy real-time provider integration.
"""

import os
import sys
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.dukascopy_realtime import DukascopyProvider
from api.base import PriceTick


class TestDukascopyProvider:
    """Test cases for DukascopyProvider."""

    @pytest.fixture
    def provider(self):
        return DukascopyProvider(cache=False, poll_interval=0.1)

    def test_initialization(self, provider):
        assert provider.cache_enabled is False
        assert provider.poll_interval == 0.1
        assert provider._running is False
        assert len(provider._subscribers) == 0

    def test_symbol_mapping(self):
        """Test that symbol mapping works correctly."""
        from data.dukascopy_realtime import DUKASCOPE_SYMBOLS

        assert DUKASCOPE_SYMBOLS["EURUSD"] == "EURUSD"
        assert DUKASCOPE_SYMBOLS["GBPUSD"] == "GBPUSD"
        assert DUKASCOPE_SYMBOLS["USDJPY"] == "USDJPY"

    @pytest.mark.asyncio
    async def test_get_session(self, provider):
        session = await provider._get_session()
        assert session is not None
        assert not session.closed
        await provider.close()

    def test_decode_bi5(self, provider):
        """Test BI5 decoding with mock data."""
        # Create mock BI5 data (20 bytes per tick)
        import struct

        mock_data = b""
        for i in range(5):  # 5 ticks
            ms = struct.pack(">I", i * 1000)
            ask = struct.pack(">I", 1120000 + i * 10)
            bid = struct.pack(">I", 1120000 - i * 10)
            ask_vol = struct.pack(">f", 1.0 + i * 0.1)
            bid_vol = struct.pack(">f", 0.9 + i * 0.1)
            mock_data += ms + ask + bid + ask_vol + bid_vol

        ticks = provider._decode_bi5(mock_data)
        assert len(ticks) == 5
        for ms, bid, ask, bv, av in ticks:
            assert ms >= 0
            assert bid > 0
            assert ask > 0

    @pytest.mark.asyncio
    async def test_fetch_ticks_mock(self, provider):
        """Test tick fetching with mocked HTTP response."""
        mock_data = b"\\x00\\x00\\x03\\xe8" + b"\\x00" * 16  # Minimal BI5 data

        with patch.object(provider, "_get_session") as mock_get_session:
            mock_session = AsyncMock()
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.read = AsyncMock(return_value=mock_data)
            mock_session.get = AsyncMock(return_value=mock_resp)
            mock_get_session.return_value = mock_session

            ticks = await provider.fetch_ticks("EURUSD", "2026-05-04")
            # Should return something (even if decode fails gracefully)
            assert isinstance(ticks, list)

        await provider.close()

    @pytest.mark.asyncio
    async def test_stream_prices(self, provider):
        """Test price streaming."""
        callback_mock = AsyncMock()
        symbols = ["EURUSD"]

        # Start streaming in background
        task = asyncio.create_task(provider.stream_prices(symbols, callback_mock))

        await asyncio.sleep(0.2)  # Let it run briefly

        provider._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        await provider.close()

    def test_get_latest_price_empty(self, provider):
        """Test getting latest price when none exists."""
        price = provider.get_latest_price("EURUSD")
        assert price is None

    @pytest.mark.asyncio
    async def test_close(self, provider):
        """Test clean shutdown."""
        await provider.close()
        assert provider._running is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
