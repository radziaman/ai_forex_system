"""
Dukascopy historical tick/OHLCV data provider.
Free tick data since 2003 via HTTP API. No authentication required.
"""
import asyncio
import struct
import lzma
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Callable
from pathlib import Path

import aiohttp

from api.base import DataProvider, PriceTick, OHLCV

logger = logging.getLogger(__name__)

# Symbol mapping: our format -> Dukascopy format
DUKASCOPE_SYMBOLS = {
    "EURUSD": "EURUSD", "GBPUSD": "GBPUSD", "USDJPY": "USDJPY",
    "AUDUSD": "AUDUSD", "USDCAD": "USDCAD", "USDCHF": "USDCHF",
    "NZDUSD": "NZDUSD", "EURJPY": "EURJPY", "GBPJPY": "GBPJPY",
    "EURGBP": "EURGBP", "XAUUSD": "XAUUSD",
}

BASE_URL = "https://www.dukascopy.com/datafeed/{symbol}/{year}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
CACHE_DIR = Path("data/dukascopy_cache")

TICK_RECORD_SIZE = 20  # bytes per tick


def _decode_bi5(data: bytes) -> List[tuple]:
    """Decode LZMA-compressed Dukascopy .bi5 tick data."""
    try:
        decompressed = lzma.decompress(data)
    except lzma.LZMAError:
        # Sometimes the data is not LZMA-compressed (rare)
        decompressed = data
    ticks = []
    for i in range(0, len(decompressed), TICK_RECORD_SIZE):
        chunk = decompressed[i:i + TICK_RECORD_SIZE]
        if len(chunk) < TICK_RECORD_SIZE:
            break
        ms = struct.unpack(">I", chunk[0:4])[0]
        ask_raw = struct.unpack(">I", chunk[4:8])[0]
        bid_raw = struct.unpack(">I", chunk[8:12])[0]
        ask_vol = struct.unpack(">f", chunk[12:16])[0]
        bid_vol = struct.unpack(">f", chunk[16:20])[0]
        bid = bid_raw / 100_000.0
        ask = ask_raw / 100_000.0
        ticks.append((ms / 1000.0, bid, ask, bid_vol, ask_vol))
    return ticks


def _aggregate_ohlcv(ticks: List[tuple], period_seconds: int = 3600) -> List[OHLCV]:
    """Aggregate ticks into OHLCV bars."""
    if not ticks:
        return []
    bars = []
    current_bar: Optional[dict] = None
    bar_start = 0
    for tick in ticks:
        ts, bid, ask, bv, av = tick
        mid = (bid + ask) / 2.0
        bar_idx = int(ts // period_seconds)
        if current_bar is None or bar_idx != bar_start:
            if current_bar is not None:
                bars.append(current_bar)
            bar_start = bar_idx
            current_bar = {
                "timestamp": bar_idx * period_seconds,
                "open": mid, "high": mid, "low": mid, "close": mid,
                "volume": bv + av,
            }
        else:
            current_bar["high"] = max(current_bar["high"], mid)
            current_bar["low"] = min(current_bar["low"], mid)
            current_bar["close"] = mid
            current_bar["volume"] += bv + av
    if current_bar is not None:
        bars.append(current_bar)
    return [OHLCV(**b) for b in bars]


class DukascopyDataProvider(DataProvider):
    """
    Free historical tick/OHLCV data provider using Dukascopy's public HTTP API.
    Data available from 2003 onwards for all major forex pairs.
    No API key required — rate-limit to ~1 request/second per Dukascopy's TOS.

    Usage:
        provider = DukascopyDataProvider()
        ohlcv = await provider.fetch_ohlcv("EURUSD", "1h", "2026-01-01", "2026-03-31")
        ticks = await provider.fetch_ticks("EURUSD", "2026-03-15")
    """

    def __init__(self, cache: bool = True, rate_limit: float = 0.3):
        self.cache_enabled = cache
        self.rate_limit = rate_limit
        if cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Tick data
    # ------------------------------------------------------------
    async def fetch_ticks(self, symbol: str, date: str) -> List[PriceTick]:
        """
        Fetch all ticks for a given date. Date format: "YYYY-MM-DD".
        Returns list of PriceTick with ~1-second granularity.
        """
        duk = DUKASCOPE_SYMBOLS.get(symbol.upper())
        if not duk:
            raise ValueError(f"Symbol {symbol} not supported by Dukascopy")

        dt = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        all_ticks: List[PriceTick] = []

        async with aiohttp.ClientSession() as session:
            for hour in range(24):
                url = BASE_URL.format(
                    symbol=duk, year=dt.year, month=dt.month,
                    day=dt.day, hour=hour,
                )
                cache_path = CACHE_DIR / f"{symbol}_{dt.strftime('%Y%m%d')}_{hour:02d}.bi5"

                data = None
                if self.cache_enabled and cache_path.exists():
                    data = cache_path.read_bytes()
                else:
                    try:
                        async with session.get(url, timeout=15) as resp:
                            if resp.status == 200:
                                data = await resp.read()
                                if self.cache_enabled and len(data) > 0:
                                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                                    cache_path.write_bytes(data)
                            else:
                                continue
                    except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                        logger.debug(f"Dukascopy hour {hour}: {e}")
                        continue
                    await asyncio.sleep(self.rate_limit)

                if not data or len(data) < 10:
                    continue

                try:
                    decoded = _decode_bi5(data)
                except Exception as e:
                    logger.warning(f"Failed to decode {url}: {e}")
                    continue

                hour_ts = dt.timestamp() + hour * 3600
                for ms_offset, bid, ask, bv, av in decoded:
                    all_ticks.append(PriceTick(
                        symbol=symbol, bid=bid, ask=ask,
                        spread=round(ask - bid, 5),
                        volume=bv + av,
                        timestamp=hour_ts + ms_offset,
                    ))

        all_ticks.sort(key=lambda t: t.timestamp)
        logger.info(f"Dukascopy: {len(all_ticks)} ticks for {symbol} on {date}")
        return all_ticks

    # ------------------------------------------------------------
    # OHLCV data
    # ------------------------------------------------------------
    async def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1h",
        start: str = "", end: str = "",
    ) -> List[OHLCV]:
        """
        Fetch historical OHLCV by aggregating Dukascopy tick data.
        Supports 1m, 5m, 15m, 30m, 1h, 4h, 1d.
        """
        period_map = {
            "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
            "1h": 3600, "4h": 14400, "1d": 86400,
        }
        period = period_map.get(timeframe, 3600)

        dt_start = datetime.fromisoformat(start) if "T" in start else datetime.strptime(start, "%Y-%m-%d")
        dt_end = datetime.fromisoformat(end) if "T" in end else datetime.strptime(end, "%Y-%m-%d")
        dt_start = dt_start.replace(tzinfo=timezone.utc)
        dt_end = dt_end.replace(tzinfo=timezone.utc)

        all_ticks = []
        current = dt_start
        while current <= dt_end:
            date_str = current.strftime("%Y-%m-%d")
            ticks = await self.fetch_ticks(symbol, date_str)
            all_ticks.extend(ticks)
            current += timedelta(days=1)

        if not all_ticks:
            return []

        tick_tuples = [
            (t.timestamp, t.bid, t.ask, t.volume, t.volume)
            for t in all_ticks
        ]
        ohlcv_bars = _aggregate_ohlcv(tick_tuples, period)

        # Filter to requested date range
        ts_start = dt_start.timestamp()
        ts_end = dt_end.timestamp()
        ohlcv_bars = [b for b in ohlcv_bars if ts_start <= b.timestamp <= ts_end]

        logger.info(f"Dukascopy: {len(ohlcv_bars)} {timeframe} bars for {symbol} ({start} to {end})")
        return ohlcv_bars

    # ------------------------------------------------------------
    # Streaming (not supported via free API)
    # ------------------------------------------------------------
    async def stream_prices(self, symbols: List[str], callback: Callable):
        raise NotImplementedError(
            "Dukascopy does not provide free real-time streaming. "
            "Use the cTrader client for live prices."
        )
