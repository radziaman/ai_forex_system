"""
Dukascopy historical tick/OHLCV data provider.
Free tick data since 2003 via HTTP API. Uses cloudscraper to bypass Cloudflare.
"""
import asyncio
import struct
import lzma
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Callable
from pathlib import Path

import cloudscraper

from api.base import DataProvider, PriceTick, OHLCV

logger = logging.getLogger(__name__)

DUKASCOPE_SYMBOLS = {
    "EURUSD": "EURUSD", "GBPUSD": "GBPUSD", "USDJPY": "USDJPY",
    "AUDUSD": "AUDUSD", "USDCAD": "USDCAD", "USDCHF": "USDCHF",
    "NZDUSD": "NZDUSD", "EURJPY": "EURJPY", "GBPJPY": "GBPJPY",
    "EURGBP": "EURGBP", "XAUUSD": "XAUUSD",
}

BASE_URL = "https://www.dukascopy.com/datafeed/{symbol}/{year}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
CACHE_DIR = Path("data/dukascopy_cache")
TICK_RECORD_SIZE = 20


def _decode_bi5(data: bytes) -> List[tuple]:
    try:
        decompressed = lzma.decompress(data)
    except lzma.LZMAError:
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
        ticks.append((ms / 1000.0, bid_raw / 100000.0, ask_raw / 100000.0, bid_vol, ask_vol))
    return ticks


def _aggregate_ohlcv(ticks: List[tuple], period_seconds: int = 3600) -> List[OHLCV]:
    if not ticks:
        return []
    bars = []
    current_bar = None
    bar_start = 0
    for ts, bid, ask, bv, av in ticks:
        mid = (bid + ask) / 2.0
        bar_idx = int(ts // period_seconds)
        if current_bar is None or bar_idx != bar_start:
            if current_bar is not None:
                bars.append(current_bar)
            bar_start = bar_idx
            current_bar = {"timestamp": bar_idx * period_seconds, "open": mid, "high": mid, "low": mid, "close": mid, "volume": bv + av}
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
    Free historical tick/OHLCV data via Dukascopy's public HTTP API.
    Uses cloudscraper to bypass Cloudflare bot protection.

    Data available: 2003-present, all major forex pairs.
    Rate limit: ~100ms between requests (configurable).

    Usage:
        provider = DukascopyDataProvider()
        ohlcv = await provider.fetch_ohlcv("EURUSD", "1h", "2026-03-01", "2026-03-02")
        ticks = await provider.fetch_ticks("EURUSD", "2026-03-02")
    """

    def __init__(self, cache: bool = True, rate_limit: float = 0.1):
        self.cache_enabled = cache
        self.rate_limit = rate_limit
        self._scraper = cloudscraper.create_scraper()
        if cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

    async def fetch_ticks(self, symbol: str, date: str) -> List[PriceTick]:
        duk = DUKASCOPE_SYMBOLS.get(symbol.upper())
        if not duk:
            raise ValueError(f"Symbol {symbol} not supported")
        dt = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        all_ticks: List[PriceTick] = []
        loop = asyncio.get_event_loop()

        for hour in range(24):
            url = BASE_URL.format(symbol=duk, year=dt.year, month=dt.month, day=dt.day, hour=hour)
            cache_path = CACHE_DIR / f"{symbol}_{dt.strftime('%Y%m%d')}_{hour:02d}.bi5"

            data = None
            if self.cache_enabled and cache_path.exists():
                data = cache_path.read_bytes()
            else:
                try:
                    resp = await loop.run_in_executor(None, lambda: self._scraper.get(url, timeout=15))
                    if resp.status_code == 200 and len(resp.content) > 10:
                        data = resp.content
                        if self.cache_enabled:
                            CACHE_DIR.mkdir(parents=True, exist_ok=True)
                            cache_path.write_bytes(data)
                except Exception as e:
                    logger.debug(f"Dukascopy hour {hour}: {e}")
                    continue
                await asyncio.sleep(self.rate_limit)

            if not data or len(data) < 10:
                continue
            try:
                decoded = _decode_bi5(data)
            except Exception:
                continue

            hour_ts = dt.timestamp() + hour * 3600
            for ms_offset, bid, ask, bv, av in decoded:
                all_ticks.append(PriceTick(symbol=symbol, bid=bid, ask=ask, spread=round(ask - bid, 5), volume=bv + av, timestamp=hour_ts + ms_offset))

        all_ticks.sort(key=lambda t: t.timestamp)
        return all_ticks

    async def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", start: str = "", end: str = "") -> List[OHLCV]:
        period_map = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}
        period = period_map.get(timeframe, 3600)
        dt_start = (datetime.fromisoformat(start) if "T" in start else datetime.strptime(start, "%Y-%m-%d")).replace(tzinfo=timezone.utc)
        dt_end = (datetime.fromisoformat(end) if "T" in end else datetime.strptime(end, "%Y-%m-%d")).replace(tzinfo=timezone.utc)

        all_ticks = []
        current = dt_start
        while current <= dt_end:
            ticks = await self.fetch_ticks(symbol, current.strftime("%Y-%m-%d"))
            all_ticks.extend(ticks)
            current += timedelta(days=1)

        if not all_ticks:
            return []
        tick_tuples = [(t.timestamp, t.bid, t.ask, t.volume, t.volume) for t in all_ticks]
        ohlcv_bars = _aggregate_ohlcv(tick_tuples, period)
        ts_start, ts_end = dt_start.timestamp(), dt_end.timestamp()
        return [b for b in ohlcv_bars if ts_start <= b.timestamp <= ts_end]

    async def stream_prices(self, symbols: List[str], callback: Callable):
        raise NotImplementedError("Dukascopy does not provide real-time streaming. Use cTrader for live prices.")
