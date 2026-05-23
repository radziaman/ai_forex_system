"""
Multi-Source Tick Provider — Unified streaming from cTrader (FX/futures) + Yahoo (futures).

Combines multiple data sources into a single tick stream for the simulation.
Supports any instrument the screener finds tradeable.

Source priority:
  1. cTrader Open API: Real-time bid/ask quotes (all cTrader-supported symbols)
  2. Dukascopy: Fallback for FX pairs when cTrader unavailable
  3. Yahoo Finance: 1-minute futures data (HO=F, CL=F, etc.)

Usage:
    provider = MultiSourceTickProvider()
    await provider.stream_prices([...], callback)
    ticks = await provider.get_instrument_data("HO=F", days=5)
"""

import os, sys, time, math, random
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Callable, Any, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
from loguru import logger

_src = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _src not in sys.path:
    sys.path.insert(0, _src)


@dataclass
class PriceTick:
    """Unified tick format for all sources."""

    symbol: str
    bid: float
    ask: float
    mid: float
    timestamp: float
    volume: float = 0.0
    source: str = ""


# ─── cTrader Read-Only Data Listener ──────────────────────────────────────


class CtradeDataOnly:
    """Wrapper around ctrader_client — read-only data, no trading capability.

    Reuses the existing, proven CtraderClient with account_id=0 to ensure
    ZERO ability to place trades. Only subscribes to spot/depth quotes.
    """

    def __init__(self):
        self._client = None
        self._callbacks = {}

    async def connect(self) -> bool:
        """Connect to cTrader with account_id=0 (read-only, no trading)."""
        try:
            from api.ctrader_client import CtraderClient
            from infrastructure.secrets import Secrets

            secrets = Secrets()
            self._client = CtraderClient(
                app_id=secrets.ctrader_app_id,
                app_secret=secrets.ctrader_app_secret,
                access_token=secrets.ctrader_access_token,
                account_id=0,  # ZERO = read-only, cannot place orders
                demo=secrets.is_demo,
            )
            connected = await self._client.start()
            if connected:
                logger.info("cTrader connected (read-only, account_id=0)")
                # Wire depth events to our dispatch
                self._client.on_depth_update = self._on_depth
            return connected
        except Exception as e:
            logger.warning(f"cTrader not available: {e}")
            return False

    async def subscribe(self, symbols: List[str], callback: Callable):
        """Subscribe to price quotes via depth subscription."""
        from api.symbol_map import get_symbol_id

        for symbol in symbols:
            self._callbacks[symbol] = callback
            symbol_id = get_symbol_id(symbol)
            if symbol_id and self._client:
                try:
                    raw = getattr(self._client, "raw", None)
                    if raw and hasattr(raw, "subscribe_depth"):
                        await raw.subscribe_depth(symbol_id)
                        logger.info(f"  Subscribed to {symbol} depth")
                except Exception as e:
                    logger.warning(f"Subscribe {symbol} failed: {e}")

    def _on_depth(self, depth_data):
        """Process depth update and dispatch as tick."""
        symbol = getattr(depth_data, "symbol", None)
        bid = getattr(depth_data, "bid", 0)
        ask = getattr(depth_data, "ask", 0)

        if symbol and symbol in self._callbacks:
            tick = PriceTick(
                symbol=symbol,
                bid=bid,
                ask=ask,
                mid=(bid + ask) / 2,
                timestamp=time.time(),
                source="ctrader",
            )
            try:
                cb = self._callbacks[symbol]
                if asyncio.iscoroutinefunction(cb):
                    asyncio.ensure_future(cb(tick))
                else:
                    cb(tick)
            except Exception:
                pass

    async def close(self):
        if self._client:
            try:
                self._client._is_connected = False
                if self._client._writer:
                    self._client._writer.close()
            except:
                pass
            self._client = None
            logger.info("cTrader data listener closed")


# ─── Yahoo Finance 1-Minute Data Provider ─────────────────────────────────


class YahooFuturesProvider:
    """Provides 1-minute futures data from Yahoo Finance.

    Yahoo Finance provides up to 7 days of 1-minute history for futures.
    This provider downloads it and replays it as ticks with realistic
    intra-minute microprice variation.
    """

    CACHE_DIR = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "tick_cache"
    )

    SUPPORTED_SYMBOLS = {
        "HO=F": "Heating Oil Futures",
        "CL=F": "Crude Oil Futures",
        "RB=F": "Gasoline Futures",
        "NG=F": "Natural Gas Futures",
        "GC=F": "Gold Futures",
        "SI=F": "Silver Futures",
        "ES=F": "S&P 500 Futures",
        "NQ=F": "Nasdaq Futures",
    }

    def __init__(self):
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        self._cache: Dict[str, pd.DataFrame] = {}
        self._replay_index: Dict[str, int] = {}
        self._replay_ticks: Dict[str, List[PriceTick]] = {}

    def is_supported(self, symbol: str) -> bool:
        return symbol in self.SUPPORTED_SYMBOLS

    async def download_1m_data(
        self, symbol: str, days: int = 7
    ) -> Optional[pd.DataFrame]:
        """Download 1-minute OHLCV data from Yahoo Finance.

        Args:
            symbol: Yahoo Finance ticker (e.g. 'HO=F')
            days: Days of 1m history to download (max 7 for 1m)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        import yfinance as yf

        cache_key = f"{symbol}_{days}d"
        if cache_key in self._cache:
            return self._cache[cache_key]

        cache_file = os.path.join(self.CACHE_DIR, f"{symbol.replace('=', '_')}_1m.csv")
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                # Check if cache is recent enough (within 1 hour)
                if df.index[-1] > pd.Timestamp.now() - pd.Timedelta(hours=1):
                    self._cache[cache_key] = df
                    logger.info(f"Loaded {len(df)} 1m bars for {symbol} from cache")
                    return df
            except:
                pass

        logger.info(f"Downloading {days}d of 1m data for {symbol}...")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{days}d", interval="1m")

            if df is None or df.empty or len(df) < 10:
                logger.warning(
                    f"Insufficient 1m data for {symbol}: {len(df) if df is not None else 0} bars"
                )
                return None

            df.to_csv(cache_file)
            self._cache[cache_key] = df
            logger.info(f"Downloaded {len(df)} 1m bars for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Failed to download {symbol}: {e}")
            return None

    def generate_ticks_from_bars(
        self, df: pd.DataFrame, symbol: str, ticks_per_bar: int = 10
    ) -> List[PriceTick]:
        """Generate realistic ticks from 1-minute OHLCV bars.

        Distributes tick prices within each bar's [low, high] range,
        weighted towards the open/close to simulate realistic price paths.
        """
        ticks = []
        base_price = (
            df["Close"].iloc[-1] if "Close" in df.columns else df.iloc[:, 0].iloc[-1]
        )
        pip_size = 0.0001 if base_price < 10 else 0.01 if base_price < 1000 else 0.1

        for idx, (_, row) in enumerate(df.iterrows()):
            # Extract OHLC
            if "Close" in row:
                o, h, l, c, v = (
                    row["Open"],
                    row["High"],
                    row["Low"],
                    row["Close"],
                    row.get("Volume", 0),
                )
            else:
                o = h = l = c = row.iloc[0]
                v = 0

            if pd.isna(o) or pd.isna(h) or pd.isna(l) or pd.isna(c):
                continue

            ts = row.name.timestamp() if hasattr(row.name, "timestamp") else time.time()

            # Generate ticks within this bar
            for _ in range(ticks_per_bar):
                # Random walk biased towards the close
                t = random.random()
                if t < 0.3:
                    price = o + random.random() * (c - o)
                elif t < 0.7:
                    price = (
                        c + random.random() * (h - c)
                        if random.random() > 0.5
                        else c - random.random() * (c - l)
                    )
                else:
                    price = random.uniform(l, h)

                price = round(price / pip_size) * pip_size  # Quantize to pip
                spread = pip_size * random.uniform(0.5, 2.0)
                tick_ts = ts + random.uniform(0, 60)  # Spread within the minute

                ticks.append(
                    PriceTick(
                        symbol=symbol,
                        bid=round(price - spread / 2, 6),
                        ask=round(price + spread / 2, 6),
                        mid=price,
                        timestamp=tick_ts,
                        volume=random.uniform(0.1, 10),
                        source="yfinance",
                    )
                )

        ticks.sort(key=lambda t: t.timestamp)
        logger.info(
            f"Generated {len(ticks)} synthetic ticks for {symbol} "
            f"({df.index[0]} to {df.index[-1]})"
        )
        return ticks

    async def get_ticks(self, symbol: str, days: int = 7) -> Optional[List[PriceTick]]:
        """Get tick data for a symbol — downloads + generates if needed."""
        if symbol in self._replay_ticks:
            return self._replay_ticks[symbol]

        df = await self.download_1m_data(symbol, days)
        if df is None:
            return None

        ticks = self.generate_ticks_from_bars(df, symbol)
        self._replay_ticks[symbol] = ticks
        self._replay_index[symbol] = 0
        return ticks


# ─── Multi-Source Provider ────────────────────────────────────────────────


class MultiSourceTickProvider:
    """Unified tick provider — streams from Dukascopy (FX) + Yahoo (futures).

    Provides a single `stream_prices(symbols, callback)` interface that
    automatically routes each symbol to the correct data source.

    Symbols are auto-detected:
      - FX pairs (EURUSD, GBPUSD, etc.) → Dukascopy
      - Futures (HO=F, CL=F, etc.) → Yahoo Finance 1m replay
    """

    def __init__(self, poll_interval: float = 1.0):
        self.poll_interval = poll_interval
        self._running = False
        self._ctrader = None  # type: ignore[assignment]
        self._dukascopy = None  # type: ignore[assignment]
        self._yahoo = YahooFuturesProvider()
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._poll_tasks: Dict[str, asyncio.Task] = {}
        self._replay_data: Dict[str, List[PriceTick]] = {}
        self._replay_pos: Dict[str, int] = {}
        self._last_tick_time: Dict[str, float] = {}
        self._ctrader_symbols = {
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "AUDUSD",
            "USDCAD",
            "USDCHF",
            "NZDUSD",
            "XAUUSD",
            "XTIUSD",
            "US500",
            "BTCUSD",
        }
        self._dukascopy_symbols = self._ctrader_symbols.copy()

    def is_futures_symbol(self, symbol: str) -> bool:
        """Check if a symbol is a futures/ETF (needs Yahoo data)."""
        return symbol.upper() not in self._dukascopy_symbols and "=" in symbol

    async def stream_prices(self, symbols: List[str], callback: Callable):
        """Start streaming prices for all symbols.

        Source priority:
          1. cTrader: Real-time bid/ask for supported symbols
          2. Dukascopy: Fallback for FX when cTrader unavailable
          3. Yahoo Finance: Futures replay for unsupported symbols
        """
        self._running = True
        futures_syms = [s for s in symbols if self.is_futures_symbol(s)]
        native_syms = [s for s in symbols if not self.is_futures_symbol(s)]

        for symbol in symbols:
            self._subscribers[symbol].append(callback)

        # Try cTrader first for ALL non-futures symbols (primary source)
        ctrader_available = False
        if native_syms:
            self._ctrader = CtradeDataOnly()
            ctrader_available = await self._ctrader.connect()
            if ctrader_available:

                def ctrader_bridge(quote):
                    tick = self._ctrader.convert_quote(quote)
                    if tick:
                        self._dispatch_tick(tick)

                await self._ctrader.subscribe(native_syms, ctrader_bridge)
                logger.info(f"  cTrader streaming {len(native_syms)} symbols")

        # Fallback: Dukascopy for FX if cTrader failed
        if not ctrader_available and native_syms:
            await self._start_dukascopy(native_syms)

        # Yahoo replay for futures (always)
        for symbol in futures_syms:
            if symbol not in self._poll_tasks:
                self._poll_tasks[symbol] = asyncio.create_task(
                    self._replay_futures(symbol)
                )

        n_sources = len(native_syms) + len(futures_syms)
        source_label = "cTrader" if ctrader_available else "Dukascopy"
        logger.info(
            f"MultiSource streaming {n_sources} symbols "
            f"({len(native_syms)} {source_label}, {len(futures_syms)} Yahoo)"
        )

    async def _start_dukascopy(self, symbols: List[str]):
        """Start Dukascopy tick streaming for FX symbols."""
        try:
            from data.dukascopy_realtime import DukascopyProvider

            self._dukascopy = DukascopyProvider(poll_interval=self.poll_interval)

            def dukascopy_callback(tick):
                """Bridge Dukascopy ticks to our subscribers."""
                symbol = tick.symbol if hasattr(tick, "symbol") else symbols[0]
                bid = tick.bid if hasattr(tick, "bid") else 0
                ask = tick.ask if hasattr(tick, "ask") else 0
                mid = (
                    (bid + ask) / 2
                    if bid and ask
                    else (tick.mid if hasattr(tick, "mid") else 0)
                )
                ts = tick.timestamp if hasattr(tick, "timestamp") else time.time()

                our_tick = PriceTick(
                    symbol=symbol,
                    bid=bid,
                    ask=ask,
                    mid=mid,
                    timestamp=ts,
                    source="dukascopy",
                )
                self._dispatch_tick(our_tick)

            await self._dukascopy.stream_prices(symbols, dukascopy_callback)
        except Exception as e:
            logger.warning(f"Dukascopy failed: {e}. Using synthetic ticks for FX.")

    async def _replay_futures(self, symbol: str):
        """Replay Yahoo 1m futures data as a tick stream."""
        if symbol not in self._replay_data:
            ticks = await self._yahoo.get_ticks(symbol)
            if not ticks:
                logger.warning(f"No data available for {symbol}")
                return
            self._replay_data[symbol] = ticks
            self._replay_pos[symbol] = 0

        ticks = self._replay_data[symbol]
        rate = (
            len(ticks) / (7 * 24 * 3600) if len(ticks) > 0 else 1.0
        )  # ticks/sec to replay in 7 days
        # Speed up for simulation: replay at 60x real-time (1 hour of data = 1 minute)
        speedup = 60.0

        logger.info(f"Replaying {len(ticks)} ticks for {symbol} at {speedup}x speed")

        while self._running and self._replay_pos.get(symbol, 0) < len(ticks):
            idx = self._replay_pos[symbol]
            tick = ticks[idx]

            # Wait appropriate time between ticks based on speedup
            if idx > 0:
                time_gap = tick.timestamp - ticks[idx - 1].timestamp
                if time_gap > 0:
                    await asyncio.sleep(time_gap / speedup)

            self._dispatch_tick(tick)
            self._replay_pos[symbol] = idx + 1

            # Check if we've exhausted data and need to refresh
            if idx >= len(ticks) - 1:
                logger.info(f"{symbol} replay complete, refreshing data...")
                await asyncio.sleep(60)  # Wait before refreshing
                ticks = await self._yahoo.get_ticks(symbol)
                if ticks:
                    self._replay_data[symbol] = ticks
                    self._replay_pos[symbol] = 0

    def _dispatch_tick(self, tick: PriceTick):
        """Send tick to all subscribers for its symbol."""
        self._last_tick_time[tick.symbol] = time.time()
        if tick.symbol in self._subscribers:
            for callback in self._subscribers[tick.symbol]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        # Schedule async callback without awaiting (fire-and-forget)
                        asyncio.ensure_future(callback(tick))
                    else:
                        callback(tick)
                except Exception as e:
                    logger.error(f"Tick callback error: {e}")

    def get_latest_price(self, symbol: str) -> Optional[PriceTick]:
        """Get the latest cached price (for fast lookups)."""
        for sym, ticks in self._replay_data.items():
            if sym == symbol and self._replay_pos.get(sym, 0) > 0:
                idx = min(self._replay_pos[sym] - 1, len(ticks) - 1)
                return ticks[idx]
        return None

    async def prepare_futures_data(self, symbols: List[str], days: int = 7):
        """Pre-download futures data so it's ready when streaming starts."""
        logger.info(f"Pre-downloading data for {symbols}...")
        for symbol in symbols:
            if self.is_futures_symbol(symbol):
                ticks = await self._yahoo.get_ticks(symbol, days=days)
                if ticks:
                    self._replay_data[symbol] = ticks
                    self._replay_pos[symbol] = 0
                    logger.info(f"  {symbol}: {len(ticks)} ticks ready")
                else:
                    logger.warning(f"  {symbol}: No data available")

    async def close(self):
        """Clean up resources."""
        self._running = False
        for task in self._poll_tasks.values():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, RuntimeError):
                pass
        if self._ctrader:
            await self._ctrader.close()
        if self._dukascopy:
            await self._dukascopy.close()
        logger.info("MultiSource provider closed")
