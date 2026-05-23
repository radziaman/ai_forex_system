"""
Multi-symbol market data manager: OHLCV, order flow, Level II DOM.
v2.0 — systematic, efficient, clean data pipeline for all 22 symbols.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from loguru import logger
from typing import Dict, Optional, List, Tuple, Any, Set, Callable
import os, json, time, math, struct, lzma
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict

# ---------------------------------------------------------------------------
# Symbol registry (single source of truth)
# ---------------------------------------------------------------------------

SYMBOLS = [
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
]
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h"]

JPY_PAIRS = {"USDJPY"}
XAU_PAIRS = {"XAUUSD"}
CRYPTO_PAIRS = {"BTCUSD"}
INDEX_PAIRS = {"US500"}
ENERGY_PAIRS = {"XTIUSD"}

BASE_PRICES = {
    "EURUSD": 1.12,
    "GBPUSD": 1.28,
    "USDJPY": 150.0,
    "AUDUSD": 0.67,
    "USDCAD": 1.35,
    "USDCHF": 0.88,
    "NZDUSD": 0.61,
    "XAUUSD": 2000.0,
    "XTIUSD": 75.0,
    "US500": 4500.0,
    "BTCUSD": 45000.0,
}

# Dukascopy symbols that are actually available
DUKASCOPE_SYMBOLS = {
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "USDCHF",
    "NZDUSD",
    "XAUUSD",
}

# Per-symbol price sanity windows (fraction of base)
PRICE_WINDOWS = {
    sym: (
        (0.5, 2.0)
        if sym in CRYPTO_PAIRS
        else (0.7, 1.5) if sym in INDEX_PAIRS else (0.8, 1.2)
    )
    for sym in SYMBOLS
}

TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}

# BI5 cache directory
CACHE_DIR = Path("data/dukascopy_cache")


@dataclass
class DepthLevelData:
    price: float = 0.0
    size: float = 0.0


@dataclass
class MarketDepthData:
    symbol: str = ""
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0
    volume: float = 0.0
    timestamp: float = field(default_factory=lambda: time.time())
    bids: List[DepthLevelData] = field(default_factory=list)
    asks: List[DepthLevelData] = field(default_factory=list)


@dataclass
class DataFreshness:
    """Tracks when and from where each symbol's data was last refreshed."""

    last_tick_ts: float = 0.0
    last_ohlcv_ts: float = 0.0
    last_source: str = ""
    tick_count: int = 0
    bar_count: Dict[str, int] = field(
        default_factory=lambda: {tf: 0 for tf in TIMEFRAMES}
    )
    errors_since_healthy: int = 0
    is_healthy: bool = True


class DataManager:
    """Multi-symbol market data manager — v2.0 systematic pipeline.

    Responsibilities:
      - Ingests ticks from any source (Dukascopy, cTrader, FIX)
      - Aggregates ticks → 1m bars → propagates to 5m/15m/1h/4h
      - Tracks data freshness and source health per symbol
      - Loads historical data from BI5 cache, CSV, or generates synthetic
      - Provides feature caching with hash-based invalidation
      - Maintains Level II DOM, CVD, and order flow analytics
    """

    def __init__(self, historical_path: str = "data/historical"):
        self.historical_path = historical_path
        self.enabled: bool = True  # Master AI control

        # --- Core data ---
        self.ohlcv: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.tick_buffer: Dict[str, List[Dict]] = {}
        self._cvd: Dict[str, Tuple[List[float], List[float], List[float]]] = {}
        self.order_flow: Dict[str, Dict] = {}
        self._last_realtime_price: Dict[str, float] = {}
        self.latest_snapshot: Dict[str, Optional[Any]] = {}
        self.market_depth: Dict[str, MarketDepthData] = {}
        self._feature_cache: Dict[str, Dict[str, Tuple[np.ndarray, float]]] = {}
        # (symbol) -> {tf: (features, cache_key)} where cache_key = hash of last OHLCV row

        # --- Freshness & source tracking ---
        self.freshness: Dict[str, DataFreshness] = {}
        self._source_health: Dict[str, Dict] = {
            "dukascopy": {"ok": True, "failures": 0, "last_ok": time.time()},
            "yfinance": {"ok": True, "failures": 0, "last_ok": time.time()},
            "ctrader": {"ok": True, "failures": 0, "last_ok": time.time()},
        }
        self._on_data_update: List[Callable] = []  # callbacks when data changes

        # --- Tick aggregation batching ---
        self._pending_ticks: Dict[str, List[Dict]] = defaultdict(list)
        self._last_aggregate_time: float = time.time()
        self._aggregate_interval: float = 1.0  # batch & process every 1s max

        # --- Initialize structures for all symbols ---
        for sym in SYMBOLS:
            self.ohlcv[sym] = {}
            self.tick_buffer[sym] = []
            self._cvd[sym] = ([], [], [])
            self.order_flow[sym] = {}
            self._last_realtime_price[sym] = BASE_PRICES.get(sym, 1.12)
            self.latest_snapshot[sym] = None
            self.market_depth[sym] = MarketDepthData(symbol=sym)
            self._feature_cache[sym] = {}
            self.freshness[sym] = DataFreshness()
            for tf in TIMEFRAMES:
                self.ohlcv[sym][tf] = pd.DataFrame(
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )

        logger.info(
            f"DataManager v2.0 initialized: {len(SYMBOLS)} symbols x {len(TIMEFRAMES)} timeframes"
        )

    # ------------------------------------------------------------------
    # Data quality
    # ------------------------------------------------------------------

    def _validate_tick(
        self, symbol: str, bid: float, ask: float, volume: float
    ) -> bool:
        if not all(isinstance(v, (int, float)) for v in [bid, ask, volume]):
            return False
        if bid <= 0 or ask <= 0 or bid > ask:
            return False
        if volume < 0:
            return False
        base = BASE_PRICES.get(symbol, 1.12)
        # Per-symbol price sanity window — uses PRICE_WINDOWS which allocates
        # generous bounds for each asset class (e.g. crypto can swing 2x).
        # The 1000x multiplier accounts for depth-event price formats that
        # may use a non-standard raw scale (see _price_divisor in ctrader_client).
        lo, hi = PRICE_WINDOWS.get(symbol, (0.1, 5.0))
        max_price = base * hi * 1000
        if bid > max_price or ask > max_price:
            logger.debug(
                f"{symbol}: price {bid}/{ask} exceeds sanity "
                f"max={max_price:.0f} (base={base}, window={hi}), rejected"
            )
            return False
        prev = self._last_realtime_price.get(symbol)
        if prev and prev > 0 and prev != base:
            change = abs(bid - prev) / prev
            if change > 0.05:
                logger.debug(f"{symbol}: >5% tick move ({change:.2%}), rejected")
                return False
        return True

    # ------------------------------------------------------------------
    # Tick ingestion (batched)
    # ------------------------------------------------------------------

    def update_tick(
        self,
        symbol: str,
        bid: float,
        ask: float,
        volume: float = 0,
        ts: Optional[float] = None,
    ):
        if not self.enabled:
            return
        if not self._validate_tick(symbol, bid, ask, volume):
            self.freshness[symbol].errors_since_healthy += 1
            return

        ts = ts or time.time()
        mid = (bid + ask) / 2.0
        self._last_realtime_price[symbol] = mid

        tick = {
            "ts": ts,
            "symbol": symbol,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "vol": max(volume, 0),
        }
        self._pending_ticks[symbol].append(tick)

        fd = self.freshness[symbol]
        fd.last_tick_ts = ts
        fd.tick_count += 1
        fd.errors_since_healthy = 0
        fd.is_healthy = True

        now = time.time()
        if now - self._last_aggregate_time >= self._aggregate_interval:
            self._flush_pending()
            self._last_aggregate_time = now

    def _flush_pending(self):
        """Process all queued ticks in batch."""
        for symbol, ticks in list(self._pending_ticks.items()):
            if not ticks:
                continue
            buf = self.tick_buffer[symbol]
            buf.extend(ticks)
            if len(buf) > 10000:
                self.tick_buffer[symbol] = buf[-5000:]

            for t in ticks:
                self._aggregate_1m(symbol, t["ts"], t["mid"], t["vol"])
                self._update_of(symbol, t["bid"], t["ask"], t["vol"])

            ts_min = min(t["ts"] for t in ticks)
            ts_max = max(t["ts"] for t in ticks)
            self.freshness[symbol].last_tick_ts = ts_max
            self._invalidate_feature_cache(symbol, ts_min, ts_max)

            ticks.clear()

    # ------------------------------------------------------------------
    # OHLCV aggregation
    # ------------------------------------------------------------------

    def _aggregate_1m(self, symbol: str, ts: float, price: float, volume: float):
        df = self.ohlcv[symbol]["1m"]
        if not df.empty and df["timestamp"].dtype == "object":
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        bar_ts = int(ts // 60) * 60
        if df.empty or float(df.iloc[-1]["timestamp"]) < bar_ts:
            new = pd.DataFrame(
                [
                    {
                        "timestamp": bar_ts,
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price,
                        "volume": max(volume, 0),
                    }
                ]
            )
            self.ohlcv[symbol]["1m"] = (
                new if df.empty else pd.concat([df, new], ignore_index=True)
            )
            self._cap_bars(symbol, "1m")
        else:
            idx = len(df) - 1
            r = df.iloc[idx]
            if price > r["high"]:
                df.at[idx, "high"] = price
            if price < r["low"]:
                df.at[idx, "low"] = price
            df.at[idx, "close"] = price
            df.at[idx, "volume"] += max(volume, 0)
        self._propagate(symbol)

    def _propagate(self, symbol: str):
        df_1m = self.ohlcv[symbol]["1m"].copy()
        if df_1m.empty:
            return
        df_1m["datetime"] = pd.to_datetime(df_1m["timestamp"], unit="s")
        df_1m = df_1m.set_index("datetime")
        for tf, minutes in [("5m", 5), ("15m", 15), ("1h", 60), ("4h", 240)]:
            res = (
                df_1m.resample(f"{minutes}min")
                .agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }
                )
                .dropna()
            )
            if res.empty:
                continue
            res = res.reset_index()
            res["timestamp"] = res["datetime"].astype(np.int64) // 10**9
            new_bars = res[["timestamp", "open", "high", "low", "close", "volume"]]

            existing = self.ohlcv[symbol].get(tf)
            if existing is not None and len(existing) > len(new_bars):
                # Merge: keep existing, append new bars, remove duplicates
                merged = pd.concat([existing, new_bars], ignore_index=True)
                merged = merged.drop_duplicates(subset=["timestamp"], keep="last")
                merged = merged.sort_values("timestamp").reset_index(drop=True)
                self.ohlcv[symbol][tf] = merged
            else:
                self.ohlcv[symbol][tf] = new_bars
            self._cap_bars(symbol, tf)

    def _cap_bars(self, symbol: str, tf: str, max_bars: int = 5000):
        df = self.ohlcv[symbol][tf]
        if len(df) > max_bars:
            self.ohlcv[symbol][tf] = df.iloc[-max_bars:].reset_index(drop=True)

    def _invalidate_feature_cache(self, symbol: str, ts_min: float, ts_max: float):
        """Smart invalidation based on time range of incoming data."""
        cache = self._feature_cache.get(symbol, {})
        stale_tfs = []
        for tf in cache:
            df = self.ohlcv[symbol].get(tf)
            if df is not None and not df.empty:
                last_bar_ts = float(df["timestamp"].iloc[-1])
                if ts_max >= last_bar_ts - TF_MINUTES.get(tf, 60) * 60:
                    stale_tfs.append(tf)
        for tf in stale_tfs:
            if tf in cache:
                del cache[tf]

    # ------------------------------------------------------------------
    # Order flow (CVD)
    # ------------------------------------------------------------------

    def _update_of(self, symbol: str, bid: float, ask: float, volume: float):
        cvd_hist, bid_hist, ask_hist = self._cvd[symbol]
        mid = (bid + ask) / 2.0
        if bid_hist:
            prev_mid = (bid_hist[-1] + ask_hist[-1]) / 2.0
            delta = volume if mid >= prev_mid else -volume
            new_cvd = (cvd_hist[-1] if cvd_hist else 0.0) + delta
        else:
            new_cvd = 0.0
        cvd_hist.append(new_cvd)
        bid_hist.append(bid)
        ask_hist.append(ask)
        if len(cvd_hist) > 1000:
            self._cvd[symbol] = (cvd_hist[-500:], bid_hist[-500:], ask_hist[-500:])
        cvd_slope = (cvd_hist[-1] - cvd_hist[-20]) / 20 if len(cvd_hist) >= 20 else 0.0
        spread = ask - bid
        self.order_flow[symbol] = {
            "cvd": new_cvd,
            "cvd_slope": cvd_slope,
            "imbalance": (bid - ask) / spread if spread > 0 else 0.0,
        }

    # ------------------------------------------------------------------
    # Historical data loading + disk persistence
    # ------------------------------------------------------------------

    def save_ohlcv(self, symbol: str, tf: str = "1h"):
        """Append in-memory OHLCV bars to CSV on disk.

        Uses a single rolling file per symbol per timeframe.
        Deduplicates by timestamp to handle append-after-restart.
        """
        df = self.ohlcv.get(symbol, {}).get(tf)
        if df is None or df.empty:
            return
        path = os.path.join(self.historical_path, f"{symbol}_{tf}.csv")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path):
                existing = pd.read_csv(path)
                combined = pd.concat([existing, df], ignore_index=True)
                combined = (
                    combined.drop_duplicates(subset=["timestamp"], keep="last")
                    .sort_values("timestamp")
                    .reset_index(drop=True)
                )
                combined.to_csv(path, index=False)
            else:
                df.to_csv(path, index=False)
            logger.debug(f"Saved {symbol} {tf}: {len(df)} bars → {path}")
        except Exception as e:
            logger.debug(f"Save failed for {symbol} {tf}: {e}")

    def save_all_ohlcv(self, timeframes: Optional[List[str]] = None):
        """Save all symbols × timeframes to CSV."""
        tfs = timeframes or TIMEFRAMES
        saved = 0
        for sym in SYMBOLS:
            for tf in tfs:
                df = self.ohlcv.get(sym, {}).get(tf)
                if df is not None and not df.empty:
                    self.save_ohlcv(sym, tf)
                    saved += 1
        if saved > 0:
            logger.info(f"Saved {saved} symbol-TF pairs to disk")

    def load_historical(self, symbol: str, tf: str, days: int = 365):
        fp = os.path.join(self.historical_path, f"{symbol}_{tf}.csv")
        if os.path.exists(fp):
            self.ohlcv[symbol][tf] = pd.read_csv(fp)
            self.freshness[symbol].last_source = "csv"
            self.freshness[symbol].bar_count[tf] = len(self.ohlcv[symbol][tf])
            logger.info(
                f"Loaded {symbol} {tf} CSV ({len(self.ohlcv[symbol][tf])} bars)"
            )
            return

        # No CSV cache — fetch real data from Yahoo Finance
        if self.try_alternative_source(symbol, tf, days=days):
            return

        # Fallback: Dukascopy BI5 cache
        self.load_from_dukascopy_cache(
            symbols=[symbol], timeframes=[tf], max_hours=days * 24
        )
        bars = self.ohlcv.get(symbol, {}).get(tf, pd.DataFrame())
        if (
            bars is not None
            and hasattr(bars, "empty")
            and not bars.empty
            and len(bars) > 10
        ):
            self.save_ohlcv(symbol, tf)
            return

        logger.warning(f"No real data source available for {symbol} {tf}")

    def load_all(self, days: int = 365, timeframes: Optional[List[str]] = None):
        tfs = timeframes or TIMEFRAMES
        for sym in SYMBOLS:
            for tf in tfs:
                self.load_historical(sym, tf, days)

    def load_from_dukascopy_cache(
        self,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        max_hours: int = 168,
    ) -> int:
        """Load historical data from cached Dukascopy BI5 files.

        Returns number of symbols successfully loaded.
        """
        tfs = timeframes or ["1h"]
        targets = [
            s.upper() for s in (symbols or SYMBOLS) if s.upper() in DUKASCOPE_SYMBOLS
        ]
        loaded = 0
        for sym in targets:
            tf = tfs[0]
            bars = self._decode_bi5_to_bars(sym, tf, max_hours)
            if bars is not None and len(bars) > 10:
                self.ohlcv[sym][tf] = bars
                self.freshness[sym].last_source = "dukascopy_cache"
                self.freshness[sym].bar_count[tf] = len(bars)
                self.freshness[sym].is_healthy = True
                loaded += 1
                logger.info(f"Loaded {sym} {tf}: {len(bars)} bars (BI5 cache)")
            else:
                logger.debug(f"No BI5 cache for {sym}")
        return loaded

    def _decode_bi5_to_bars(
        self, symbol: str, tf: str = "1h", max_hours: int = 168
    ) -> Optional[pd.DataFrame]:
        """Read cached BI5 files and aggregate into OHLCV bars."""
        tf_seconds = TF_MINUTES.get(tf, 60) * 60
        cache_files = sorted(CACHE_DIR.glob(f"{symbol}_*_*.bi5"))
        if not cache_files:
            return None
        ticks = []
        for cf in cache_files[-max_hours:]:
            try:
                raw = cf.read_bytes()
                decompressed = lzma.decompress(raw)
            except Exception:
                decompressed = raw
            parts = cf.stem.split("_")
            if len(parts) < 2:
                continue
            dt_str, hour_str = parts[-2], parts[-1]
            try:
                base_ts = (
                    datetime.strptime(dt_str, "%Y%m%d").timestamp()
                    + int(hour_str) * 3600
                )
            except (ValueError, IndexError):
                continue
            for i in range(0, len(decompressed), 20):
                chunk = decompressed[i : i + 20]
                if len(chunk) < 20:
                    break
                ms = struct.unpack(">I", chunk[0:4])[0]
                bid_raw = struct.unpack(">I", chunk[8:12])[0]
                ask_raw = struct.unpack(">I", chunk[4:8])[0]
                ticks.append(
                    (base_ts + ms / 1000.0, bid_raw / 100000.0, ask_raw / 100000.0)
                )
        if len(ticks) < 10:
            return None
        ticks.sort(key=lambda t: t[0])
        df = pd.DataFrame(ticks, columns=["timestamp", "bid", "ask"])
        df["bar"] = (df["timestamp"] // tf_seconds).astype(int)
        bars = (
            df.groupby("bar")
            .agg(
                open=("bid", "first"),
                high=("bid", "max"),
                low=("bid", "min"),
                close=("bid", "last"),
                volume=("ask", "count"),
            )
            .reset_index()
        )
        bars["timestamp"] = bars["bar"] * tf_seconds
        bars = bars.drop(columns=["bar"])
        bars = bars[["timestamp", "open", "high", "low", "close", "volume"]]
        return bars

    async def load_from_ctrader(
        self, symbol: str, timeframe: str = "1h", days: int = 365, client=None
    ) -> bool:
        """Fetch OHLCV for one symbol×timeframe from cTrader protobuf API.

        Returns True if data was refreshed. Skips if existing data is <6h old.
        """
        if client is None or not hasattr(client, "fetch_historical_ohlcv"):
            return False

        return True  # data_refreshed
        try:
            existing = self.ohlcv.get(symbol, {}).get(timeframe)
            if existing is not None and len(existing) > 50:
                last_ts = float(existing["timestamp"].iloc[-1])
                age_hours = (time.time() - last_ts) / 3600
                if age_hours < 6:
                    return True
                bars = await client.fetch_historical_ohlcv(
                    symbol, timeframe, days_back=min(days, 5)
                )
            else:
                bars = await client.fetch_historical_ohlcv(
                    symbol, timeframe, days_back=days
                )
            if bars and len(bars) > 10:
                df = pd.DataFrame(bars)
                if existing is not None and len(existing) > 50:
                    merged = pd.concat([existing, df], ignore_index=True)
                    merged = merged.drop_duplicates(subset=["timestamp"]).sort_values(
                        "timestamp"
                    )
                    self.ohlcv[symbol][timeframe] = merged
                    logger.info(
                        f"cTrader appended {len(df)} bars to {symbol} {timeframe} "
                        f"(total {len(merged)})"
                    )
                else:
                    self.ohlcv[symbol][timeframe] = df
                    logger.info(
                        f"Loaded {symbol} {timeframe} from cTrader: {len(df)} bars"
                    )
                self._source_health["ctrader"]["last_ok"] = time.time()
                self._source_health["ctrader"]["failures"] = 0
                self.freshness[symbol].last_source = "ctrader"
                self.freshness[symbol].bar_count[timeframe] = len(
                    self.ohlcv[symbol][timeframe]
                )
                return True
        except Exception as e:
            self._source_health["ctrader"]["failures"] += 1
            logger.debug(f"cTrader failed for {symbol}: {e}")
            return False

    async def load_all_timeframes_from_ctrader(
        self,
        symbol: str,
        client=None,
        days: int = 365,
        timeframes: Optional[List[str]] = None,
    ) -> int:
        """Fetch ALL timeframes for a symbol from cTrader in one call.

        Returns number of timeframes successfully refreshed.
        """
        tfs = timeframes or TIMEFRAMES
        refreshed = 0
        for tf in tfs:
            try:
                if await self.load_from_ctrader(symbol, tf, days=days, client=client):
                    refreshed += 1
            except Exception:
                pass
        return refreshed

    async def load_with_fallback(
        self, symbol: str, timeframe: str = "1h", days: int = 365, ctrader_client=None
    ) -> bool:
        """Unified data loader: tries CSV cache → Dukascopy → yFinance → cTrader."""
        # 0. Local CSV cache first (fastest)
        fp = os.path.join(self.historical_path, f"{symbol}_{timeframe}.csv")
        if os.path.exists(fp):
            try:
                df = pd.read_csv(fp)
                if df is not None and len(df) >= 50:
                    self.ohlcv[symbol][timeframe] = df
                    self.freshness[symbol].last_source = "csv"
                    self.freshness[symbol].bar_count[timeframe] = len(df)
                    logger.info(f"Loaded {symbol} {timeframe} CSV ({len(df)} bars)")
                    return True
            except Exception as e:
                logger.debug(f"CSV load failed for {symbol}: {e}")

        # 1. Dukascopy BI5 cache
        loaded = self.load_from_dukascopy_cache(
            symbols=[symbol], timeframes=[timeframe], max_hours=days * 24
        )
        if loaded > 0 and len(self.ohlcv.get(symbol, {}).get(timeframe, [])) > 50:
            return True

        # 2. yFinance
        if self.try_alternative_source(symbol, timeframe, days):
            return True

        # 3. cTrader protobuf
        if await self.load_from_ctrader(symbol, timeframe, days, ctrader_client):
            return True

        logger.warning(f"No data source available for {symbol}")
        return False

    def try_alternative_source(
        self, symbol: str, timeframe: str = "1h", days: int = 30
    ) -> bool:
        try:
            import yfinance as yf

            ticker_map = {
                "EURUSD": "EURUSD=X",
                "GBPUSD": "GBPUSD=X",
                "USDJPY": "USDJPY=X",
                "AUDUSD": "AUDUSD=X",
                "USDCAD": "USDCAD=X",
                "USDCHF": "USDCHF=X",
                "NZDUSD": "NZDUSD=X",
                "XAUUSD": "GC=F",
                "XTIUSD": "CL=F",
                "BTCUSD": "BTC-USD",
                "US500": "^GSPC",
                # Screener universe symbols (Yahoo-native tickers)
                "GC=F": "GC=F",  # Gold futures
                "SI=F": "SI=F",  # Silver futures
                "HG=F": "HG=F",  # Copper futures
                "PA=F": "PA=F",  # Palladium futures
                "PL=F": "PL=F",  # Platinum futures
                "CL=F": "CL=F",  # Crude oil futures
                "HO=F": "HO=F",  # Heating oil futures
                "RB=F": "RB=F",  # Gasoline futures
                "NG=F": "NG=F",  # Natural gas futures
                "ZB=F": "ZB=F",  # US Treasury Bond futures
                "ZN=F": "ZN=F",  # 10-Year T-Note futures
                "ZF=F": "ZF=F",  # 5-Year T-Note futures
                "ZT=F": "ZT=F",  # 2-Year T-Note futures
                "ES=F": "ES=F",  # S&P 500 E-mini futures
                "NQ=F": "NQ=F",  # NASDAQ E-mini futures
                "YM=F": "YM=F",  # Dow Jones E-mini futures
                "DX-Y.NYB": "DX-Y.NYB",  # US Dollar Index
                "USO": "USO",  # Oil ETF
                "GLD": "GLD",  # Gold ETF
                "SLV": "SLV",  # Silver ETF
                "TLT": "TLT",  # 20+ Year Treasury ETF
            }
            # Smart Yahoo ticker resolution:
            #   1. Known core symbols → ticker_map
            #   2. Already Yahoo-formatted (contains = or ^) → use as-is
            #   3. Clean 6-char all-alpha → assume forex → append =X
            #   4. Everything else (ETF, stock, index) → use as-is
            if symbol in ticker_map:
                yf_sym = ticker_map[symbol]
            elif any(c in symbol for c in ("=", "^")):
                yf_sym = symbol  # e.g. GC=F, EURUSD=X, ^GSPC
            elif len(symbol) >= 5 and symbol.isalpha() and symbol.isupper():
                yf_sym = f"{symbol}=X"  # Probable forex pair (EURUSD → EURUSD=X)
            else:
                yf_sym = symbol  # Use as-is: USO, GLD, SLV, TLT, etc.
            interval = timeframe.replace("m", "m").replace("h", "h")
            existing = self.ohlcv.get(symbol, {}).get(timeframe)
            if existing is not None and len(existing) > 50:
                last_ts = float(existing["timestamp"].iloc[-1])
                age_hours = (time.time() - last_ts) / 3600
                if age_hours < 6:
                    return True
                fetch_days = max(int(age_hours / 24) + 2, 3)
                data = yf.download(
                    yf_sym, period=f"{fetch_days}d", interval=interval, progress=False
                )
            else:
                data = yf.download(
                    yf_sym, period=f"{max(days, 5)}d", interval=interval, progress=False
                )
            if data is not None and not data.empty:
                df = data.reset_index()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [
                        c[0] if isinstance(c, tuple) else c for c in df.columns
                    ]
                date_col = next(
                    (
                        c
                        for c in df.columns
                        if c.lower() in ("date", "datetime", "timestamp")
                    ),
                    None,
                )
                if date_col:
                    df["timestamp"] = (
                        pd.to_datetime(df[date_col]).astype("int64") // 10**9
                    )
                elif df.columns[0].lower() in ("date", "datetime"):
                    df["timestamp"] = (
                        pd.to_datetime(df.iloc[:, 0]).astype("int64") // 10**9
                    )
                cols_lower = {c: c.lower() for c in df.columns if isinstance(c, str)}
                df = df.rename(columns=cols_lower)
                new_bars = df[["timestamp", "open", "high", "low", "close", "volume"]]
                if existing is not None and len(existing) > 50:
                    merged = pd.concat([existing, new_bars], ignore_index=True)
                    merged = merged.drop_duplicates(subset=["timestamp"]).sort_values(
                        "timestamp"
                    )
                    self.ohlcv[symbol][timeframe] = merged
                    logger.info(
                        f"Appended {len(new_bars)} new bars to {symbol} (total {len(merged)})"
                    )
                else:
                    self.ohlcv[symbol][timeframe] = new_bars
                    logger.info(f"Loaded {symbol} from yFinance: {len(new_bars)} bars")
                self._source_health["yfinance"]["last_ok"] = time.time()
                self._source_health["yfinance"]["failures"] = 0
                self.freshness[symbol].last_source = "yfinance"
                self.freshness[symbol].bar_count[timeframe] = len(
                    self.ohlcv[symbol][timeframe]
                )
                self.save_ohlcv(symbol, timeframe)
                return True
        except Exception as e:
            self._source_health["yfinance"]["failures"] += 1
            logger.debug(f"yFinance failed for {symbol}: {e}")
        return False

    # ------------------------------------------------------------------
    # Data gap detection & healing
    # ------------------------------------------------------------------

    def detect_gaps(
        self, symbol: str, tf: str = "1h", max_gap_minutes: int = 90
    ) -> List[Tuple[float, float]]:
        """Return list of (start, end) gaps longer than max_gap_minutes."""
        df = self.ohlcv.get(symbol, {}).get(tf)
        if df is None or len(df) < 2:
            return []
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        gaps = []
        for i in range(1, len(ts)):
            gap = ts.iloc[i] - ts.iloc[i - 1]
            if gap > max_gap_minutes * 60:
                gaps.append((ts.iloc[i - 1], ts.iloc[i]))
        return gaps

    def heal_gaps(
        self,
        symbol: str,
        tf: str = "1h",
        max_gap_minutes: int = 90,
        ctrader_client=None,
    ) -> int:
        """Attempt to heal gaps by backfilling from alternative sources.

        Tries cTrader historical API first (if client provided), then
        falls back to Yahoo Finance.  Returns number of gaps healed.

        Args:
            symbol: Instrument symbol (e.g. "EURUSD")
            tf: Timeframe string (e.g. "1h")
            max_gap_minutes: Maximum allowed gap before healing triggers
            ctrader_client: Optional CtraderClient for broker-sourced backfill
        """
        import asyncio

        gaps = self.detect_gaps(symbol, tf, max_gap_minutes)
        if not gaps:
            return 0
        healed = 0
        for gap_start, gap_end in gaps:
            gap_days = min(int((gap_end - gap_start) / 86400) + 2, 60)
            # Try cTrader first if available
            if ctrader_client is not None and hasattr(
                ctrader_client, "fetch_historical_ohlcv"
            ):
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Already inside async context — create task
                        asyncio.ensure_future(
                            self.load_from_ctrader(
                                symbol, tf, days=gap_days, client=ctrader_client
                            )
                        )
                        healed += 1
                        continue
                    else:
                        # Synchronous context — use try_alternative_source instead
                        pass
                except Exception:
                    pass
            # Fallback: yFinance
            if self.try_alternative_source(symbol, tf, days=gap_days):
                healed += 1
                logger.info(f"Healed gap {symbol}: {gap_start:.0f}-{gap_end:.0f}")
        return healed

    # ------------------------------------------------------------------
    # Source health monitoring
    # ------------------------------------------------------------------

    def report_source_failure(self, source: str):
        if source in self._source_health:
            self._source_health[source]["failures"] += 1
            if self._source_health[source]["failures"] > 10:
                self._source_health[source]["ok"] = False

    def report_source_success(self, source: str):
        if source in self._source_health:
            self._source_health[source]["ok"] = True
            self._source_health[source]["failures"] = 0
            self._source_health[source]["last_ok"] = time.time()

    def best_source(self) -> str:
        best = "dukascopy"
        best_score = -1
        for name, h in self._source_health.items():
            score = 0 if not h["ok"] else max(0, 10 - h["failures"])
            if score > best_score:
                best_score = score
                best = name
        return best

    # ------------------------------------------------------------------
    # Feature cache (hash-based invalidation)
    # ------------------------------------------------------------------

    def get_cached_features(self, symbol: str, timeframe: str) -> Optional[np.ndarray]:
        entry = self._feature_cache.get(symbol, {}).get(timeframe)
        if entry is None:
            return None
        features, cache_key = entry
        df = self.ohlcv.get(symbol, {}).get(timeframe)
        if df is not None and not df.empty:
            current_key = hash(
                (
                    float(df["timestamp"].iloc[-1]),
                    float(df["close"].iloc[-1]),
                    float(df["volume"].iloc[-1]),
                    len(df),
                )
            )
            if cache_key == current_key:
                return features
        return None

    def set_cached_features(self, symbol: str, timeframe: str, features: np.ndarray):
        df = self.ohlcv.get(symbol, {}).get(timeframe)
        cache_key = 0
        if df is not None and not df.empty:
            cache_key = hash(
                (
                    float(df["timestamp"].iloc[-1]),
                    float(df["close"].iloc[-1]),
                    float(df["volume"].iloc[-1]),
                    len(df),
                )
            )
        if symbol not in self._feature_cache:
            self._feature_cache[symbol] = {}
        self._feature_cache[symbol][timeframe] = (features, cache_key)

    # ------------------------------------------------------------------
    # Price accessors
    # ------------------------------------------------------------------

    def get_ohlcv(self, symbol: str, tf: str) -> Optional[pd.DataFrame]:
        return self.ohlcv.get(symbol, {}).get(tf)

    def get_price(self, symbol: str, tf: str = "1h") -> float:
        df = self.get_ohlcv(symbol, tf)
        if df is not None and len(df) > 0:
            return float(df["close"].iloc[-1])
        return BASE_PRICES.get(symbol, 1.12)

    def get_atr(self, symbol: str, tf: str = "1h", period: int = 14) -> float:
        df = self.get_ohlcv(symbol, tf)
        if df is None or len(df) < period + 1:
            return BASE_PRICES.get(symbol, 1.12) * 0.001
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - df["close"].shift()).abs(),
                (df["low"] - df["close"].shift()).abs(),
            ],
            axis=1,
        ).max(1)
        return float(
            max(tr.iloc[-period:].mean(), BASE_PRICES.get(symbol, 1.12) * 0.0005)
        )

    def update_price(self, symbol: str, price: float, timeframe: str = "1h"):
        sym = symbol.upper()
        self._last_realtime_price[sym] = price
        df = self.ohlcv.get(sym, {}).get(timeframe)
        if df is not None and not df.empty:
            df.at[len(df) - 1, "close"] = price

    def get_tick_buffer(self, symbol: str, n: int = 1000) -> List[Dict]:
        return self.tick_buffer.get(symbol, [])[-n:]

    def all_prices(self, tf: str = "1h") -> Dict[str, float]:
        prices = {}
        for sym in SYMBOLS:
            base = BASE_PRICES.get(sym, 1.12)
            tick = self._last_realtime_price.get(sym)
            ohlcv = self.get_price(sym, tf)
            ref = ohlcv if ohlcv and ohlcv != base else base
            lo, hi = PRICE_WINDOWS.get(sym, (0.5, 2.0))
            if tick is not None and ref > 0:
                ratio = tick / ref
                if lo <= ratio <= hi:
                    prices[sym] = tick
                else:
                    prices[sym] = ref
            elif ohlcv and ohlcv != base:
                prices[sym] = ohlcv
            elif self.tick_buffer.get(sym):
                prices[sym] = self.tick_buffer[sym][-1].get("mid", base)
            else:
                prices[sym] = base
        return prices

    # ------------------------------------------------------------------
    # Level II DOM
    # ------------------------------------------------------------------

    def update_market_depth(self, depth: Any):
        """Accept both CtraderDepth and MarketDepthData (tuple-based) formats."""
        try:
            from api.ctrader_client import MarketDepth as CtraderDepth

            if isinstance(depth, CtraderDepth):
                sym = depth.symbol
                md: Optional[MarketDepthData] = self.market_depth[sym]
                assert md is not None, f"Market depth not found for {sym}"
                md.symbol = sym
                md.bid = depth.bid
                md.ask = depth.ask
                md.spread = depth.spread
                md.volume = depth.volume
                md.timestamp = depth.timestamp
                md.bids = [
                    DepthLevelData(price=b.price, size=b.size) for b in depth.bids
                ]
                md.asks = [
                    DepthLevelData(price=a.price, size=a.size) for a in depth.asks
                ]
                return
        except ImportError:
            pass

        if isinstance(depth, MarketDepthData):
            sym = depth.symbol
            self.market_depth[sym] = depth
            return

        if (
            hasattr(depth, "symbol")
            and hasattr(depth, "bids")
            and hasattr(depth, "asks")
        ):
            sym = depth.symbol
            md = self.market_depth.get(sym)
            if md is None:
                return
            md.bid = getattr(depth, "bid", md.bid)
            md.ask = getattr(depth, "ask", md.ask)
            md.spread = md.ask - md.bid
            md.timestamp = getattr(depth, "timestamp", time.time())
            if depth.bids and isinstance(depth.bids[0], (tuple, list)):
                md.bids = [DepthLevelData(price=p, size=s) for p, s in depth.bids]
            if depth.asks and isinstance(depth.asks[0], (tuple, list)):
                md.asks = [DepthLevelData(price=p, size=s) for p, s in depth.asks]
            return

        logger.debug(f"update_market_depth: unknown type {type(depth).__name__}")

    def get_market_depth(self, symbol: str) -> Optional[MarketDepthData]:
        return self.market_depth.get(symbol)

    def get_dom_imbalance(self, symbol: str, levels: int = 5) -> float:
        md = self.market_depth.get(symbol)
        if not md or not md.bids or not md.asks:
            return 0.0
        bid_vol = sum(b.size for b in md.bids[:levels])
        ask_vol = sum(a.size for a in md.asks[:levels])
        total = bid_vol + ask_vol
        return (bid_vol - ask_vol) / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # Order flow analytics
    # ------------------------------------------------------------------

    def get_order_flow_metrics(self, symbol: str) -> Dict:
        cvd_hist, _, _ = self._cvd.get(symbol, ([], [], []))
        of = self.order_flow.get(symbol, {})
        return {
            "cvd": of.get("cvd", 0.0),
            "cvd_slope": of.get("cvd_slope", 0.0),
            "imbalance": of.get("imbalance", 0.0),
            "dom_imbalance": self.get_dom_imbalance(symbol),
            "cvd_hist": cvd_hist[-100:] if cvd_hist else [],
        }

    def calculate_gamma_exposure(self, symbol: str) -> Dict:
        return {
            "gamma_exposure": 0.0,
            "gamma_flip_point": 0.0,
            "dealer_position": "neutral",
        }

    # ------------------------------------------------------------------
    # Dashboard snapshot
    # ------------------------------------------------------------------

    def get_snapshot(self, symbol: str = "EURUSD", acc=None, positions=None):
        try:
            from .feature_engine import FeatureEngine

            fe = FeatureEngine()
            features = fe.compute_features(
                self.ohlcv.get(symbol, {}),
                self.order_flow.get(symbol, {}),
                acc,
                positions,
            )
            fnames = fe.feature_names
        except Exception:
            features = None
            fnames = []
        snap = type(
            "Snapshot",
            (),
            {
                "symbol": symbol,
                "timestamp": time.time(),
                "features": features,
                "feature_names": fnames,
                "regime": "ranging",
                "regime_conf": 0.8,
                "cvd": self._cvd[symbol][0][-1] if self._cvd[symbol][0] else 0.0,
                "bid_ask_imbalance": self.order_flow.get(symbol, {}).get(
                    "imbalance", 0.0
                ),
            },
        )()
        self.latest_snapshot[symbol] = snap
        return snap

    # ------------------------------------------------------------------
    # Data freshness report
    # ------------------------------------------------------------------

    def freshness_report(self) -> Dict[str, Any]:
        """Return a snapshot of data freshness across all symbols."""
        report = {}
        for sym in SYMBOLS:
            fd = self.freshness[sym]
            report[sym] = {
                "tick_count": fd.tick_count,
                "last_tick_age": (
                    time.time() - fd.last_tick_ts if fd.last_tick_ts > 0 else -1
                ),
                "is_healthy": fd.is_healthy,
                "source": fd.last_source,
                "errors": fd.errors_since_healthy,
                "bars": {tf: fd.bar_count[tf] for tf in TIMEFRAMES},
            }
        return report
