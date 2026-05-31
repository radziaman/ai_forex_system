"""
Historical data loading, disk persistence, gap detection & healing,
and multi-source fallback (CSV, Dukascopy BI5, yFinance, cTrader).
"""

import asyncio
import lzma
import os
import struct
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .data_manager import (
    BASE_PRICES,
    CACHE_DIR,
    DUKASCOPE_SYMBOLS,
    SYMBOLS,
    TIMEFRAMES,
    TF_MINUTES,
)


class HistoricalLoader:
    """Handles historical data loading, saving, gap detection, and source fallback."""

    def __init__(self, dm):
        self.dm = dm  # Reference to DataManager for shared state

    # ------------------------------------------------------------------
    # Disk persistence
    # ------------------------------------------------------------------

    def save_ohlcv(self, symbol: str, tf: str = "1h"):
        """Append in-memory OHLCV bars to CSV on disk.

        Uses a single rolling file per symbol per timeframe.
        Deduplicates by timestamp to handle append-after-restart.
        """
        df = self.dm.ohlcv.get(symbol, {}).get(tf)
        if df is None or df.empty:
            return
        path = os.path.join(self.dm.historical_path, f"{symbol}_{tf}.csv")
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
                df = self.dm.ohlcv.get(sym, {}).get(tf)
                if df is not None and not df.empty:
                    self.save_ohlcv(sym, tf)
                    saved += 1
        if saved > 0:
            logger.info(f"Saved {saved} symbol-TF pairs to disk")

    # ------------------------------------------------------------------
    # Historical loading
    # ------------------------------------------------------------------

    def load_historical(self, symbol: str, tf: str, days: int = 365):
        """Load data for one symbol × timeframe from CSV cache or fallback."""
        fp = os.path.join(self.dm.historical_path, f"{symbol}_{tf}.csv")
        if os.path.exists(fp):
            self.dm.ohlcv[symbol][tf] = pd.read_csv(fp)
            self.dm.freshness[symbol].last_source = "csv"
            self.dm.freshness[symbol].bar_count[tf] = len(self.dm.ohlcv[symbol][tf])
            logger.info(
                f"Loaded {symbol} {tf} CSV ({len(self.dm.ohlcv[symbol][tf])} bars)"
            )
            return

        # No CSV cache — fetch real data from Yahoo Finance
        if self.try_alternative_source(symbol, tf, days=days):
            return

        # Fallback: Dukascopy BI5 cache
        self.load_from_dukascopy_cache(
            symbols=[symbol], timeframes=[tf], max_hours=days * 24
        )
        bars = self.dm.ohlcv.get(symbol, {}).get(tf, pd.DataFrame())
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
        """Load all symbols × timeframes."""
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
                self.dm.ohlcv[sym][tf] = bars
                self.dm.freshness[sym].last_source = "dukascopy_cache"
                self.dm.freshness[sym].bar_count[tf] = len(bars)
                self.dm.freshness[sym].is_healthy = True
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

        try:
            existing = self.dm.ohlcv.get(symbol, {}).get(timeframe)
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
                    self.dm.ohlcv[symbol][timeframe] = merged
                    logger.info(
                        f"cTrader appended {len(df)} bars to {symbol} {timeframe} "
                        f"(total {len(merged)})"
                    )
                else:
                    self.dm.ohlcv[symbol][timeframe] = df
                    logger.info(
                        f"Loaded {symbol} {timeframe} from cTrader: {len(df)} bars"
                    )
                self.dm._source_health["ctrader"]["last_ok"] = time.time()
                self.dm._source_health["ctrader"]["failures"] = 0
                self.dm.freshness[symbol].last_source = "ctrader"
                self.dm.freshness[symbol].bar_count[timeframe] = len(
                    self.dm.ohlcv[symbol][timeframe]
                )
                return True
            return False
        except Exception as e:
            self.dm._source_health["ctrader"]["failures"] += 1
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
        fp = os.path.join(self.dm.historical_path, f"{symbol}_{timeframe}.csv")
        if os.path.exists(fp):
            try:
                df = pd.read_csv(fp)
                if df is not None and len(df) >= 50:
                    self.dm.ohlcv[symbol][timeframe] = df
                    self.dm.freshness[symbol].last_source = "csv"
                    self.dm.freshness[symbol].bar_count[timeframe] = len(df)
                    logger.info(f"Loaded {symbol} {timeframe} CSV ({len(df)} bars)")
                    return True
            except Exception as e:
                logger.debug(f"CSV load failed for {symbol}: {e}")

        # 1. Dukascopy BI5 cache
        loaded = self.load_from_dukascopy_cache(
            symbols=[symbol], timeframes=[timeframe], max_hours=days * 24
        )
        if loaded > 0 and len(self.dm.ohlcv.get(symbol, {}).get(timeframe, [])) > 50:
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
        """Try fetching OHLCV from Yahoo Finance as alternative source."""
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
                "GC=F": "GC=F",
                "SI=F": "SI=F",
                "HG=F": "HG=F",
                "PA=F": "PA=F",
                "PL=F": "PL=F",
                "CL=F": "CL=F",
                "HO=F": "HO=F",
                "RB=F": "RB=F",
                "NG=F": "NG=F",
                "ZB=F": "ZB=F",
                "ZN=F": "ZN=F",
                "ZF=F": "ZF=F",
                "ZT=F": "ZT=F",
                "ES=F": "ES=F",
                "NQ=F": "NQ=F",
                "YM=F": "YM=F",
                "DX-Y.NYB": "DX-Y.NYB",
                "USO": "USO",
                "GLD": "GLD",
                "SLV": "SLV",
                "TLT": "TLT",
            }
            # Smart Yahoo ticker resolution
            if symbol in ticker_map:
                yf_sym = ticker_map[symbol]
            elif any(c in symbol for c in ("=", "^")):
                yf_sym = symbol
            elif len(symbol) >= 5 and symbol.isalpha() and symbol.isupper():
                yf_sym = f"{symbol}=X"
            else:
                yf_sym = symbol
            interval = timeframe.replace("m", "m").replace("h", "h")
            existing = self.dm.ohlcv.get(symbol, {}).get(timeframe)
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
                    yf_sym,
                    period=f"{max(days, 5)}d",
                    interval=interval,
                    progress=False,
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
                    self.dm.ohlcv[symbol][timeframe] = merged
                    logger.info(
                        f"Appended {len(new_bars)} new bars to {symbol} "
                        f"(total {len(merged)})"
                    )
                else:
                    self.dm.ohlcv[symbol][timeframe] = new_bars
                    logger.info(f"Loaded {symbol} from yFinance: {len(new_bars)} bars")
                self.dm._source_health["yfinance"]["last_ok"] = time.time()
                self.dm._source_health["yfinance"]["failures"] = 0
                self.dm.freshness[symbol].last_source = "yfinance"
                self.dm.freshness[symbol].bar_count[timeframe] = len(
                    self.dm.ohlcv[symbol][timeframe]
                )
                self.save_ohlcv(symbol, timeframe)
                return True
        except Exception as e:
            self.dm._source_health["yfinance"]["failures"] += 1
            logger.debug(f"yFinance failed for {symbol}: {e}")
        return False

    # ------------------------------------------------------------------
    # Data gap detection & healing
    # ------------------------------------------------------------------

    def detect_gaps(
        self, symbol: str, tf: str = "1h", max_gap_minutes: int = 90
    ) -> List[Tuple[float, float]]:
        """Return list of (start, end) gaps longer than max_gap_minutes."""
        df = self.dm.ohlcv.get(symbol, {}).get(tf)
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
                        asyncio.ensure_future(
                            self.load_from_ctrader(
                                symbol, tf, days=gap_days, client=ctrader_client
                            )
                        )
                        healed += 1
                        continue
                    else:
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
        """Record a failure for a data source."""
        if source in self.dm._source_health:
            self.dm._source_health[source]["failures"] += 1
            if self.dm._source_health[source]["failures"] > 10:
                self.dm._source_health[source]["ok"] = False

    def report_source_success(self, source: str):
        """Record a success for a data source."""
        if source in self.dm._source_health:
            self.dm._source_health[source]["ok"] = True
            self.dm._source_health[source]["failures"] = 0
            self.dm._source_health[source]["last_ok"] = time.time()

    def best_source(self) -> str:
        """Return the healthiest data source name."""
        best = "dukascopy"
        best_score = -1
        for name, h in self.dm._source_health.items():
            score = 0 if not h["ok"] else max(0, 10 - h["failures"])
            if score > best_score:
                best_score = score
                best = name
        return best
