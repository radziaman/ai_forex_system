"""
Tick ingestion, validation, OHLCV aggregation, and bar propagation.
"""

import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from .data_manager import BASE_PRICES, PRICE_WINDOWS, TF_MINUTES
from .microstructure_features import PriceTick


class TickIngester:
    """Handles tick validation, ingestion, OHLCV aggregation, and bar propagation."""

    def __init__(self, dm):
        self.dm = dm  # Reference to DataManager for shared state

    def validate_tick(self, symbol: str, bid: float, ask: float, volume: float) -> bool:
        """Validate tick price/volume sanity before ingestion."""
        if not all(isinstance(v, (int, float)) for v in [bid, ask, volume]):
            return False
        if bid <= 0 or ask <= 0 or bid > ask:
            return False
        if volume < 0:
            return False
        base = BASE_PRICES.get(symbol, 1.12)
        lo, hi = PRICE_WINDOWS.get(symbol, (0.1, 5.0))
        max_price = base * hi * 1000
        if bid > max_price or ask > max_price:
            logger.debug(
                f"{symbol}: price {bid}/{ask} exceeds sanity "
                f"max={max_price:.0f} (base={base}, window={hi}), rejected"
            )
            return False
        prev = self.dm._last_realtime_price.get(symbol)
        if prev and prev > 0 and prev != base:
            change = abs(bid - prev) / prev
            if change > 0.05:
                logger.debug(f"{symbol}: >5% tick move ({change:.2%}), rejected")
                return False
        return True

    def update_tick(
        self,
        symbol: str,
        bid: float,
        ask: float,
        volume: float = 0,
        ts: Optional[float] = None,
    ):
        """Validate and ingest a tick, batching for aggregation."""
        if not self.dm.enabled:
            return
        if not self.validate_tick(symbol, bid, ask, volume):
            self.dm.freshness[symbol].errors_since_healthy += 1
            return

        with self.dm._tick_lock:
            ts = ts or time.time()
            mid = (bid + ask) / 2.0
            self.dm._last_realtime_price[symbol] = mid

            tick = {
                "ts": ts,
                "symbol": symbol,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "vol": max(volume, 0),
            }
            self.dm._pending_ticks[symbol].append(tick)

            # Feed microstructure engine
            self.dm.microstructure.ingest_tick(
                PriceTick(
                    symbol=symbol,
                    bid=bid,
                    ask=ask,
                    mid=mid,
                    timestamp=ts,
                    volume=max(volume, 0),
                )
            )

            fd = self.dm.freshness[symbol]
            fd.last_tick_ts = ts
            fd.tick_count += 1
            fd.errors_since_healthy = 0
            fd.is_healthy = True

            now = time.time()
            if now - self.dm._last_aggregate_time >= self.dm._aggregate_interval:
                self.flush_pending()
                self.dm._last_aggregate_time = now

    def flush_pending(self):
        """Process all queued ticks in batch: aggregate bars, update CVD, invalidate cache."""
        for symbol, ticks in list(self.dm._pending_ticks.items()):
            if not ticks:
                continue
            buf = self.dm.tick_buffer[symbol]
            buf.extend(ticks)
            if len(buf) > 10000:
                self.dm.tick_buffer[symbol] = buf[-5000:]

            for t in ticks:
                self.aggregate_1m(symbol, t["ts"], t["mid"], t["vol"])
                self.dm.dom_engine.update_of(symbol, t["bid"], t["ask"], t["vol"])

            ts_min = min(t["ts"] for t in ticks)
            ts_max = max(t["ts"] for t in ticks)
            self.dm.freshness[symbol].last_tick_ts = ts_max
            self.dm.feature_cache_mgr.invalidate(symbol, ts_min, ts_max)

            ticks.clear()

    def aggregate_1m(self, symbol: str, ts: float, price: float, volume: float):
        """Aggregate a tick into the 1-minute OHLCV bar."""
        df = self.dm.ohlcv[symbol]["1m"]
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
            self.dm.ohlcv[symbol]["1m"] = (
                new if df.empty else pd.concat([df, new], ignore_index=True)
            )
            self.cap_bars(symbol, "1m")
        else:
            idx = len(df) - 1
            r = df.iloc[idx]
            if price > r["high"]:
                df.at[idx, "high"] = price
            if price < r["low"]:
                df.at[idx, "low"] = price
            df.at[idx, "close"] = price
            df.at[idx, "volume"] += max(volume, 0)
        self.propagate(symbol)

    def propagate(self, symbol: str):
        """Propagate 1m bars into higher timeframes (5m, 15m, 1h, 4h)."""
        df_1m = self.dm.ohlcv[symbol]["1m"].copy()
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

            existing = self.dm.ohlcv[symbol].get(tf)
            if existing is not None and len(existing) > len(new_bars):
                merged = pd.concat([existing, new_bars], ignore_index=True)
                merged = merged.drop_duplicates(subset=["timestamp"], keep="last")
                merged = merged.sort_values("timestamp").reset_index(drop=True)
                self.dm.ohlcv[symbol][tf] = merged
            else:
                self.dm.ohlcv[symbol][tf] = new_bars
            self.cap_bars(symbol, tf)

    def cap_bars(self, symbol: str, tf: str, max_bars: int = 5000):
        """Cap number of bars stored per symbol per timeframe."""
        df = self.dm.ohlcv[symbol][tf]
        if len(df) > max_bars:
            self.dm.ohlcv[symbol][tf] = df.iloc[-max_bars:].reset_index(drop=True)
