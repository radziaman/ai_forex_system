"""
DataService — centralized data pipeline for the trading system.
"""
import numpy as np
import pandas as pd
import time
import asyncio
from typing import Dict, Optional, List, Tuple
from loguru import logger
from pathlib import Path
from datetime import datetime, timezone, timedelta

from data.data_manager import DataManager, SYMBOLS, BASE_PRICES
from rts_ai_fx.features_unified import FeaturePipeline
from rts_ai_fx.regime_detector import HMMRegimeDetector


class DataService:
    """Centralized data pipeline — fetching, caching, feature engineering."""

    def __init__(self, config, provider=None):
        self.config = config
        self.provider = provider
        self.data_manager = DataManager(
            historical_path=config.data.historical_path,
        )
        self.feature_pipeline = FeaturePipeline(
            lookback=30, timeframes=["1h"],
            use_microstructure=True, use_cross_asset=False,
        )
        self.regime_detector = HMMRegimeDetector(n_regimes=4, lookback=60)
        self._regimes: Dict[str, str] = {}
        self._regime_detectors: Dict[str, HMMRegimeDetector] = {}
        self._download_queue: List[str] = []
        self._download_retries: Dict[str, int] = {}
        self._last_data_download = 0.0

    def get_price(self, symbol: str, timeframe: str = "1h") -> Optional[float]:
        return self.data_manager.get_price(symbol, timeframe)

    def get_price_or_default(self, symbol: str) -> float:
        price = self.data_manager.get_price(symbol, "1h")
        if price is None or price == 0:
            return BASE_PRICES.get(symbol.upper(), 1.12)
        return price

    def get_ohlcv(self, symbol: str, timeframe: str = "1h"):
        return self.data_manager.get_ohlcv(symbol, timeframe)

    def get_atr(self, symbol: str, timeframe: str = "1h", period: int = 14) -> float:
        return self.data_manager.get_atr(symbol, timeframe, period)

    def get_tick_buffer(self, symbol: str, n: int = 1000):
        return self.data_manager.get_tick_buffer(symbol, n)

    def detect_regime(self, symbol: str, df) -> str:
        if symbol not in self._regime_detectors:
            rd = HMMRegimeDetector(n_regimes=4, lookback=60)
            rd.fit(df)
            self._regime_detectors[symbol] = rd
            regime = rd.detect_regime(df)
        else:
            regime = self._regime_detectors[symbol].detect_regime(df)
        self._regimes[symbol] = regime
        return regime

    def get_regime(self, symbol: str) -> str:
        return self._regimes.get(symbol, "ranging")

    def all_prices(self) -> Dict:
        return self.data_manager.all_prices()

    def latest_snapshot(self) -> Dict:
        return self.data_manager.latest_snapshot

    def get_current_spread(self, symbol: str) -> float:
        try:
            tick = self.data_manager.latest_snapshot.get(symbol)
            if tick and hasattr(tick, 'bid') and hasattr(tick, 'ask'):
                pip_size = 0.0001 if "JPY" not in symbol.upper() else 0.01
                return float((tick.ask - tick.bid) / pip_size)
            return 1.0
        except Exception:
            return 1.0

    async def load_historical_data(self):
        """Load cached Dukascopy data for all symbols."""
        loaded = 0
        from data.dukascopy_realtime import CACHE_DIR
        import lzma, struct

        logger.info("Loading cached historical data from Dukascopy...")
        for sym in SYMBOLS:
            if self.data_manager.get_ohlcv(sym, "1h") is not None and len(self.data_manager.get_ohlcv(sym, "1h")) > 200:
                loaded += 1
                continue

            cache_files = sorted(CACHE_DIR.glob(f"{sym}_*_*.bi5"))
            if not cache_files:
                logger.debug(f"No cached data for {sym}")
                continue

            ticks = []
            for cf in cache_files[-168:]:
                try:
                    raw = cf.read_bytes()
                    decompressed = lzma.decompress(raw)
                except Exception:
                    decompressed = raw
                parts = cf.stem.split("_")
                dt_str, hour_str = parts[-2], parts[-1]
                base_ts = datetime.strptime(dt_str, "%Y%m%d").timestamp() + int(hour_str) * 3600
                for i in range(0, len(decompressed), 20):
                    chunk = decompressed[i:i+20]
                    if len(chunk) < 20:
                        break
                    ms = struct.unpack(">I", chunk[0:4])[0]
                    ask_raw = struct.unpack(">I", chunk[4:8])[0]
                    bid_raw = struct.unpack(">I", chunk[8:12])[0]
                    ticks.append((base_ts + ms/1000.0, bid_raw/100000.0, ask_raw/100000.0))

            if len(ticks) < 10:
                logger.debug(f"Too few cached ticks for {sym}")
                continue

            ticks.sort(key=lambda t: t[0])
            df = pd.DataFrame(ticks, columns=["timestamp", "bid", "ask"])
            df["bar"] = (df["timestamp"] // 3600).astype(int)
            bars = df.groupby("bar").agg(
                open=("bid", "first"), high=("bid", "max"), low=("bid", "min"),
                close=("bid", "last"), volume=("ask", "count")
            ).reset_index()
            bars["timestamp"] = bars["bar"] * 3600
            bars = bars.drop(columns=["bar"])
            self.data_manager.ohlcv[sym]["1h"] = bars
            loaded += 1
            logger.info(f"Loaded {len(bars)} 1h bars for {sym} (cached)")

        if loaded < len(SYMBOLS):
            logger.warning(f"Only {loaded}/{len(SYMBOLS)} pairs in cache.")
            self._download_queue = [
                s for s in SYMBOLS
                if self.data_manager.get_ohlcv(s, "1h") is None or len(self.data_manager.get_ohlcv(s, "1h")) < 50
            ]
        return loaded

    async def background_download_step(self) -> bool:
        """Download one missing symbol. Returns True if queue is now empty."""
        if not self._download_queue or time.time() - self._last_data_download < 30:
            return False
        sym = self._download_queue[0]
        logger.info(f"Background downloading {sym} from Dukascopy...")
        self._last_data_download = time.time()
        success = False
        from data.dukascopy_realtime import DukascopyProvider, CACHE_DIR
        import lzma, struct

        cache_files = sorted(CACHE_DIR.glob(f"{sym}_*_*.bi5"))
        if cache_files:
            ticks = []
            for cf in cache_files[-720:]:
                raw = cf.read_bytes()
                try:
                    decompressed = lzma.decompress(raw)
                except Exception:
                    decompressed = raw
                parts = cf.stem.split("_")
                dt_str, hour_str = parts[-2], parts[-1]
                base_ts = datetime.strptime(dt_str, "%Y%m%d").timestamp() + int(hour_str) * 3600
                for i in range(0, len(decompressed), 20):
                    chunk = decompressed[i:i+20]
                    if len(chunk) < 20:
                        break
                    ms = struct.unpack(">I", chunk[0:4])[0]
                    _, ask_raw, bid_raw = struct.unpack(">III", chunk[0:12])
                    ticks.append((base_ts + ms/1000.0, bid_raw/100000.0, ask_raw/100000.0))
            if len(ticks) > 50:
                ticks.sort(key=lambda t: t[0])
                df = pd.DataFrame(ticks, columns=["timestamp", "bid", "ask"])
                df["bar"] = (df["timestamp"] // 3600).astype(int)
                bars = df.groupby("bar").agg(
                    open=("bid", "first"), high=("bid", "max"),
                    low=("bid", "min"), close=("bid", "last"), volume=("ask", "count")
                ).reset_index()
                bars["timestamp"] = bars["bar"] * 3600
                bars = bars.drop(columns=["bar"])
                self.data_manager.ohlcv[sym]["1h"] = bars
                logger.info(f"Background loaded {sym}: {len(bars)} bars (cached)")
                success = True
        else:
            provider = DukascopyProvider(cache=True)
            end = datetime.now(timezone.utc) - timedelta(days=1)
            start = end - timedelta(days=30)
            ohlcv = await provider.fetch_ohlcv(sym, "1h", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
            await provider.close()
            if ohlcv and len(ohlcv) > 50:
                df = pd.DataFrame([{
                    "timestamp": o.timestamp, "open": o.open, "high": o.high,
                    "low": o.low, "close": o.close, "volume": o.volume,
                } for o in ohlcv])
                self.data_manager.ohlcv[sym]["1h"] = df
                logger.info(f"Background download complete for {sym}: {len(df)} bars")
                success = True

        if not success:
            import yfinance as yf
            yf_sym = {"XAUUSD": "GC=F", "XAGUSD": "SI=F", "XTIUSD": "CL=F",
                      "XBRUSD": "BZ=F", "XNGUSD": "NG=F", "BTCUSD": "BTC-USD",
                      "ETHUSD": "ETH-USD"}.get(sym, f"{sym}=X")
            try:
                yf_data = yf.download(yf_sym, period="1mo", interval="1h", progress=False)
                if not yf_data.empty:
                    df = yf_data.reset_index()
                    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                            "Close": "close", "Volume": "volume"})
                    df["timestamp"] = pd.to_datetime(df["Date"]).astype(int) // 10**9
                    self.data_manager.ohlcv[sym]["1h"] = df[["timestamp", "open", "high", "low", "close", "volume"]]
                    logger.info(f"yFinance fallback loaded {sym}: {len(df)} bars")
                    success = True
            except Exception as yf_err:
                logger.debug(f"yFinance fallback failed for {sym}: {yf_err}")

        if success:
            self._download_queue.pop(0)
        else:
            self._download_retries[sym] = self._download_retries.get(sym, 0) + 1
            if self._download_retries[sym] >= 3:
                logger.warning(f"Giving up on {sym} after 3 failures")
                self._download_queue.pop(0)
                self._download_queue.append(sym)
                self._download_retries[sym] = 0

        return len(self._download_queue) == 0

    def fit_regime_detectors(self):
        """Fit HMM regime detector per symbol."""
        for sym in SYMBOLS:
            df = self.data_manager.get_ohlcv(sym, "1h")
            if df is not None and len(df) > 200:
                rd = HMMRegimeDetector(n_regimes=4, lookback=60)
                rd.fit(df)
                self._regime_detectors[sym] = rd
                self._regimes[sym] = rd.detect_regime(df)
                if self.regime_detector.model is None:
                    self.regime_detector.fit(df)
                logger.info(f"HMM regime detector fitted for {sym}")

    def fit_feature_pipeline(self):
        self.feature_pipeline.fit_all(self.data_manager.ohlcv)
        logger.info("Feature pipeline fitted on all symbols")

    def get_actual_state_dim(self, external_signals: np.ndarray) -> int:
        """Compute actual flattened feature dimension."""
        for sym in SYMBOLS:
            df_1h = self.data_manager.get_ohlcv(sym, "1h")
            if df_1h is not None and len(df_1h) > self.feature_pipeline.lookback + 10:
                tick_buffer = self.data_manager.get_tick_buffer(sym, 1000)
                features = self.feature_pipeline.transform(
                    self.data_manager.ohlcv, symbol=sym,
                    tick_buffer=tick_buffer,
                    external_signals=external_signals,
                )
                if features is not None:
                    return len(features.flatten())
        return 55 * 30
