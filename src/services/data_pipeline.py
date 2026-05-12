"""Data pipeline: tick ingestion, OHLCV aggregation, feature caching + emission."""

import asyncio
import time
from typing import Dict, List, Optional, Any
from loguru import logger

from infrastructure.service_base import TradingService
from infrastructure.event_bus import get_event_bus, EventType
from infrastructure.config_v2 import AppConfig
from data.data_manager import DataManager, SYMBOLS
from api.base import PriceTick
from rts_ai_fx.features_unified import FeaturePipeline
from services import FeatureUpdate


class DataPipeline(TradingService):
    """Ingests market data, aggregates OHLCV, computes + caches features."""

    def __init__(self, config: AppConfig):
        super().__init__("data_pipeline")
        self.config = config
        self.data_manager = DataManager(historical_path=config.data.historical_path)
        self.feature_pipeline = FeaturePipeline(
            lookback=config.features.lookback,
            timeframes=config.features.timeframes,
            use_microstructure=config.features.use_microstructure,
        )
        self.event_bus = get_event_bus()
        self._features_dirty: Dict[str, bool] = {}
        self._last_bar_ts: Dict[str, float] = {}
        self.tick_counter: int = 0

    async def start(self) -> None:
        self._running = True
        logger.info(
            f"DataPipeline: {len(SYMBOLS)} symbols x {self.config.features.timeframes}"
        )

    async def stop(self) -> None:
        self._running = False

    def ingest_tick(self, tick: PriceTick) -> None:
        if not self._running:
            return
        self.tick_counter += 1
        sym = tick.symbol
        self.data_manager.update_tick(
            sym, tick.bid, tick.ask, tick.volume, tick.timestamp
        )
        df = self.data_manager.get_ohlcv(sym, "1m")
        if df is not None and not df.empty:
            current_bar_ts = float(df["timestamp"].iloc[-1])
            if self._last_bar_ts.get(sym, 0) != current_bar_ts:
                self._last_bar_ts[sym] = current_bar_ts
                self._features_dirty[sym] = True

    def on_bar_close(self, symbol: str) -> None:
        features = self.get_features(symbol)
        if features is not None:
            df = self.data_manager.get_ohlcv(symbol, "1h")
            price = self.data_manager.get_price(symbol, "1h")
            logger.info(
                f"[data] {symbol}: features ready (price={price:.5f}, features.shape={features.shape})"
            )
            update = FeatureUpdate(
                symbol=symbol,
                timeframe="1h",
                features=features,
                ohlcv=df,
                price=price,
                timestamp=time.time(),
            )
            asyncio.ensure_future(
                self.event_bus.emit(
                    EventType.FEATURES_READY, update, source="data_pipeline"
                )
            )
            self._features_dirty[symbol] = False
        else:
            logger.info(f"[data] {symbol}: no features (insufficient data)")

    def get_features(self, symbol: str) -> Optional[Any]:
        cached = self.data_manager.get_cached_features(symbol, "1h")
        if cached is not None and not self._features_dirty.get(symbol, True):
            return cached

        df = self.data_manager.get_ohlcv(symbol, "1h")
        if df is None or len(df) < self.config.features.lookback + 10:
            return None

        try:
            features = self.feature_pipeline.transform(
                self.data_manager.ohlcv,
                symbol=symbol,
            )
            if features is not None:
                self.data_manager.set_cached_features(symbol, "1h", features)
            return features
        except Exception as e:
            logger.debug(f"Feature computation failed for {symbol}: {e}")
            return None

    def get_price(self, symbol: str) -> float:
        return self.data_manager.get_price(symbol, "1h")

    def get_live_price(self, symbol: str) -> float:
        """Latest mid-price from tick stream. Falls back to 1h close if no tick yet."""
        from data.data_manager import BASE_PRICES

        latest = self.data_manager._last_realtime_price.get(symbol)
        if latest and latest > 0:
            default = BASE_PRICES.get(symbol, 0)
            if default > 0 and latest == default:
                return self.get_price(symbol)
            return latest
        return self.get_price(symbol)

    def get_atr(self, symbol: str, period: int = 14) -> float:
        return self.data_manager.get_atr(symbol, "1h", period)

    def get_ohlcv(self, symbol: str, tf: str = "1h") -> Optional[Any]:
        return self.data_manager.get_ohlcv(symbol, tf)

    def all_prices(self) -> Dict[str, float]:
        return self.data_manager.all_prices()
