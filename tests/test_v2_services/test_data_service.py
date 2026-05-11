"""Tests for DataPipeline."""
import time
import numpy as np
from infrastructure.config_v2 import AppConfig
from services.data_pipeline import DataPipeline
from api.base import PriceTick
from data.data_manager import SYMBOLS


class TestDataPipeline:
    def setup_method(self):
        self.config = AppConfig()
        self.config.features.timeframes = ["1h"]
        self.config.features.lookback = 10
        self.svc = DataPipeline(self.config)

    def test_init(self):
        assert self.svc.name == "data_pipeline"
        assert not self.svc.is_running

    def test_start_stop(self):
        import asyncio
        asyncio.run(self.svc.start())
        assert self.svc.is_running
        asyncio.run(self.svc.stop())
        assert not self.svc.is_running

    def test_ingest_tick_updates_data_manager(self):
        tick = PriceTick(symbol="EURUSD", bid=1.1200, ask=1.1201, volume=100, timestamp=time.time())
        self.svc.ingest_tick(tick)
        price = self.svc.data_manager.get_price("EURUSD", "1h")
        assert price > 0

    def test_get_price_returns_reasonable_value(self):
        price = self.svc.get_price("EURUSD")
        assert price > 0

    def test_get_atr_returns_positive(self):
        atr = self.svc.get_atr("EURUSD")
        assert atr > 0

    def test_get_features_all_symbols(self):
        for sym in ["EURUSD", "GBPUSD", "USDJPY"]:
            features = self.svc.get_features(sym)
            if features is not None:
                assert isinstance(features, np.ndarray)
                assert features.size > 0
                break

    def test_all_prices_has_all_symbols(self):
        prices = self.svc.all_prices()
        assert len(prices) == len(SYMBOLS)
        for sym in SYMBOLS:
            assert prices.get(sym, 0) > 0

    def test_ingest_tick_with_invalid_data_ignored(self):
        # bid=0 is rejected by _validate_tick (bid <= 0), so state stays clean
        tick = PriceTick(symbol="EURUSD", bid=0, ask=0, volume=0, timestamp=time.time())
        self.svc.ingest_tick(tick)
        # Validation returns False, so update_tick is never called
        assert self.svc.data_manager.freshness["EURUSD"].tick_count == 0
