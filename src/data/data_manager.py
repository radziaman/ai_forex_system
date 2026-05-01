import numpy as np
import pandas as pd
from loguru import logger
from typing import Dict, Optional, List
import os, json


class DataManager:
    """Manages market data across 5 timeframes + order flow."""

    def __init__(self, historical_path="data/historical", redis_url=None):
        self.historical_path = historical_path
        self.ohlcv: Dict[str, pd.DataFrame] = {}
        self.order_flow: Dict = {}
        self.latest_snapshot = None
        self.tick_buffer = []
        self._cvd_state = 0.0
        self._cvd_hist = []
        self._bid_hist = []
        self._ask_hist = []
        for tf in ["1m", "5m", "15m", "1h", "4h"]:
            self.ohlcv[tf] = pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
        logger.info("DataManager initialized")

    def update_tick(self, symbol, bid, ask, volume=0, ts=None):
        ts = ts or pd.Timestamp.now().timestamp()
        mid = (bid + ask) / 2.0
        self.tick_buffer.append(
            {
                "ts": ts,
                "symbol": symbol,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "vol": volume,
            }
        )
        self._aggregate_1m(symbol, ts, mid, volume)
        self._update_of(bid, ask, volume)
        if len(self.tick_buffer) > 10000:
            self.tick_buffer = self.tick_buffer[-5000:]

    def _aggregate_1m(self, symbol, ts, price, volume):
        df = self.ohlcv["1m"]
        bar_ts = pd.Timestamp.fromtimestamp(ts).floor("1min").timestamp()
        if df.empty or df.iloc[-1]["timestamp"] < bar_ts:
            new = pd.DataFrame(
                [
                    {
                        "timestamp": bar_ts,
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price,
                        "volume": volume,
                    }
                ]
            )
            self.ohlcv["1m"] = pd.concat([df, new], ignore_index=True)
        else:
            idx = len(df) - 1
            df.at[idx, "high"] = max(df.iloc[idx]["high"], price)
            df.at[idx, "low"] = min(df.iloc[idx]["low"], price)
            df.at[idx, "close"] = price
            df.at[idx, "volume"] += volume
        self._propagate()

    def _propagate(self):
        df_1m = self.ohlcv["1m"].copy()
        if df_1m.empty:
            return
        df_1m["datetime"] = pd.to_datetime(df_1m["timestamp"], unit="s")
        df_1m = df_1m.set_index("datetime")
        for tf, minutes in [("5m", 5), ("15m", 15), ("1h", 60), ("4h", 240)]:
            res = (
                df_1m.resample(f"{minutes}T")
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
            res = res.reset_index()
            res["timestamp"] = res["datetime"].astype(np.int64) // 10**9
            self.ohlcv[tf] = res[
                ["timestamp", "open", "high", "low", "close", "volume"]
            ]
        for tf in self.ohlcv:
            if len(self.ohlcv[tf]) > 5000:
                self.ohlcv[tf] = self.ohlcv[tf].iloc[-5000:].reset_index(drop=True)

    def _update_of(self, bid, ask, volume):
        if not self._bid_hist:
            self._cvd_state = 0.0
            self._cvd_hist = []
        mid = (bid + ask) / 2.0
        if self._bid_hist:
            delta = (
                volume
                if mid > (self._bid_hist[-1] + self._ask_hist[-1]) / 2.0
                else -volume
            )
            self._cvd_state += delta
        self._cvd_hist.append(self._cvd_state)
        self._bid_hist.append(bid)
        self._ask_hist.append(ask)
        for tf in ["1m", "5m", "15m", "1h", "4h"]:
            self.order_flow[f"{tf}_cvd"] = self._cvd_state
            self.order_flow[f"{tf}_cvd_slope"] = (
                (self._cvd_hist[-1] - self._cvd_hist[-20]) / 20
                if len(self._cvd_hist) >= 20
                else 0.0
            )
            self.order_flow[f"{tf}_imbalance"] = (
                (bid - ask) / (ask - bid) if (ask - bid) > 0 else 0.0
            )
            self.order_flow[f"{tf}_large_z"] = 0.0

    def get_snapshot(self, symbol="EURUSD", acc=None, positions=None):
        try:
            from src.data.feature_engine import FeatureEngine
            fe = FeatureEngine()
            features = fe.compute_features(self.ohlcv, self.order_flow, acc, positions)
            feature_names = fe.feature_names
        except Exception:
            features = None
            feature_names = []
        snapshot = type(
            "Snapshot",
            (),
            {
                "symbol": symbol,
                "timestamp": pd.Timestamp.now().timestamp(),
                "features": features,
                "feature_names": feature_names,
                "regime": "trending_up",
                "regime_conf": 0.8,
                "cvd": self._cvd_state,
                "bid_ask_imbalance": self.order_flow.get("1m_imbalance", 0.0),
            },
        )()
        self.latest_snapshot = snapshot
        return snapshot

    def load_historical(self, tf: str, days: int = 365, symbol: str = "EURUSD"):
        fp = os.path.join(self.historical_path, f"{symbol}_{tf}_{days}d.csv")
        if os.path.exists(fp):
            self.ohlcv[tf] = pd.read_csv(fp)
            logger.info(f"Loaded {tf} historical data ({len(self.ohlcv[tf])} bars)")
        else:
            self._gen_synthetic(symbol, tf, days)
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            self.ohlcv[tf].to_csv(fp, index=False)
            logger.info(f"Generated synthetic {tf} data ({len(self.ohlcv[tf])} bars)")

    def _gen_synthetic(self, symbol, tf, days):
        secs = (
            self.feature_engine.TF_MULT.get(tf, 60)
            if hasattr(self, "feature_engine")
            else 60
        )
        periods = days * 24 * 3600 // secs
        dates = pd.date_range(end=pd.Timestamp.now(), periods=periods, freq=f"{secs}S")
        base = 1.12 if "JPY" not in symbol else 150.0
        if "XAU" in symbol:
            base = 2000.0
        prices = base * np.exp(np.cumsum(np.random.normal(0, 0.0001, periods)))
        data = [
            {
                "timestamp": d.timestamp(),
                "open": p,
                "high": p * 1.0001,
                "low": p * 0.9999,
                "close": p * 1.00005,
                "volume": int(np.random.exponential(50000)),
            }
            for d, p in zip(dates, prices)
        ]
        self.ohlcv[tf] = pd.DataFrame(data)

    def get_feature_dim(self):
        from src.data.feature_engine import FeatureEngine

        return len(FeatureEngine().feature_names)
