import numpy as np
import pandas as pd
from loguru import logger
from typing import Dict, Optional, List, Tuple
import os, json


SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h"]

JPY_PAIRS = {"USDJPY", "EURJPY", "GBPJPY"}
XAU_PAIRS = {"XAUUSD", "XAGUSD"}

BASE_PRICES = {
    "EURUSD": 1.12, "GBPUSD": 1.28, "USDJPY": 150.0, "AUDUSD": 0.67,
    "USDCAD": 1.35, "USDCHF": 0.88, "NZDUSD": 0.61,
    "EURJPY": 162.0, "GBPJPY": 190.0, "EURGBP": 0.86, "XAUUSD": 2000.0,
}


class DataManager:
    """Multi-symbol market data manager: OHLCV for 7+ pairs across 5 timeframes."""

    def __init__(self, historical_path="data/historical"):
        self.historical_path = historical_path
        self.ohlcv: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.order_flow: Dict[str, Dict] = {}
        self.tick_buffer: Dict[str, List[Dict]] = {}
        self._cvd: Dict[str, Tuple[List[float], List[float], List[float]]] = {}
        self.latest_snapshot: Dict[str, Optional[object]] = {}

        for sym in SYMBOLS:
            self.ohlcv[sym] = {}
            self.order_flow[sym] = {}
            self.tick_buffer[sym] = []
            self._cvd[sym] = ([], [], [])  # cvd_hist, bid_hist, ask_hist
            self.latest_snapshot[sym] = None
            for tf in TIMEFRAMES:
                self.ohlcv[sym][tf] = pd.DataFrame(
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )

        logger.info(f"DataManager initialized: {len(SYMBOLS)} symbols × {len(TIMEFRAMES)} timeframes")

    # ------------------------------------------------------------------
    # Live tick ingestion
    # ------------------------------------------------------------------

    def update_tick(self, symbol: str, bid: float, ask: float, volume: float = 0, ts: Optional[float] = None):
        ts = ts or pd.Timestamp.now().timestamp()
        mid = (bid + ask) / 2.0
        self.tick_buffer[symbol].append({
            "ts": ts, "symbol": symbol, "bid": bid, "ask": ask, "mid": mid, "vol": volume,
        })
        self._aggregate_1m(symbol, ts, mid, volume)
        self._update_of(symbol, bid, ask, volume)
        if len(self.tick_buffer[symbol]) > 10000:
            self.tick_buffer[symbol] = self.tick_buffer[symbol][-5000:]

    def _aggregate_1m(self, symbol: str, ts: float, price: float, volume: float):
        df = self.ohlcv[symbol]["1m"]
        bar_ts = pd.Timestamp.fromtimestamp(ts).floor("1min").timestamp()
        if df.empty or df.iloc[-1]["timestamp"] < bar_ts:
            new = pd.DataFrame([{
                "timestamp": bar_ts, "open": price, "high": price,
                "low": price, "close": price, "volume": volume,
            }])
            self.ohlcv[symbol]["1m"] = pd.concat([df, new], ignore_index=True)
        else:
            idx = len(df) - 1
            df.at[idx, "high"] = max(df.iloc[idx]["high"], price)
            df.at[idx, "low"] = min(df.iloc[idx]["low"], price)
            df.at[idx, "close"] = price
            df.at[idx, "volume"] += volume
        self._propagate(symbol)

    def _propagate(self, symbol: str):
        df_1m = self.ohlcv[symbol]["1m"].copy()
        if df_1m.empty:
            return
        df_1m["datetime"] = pd.to_datetime(df_1m["timestamp"], unit="s")
        df_1m = df_1m.set_index("datetime")
        for tf, minutes in [("5m", 5), ("15m", 15), ("1h", 60), ("4h", 240)]:
            res = (
                df_1m.resample(f"{minutes}T")
                .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
                .dropna()
            )
            res = res.reset_index()
            res["timestamp"] = res["datetime"].astype(np.int64) // 10**9
            self.ohlcv[symbol][tf] = res[["timestamp", "open", "high", "low", "close", "volume"]]
        for tf in TIMEFRAMES:
            if len(self.ohlcv[symbol][tf]) > 5000:
                self.ohlcv[symbol][tf] = self.ohlcv[symbol][tf].iloc[-5000:].reset_index(drop=True)

    def _update_of(self, symbol: str, bid: float, ask: float, volume: float):
        cvd_hist, bid_hist, ask_hist = self._cvd[symbol]
        mid = (bid + ask) / 2.0
        if bid_hist:
            delta = volume if mid > (bid_hist[-1] + ask_hist[-1]) / 2.0 else -volume
            new_cvd = (cvd_hist[-1] if cvd_hist else 0.0) + delta
        else:
            new_cvd = 0.0
        cvd_hist.append(new_cvd)
        bid_hist.append(bid)
        ask_hist.append(ask)
        if len(cvd_hist) > 1000:
            self._cvd[symbol] = (cvd_hist[-500:], bid_hist[-500:], ask_hist[-500:])
        self.order_flow[symbol] = {
            "cvd": new_cvd,
            "cvd_slope": (cvd_hist[-1] - cvd_hist[-20]) / 20 if len(cvd_hist) >= 20 else 0.0,
            "imbalance": (bid - ask) / (ask - bid) if (ask - bid) > 0 else 0.0,
        }

    def get_tick_buffer(self, symbol: str, n: int = 1000) -> List[Dict]:
        return self.tick_buffer.get(symbol, [])[-n:]

    # ------------------------------------------------------------------
    # Historical data loading
    # ------------------------------------------------------------------

    def load_historical(self, symbol: str, tf: str, days: int = 365):
        fp = os.path.join(self.historical_path, f"{symbol}_{tf}_{days}d.csv")
        if os.path.exists(fp):
            self.ohlcv[symbol][tf] = pd.read_csv(fp)
            logger.info(f"Loaded {symbol} {tf} ({len(self.ohlcv[symbol][tf])} bars)")
        else:
            self._gen_synthetic(symbol, tf, days)
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            self.ohlcv[symbol][tf].to_csv(fp, index=False)
            logger.info(f"Generated synthetic {symbol} {tf} ({len(self.ohlcv[symbol][tf])} bars)")

    def load_all(self, days: int = 365, timeframes: Optional[List[str]] = None):
        tfs = timeframes or TIMEFRAMES
        for sym in SYMBOLS:
            for tf in tfs:
                self.load_historical(sym, tf, days)

    def _gen_synthetic(self, symbol: str, tf: str, days: int):
        tf_map = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
        secs = tf_map.get(tf, 3600)
        periods = days * 86400 // secs
        dates = pd.date_range(end=pd.Timestamp.now(), periods=periods, freq=f"{secs}S")
        base = BASE_PRICES.get(symbol, 1.12)
        prices = base * np.exp(np.cumsum(np.random.normal(0, 0.0001, periods)))
        spread = 0.0001 if "JPY" not in symbol.upper() else 0.01
        data = [{
            "timestamp": d.timestamp(),
            "open": p, "high": p * (1 + spread),
            "low": p * (1 - spread), "close": p * (1 + np.random.normal(0, spread/2)),
            "volume": int(np.random.exponential(50000)),
        } for d, p in zip(dates, prices)]
        self.ohlcv[symbol][tf] = pd.DataFrame(data)

    # ------------------------------------------------------------------
    # Convenience accessors
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
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ], axis=1).max(1)
        return float(max(tr.iloc[-period:].mean(), BASE_PRICES.get(symbol, 1.12) * 0.0005))

    def get_ohlcv_dict(self, symbol: str) -> Dict[str, pd.DataFrame]:
        return self.ohlcv.get(symbol, {})

    def all_prices(self, tf: str = "1h") -> Dict[str, float]:
        return {sym: self.get_price(sym, tf) for sym in SYMBOLS}

    # ------------------------------------------------------------------
    # Dashboard snapshot
    # ------------------------------------------------------------------

    def get_snapshot(self, symbol: str = "EURUSD", acc=None, positions=None):
        try:
            from src.data.feature_engine import FeatureEngine
            fe = FeatureEngine()
            features = fe.compute_features(
                self.ohlcv.get(symbol, {}), self.order_flow.get(symbol, {}), acc, positions
            )
            feature_names = fe.feature_names
        except Exception:
            features = None
            feature_names = []
        snap = type("Snapshot", (), {
            "symbol": symbol,
            "timestamp": pd.Timestamp.now().timestamp(),
            "features": features,
            "feature_names": feature_names,
            "regime": "ranging",
            "regime_conf": 0.8,
            "cvd": self._cvd[symbol][0][-1] if self._cvd[symbol][0] else 0.0,
            "bid_ask_imbalance": self.order_flow.get(symbol, {}).get("imbalance", 0.0),
        })()
        self.latest_snapshot[symbol] = snap
        return snap
