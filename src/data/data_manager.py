import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from loguru import logger
from typing import Dict, Optional, List, Tuple, Any
import os, json


# Multi-asset symbols: Forex, Commodities, Indices, Crypto (IC Markets / cTrader)
SYMBOLS = [
    # Forex Majors
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    # Forex Minors
    "EURJPY", "GBPJPY", "EURGBP",
    # Commodities (Metals & Energy)
    "XAUUSD", "XAGUSD", "XTIUSD", "XBRUSD", "XNGUSD",
    # Indices
    "US500", "US30", "USTEC", "UK100", "DE40",
    # Crypto
    "BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD",
]
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h"]

JPY_PAIRS = {"USDJPY", "EURJPY", "GBPJPY"}
XAU_PAIRS = {"XAUUSD", "XAGUSD"}
CRYPTO_PAIRS = {"BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD"}
INDEX_PAIRS = {"US500", "US30", "USTEC", "UK100", "DE40"}
ENERGY_PAIRS = {"XTIUSD", "XBRUSD", "XNGUSD"}

BASE_PRICES = {
    "EURUSD": 1.12, "GBPUSD": 1.28, "USDJPY": 150.0, "AUDUSD": 0.67,
    "USDCAD": 1.35, "USDCHF": 0.88, "NZDUSD": 0.61,
    "EURJPY": 162.0, "GBPJPY": 190.0, "EURGBP": 0.86,
    "XAUUSD": 2000.0, "XAGUSD": 24.0,
    "XTIUSD": 75.0, "XBRUSD": 80.0, "XNGUSD": 3.0,
    "US500": 4500.0, "US30": 35000.0, "USTEC": 15000.0,
    "UK100": 7500.0, "DE40": 17000.0,
    "BTCUSD": 45000.0, "ETHUSD": 2500.0, "LTCUSD": 80.0, "XRPUSD": 0.60,
}


@dataclass
class DepthLevelData:
    """Single price level in the order book."""
    price: float = 0.0
    size: float = 0.0  # in units


@dataclass
class MarketDepthData:
    """Full Level II market depth with bid/ask stacks."""
    symbol: str = ""
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0
    volume: float = 0.0
    timestamp: float = field(default_factory=lambda: pd.Timestamp.now().timestamp())
    bids: List[DepthLevelData] = field(default_factory=list)  # Sorted best first
    asks: List[DepthLevelData] = field(default_factory=list)  # Sorted best first


class DataManager:
    """Multi-symbol market data manager: OHLCV, order flow, and Level II DOM."""

    def __init__(self, historical_path="data/historical"):
        self.historical_path = historical_path
        self.ohlcv: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.order_flow: Dict[str, Dict] = {}
        self.tick_buffer: Dict[str, List[Dict]] = {}
        self._cvd: Dict[str, Tuple[List[float], List[float], List[float]]] = {}
        self.latest_snapshot: Dict[str, Optional[object]] = {}
        self.market_depth: Dict[str, MarketDepthData] = {}

        for sym in SYMBOLS:
            self.ohlcv[sym] = {}
            self.order_flow[sym] = {}
            self.tick_buffer[sym] = []
            self._cvd[sym] = ([], [], [])
            self.latest_snapshot[sym] = None
            self.market_depth[sym] = MarketDepthData(symbol=sym)
            for tf in TIMEFRAMES:
                self.ohlcv[sym][tf] = pd.DataFrame(
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )

        logger.info(f"DataManager initialized: {len(SYMBOLS)} symbols × {len(TIMEFRAMES)} timeframes + Level II DOM")

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
        # Ensure timestamp column is numeric (float)
        if not df.empty and df["timestamp"].dtype == "object":
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        bar_ts = pd.Timestamp.fromtimestamp(ts).floor("1min").timestamp()
        if df.empty or float(df.iloc[-1]["timestamp"]) < bar_ts:
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

    def update_price(self, symbol: str, price: float, timeframe: str = "1h"):
        """Update the latest price for a symbol in the given timeframe."""
        symbol = symbol.upper()
        if symbol not in self.ohlcv:
            return
        df = self.ohlcv[symbol].get(timeframe)
        if df is not None and not df.empty:
            df.at[len(df) - 1, "close"] = price

    def _propagate(self, symbol: str):
        df_1m = self.ohlcv[symbol]["1m"].copy()
        if df_1m.empty:
            return
        df_1m["datetime"] = pd.to_datetime(df_1m["timestamp"], unit="s")
        df_1m = df_1m.set_index("datetime")
        for tf, minutes in [("5m", 5), ("15m", 15), ("1h", 60), ("4h", 240)]:
            res = (
                df_1m.resample(f"{minutes}min")
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
        dates = pd.date_range(end=pd.Timestamp.now(), periods=periods, freq=f"{secs}s")
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
    # Level II DOM (Depth of Market)
    # ------------------------------------------------------------------

    def update_market_depth(self, depth: Any):
        """Update Level II DOM data from cTrader or other provider."""
        try:
            from src.api.ctrader_client import MarketDepth as CtraderDepth
            if isinstance(depth, CtraderDepth):
                sym = depth.symbol
                md = self.market_depth[sym]
                md.symbol = sym
                md.bid = depth.bid
                md.ask = depth.ask
                md.spread = depth.spread
                md.volume = depth.volume
                md.timestamp = depth.timestamp
                md.bids = [DepthLevelData(price=b.price, size=b.size) for b in depth.bids]
                md.asks = [DepthLevelData(price=a.price, size=a.size) for a in depth.asks]
                logger.debug(f"DOM updated: {sym} - {len(md.bids)} bids, {len(md.asks)} asks")
        except ImportError:
            logger.debug("cTrader client not available for DOM update")

    def get_market_depth(self, symbol: str) -> Optional[MarketDepthData]:
        """Get Level II market depth for a symbol."""
        return self.market_depth.get(symbol)

    def get_dom_imbalance(self, symbol: str, levels: int = 5) -> float:
        """Calculate bid/ask imbalance from top N levels of DOM."""
        md = self.market_depth.get(symbol)
        if not md or not md.bids or not md.asks:
            return 0.0
        bid_vol = sum(b.size for b in md.bids[:levels])
        ask_vol = sum(a.size for a in md.asks[:levels])
        if bid_vol + ask_vol == 0:
            return 0.0
        return (bid_vol - ask_vol) / (bid_vol + ask_vol)

    # ------------------------------------------------------------------
    # Order Flow Analytics (CVD, Delta, Footprint)
    # ------------------------------------------------------------------

    def get_order_flow_metrics(self, symbol: str) -> Dict:
        """Get comprehensive order flow metrics."""
        cvd_hist, bid_hist, ask_hist = self._cvd.get(symbol, ([], [], []))
        of = self.order_flow.get(symbol, {})

        return {
            "cvd": of.get("cvd", 0.0),
            "cvd_slope": of.get("cvd_slope", 0.0),
            "imbalance": of.get("imbalance", 0.0),
            "dom_imbalance": self.get_dom_imbalance(symbol),
            "cvd_hist": cvd_hist[-100:] if cvd_hist else [],
        }

    # ------------------------------------------------------------------
    # Gamma Exposure Mapping (Institutional Feature)
    # ------------------------------------------------------------------

    def calculate_gamma_exposure(self, symbol: str) -> Dict:
        """
        Estimate gamma exposure from options flow.
        Note: Requires options data (not available from cTrader).
        This is a placeholder for when options data source is added.
        """
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
