"""Data fetching and preprocessing module using yfinance and ccxt"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


class DataFetcher:
    """Fetch forex data from yfinance or ccxt"""

    def __init__(self, source: str = "yfinance"):
        self.source = source

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        start: str = "2015-01-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol"""
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        if self.source == "yfinance":
            return self._fetch_yfinance(symbol, timeframe, start, end)
        elif self.source == "ccxt":
            return self._fetch_ccxt(symbol, timeframe, start, end)
        else:
            raise ValueError(f"Unknown source: {self.source}")

    def _fetch_yfinance(
        self, symbol: str, timeframe: str, start: str, end: str
    ) -> pd.DataFrame:
        """Fetch from yfinance"""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, interval=timeframe)
            if df.empty:
                raise ValueError(f"No data for {symbol}")
            df.columns = [c.lower() for c in df.columns]
            df.index.name = "timestamp"
            return df
        except ImportError:
            raise ImportError("Install yfinance: pip install yfinance")

    def _fetch_ccxt(
        self, symbol: str, timeframe: str, start: str, end: str
    ) -> pd.DataFrame:
        """Fetch from ccxt (crypto-capable)"""
        try:
            import ccxt

            exchange = ccxt.binance()
            timeframe_map = {"1h": "1h", "15m": "15m", "5m": "5m", "1d": "1d"}
            tf = timeframe_map.get(timeframe, "1h")

            since = int(pd.Timestamp(start).timestamp() * 1000)
            ohlcv = exchange.fetch_ohlcv(symbol, tf, since=since, limit=1000)

            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            return df
        except ImportError:
            raise ImportError("Install ccxt: pip install ccxt")

    def fetch_multiple_pairs(
        self, symbols: List[str], timeframe: str = "1h", start: str = "2015-01-01"
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple pairs (Zenox: 16 pairs)"""
        data = {}
        for symbol in symbols:
            try:
                df = self.fetch_ohlcv(symbol, timeframe, start)
                data[symbol] = df
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
        return data

    def calculate_correlation_matrix(
        self, data_dict: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Calculate correlation matrix between pairs"""
        close_prices = pd.DataFrame()
        for symbol, df in data_dict.items():
            close_prices[symbol] = df["close"]
        return close_prices.corr()

    def save_data(self, df: pd.DataFrame, filepath: str):
        df.to_csv(filepath)

    def load_data(self, filepath: str) -> pd.DataFrame:
        return pd.read_csv(filepath, index_col=0, parse_dates=True)


class DataPreprocessor:
    """Clean and prepare data for model training"""

    def __init__(self, lookback: int = 30):
        self.lookback = lookback

    def create_sequences(self, data: np.ndarray, target_col: int = 3) -> tuple:
        """Create sequences for LSTM-CNN (30 bars lookback)"""
        X, y = [], []
        for i in range(self.lookback, len(data)):
            start_idx = i - self.lookback
            X.append(data[start_idx:i])
            y.append(data[i, target_col])
        return np.array(X), np.array(y)

    def train_test_split(
        self, df: pd.DataFrame, train_end: str = "2020-01-01", target_col: str = "close"
    ) -> tuple:
        """Split with gap validation (Aurum AI methodology)"""
        train = df[df.index < train_end]
        test = df[df.index >= train_end]

        train_data = train.values
        test_data = test.values

        X_train, y_train = self.create_sequences(train_data)
        X_test, y_test = self.create_sequences(test_data)

        return (X_train, X_test), (y_train, y_test), (train, test)

    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features for neural network"""
        normalized = df.copy()
        exclude_cols = ["open", "high", "low", "close", "volume"]

        for col in df.columns:
            if col not in exclude_cols:
                normalized[col] = (df[col] - df[col].mean()) / df[col].std()

        for col in ["open", "high", "low", "close"]:
            if col in normalized.columns:
                normalized[col] = (df[col] - df[col].mean()) / df[col].std()

        return normalized
