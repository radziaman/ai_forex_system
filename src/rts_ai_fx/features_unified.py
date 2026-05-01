"""
Unified feature pipeline — single source of truth for all feature engineering.
Multi-timeframe support, proper cyclical encoding, ADX, market microstructure.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.ewm(span=period).mean() / atr)
    minus_di = 100 * (minus_dm.abs().ewm(span=period).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(period).mean()


def _hurst(series: pd.Series, max_lag: int = 20) -> float:
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
    if not tau or any(t == 0 for t in tau):
        return 0.5
    return np.polyfit(np.log(lags), np.log(tau), 1)[0]


def apply_cyclical_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Replace raw hour/dow/month with sin/cos encoding."""
    if "hour" in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df.drop("hour", axis=1, inplace=True)
    if "day_of_week" in df.columns:
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df.drop("day_of_week", axis=1, inplace=True)
    if "month" in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df.drop("month", axis=1, inplace=True)
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute full feature set: technical, momentum, volatility, trend, regime."""
    df = df.copy()
    # Price dynamics
    df["body"] = abs(df["close"] - df["open"]) / df["open"].replace(0, np.nan)
    df["range"] = (df["high"] - df["low"]) / df["open"].replace(0, np.nan)
    df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(1)) / df["open"].replace(0, np.nan)
    df["lower_shadow"] = (df[["open", "close"]].min(1) - df["low"]) / df["open"].replace(0, np.nan)

    # Momentum
    df["rsi_14"] = _rsi(df["close"], 14)
    df["rsi_21"] = _rsi(df["close"], 21)
    e12 = df["close"].ewm(span=12).mean()
    e26 = df["close"].ewm(span=26).mean()
    df["macd"] = e12 - e26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["mom_1"] = df["close"].pct_change(1)
    df["mom_5"] = df["close"].pct_change(5)
    df["mom_10"] = df["close"].pct_change(10)

    # Volatility
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_21"] = tr.rolling(21).mean()
    df["vol_ratio"] = df["atr_14"] / df["atr_21"].replace(0, np.nan)
    df["volatility_20"] = df["close"].rolling(20).std() / df["close"].replace(0, np.nan)
    df["bb_mid"] = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, np.nan)
    df["bb_pos"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, 0.5)

    # Trend
    for p in [20, 50, 200]:
        df[f"ema_{p}"] = df["close"].ewm(span=p).mean()
        df[f"ema_ratio_{p}"] = df["close"] / df[f"ema_{p}"].replace(0, np.nan)
    df["adx_14"] = _adx(df, 14)
    # SMA distances
    for p in [20, 50, 200]:
        sma = df["close"].rolling(p).mean()
        df[f"dist_sma{p}"] = (df["close"] - sma) / sma.replace(0, np.nan)

    # Stochastic
    low14 = df["low"].rolling(14).min()
    high14 = df["high"].rolling(14).max()
    df["stoch_k"] = (df["close"] - low14) / (high14 - low14).replace(0, 1) * 100
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # Time features — handle both DatetimeIndex and timestamp columns
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            dt_series = pd.to_datetime(df["timestamp"], unit="s")
            df["hour"] = dt_series.dt.hour
            df["day_of_week"] = dt_series.dt.dayofweek
            df["month"] = dt_series.dt.month
            df["quarter"] = dt_series.dt.quarter
    else:
        df["hour"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month
        df["quarter"] = df.index.quarter

    df = apply_cyclical_encoding(df)

    # Hurst exponent (rolling estimate)
    df["hurst"] = df["close"].rolling(50).apply(lambda x: _hurst(x.values), raw=False)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


TARGET_COLS = {"open", "high", "low", "close", "volume"}


class FeaturePipeline:
    """
    Multi-timeframe feature pipeline.
    Computes features independently on each timeframe, then fuses via flattening.
    """

    def __init__(self, lookback: int = 30, timeframes: Optional[List[str]] = None):
        self.lookback = lookback
        self.timeframes = timeframes or ["1h"]
        self._means: Dict[str, np.ndarray] = {}
        self._stds: Dict[str, np.ndarray] = {}
        self._feature_cols: List[str] = []

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if c not in TARGET_COLS]

    def fit(self, dfs: Dict[str, pd.DataFrame]):
        """Fit scalers on training data for each timeframe."""
        all_features = []
        for tf in self.timeframes:
            df = dfs.get(tf)
            if df is None or len(df) < self.lookback + 5:
                continue
            processed = compute_features(df)
            cols = self._get_feature_columns(processed)
            if not self._feature_cols:
                self._feature_cols = cols
            vals = processed[cols].values
            # Remove NaN rows
            mask = ~np.any(np.isnan(vals), axis=1)
            vals = vals[mask]
            if len(vals) > self.lookback:
                all_features.append(vals)
        if not all_features:
            return
        stacked = np.vstack(all_features)
        self._means[tf] = np.nanmean(stacked, axis=0)
        self._stds[tf] = np.nanstd(stacked, axis=0)
        self._stds[tf] = np.where(self._stds[tf] == 0, 1.0, self._stds[tf])

    def transform(self, dfs: Dict[str, pd.DataFrame]) -> Optional[np.ndarray]:
        """Transform multi-timeframe data into a single fused feature vector (last lookback window)."""
        tf_vectors = []
        for tf in self.timeframes:
            df = dfs.get(tf)
            if df is None or len(df) < self.lookback:
                return None
            processed = compute_features(df)
            cols = self._feature_cols if self._feature_cols else self._get_feature_columns(processed)
            missing = [c for c in cols if c not in processed.columns]
            if missing:
                return None
            vals = processed[cols].values
            # Z-score normalize
            if tf in self._means and tf in self._stds:
                vals = (vals - self._means[tf]) / self._stds[tf]
            # Get last lookback window
            window = vals[-self.lookback:]
            if len(window) < self.lookback:
                return None
            # Mean-pool to handle any remaining NaN
            window = np.nan_to_num(window, nan=0.0)
            tf_vectors.append(window)
        if not tf_vectors:
            return None
        # Flatten: (lookback, n_features * n_timeframes)
        return np.concatenate(tf_vectors, axis=1)

    def create_sequences(self, dfs: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for supervised learning."""
        if not self._feature_cols:
            self.fit(dfs)
            if not self._feature_cols:
                return np.array([]), np.array([])
        # Get the primary timeframe data for labels
        primary_tf = self.timeframes[0]
        primary = dfs.get(primary_tf)
        if primary is None:
            return np.array([]), np.array([])
        prices = primary["close"].values
        sequences, targets = [], []
        for i in range(self.lookback, len(prices)):
            tf_data = {}
            for tf in self.timeframes:
                df = dfs.get(tf)
                if df is None:
                    continue
                start = max(0, i - self.lookback)
                tf_data[tf] = df.iloc[start:i]
            feat = self.transform(tf_data)
            if feat is not None:
                sequences.append(feat.flatten())
                targets.append(prices[i])
        if not sequences:
            return np.array([]), np.array([])
        return np.array(sequences), np.array(targets)
