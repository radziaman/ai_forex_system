"""
Unified feature pipeline — single source of truth for all feature engineering.
Multi-timeframe support, proper cyclical encoding, ADX, market microstructure,
order flow dynamics, and cross-asset features.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable


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


def compute_microstructure_features(df: pd.DataFrame, tick_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Market microstructure features from tick data and OHLCV."""
    df = df.copy()
    if "volume" in df.columns and df["volume"].sum() > 0:
        vol = df["volume"].values
        df["log_volume"] = np.log1p(vol)
        df["volume_ma_ratio"] = vol / (pd.Series(vol).rolling(20).mean().values + 1e-8)
        df["volume_shock"] = (vol - pd.Series(vol).rolling(5).mean().values) / (pd.Series(vol).rolling(5).std().values + 1e-8)
        dollar_vol = vol * df["close"].values
        df["dollar_vol_rank"] = pd.Series(dollar_vol).rank(pct=True).values
    if all(c in df.columns for c in ["open", "high", "low", "close"]):
        high_low = df["high"].values - df["low"].values
        high_c = np.abs(df["high"].values - df["close"].shift(1).values)
        low_c = np.abs(df["low"].values - df["close"].shift(1).values)
        tr = np.maximum(high_low, np.maximum(high_c, low_c))
        df["tr_scaled"] = tr / df["close"].values
        df["position_in_bar"] = (df["close"].values - df["low"].values) / (df["high"].values - df["low"].values + 1e-8)
        df["intraday_volatility"] = (df["high"].values - df["low"].values) / df["open"].values
        df["gap"] = (df["open"].values - df["close"].shift(1).values) / df["close"].shift(1).values
    if tick_data is not None and len(tick_data) > 100:
        df = _add_tick_features(df, tick_data)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def _add_tick_features(df: pd.DataFrame, ticks: pd.DataFrame) -> pd.DataFrame:
    """Add tick-level features: order flow imbalance, trade intensity, CVD."""
    if "price" not in ticks.columns and "mid" in ticks.columns:
        ticks = ticks.copy()
        ticks["price"] = ticks["mid"]
    if "price" not in ticks.columns:
        return df
    ticks = ticks.dropna(subset=["price"]).copy()
    if len(ticks) < 100:
        return df
    tick_prices = ticks["price"].values
    tick_volumes = ticks.get("volume", pd.Series(np.ones(len(ticks)))).values
    price_changes = np.diff(tick_prices)
    buy_volume = np.sum(tick_volumes[1:][price_changes > 0])
    sell_volume = np.sum(tick_volumes[1:][price_changes < 0])
    total_vol = buy_volume + sell_volume
    if total_vol > 0:
        df["ofi"] = (buy_volume - sell_volume) / total_vol
        df["buy_ratio"] = buy_volume / total_vol
    cvd = np.cumsum(np.where(price_changes > 0, tick_volumes[1:], np.where(price_changes < 0, -tick_volumes[1:], 0)))
    if len(cvd) > 0:
        df["cvd"] = cvd[-1]
        if len(cvd) > 20:
            df["cvd_slope"] = (cvd[-1] - cvd[-20]) / 20
            df["cvd_volatility"] = np.std(cvd[-20:])
    trade_intensity = len(ticks) / max(
        ticks["timestamp"].max() - ticks["timestamp"].min(), 1
    )
    df["trade_intensity"] = trade_intensity
    if len(price_changes) > 10:
        df["tick_volatility"] = np.std(price_changes[-100:])
        df["avg_tick_size"] = np.mean(np.abs(price_changes[-100:]))
    return df


def compute_cross_asset_features(
    df: pd.DataFrame,
    external_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Add cross-asset features: correlations, ratios, spreads."""
    df = df.copy()
    if external_data is None:
        return df
    for name, ext_df in external_data.items():
        if ext_df is None or len(ext_df) < 20:
            continue
        ext_close = ext_df["close"].values
        min_len = min(len(df), len(ext_close))
        if min_len < 20:
            continue
        our_close = df["close"].values[-min_len:]
        their_close = ext_close[-min_len:]
        corr = np.corrcoef(our_close, their_close)[0, 1]
        df[f"corr_{name}"] = corr
        ratio = our_close / (their_close + 1e-10)
        df[f"ratio_{name}"] = ratio[-1] if hasattr(ratio, "__len__") else ratio
        ret_diff = (
            np.diff(our_close) / our_close[:-1]
            - np.diff(their_close) / their_close[:-1]
        )
        df[f"ret_divergence_{name}"] = np.mean(ret_diff[-10:]) if len(ret_diff) >= 10 else 0.0
        spread = our_close - their_close
        z = (spread[-1] - np.mean(spread)) / (np.std(spread) + 1e-10)
        df[f"zscore_{name}"] = z
    return df


TARGET_COLS = {"open", "high", "low", "close", "volume"}


class FeaturePipeline:
    """
    Multi-timeframe feature pipeline.
    Computes features independently on each timeframe, then fuses via flattening.
    Incorporates microstructure, cross-asset, and external signal features.
    """

    def __init__(
        self,
        lookback: int = 30,
        timeframes: Optional[List[str]] = None,
        use_microstructure: bool = True,
        use_cross_asset: bool = False,
        cross_asset_data: Optional[Dict[str, pd.DataFrame]] = None,
        tick_buffer: Optional[List[Dict]] = None,
    ):
        self.lookback = lookback
        self.timeframes = timeframes or ["1h"]
        self.use_microstructure = use_microstructure
        self.use_cross_asset = use_cross_asset
        self.cross_asset_data = cross_asset_data or {}
        self.tick_buffer = tick_buffer or []
        self._means: Dict[str, np.ndarray] = {}
        self._stds: Dict[str, np.ndarray] = {}
        self._feature_cols: List[str] = []
        self._external_feature_columns: List[str] = []

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if c not in TARGET_COLS]

    def _extract_symbol_data(
        self, dfs: Dict[str, pd.DataFrame], symbol: str = "EURUSD"
    ) -> Dict[str, pd.DataFrame]:
        """Extract per-symbol data from nested dict or pass through flat dict."""
        if symbol in dfs:
            return dfs[symbol]
        first_val = next(iter(dfs.values()))
        if isinstance(first_val, dict):
            return dfs.get(symbol, {})
        return dfs

    def _norm_key(self, symbol: str, tf: str) -> str:
        return f"{symbol}_{tf}"

    def fit(self, dfs: Dict[str, pd.DataFrame], symbol: str = "EURUSD"):
        """Fit per-symbol scalers on training data for each timeframe."""
        symbol_data = self._extract_symbol_data(dfs, symbol)
        all_features = []
        for tf in self.timeframes:
            df = symbol_data.get(tf)
            if df is None or len(df) < self.lookback + 5:
                continue
            processed = compute_features(df)
            if self.use_microstructure:
                processed = compute_microstructure_features(processed)
            cols = self._get_feature_columns(processed)
            if not self._feature_cols:
                self._feature_cols = cols
            vals = processed[cols].values
            mask = ~np.any(np.isnan(vals), axis=1)
            vals = vals[mask]
            if len(vals) > self.lookback:
                all_features.append(vals)
        if not all_features:
            return
        stacked = np.vstack(all_features)
        key = self._norm_key(symbol, tf)
        self._means[key] = np.nanmean(stacked, axis=0)
        self._stds[key] = np.nanstd(stacked, axis=0)
        self._stds[key] = np.where(self._stds[key] == 0, 1.0, self._stds[key])

    def fit_all(self, dfs: Dict[str, Dict[str, pd.DataFrame]]):
        """Fit scalers for all symbols at once."""
        for symbol in dfs:
            self.fit(dfs, symbol)

    def transform(
        self,
        dfs: Dict[str, pd.DataFrame],
        symbol: str = "EURUSD",
        tick_buffer: Optional[List[Dict]] = None,
        external_signals: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Transform multi-timeframe data into a fused feature vector."""
        symbol_data = self._extract_symbol_data(dfs, symbol)
        tf_vectors = []
        for tf in self.timeframes:
            df = symbol_data.get(tf)
            if df is None or len(df) < self.lookback:
                return None
            processed = compute_features(df)
            if self.use_microstructure:
                ticks_df = None
                if tick_buffer and len(tick_buffer) > 100:
                    ticks_df = pd.DataFrame(tick_buffer)
                processed = compute_microstructure_features(processed, ticks_df)
            if self.use_cross_asset and self.cross_asset_data:
                processed = compute_cross_asset_features(processed, self.cross_asset_data)
            cols = self._feature_cols if self._feature_cols else self._get_feature_columns(processed)
            missing = [c for c in cols if c not in processed.columns]
            if missing:
                return None
            vals = processed[cols].values
            key = self._norm_key(symbol, tf)
            if key in self._means and key in self._stds:
                vals = (vals - self._means[key]) / self._stds[key]
            window = vals[-self.lookback:]
            if len(window) < self.lookback:
                return None
            window = np.nan_to_num(window, nan=0.0)
            tf_vectors.append(window)
        if not tf_vectors:
            return None
        fused = np.concatenate(tf_vectors, axis=1)
        if external_signals is not None:
            ext_tiled = np.tile(external_signals, (fused.shape[0], 1))
            fused = np.concatenate([fused, ext_tiled], axis=1)
        return fused

    def create_sequences(
        self,
        dfs: Dict[str, pd.DataFrame],
        symbol: str = "EURUSD",
        tick_buffer: Optional[List[Dict]] = None,
        external_signals: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for supervised learning."""
        symbol_data = self._extract_symbol_data(dfs, symbol)
        if not self._feature_cols:
            self.fit(dfs, symbol)
            if not self._feature_cols:
                return np.array([]), np.array([])
        primary_tf = self.timeframes[0]
        primary = symbol_data.get(primary_tf)
        if primary is None:
            return np.array([]), np.array([])
        prices = primary["close"].values
        sequences, targets = [], []
        for i in range(self.lookback, len(prices)):
            tf_data = {}
            for tf in self.timeframes:
                df = symbol_data.get(tf)
                if df is None:
                    continue
                start = max(0, i - self.lookback)
                tf_data[tf] = df.iloc[start:i]
            feat = self.transform(tf_data, symbol, tick_buffer, external_signals)
            if feat is not None:
                sequences.append(feat.flatten())
                targets.append(prices[i])
        if not sequences:
            return np.array([]), np.array([])
        return np.array(sequences), np.array(targets)

    def fit_transform(
        self,
        dfs: Dict[str, pd.DataFrame],
        symbol: str = "EURUSD",
        tick_buffer: Optional[List[Dict]] = None,
        external_signals: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform in one call."""
        self.fit(dfs, symbol)
        return self.create_sequences(dfs, symbol, tick_buffer, external_signals)


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
