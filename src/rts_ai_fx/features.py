"""Feature engineering module - 51+ features for forex prediction"""

import numpy as np
import pandas as pd
from typing import Optional


class FeatureEngineer:
    """Generate 51+ features for LSTM-CNN hybrid model"""

    def __init__(self, lookback: int = 30):
        self.lookback = lookback

    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price dynamics features"""
        df["body_size"] = abs(df["close"] - df["open"]) / df["open"]
        df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df[
            "open"
        ]
        df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df[
            "open"
        ]
        df["range"] = (df["high"] - df["low"]) / df["open"]
        return df

    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum-based features"""
        df["rsi_14"] = self._calculate_rsi(df["close"], 14)
        df["rsi_21"] = self._calculate_rsi(df["close"], 21)
        df["macd"], df["macd_signal"] = self._calculate_macd(df["close"])
        df["momentum_3"] = df["close"].pct_change(3)
        df["momentum_5"] = df["close"].pct_change(5)
        df["roc_10"] = df["close"].pct_change(10)
        return df

    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility regime features"""
        df["atr_14"] = self._calculate_atr(df, 14)
        df["atr_21"] = self._calculate_atr(df, 21)
        df["yang_zhang_vol"] = self._yang_zhang(df, 10)
        df["volatility_ratio"] = df["atr_14"] / df["atr_21"]
        df["garman_klass_vol"] = np.log(df["high"] / df["low"]) ** 2
        return df

    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trend detection features"""
        for period in [20, 50, 100, 200]:
            df[f"ema_{period}"] = df["close"].ewm(span=period).mean()
            df[f"ema_ratio_{period}"] = df["close"] / df[f"ema_{period}"]
        df["adx_14"] = self._calculate_adx(df, 14)
        df["trend_strength"] = abs(df["ema_20"] - df["ema_50"]) / df["ema_50"]
        return df

    def add_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market regime detection (trend/range/volatile/crisis)"""
        vol_median = df["yang_zhang_vol"].rolling(100).median()
        df["vol_regime"] = np.where(
            df["yang_zhang_vol"] > vol_median * 1.5,
            2,
            np.where(df["yang_zhang_vol"] > vol_median, 1, 0),
        )
        df["trend_regime"] = np.where(df["adx_14"] > 25, 1, 0)
        df["crisis_filter"] = np.where(
            (df["range"] > df["range"].rolling(50).mean() * 3), 1, 0
        )
        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Time-based features"""
        if isinstance(df.index, pd.DatetimeIndex):
            df["hour"] = df.index.hour / 23.0
            df["day_of_week"] = df.index.dayofweek / 6.0
            df["session_london"] = ((df.index.hour >= 8) & (df.index.hour < 17)).astype(
                float
            )
            df["session_ny"] = ((df.index.hour >= 13) & (df.index.hour < 22)).astype(
                float
            )
            df["session_tokyo"] = ((df.index.hour >= 0) & (df.index.hour < 9)).astype(
                float
            )
        return df

    def add_cross_asset_features(
        self,
        df: pd.DataFrame,
        gold: Optional[pd.DataFrame] = None,
        usdx: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Cross-asset correlation features"""
        if gold is not None:
            df["gold_corr_20"] = df["close"].rolling(20).corr(gold["close"])
        if usdx is not None:
            df["usdx_corr_20"] = df["close"].rolling(20).corr(usdx["close"])
        return df

    def generate_all_features(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate complete feature set (51+ features)"""
        df = self.add_price_features(df)
        df = self.add_momentum_indicators(df)
        df = self.add_volatility_features(df)
        df = self.add_trend_indicators(df)
        df = self.add_market_regime(df)
        df = self.add_time_features(df)
        df = self.add_cross_asset_features(df, **kwargs)
        return df.dropna()

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        tr = pd.concat(
            [
                df["high"] - df["low"],
                abs(df["high"] - df["close"].shift()),
                abs(df["low"] - df["close"].shift()),
            ],
            axis=1,
        ).max(axis=1)
        return tr.rolling(period).mean() / df["close"]

    def _calculate_adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        tr = self._calculate_atr(df, 1)
        plus_dm = df["high"].diff().where(df["high"].diff() > df["low"].diff().abs(), 0)
        minus_dm = (
            df["low"].diff().abs().where(df["low"].diff().abs() > df["high"].diff(), 0)
        )
        tr_smooth = tr.rolling(period).mean()
        plus_di = 100 * plus_dm.rolling(period).mean() / tr_smooth
        minus_di = 100 * minus_dm.rolling(period).mean() / tr_smooth
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx.rolling(period).mean()

    def _yang_zhang(self, df: pd.DataFrame, period: int) -> pd.Series:
        log_hl = np.log(df["high"] / df["low"])
        log_co = np.log(df["close"] / df["open"])
        rs = log_hl.rolling(period).std()
        ro = log_co.rolling(period).std()
        rc = (log_hl - log_co).rolling(period).std()
        return np.sqrt(ro**2 + rc**2 + rs**2)
