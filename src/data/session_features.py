"""Session behavior features for RTS AI Forex Trading System.

Computes session-type classification, volatility expansion,
volume surges, range contraction, weekend gap risk, and momentum bias.
"""

from typing import Dict, Optional

import pandas as pd


class SessionBehaviorEngine:
    """Computes session-based behavioral features from OHLCV data."""

    def __init__(self, symbol: str = ""):
        self.symbol = symbol

    def compute_session_features(
        self, df: Optional[pd.DataFrame], timestamp: Optional[float] = None
    ) -> Dict[str, float]:
        """Return session behavior features.

        Args:
            df: OHLCV DataFrame with optional ``timestamp`` column.
            timestamp: Unix timestamp in seconds (defaults to now).

        Returns:
            Dict of float-valued session features.
        """
        if timestamp is None:
            timestamp = pd.Timestamp.now("UTC").timestamp()

        ts = pd.Timestamp(timestamp, unit="s", tz="UTC")
        hour = ts.hour
        dayofweek = ts.dayofweek  # 0=Monday … 6=Sunday

        # Session type encoding
        if 12 <= hour < 16:
            session_type = 1.0  # ny_overlap
        elif 7 <= hour < 12:
            session_type = 2.0  # london_open
        elif hour >= 21 or hour < 8:
            session_type = 3.0  # asian
        else:
            session_type = 0.0  # overnight

        # Weekend gap risk: Friday close (>=20 UTC) or Sunday open (<22 UTC)
        weekend_gap_risk = (
            1.0
            if ((dayofweek == 4 and hour >= 20) or (dayofweek == 6 and hour < 22))
            else 0.0
        )

        if df is None or df.empty or len(df) < 10:
            return {
                "session_type": session_type,
                "london_vol_expansion": 1.0,
                "ny_overlap_volume_surge": 1.0,
                "asian_range_contraction": 1.0,
                "weekend_gap_risk": weekend_gap_risk,
                "session_momentum_bias": 0.0,
            }

        df = df.copy()
        # Extract hour from whichever datetime source is available
        if "timestamp" in df.columns:
            df["hour"] = pd.to_datetime(df["timestamp"], unit="s").dt.hour
        elif not isinstance(df.index, pd.DatetimeIndex):
            df["hour"] = pd.to_datetime(df.index).hour
        else:
            df["hour"] = df.index.hour

        # -- ATR helper (mean true range) ----------------------------------
        def _mean_tr(data: pd.DataFrame) -> float:
            if len(data) < 2:
                return 0.0
            tr1 = data["high"] - data["low"]
            tr2 = (data["high"] - data["close"].shift()).abs()
            tr3 = (data["low"] - data["close"].shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return float(tr.mean())

        london_mask = (df["hour"] >= 7) & (df["hour"] < 12)
        asian_mask = (df["hour"] >= 21) | (df["hour"] < 8)
        ny_mask = (df["hour"] >= 12) & (df["hour"] < 16)

        london_atr = _mean_tr(df[london_mask]) if london_mask.any() else 0.0
        asian_atr = _mean_tr(df[asian_mask]) if asian_mask.any() else 0.0
        london_vol_expansion = london_atr / asian_atr if asian_atr > 0 else 1.0

        # NY overlap volume surge vs 24 h average
        ny_vol = df.loc[ny_mask, "volume"].mean() if ny_mask.any() else 0.0
        avg_vol = df["volume"].mean() if df["volume"].sum() > 0 else 1.0
        ny_overlap_volume_surge = ny_vol / avg_vol if avg_vol > 0 else 1.0

        # Asian range contraction as % of full-day range
        asian_range = (
            (df.loc[asian_mask, "high"] - df.loc[asian_mask, "low"]).mean()
            if asian_mask.any()
            else 0.0
        )
        full_range = (df["high"] - df["low"]).mean() if len(df) > 0 else 1.0
        asian_range_contraction = asian_range / full_range if full_range > 0 else 1.0

        # Momentum bias: +1 London trend, -1 Asian mean-reversion
        if session_type == 2.0:  # london_open
            session_momentum_bias = 1.0
        elif session_type == 3.0:  # asian
            session_momentum_bias = -1.0
        else:
            session_momentum_bias = 0.0

        return {
            "session_type": session_type,
            "london_vol_expansion": london_vol_expansion,
            "ny_overlap_volume_surge": ny_overlap_volume_surge,
            "asian_range_contraction": asian_range_contraction,
            "weekend_gap_risk": weekend_gap_risk,
            "session_momentum_bias": session_momentum_bias,
        }
