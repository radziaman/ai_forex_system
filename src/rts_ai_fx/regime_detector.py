"""
HMM-based Market Regime Detector — replaces hardcoded ADX thresholds.
Learns regime transition probabilities from historical data.
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False


class HMMRegimeDetector:
    """
    4-state HMM regime detector (trending, ranging, volatile, crisis).
    Fit on historical returns + volatility features.
    """

    REGIME_NAMES = ["trending", "ranging", "volatile", "crisis"]

    def __init__(self, n_regimes: int = 4, lookback: int = 60):
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.model: Optional[hmm.GaussianHMM] = None
        self.current_regime: Optional[str] = None
        self.regime_history: List[str] = []

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for HMM: returns, volatility, volume change."""
        prices = df["close"].values
        returns = np.diff(prices) / prices[:-1]
        vol = df.get("atr_14", df["close"].rolling(14).std()).values
        vol_ratio = vol / np.mean(vol[-60:]) if len(vol) > 60 else np.ones_like(vol)
        features = np.column_stack([
            returns,
            vol_ratio[1:],
            np.abs(returns),  # absolute returns (volatility proxy)
        ])
        features = np.nan_to_num(features, nan=0.0)
        # Remove inf
        features = np.clip(features, -10, 10)
        return features

    def fit(self, df: pd.DataFrame):
        """Fit HMM on historical data."""
        if not HMM_AVAILABLE:
            return
        features = self._extract_features(df)
        if len(features) < 100:
            return
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="diag",
            n_iter=200,
            random_state=42,
            tol=1e-4,
        )
        self.model.fit(features)

    def detect_regime(self, df: pd.DataFrame) -> str:
        """Detect current regime using Viterbi decoding."""
        if not HMM_AVAILABLE or self.model is None:
            return self._fallback_regime(df)
        features = self._extract_features(df)
        if len(features) < 5:
            return "ranging"
        # Use last window of features for decoding
        recent = features[-self.lookback:]
        if len(recent) < 5:
            return "ranging"
        hidden_states = self.model.predict(recent)
        current_state = int(hidden_states[-1])
        # Map states to regime names using mean returns ordering
        if self.model.means_ is not None:
            mean_returns = self.model.means_[:, 0]
            state_order = np.argsort(mean_returns)
            regime_idx = np.where(state_order == current_state)[0][0]
            if regime_idx >= len(self.REGIME_NAMES):
                regime_idx = len(self.REGIME_NAMES) - 1
            regime = self.REGIME_NAMES[regime_idx]
        else:
            regime = self.REGIME_NAMES[current_state % len(self.REGIME_NAMES)]
        self.current_regime = regime
        self.regime_history.append(regime)
        return regime

    def _fallback_regime(self, df: pd.DataFrame) -> str:
        """Fallback rule-based regime detection when HMM unavailable."""
        recent = df.iloc[-self.lookback:] if len(df) >= self.lookback else df
        if len(recent) < 10:
            return "ranging"
        adx = recent.get("adx_14", pd.Series([25] * len(recent))).iloc[-1]
        bb_width = recent.get("bb_width", pd.Series([0.02] * len(recent))).iloc[-1]
        rvol = recent.get("vol_ratio", pd.Series([1.0] * len(recent))).iloc[-1]
        if pd.isna(adx):
            adx = 25
        if pd.isna(bb_width):
            bb_width = 0.02
        if pd.isna(rvol):
            rvol = 1.0
        if rvol > 2.0 or (bb_width > 0.05 and adx < 20):
            regime = "crisis"
        elif adx > 30 and bb_width > 0.03:
            regime = "trending"
        elif adx < 20 and bb_width < 0.02:
            regime = "ranging"
        elif rvol > 1.5:
            regime = "volatile"
        else:
            regime = "ranging"
        self.current_regime = regime
        self.regime_history.append(regime)
        return regime

    def get_regime_params(self, regime: str) -> dict:
        params = {
            "trending": {"sl_pct": 0.025, "tp_pct": 0.05, "pos_mult": 1.0, "min_conf": 0.60},
            "ranging": {"sl_pct": 0.015, "tp_pct": 0.03, "pos_mult": 1.0, "min_conf": 0.65},
            "volatile": {"sl_pct": 0.02, "tp_pct": 0.04, "pos_mult": 0.5, "min_conf": 0.75},
            "crisis": {"sl_pct": 0.01, "tp_pct": 0.02, "pos_mult": 0.0, "min_conf": 0.95},
        }
        return params.get(regime, params["ranging"])

    def should_trade(self, regime: str) -> bool:
        return self.get_regime_params(regime)["pos_mult"] > 0
