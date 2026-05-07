"""
HMM-based Market Regime Detector — replaces hardcoded ADX thresholds.
Learns regime transition probabilities from historical data.
Enhanced with smart regime transition trading (Enhancement #12).
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict

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
            "trending": {"sl_atr": 2.0, "tp_atr": 4.0, "pos_mult": 1.0, "min_conf": 0.60},
            "ranging": {"sl_atr": 1.5, "tp_atr": 3.0, "pos_mult": 1.0, "min_conf": 0.65},
            "volatile": {"sl_atr": 2.5, "tp_atr": 5.0, "pos_mult": 0.5, "min_conf": 0.75},
            "crisis": {"sl_atr": 1.0, "tp_atr": 2.0, "pos_mult": 0.0, "min_conf": 0.95},
        }
        return params.get(regime, params["ranging"])
    
    def get_position_size_multiplier(self, regime: str, sentiment_score: float = 0.0) -> float:
        """Get position size multiplier based on regime and sentiment."""
        base_mult = self.get_regime_params(regime).get("pos_mult", 1.0)
        # Adjust based on sentiment (positive sentiment = increase size)
        sentiment_adj = 0.5 + 0.5 * max(-1, min(1, sentiment_score))
        return base_mult * sentiment_adj

    def should_trade(self, regime: str) -> bool:
        return self.get_regime_params(regime)["pos_mult"] > 0

    # Enhancement #12: Smart Regime Transition Trading
    def detect_transition(self, symbol: str = None) -> Dict:
        """
        Detect regime transitions and provide trading signals.
        Trade the transition, not just the regime.
        """
        if not self.regime_history or len(self.regime_history) < 2:
            return {
                "in_transition": False,
                "from_regime": None,
                "to_regime": None,
                "confidence": 0.0,
                "should_trade": False,
                "sl_tp_adjustment": 1.0,
            }

        current = self.regime_history[-1]
        previous = self.regime_history[-2]

        in_transition = current != previous
        transition_key = f"{previous}_to_{current}"

        # Calculate transition probability (simplified)
        transition_count = sum(
            1 for i in range(1, len(self.regime_history))
            if self.regime_history[i] != self.regime_history[i-1]
        )
        total_observations = len(self.regime_history) - 1
        transition_freq = transition_count / max(total_observations, 1)

        # Higher confidence if transition is rare (more predictable)
        confidence = min(1.0, transition_freq * 5) if in_transition else 0.0

        # Adjust SL/TP during transitions (higher volatility)
        if in_transition:
            # Volatile transitions = wider stops
            if current == "volatile" or previous == "volatile":
                sl_tp_adj = 1.5
            # Trend-to-range = tighter stops
            elif (previous == "trending" and current == "ranging") or \
                 (previous == "ranging" and current == "trending"):
                sl_tp_adj = 1.2
            else:
                sl_tp_adj = 1.3
        else:
            sl_tp_adj = 1.0

        # Should we trade during transition?
        # Yes for trending->volatile (momentum)
        # No for volatile->crisis (too dangerous)
        if in_transition:
            if previous == "volatile" and current == "crisis":
                should_trade = False
            elif previous == "crisis" and current == "ranging":
                should_trade = False  # Wait for stability
            else:
                should_trade = True
        else:
            should_trade = True

        return {
            "in_transition": in_transition,
            "from_regime": previous,
            "to_regime": current,
            "transition_key": transition_key,
            "confidence": confidence,
            "should_trade": should_trade,
            "sl_tp_adjustment": sl_tp_adj,
            "transition_frequency": transition_freq,
        }

    def get_pause_conditions(self) -> Dict:
        """
        Determine if we should pause trading due to high uncertainty.
        Enhancement #12: Pause during high-uncertainty transitions.
        """
        if not self.regime_history or len(self.regime_history) < 5:
            return {"should_pause": False, "reason": "insufficient_history"}

        recent = self.regime_history[-5:]

        # Count transitions in last 5 observations
        transitions = sum(
            1 for i in range(1, len(recent))
            if recent[i] != recent[i-1]
        )

        # Pause if too many transitions (high uncertainty)
        if transitions >= 3:
            return {
                "should_pause": True,
                "reason": f"high_transition_frequency: {transitions} transitions in 5 periods",
                "transitions": transitions,
            }

        # Pause if current regime is crisis
        if recent[-1] == "crisis":
            return {
                "should_pause": True,
                "reason": "crisis_regime_detected",
                "regime": "crisis",
            }

        return {"should_pause": False, "reason": "normal_conditions"}

    def get_transition_trading_signal(self, symbol: str = None) -> Dict:
        """
        Get trading signal specifically for regime transitions.
        Returns action to take during transitions.
        """
        transition = self.detect_transition(symbol)

        if not transition["in_transition"]:
            return {"action": "HOLD", "reason": "no_transition"}

        from_r = transition["from_regime"]
        to_r = transition["to_regime"]

        # Trending -> Volatile: Trade momentum
        if from_r == "trending" and to_r == "volatile":
            return {
                "action": "BUY" if self._get_trend_direction() > 0 else "SELL",
                "reason": "momentum_during_transition",
                "confidence": transition["confidence"],
                "use_wider_stops": True,
            }

        # Ranging -> Trending: Follow the new trend
        elif from_r == "ranging" and to_r == "trending":
            return {
                "action": "BUY" if self._get_trend_direction() > 0 else "SELL",
                "reason": "breakout_from_range",
                "confidence": transition["confidence"] * 1.2,
                "use_wider_stops": False,
            }

        # Volatile -> Ranging: Mean reversion
        elif from_r == "volatile" and to_r == "ranging":
            return {
                "action": "HOLD",
                "reason": "wait_for_range_confirmation",
                "confidence": 0.5,
                "use_wider_stops": False,
            }

        # Any -> Crisis: Get out
        elif to_r == "crisis":
            return {
                "action": "CLOSE_ALL",
                "reason": "entering_crisis_regime",
                "confidence": 1.0,
                "use_wider_stops": False,
            }

        else:
            return {
                "action": "HOLD",
                "reason": f"unhandled_transition: {transition['transition_key']}",
                "confidence": 0.3,
            }

    def _get_trend_direction(self) -> int:
        """Get current trend direction (1 for up, -1 for down)."""
        if not self.regime_history:
            return 0
        # Simplified: check last regime
        if self.regime_history[-1] == "trending":
            return 1
        return 0
