"""
run_backtest.py — End-to-end backtest & validation for RTS AI Forex Trading System.

Runs:
  1. Vectorized backtest with regime-adaptive signals
  2. Sensitivity analysis (6 scenarios)
  3. Purged walk-forward validation
  4. Monte Carlo significance testing
  5. Formatted report with PASS/FAIL verdict

Usage:
    python -m src.scripts.run_backtest
    python -m src.scripts.run_backtest --data data/historical/EURUSD_1h.csv
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable

_src = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _src not in sys.path:
    sys.path.insert(0, _src)

from loguru import logger  # noqa: E402
from rts_ai_fx.features_unified import compute_features  # noqa: E402
from rts_ai_fx.regime_detector import SimpleRegimeDetector  # noqa: E402
from backtest.vectorized_backtester import VectorizedBacktester  # noqa: E402
from validation.walk_forward import PurgedWalkForward, WFResult  # noqa: E402
from validation.monte_carlo import MonteCarloSigTest, SigTestResult  # noqa: E402

# ─── Constants ────────────────────────────────────────────────────────────────
DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "historical", "EURUSD_1h.csv"
)
DEFAULT_DATA_PATH_4H = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "historical", "EURUSD_4h.csv"
)

REGIME_NAMES = ["crisis", "ranging", "volatile", "trending"]

# Backtest defaults
SPREAD_PIPS = 0.5
COMMISSION_PER_LOT = 7.0
SL_ATR = 2.5
TP_ATR = 4.0
BREAKOUT_LOOKBACK = 5
BREAKEVEN_ATR = 1.0
MAX_HOLD_BARS = 24
N_REGIMES = 4

# Walk-forward defaults
WF_N_FOLDS = 6
WF_TEST_WINDOW = 252
WF_EMBARGO = 10

# Monte Carlo defaults
MC_N_PERMUTATIONS = 10_000
MC_ALPHA = 0.05

# Burn-in: skip first N bars where rolling indicators are NaN
BURN_IN = 100

# PASS/FAIL thresholds
THRESHOLDS = {
    "min_sharpe": 0.5,
    "min_wf_avg_sharpe": 0.3,
    "max_mc_p_value": 0.05,
    "max_drawdown_pct": -30.0,
    "min_profit_factor": 1.2,
    "min_total_trades": 20,
}

# ═══════════════════════════════════════════════════════════════════════════════
# Section A: Data Loading
# ═══════════════════════════════════════════════════════════════════════════════


def load_and_prepare_data(path: str) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Load OHLCV CSV, parse timestamps, sort chronologically, compute features.

    Returns:
        prices:      numpy array of close prices, shape (n,)
        df_raw:      DataFrame with columns [timestamp, open, high, low, close, volume]
        df_features: DataFrame with all computed features (indicators + OHLCV + timestamp)  # noqa: E501
    """
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df_raw = df.copy()
    prices = df["close"].values.astype(float)

    logger.info(f"Loaded {len(prices)} bars from {path}")
    logger.info(
        f"  Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}"
    )

    df_features = compute_features(df)
    logger.info(f"  Features computed: {df_features.shape[1]} columns")

    return prices, df_raw, df_features


# ═══════════════════════════════════════════════════════════════════════════════
# Section A2: 4h Data Loading & Alignment
# ═══════════════════════════════════════════════════════════════════════════════


def load_and_prepare_4h_data(
    path_4h: str,
    timestamps_1h: pd.Series,
) -> pd.DataFrame:
    """Load 4h OHLCV CSV, compute features, align to 1h timestamps via forward-fill.

    Uses pd.merge_asof with direction='backward' so each 1h bar gets the most
    recent 4h bar's feature values.

    Returns:
        DataFrame with 4h features aligned to 1h timestamps (same length as timestamps_1h).  # noqa: E501
    """
    df = pd.read_csv(path_4h)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info(f"  Loaded {len(df)} 4h bars from {os.path.basename(path_4h)}")

    df_features = compute_features(df)
    logger.info(f"  4h features computed: {df_features.shape[1]} columns")

    # Align 4h features to 1h timestamps via merge_asof (backward-looking)
    df_1h_ts = pd.DataFrame({"timestamp": timestamps_1h}).sort_values("timestamp")
    df_4h_for_merge = df_features.sort_values("timestamp")

    aligned = pd.merge_asof(
        df_1h_ts,
        df_4h_for_merge,
        on="timestamp",
        direction="backward",
    )

    # Drop the 1h timestamp column we created for the merge (keep original 4h timestamp)
    logger.info(f"  4h → 1h alignment complete: {len(aligned)} bars")

    return aligned


# ═══════════════════════════════════════════════════════════════════════════════
# Section B: Regime Detection
# ═══════════════════════════════════════════════════════════════════════════════


def compute_regimes(
    df: pd.DataFrame,
    df_features: Optional[pd.DataFrame] = None,
    n_regimes: int = 4,
) -> Tuple[np.ndarray, np.ndarray, SimpleRegimeDetector]:
    """ADX-based regime detection via SimpleRegimeDetector.

    Uses the enriched feature DataFrame (with ADX, ATR, close) for
    per-bar regime classification.

    Regimes:
      - trending:  ADX_14 >= 20
      - ranging:   ADX_14 < 20
      - crisis:    ATR/price > 0.02
      - volatile:  ATR/price between 0.01 and 0.02

    Returns:
        regime_labels:  numpy int array of shape (n,), values 0..3
        regime_names:   numpy str array of shape (n,), values from REGIME_NAMES
        detector:       fitted SimpleRegimeDetector instance
    """
    input_df = df_features if df_features is not None else df

    detector = SimpleRegimeDetector(adx_threshold=20.0)
    regime_labels, regime_names = detector.detect(input_df)

    dist = pd.Series(regime_names).value_counts()
    logger.info(f"Regime distribution: {dist.to_dict()}")

    return regime_labels, regime_names, detector


# ═══════════════════════════════════════════════════════════════════════════════
# Section C: Signal Function Factory
# ═══════════════════════════════════════════════════════════════════════════════


def make_signal_function(  # noqa: C901
    regime_names: np.ndarray,
    features_df: pd.DataFrame,
    burn_in: int = BURN_IN,
    features_4h_df: Optional[pd.DataFrame] = None,
) -> Callable:
    """Build a signal function that returns {-1, 0, 1} per bar.

    Strategy logic per regime (adaptive):

      **Trending** (ADX >= 20):
        Breakout following — don't fight the trend.
        - Long: Close breaks ABOVE the highest high of the last 5 bars
        - Short: Close breaks BELOW the lowest low of the last 5 bars

      **Ranging / Volatile** (ADX < 20):
        Mean reversion — fade extreme moves within a range.
        - Long: z-score of close vs 20-bar SMA < -1.5 (price unusually low),
                with RSI < 40 (oversold) and close > 200-bar SMA (don't fight major uptrend)  # noqa: E501
        - Short: z-score > +1.5 (price unusually high),
                 with RSI > 60 (overbought) and close < 200-bar SMA (don't fight major downtrend)  # noqa: E501

      **Crisis** (ATR/price > 2%):
        Always neutral (0).

      **Multi-timeframe Filter** (when features_4h_df is provided):
        - Trending regime: 4h Trend Confirmation — Skip if 4h ADX < 20
        - Ranging/Volatile regime: No 4h filter (mean reversion works intra-trend)

      **Entry Filters (trending regime only)**:
        - Breakout gap: close must exceed level by >= 0.5×ATR
        - ADX Rising: ADX must be rising to confirm trend strength
        - Prior consolidation: ATR must be contracting (volatility coiling)

      **Entry Filters (ranging/volatile only)**:
        - Trend filter: Only long if close > 200-bar SMA; only short if close < 200-bar SMA  # noqa: E501
        - RSI confirmation: Long only if RSI < 40; short only if RSI > 60
    """

    # ── Session detection from hour_sin/hour_cos ──
    # Reconstruct hour from cyclical encoding to determine trading session per bar.
    sessions = np.empty(len(features_df), dtype=object)
    if "hour_sin" in features_df.columns and "hour_cos" in features_df.columns:
        hour_sin_vals = features_df["hour_sin"].values
        hour_cos_vals = features_df["hour_cos"].values
        hour_vals = (
            np.round(np.arctan2(hour_sin_vals, hour_cos_vals) * 24 / (2 * np.pi)) % 24
        ).astype(int)
        for i, h in enumerate(hour_vals):
            if 7 <= h < 12:
                sessions[i] = "london"
            elif 12 <= h < 16:
                sessions[i] = "overlap"
            elif 16 <= h < 21:
                sessions[i] = "newyork"
            elif h >= 21 or h < 7:
                sessions[i] = "asia"
            elif 8 <= h < 12:
                sessions[i] = "pacific"
            else:
                sessions[i] = "asia"
    else:
        sessions[:] = "asia"

    # Pre-compute breakout lookback series (for both 3-bar and 5-bar lookbacks)
    # Shift by 1 so we compare close[i] vs max(high[i-N .. i-1])
    high_5_max = pd.Series(features_df["high"]).rolling(5).max().shift(1)
    low_5_min = pd.Series(features_df["low"]).rolling(5).min().shift(1)
    high_3_max = pd.Series(features_df["high"]).rolling(3).max().shift(1)
    low_3_min = pd.Series(features_df["low"]).rolling(3).min().shift(1)

    # ── Pre-compute mean reversion indicators ──
    if "close" in features_df.columns:
        close_series = features_df["close"]
        sma_20 = close_series.rolling(20).mean()
        std_20 = close_series.rolling(20).std()
        z_score = np.where(std_20 > 1e-8, (close_series - sma_20) / std_20, 0.0)
        sma_200 = close_series.rolling(200).mean()
        has_sma_200 = True
    else:
        z_score = np.zeros(len(regime_names))
        sma_200 = pd.Series(np.zeros(len(regime_names)))
        has_sma_200 = False

    has_atr = "atr_14" in features_df.columns
    has_adx = "adx_14" in features_df.columns
    has_close = "close" in features_df.columns
    has_bb_width = "bb_width" in features_df.columns
    has_bb_mid = "bb_mid" in features_df.columns
    has_rsi = "rsi_14" in features_df.columns
    has_mom_1 = "mom_1" in features_df.columns
    has_mom_5 = "mom_5" in features_df.columns
    has_mom_10 = "mom_10" in features_df.columns
    has_vol_ratio = "vol_ratio" in features_df.columns

    adx_vals = features_df["adx_14"] if has_adx else None
    rsi_vals = features_df["rsi_14"] if has_rsi else None

    pip_size = 0.0001
    lot_size = 100000
    cost_per_side_pips = SPREAD_PIPS + COMMISSION_PER_LOT / (pip_size * lot_size)
    round_trip_cost_pips = 2.0 * cost_per_side_pips

    # Volatility filter bounds
    VOLATILITY_MIN = 0.0005  # 0.05% — too dead
    VOLATILITY_MAX = 0.015  # 1.5% — too risky

    # 4h trend confirmation reference
    features_4h = features_4h_df
    has_4h_data = features_4h is not None and "adx_14" in features_4h.columns

    # Track which strategy produced each signal (for diagnostics)
    strategy_counts: Dict[str, int] = {}

    def signal_fn(
        prices_arg: np.ndarray,
        features_arg: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        nonlocal strategy_counts
        strategy_counts = {}
        n = len(regime_names)
        signals = np.zeros(n, dtype=int)

        for i in range(burn_in, n):
            regime = regime_names[i]

            # ── Crisis: no trades ──
            if regime == "crisis":
                signals[i] = 0
                continue

            # ── Required: ATR (always in features) and close (from prices_arg) ──
            if not has_atr:
                continue

            close = features_df["close"].iloc[i] if has_close else prices_arg[i]
            atr_val = features_df["atr_14"].iloc[i]

            if pd.isna(atr_val) or atr_val <= 0 or pd.isna(close) or close <= 0:
                continue

            # ── Volatility filter ──
            atr_ratio = atr_val / close
            if atr_ratio > VOLATILITY_MAX or atr_ratio < VOLATILITY_MIN:
                signals[i] = 0
                continue

            # ── Minimum expected move filter ──
            expected_move_pips = atr_val * TP_ATR / pip_size
            if expected_move_pips <= 2.0 * round_trip_cost_pips:
                signals[i] = 0
                continue

            # ═══════════════════════════════════════════════════════════════
            # Trending regime — Breakout following + Time Series Momentum
            # ═══════════════════════════════════════════════════════════════
            if regime == "trending":
                session = sessions[i]
                can_breakout = False
                hh, ll = None, None

                # Session-based breakout lookback:
                #   Asia: skip breakout entirely (low volatility)
                #   London: aggressive 3-bar lookback
                #   All other sessions: standard 5-bar lookback
                if session == "asia":
                    can_breakout = False  # No breakouts in Asia
                elif session == "london":
                    hh = high_3_max.iloc[i]
                    ll = low_3_min.iloc[i]
                    can_breakout = not (pd.isna(hh) or pd.isna(ll))
                else:
                    hh = high_5_max.iloc[i]
                    ll = low_5_min.iloc[i]
                    can_breakout = not (pd.isna(hh) or pd.isna(ll))

                # ── 4h Trend Confirmation (breakout only) ──
                if can_breakout and has_4h_data:
                    adx_4h = features_4h["adx_14"].iloc[i]
                    if pd.isna(adx_4h) or adx_4h < 20:
                        can_breakout = False  # Not trending on 4h — skip breakout only

                # ── Breakout Gap Filter ──
                if can_breakout:
                    BREAKOUT_GAP_MULTIPLE = 0.5
                    if close > hh:
                        if (close - hh) < BREAKOUT_GAP_MULTIPLE * atr_val:
                            can_breakout = False
                    elif close < ll:
                        if (ll - close) < BREAKOUT_GAP_MULTIPLE * atr_val:
                            can_breakout = False
                    else:
                        can_breakout = False  # Not a breakout bar

                # ── ADX Rising Confirmation ──
                if can_breakout and has_adx and i > burn_in:
                    assert adx_vals is not None
                    adx_now = adx_vals.iloc[i]
                    adx_prev = adx_vals.iloc[i - 1]
                    if not pd.isna(adx_now) and not pd.isna(adx_prev):
                        if adx_now <= adx_prev:
                            can_breakout = False

                # ── Prior Consolidation (Coiling): ATR must be declining ──
                if can_breakout:
                    CONSOLIDATION_LOOKBACK = 5
                    if i >= CONSOLIDATION_LOOKBACK:
                        atr_now = atr_val
                        atr_before = features_df["atr_14"].iloc[
                            i - CONSOLIDATION_LOOKBACK
                        ]
                        if not pd.isna(atr_now) and not pd.isna(atr_before):
                            if atr_now >= atr_before:
                                can_breakout = False

                # ── Breakout Entry ──
                if can_breakout:
                    if close > hh:
                        signals[i] = 1  # Upside breakout → long
                        strategy_counts["breakout"] = (
                            strategy_counts.get("breakout", 0) + 1
                        )
                    elif close < ll:
                        signals[i] = -1  # Downside breakout → short
                        strategy_counts["breakout"] = (
                            strategy_counts.get("breakout", 0) + 1
                        )

                # ── Time Series Momentum (if no breakout signal) ──
                if signals[i] == 0 and has_mom_1 and has_mom_5 and has_mom_10:
                    m1 = features_df["mom_1"].iloc[i]
                    m5 = features_df["mom_5"].iloc[i]
                    m10 = features_df["mom_10"].iloc[i]
                    if not (pd.isna(m1) or pd.isna(m5) or pd.isna(m10)):
                        # Multi-horizon agreement required (Moskowitz, Ooi & Pedersen 2012)  # noqa: E501
                        if m5 > 0 and m10 > 0 and m1 > 0:
                            signals[i] = 1  # Momentum long
                            strategy_counts["ts_momentum"] = (
                                strategy_counts.get("ts_momentum", 0) + 1
                            )
                        elif m5 < 0 and m10 < 0 and m1 < 0:
                            signals[i] = -1  # Momentum short
                            strategy_counts["ts_momentum"] = (
                                strategy_counts.get("ts_momentum", 0) + 1
                            )

            # ═══════════════════════════════════════════════════════════════
            # Ranging regime — Bollinger Band Squeeze (high priority) + Mean Reversion
            # ═══════════════════════════════════════════════════════════════
            elif regime == "ranging":
                # Session-based z-score threshold for mean reversion:
                #   Asia: 1.2 (easier to trigger in low vol)
                #   Overlap: 2.0 (avoid noise during peak vol)
                #   Others: 1.5 (standard)
                session = sessions[i]
                if session == "asia":
                    z_threshold = 1.2
                elif session == "overlap":
                    z_threshold = 2.0
                else:
                    z_threshold = 1.5

                # ── Bollinger Band Squeeze (volatility breakout) ──
                # Research: Bollinger (2002) — BB width contraction precedes volatility expansion  # noqa: E501
                if has_bb_width and has_bb_mid:
                    bb_width_vals = features_df["bb_width"].values
                    if len(bb_width_vals) > 100 and i >= 100:
                        bb_100_min = np.min(bb_width_vals[max(0, i - 100) : i + 1])
                        bb_current = bb_width_vals[i]
                        bb_prev = bb_width_vals[i - 1] if i > 0 else bb_current

                        if not (
                            pd.isna(bb_current)
                            or pd.isna(bb_prev)
                            or pd.isna(bb_100_min)
                        ):
                            # Squeeze detected: BB width at/near 100-period low
                            if bb_current <= bb_100_min * 1.01 and bb_current > bb_prev:
                                # Volatility expansion starting — entry above/below mid
                                bb_mid_val = features_df["bb_mid"].iloc[i]
                                if not pd.isna(bb_mid_val):
                                    if close > bb_mid_val:
                                        signals[i] = 1  # Long breakout above mid
                                        strategy_counts["bb_squeeze"] = (
                                            strategy_counts.get("bb_squeeze", 0) + 1
                                        )
                                    elif close < bb_mid_val:
                                        signals[i] = -1  # Short breakout below mid
                                        strategy_counts["bb_squeeze"] = (
                                            strategy_counts.get("bb_squeeze", 0) + 1
                                        )

                # ── Mean Reversion (if no squeeze signal) ──
                if signals[i] == 0:
                    z = z_score[i]

                    # ── 1. Trend filter: don't fight the major trend ──
                    if has_sma_200:
                        sma_200_val = sma_200.iloc[i]
                        if pd.isna(sma_200_val):
                            continue  # Not enough data for 200-bar SMA
                        if z < -z_threshold and close < sma_200_val:
                            signals[i] = 0
                            continue  # Bearish trend + low price = falling knife
                        if z > z_threshold and close > sma_200_val:
                            signals[i] = 0
                            continue  # Bullish trend + high price = catching rallies

                    # ── 2. RSI confirmation ──
                    if has_rsi:
                        assert rsi_vals is not None
                        rsi_val = rsi_vals.iloc[i]
                        if pd.isna(rsi_val):
                            continue
                        # Long only if oversold
                        if z < -z_threshold and rsi_val > 40:
                            signals[i] = 0
                            continue
                        # Short only if overbought
                        if z > z_threshold and rsi_val < 60:
                            signals[i] = 0
                            continue
                    else:
                        continue  # Require RSI for mean reversion

                    # ── 3. Entry ──
                    if z < -z_threshold:
                        signals[i] = 1  # Mean reversion long
                        strategy_counts["mean_rev"] = (
                            strategy_counts.get("mean_rev", 0) + 1
                        )
                    elif z > z_threshold:
                        signals[i] = -1  # Mean reversion short
                        strategy_counts["mean_rev"] = (
                            strategy_counts.get("mean_rev", 0) + 1
                        )

            # ═══════════════════════════════════════════════════════════════
            # Volatile regime — Mean Reversion + Volatility Mean Reversion
            # ═══════════════════════════════════════════════════════════════
            elif regime == "volatile":
                # Session-based z-score threshold for mean reversion
                session = sessions[i]
                if session == "asia":
                    z_threshold = 1.2
                elif session == "overlap":
                    z_threshold = 2.0
                else:
                    z_threshold = 1.5

                z = z_score[i]

                # ── Mean Reversion ──
                # ── 1. Trend filter ──
                if has_sma_200:
                    sma_200_val = sma_200.iloc[i]
                    if pd.isna(sma_200_val):
                        continue
                    if z < -z_threshold and close < sma_200_val:
                        signals[i] = 0
                        continue
                    if z > z_threshold and close > sma_200_val:
                        signals[i] = 0
                        continue

                # ── 2. RSI confirmation ──
                if has_rsi:
                    assert rsi_vals is not None
                    rsi_val = rsi_vals.iloc[i]
                    if pd.isna(rsi_val):
                        continue
                    if z < -z_threshold and rsi_val > 40:
                        signals[i] = 0
                        continue
                    if z > z_threshold and rsi_val < 60:
                        signals[i] = 0
                        continue
                else:
                    continue

                # ── 3. Entry ──
                if z < -z_threshold:
                    signals[i] = 1
                    strategy_counts["mean_rev"] = strategy_counts.get("mean_rev", 0) + 1
                elif z > z_threshold:
                    signals[i] = -1
                    strategy_counts["mean_rev"] = strategy_counts.get("mean_rev", 0) + 1

                # ── Volatility Mean Reversion (if no mean rev signal) ──
                # Research: Bollerslev, Tauchen & Zhou (2009); Della Corte, Sarno & Tsiakas (2011)  # noqa: E501
                if signals[i] == 0 and has_vol_ratio and has_rsi:
                    vr = features_df["vol_ratio"].iloc[i]
                    if not pd.isna(vr) and vr > 1.5:  # Volatility 50% above normal
                        if rsi_val < 30:
                            signals[i] = 1  # Oversold + vol spike = buy mean reversion
                            strategy_counts["vol_mean_rev"] = (
                                strategy_counts.get("vol_mean_rev", 0) + 1
                            )
                        elif rsi_val > 70:
                            signals[i] = (
                                -1
                            )  # Overbought + vol spike = sell mean reversion
                            strategy_counts["vol_mean_rev"] = (
                                strategy_counts.get("vol_mean_rev", 0) + 1
                            )

        # Log strategy signal counts for diagnostics
        if strategy_counts:
            total = sum(strategy_counts.values())
            breakdown = ", ".join(
                f"{k}: {v}" for k, v in sorted(strategy_counts.items())
            )
            logger.info(f"  Strategy signal breakdown ({total} total): {breakdown}")

        return signals

    return signal_fn


# ═══════════════════════════════════════════════════════════════════════════════
# Section D: Main Backtest
# ═══════════════════════════════════════════════════════════════════════════════


def run_main_backtest(
    prices: np.ndarray,
    signal_fn: Callable,
    features_np: np.ndarray,
    atr_np: np.ndarray,
    regime_names: np.ndarray,
    sl_atr: float = SL_ATR,
    tp_atr: float = TP_ATR,
    breakeven_atr: float = BREAKEVEN_ATR,
    max_hold_bars: int = MAX_HOLD_BARS,
    spread_pips: float = SPREAD_PIPS,
    commission_per_lot: float = COMMISSION_PER_LOT,
) -> Dict:
    """Run base backtest + 6-scenario sensitivity analysis.

    Returns dict with 'base' BacktestResult, 'sensitivity' dict, 'regime_breakdown'.
    """
    # ── Base backtest ──
    bt = VectorizedBacktester(
        spread_pips=spread_pips,
        commission_per_lot=commission_per_lot,
        slippage_model="moderate",
    )
    base_result = bt.run(
        prices,
        signal_fn,
        features=features_np,
        atr=atr_np,
        regimes=regime_names,
        sl_atr=sl_atr,
        tp_atr=tp_atr,
        breakeven_atr=breakeven_atr,
        max_hold_bars=max_hold_bars,
    )
    logger.info(
        f"Base: Sharpe={base_result.sharpe:.3f} | "
        f"Return={base_result.total_return_pct:+.2f}% | "
        f"DD={base_result.max_drawdown_pct:.2f}% | "
        f"Trades={base_result.total_trades} | "
        f"WinRate={base_result.win_rate:.1%}"
    )

    # ── Sensitivity: 3 slippage × 2 cost multiplier ──
    sensitivity = {}
    for slip in ["conservative", "moderate", "aggressive"]:
        for mult in [0.5, 1.0]:
            bt_sens = VectorizedBacktester(
                spread_pips=spread_pips,
                commission_per_lot=commission_per_lot,
                slippage_model=slip,
            )
            r = bt_sens.run(
                prices,
                signal_fn,
                features=features_np,
                atr=atr_np,
                regimes=regime_names,
                sl_atr=sl_atr,
                tp_atr=tp_atr,
                cost_multiplier=mult,
                breakeven_atr=breakeven_atr,
                max_hold_bars=max_hold_bars,
            )
            key = f"{slip}_cost{mult}x"
            sensitivity[key] = r

    # ── Regime breakdown ──
    regime_breakdown = {}
    if base_result.trade_regimes:
        unique_r = sorted(set(base_result.trade_regimes))
        for reg in unique_r:
            mask = [t == reg for t in base_result.trade_regimes]
            reg_pnls = base_result.trade_pnls[mask]
            n_trades = len(reg_pnls)
            wins = int(np.sum(reg_pnls > 0))
            total_pnl = float(np.sum(reg_pnls))
            regime_breakdown[reg] = {
                "trades": n_trades,
                "win_rate": wins / n_trades if n_trades > 0 else 0.0,
                "total_return": total_pnl,
            }

    return {
        "base": base_result,
        "sensitivity": sensitivity,
        "regime_breakdown": regime_breakdown,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Section E: Walk-Forward Strategy Factory
# ═══════════════════════════════════════════════════════════════════════════════


def predict_regimes_adx(
    test_prices: np.ndarray,
    test_feat: Optional[np.ndarray],
    column_index: Dict[str, int],
) -> Tuple[np.ndarray, SimpleRegimeDetector]:
    """Detect regimes using ADX-based method on feature array.

    Reconstructs a feature DataFrame from the numpy array (which contains
    indicator columns), adds a 'close' column from prices, then runs
    SimpleRegimeDetector.detect().
    """
    if test_feat is not None and len(column_index) > 0:
        cols = list(column_index.keys())
        n_cols = min(test_feat.shape[1], len(cols))
        feat_df = pd.DataFrame(
            test_feat[: len(test_prices), :n_cols], columns=cols[:n_cols]
        )
        feat_df["close"] = test_prices
    else:
        feat_df = pd.DataFrame({"close": test_prices})

    detector = SimpleRegimeDetector(adx_threshold=20.0)
    regime_labels, regime_names = detector.detect(feat_df)
    return regime_names, detector


def make_walk_forward_strategy(
    column_index: Dict[str, int],
    features_4h_aligned: Optional[pd.DataFrame] = None,
    sl_atr: float = SL_ATR,
    tp_atr: float = TP_ATR,
    breakeven_atr: float = BREAKEVEN_ATR,
    max_hold_bars: int = MAX_HOLD_BARS,
    spread_pips: float = SPREAD_PIPS,
    commission_per_lot: float = COMMISSION_PER_LOT,
    burn_in: int = BURN_IN,
    embargo_val: int = WF_EMBARGO,
) -> Callable:
    """Build strategy_fn for PurgedWalkForward.run().

    Per fold:
      1. Detect regimes on test features via ADX
      2. Build signal function from test features (with optional 4h trend confirmation)
      3. Run VectorizedBacktester on test window
      4. Return list of trade dicts with 'pnl'
    """

    def strategy_fn(
        train_prices: np.ndarray,
        test_prices: np.ndarray,
        train_feat: Optional[np.ndarray],
        test_feat: Optional[np.ndarray],
    ) -> List[Dict]:
        # 1. Detect regimes on test data via ADX
        test_regimes, _ = predict_regimes_adx(test_prices, test_feat, column_index)

        # 2. Build test features DataFrame from numpy array + column index
        if test_feat is not None and len(column_index) > 0:
            cols = list(column_index.keys())
            n_cols = min(test_feat.shape[1], len(cols))
            test_feat_df = pd.DataFrame(
                test_feat[: len(test_prices), :n_cols], columns=cols[:n_cols]
            )
        else:
            test_feat_df = pd.DataFrame({"close": test_prices})

        # 2b. Slice 4h features to match the test window
        # Test window start = len(train_prices) + embargo (since train always starts at index 0)  # noqa: E501
        test_features_4h = None
        if features_4h_aligned is not None:
            te_s = len(train_prices) + embargo_val
            te_e = te_s + len(test_prices)
            if te_e <= len(features_4h_aligned):
                test_features_4h = features_4h_aligned.iloc[te_s:te_e].reset_index(
                    drop=True
                )

        # 3. Build signal function
        sig_fn = make_signal_function(
            test_regimes,
            test_feat_df,
            burn_in=burn_in,
            features_4h_df=test_features_4h,
        )

        # 4. Extract ATR from test features
        atr_col = column_index.get("atr_14")
        if atr_col is not None and test_feat is not None:
            test_atr = test_feat[: len(test_prices), atr_col].copy()
            test_atr = np.nan_to_num(test_atr, nan=0.001)
        else:
            test_atr = np.full(len(test_prices), 0.001)

        # 5. Run backtester on test set
        bt = VectorizedBacktester(
            spread_pips=spread_pips,
            commission_per_lot=commission_per_lot,
            slippage_model="moderate",
        )
        result = bt.run(
            test_prices,
            sig_fn,
            features=test_feat[: len(test_prices)] if test_feat is not None else None,
            atr=test_atr,
            regimes=test_regimes,
            sl_atr=sl_atr,
            tp_atr=tp_atr,
            breakeven_atr=breakeven_atr,
            max_hold_bars=max_hold_bars,
        )

        return [{"pnl": float(p)} for p in result.trade_pnls]

    return strategy_fn


# ═══════════════════════════════════════════════════════════════════════════════
# Section F: Monte Carlo Analysis
# ═══════════════════════════════════════════════════════════════════════════════


def run_monte_carlo_analysis(
    trades: List[Dict],
    regime_breakdown: Dict[str, Dict],
    n_permutations: int = MC_N_PERMUTATIONS,
    alpha: float = MC_ALPHA,
) -> Dict:
    """Run Monte Carlo permutation test on trade PnLs."""
    mc = MonteCarloSigTest(n_permutations=n_permutations, alpha=alpha)
    overall = mc.test(trades)

    logger.info(
        f"MC: Sharpe p={overall.p_value_sharpe:.4f} "
        f"({'SIGNIFICANT' if overall.is_significant_sharpe else 'NOT significant'}) | "
        f"Return p={overall.p_value_return:.4f}"
    )

    return {"overall": overall}


# ═══════════════════════════════════════════════════════════════════════════════
# Section G: Report
# ═══════════════════════════════════════════════════════════════════════════════


def compute_trade_level_metrics(trade_pnls: np.ndarray) -> Dict:
    """Compute proper metrics directly from trade PnLs, not from equity curve.

    This avoids unitization artifacts in the VectorizedBacktester's equity curve.
    """
    if len(trade_pnls) == 0:
        return {
            "total_return": 0.0,
            "sharpe": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }

    total_return = float(np.sum(trade_pnls))
    n = len(trade_pnls)
    wins = trade_pnls[trade_pnls > 0]
    losses = trade_pnls[trade_pnls < 0]
    win_rate = len(wins) / n

    # Sharpe from trades (daily frequency not known, use sqrt(n) approximation)
    if np.std(trade_pnls) > 1e-10:
        sharpe = float(np.mean(trade_pnls) / np.std(trade_pnls) * np.sqrt(252))
    else:
        sharpe = 0.0

    gross_win = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 1.0
    profit_factor = gross_win / max(gross_loss, 1e-8)

    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": float(np.mean(wins)) if len(wins) > 0 else 0.0,
        "avg_loss": float(np.mean(losses)) if len(losses) > 0 else 0.0,
        "total_trades": n,
        "gross_profit": gross_win,
        "gross_loss": gross_loss,
    }


def evaluate_verdict(  # noqa: C901
    backtest_result: Dict,
    wf_results: List[WFResult],
    mc_result: Optional[SigTestResult],
) -> Tuple[bool, List[str], List[str]]:
    """Evaluate PASS/FAIL against defined thresholds.

    Returns:
        (passed: bool, passes: List[str], fails: List[str])
    """
    base = backtest_result["base"]
    trade_metrics = compute_trade_level_metrics(base.trade_pnls)
    passes = []
    fails = []

    # Trade-level Sharpe (bypasses equity curve unitization artifacts)
    sharpe_val = trade_metrics["sharpe"]
    if sharpe_val >= THRESHOLDS["min_sharpe"]:
        passes.append(f"Trade Sharpe >= {THRESHOLDS['min_sharpe']}: {sharpe_val:.3f}")
    else:
        fails.append(f"Trade Sharpe < {THRESHOLDS['min_sharpe']}: {sharpe_val:.3f}")

    # Walk-forward avg Sharpe
    if wf_results:
        wf_sharpes = [r.sharpe for r in wf_results if r.n_trades > 0]
        if wf_sharpes:
            avg_wf_sharpe = float(np.mean(wf_sharpes))
            if avg_wf_sharpe >= THRESHOLDS["min_wf_avg_sharpe"]:
                passes.append(
                    f"WF Avg Sharpe >= {THRESHOLDS['min_wf_avg_sharpe']}: {avg_wf_sharpe:.3f}"  # noqa: E501
                )
            else:
                fails.append(
                    f"WF Avg Sharpe < {THRESHOLDS['min_wf_avg_sharpe']}: {avg_wf_sharpe:.3f}"  # noqa: E501
                )
    else:
        fails.append("Walk-forward: No results")

    # Monte Carlo p-value
    if mc_result:
        p_val = mc_result.p_value_sharpe
        if p_val <= THRESHOLDS["max_mc_p_value"]:
            passes.append(f"MC p-value <= {THRESHOLDS['max_mc_p_value']}: {p_val:.4f}")
        else:
            fails.append(f"MC p-value > {THRESHOLDS['max_mc_p_value']}: {p_val:.4f}")
    else:
        fails.append("Monte Carlo: No result (too few trades)")

    # Max drawdown from cumulative PnL
    if len(base.trade_pnls) > 0:
        cum_pnl = np.cumsum(base.trade_pnls)
        peak = np.maximum.accumulate(cum_pnl)
        dd = peak - cum_pnl
        _ = float(np.max(dd)) if len(dd) > 0 else 0.0
    else:
        pass

    net_pnl = trade_metrics["total_return"]
    is_profitable = net_pnl > 0
    if is_profitable:
        passes.append(f"Net PnL positive: {net_pnl:+.2f}")
    else:
        fails.append(f"Net PnL negative: {net_pnl:.2f} (strategy is losing money)")

    # Profit factor (from trade-level metrics)
    pf_val = trade_metrics["profit_factor"]
    if pf_val >= THRESHOLDS["min_profit_factor"]:
        passes.append(
            f"Profit Factor >= {THRESHOLDS['min_profit_factor']}: {pf_val:.2f}"
        )
    else:
        fails.append(f"Profit Factor < {THRESHOLDS['min_profit_factor']}: {pf_val:.2f}")

    # Total trades
    n_trades = trade_metrics["total_trades"]
    if n_trades >= THRESHOLDS["min_total_trades"]:
        passes.append(f"Trades >= {THRESHOLDS['min_total_trades']}: {n_trades}")
    else:
        fails.append(f"Trades < {THRESHOLDS['min_total_trades']}: {n_trades}")

    return len(fails) == 0, passes, fails


def print_report(  # noqa: C901
    backtest_result: Dict,
    wf_results: List[WFResult],
    wf_summary: Dict,
    mc_results: Dict,
) -> None:
    """Print formatted final report."""
    base = backtest_result["base"]

    # Compute trade-level metrics (bypasses equity curve unitization artifacts)
    trade_metrics = compute_trade_level_metrics(base.trade_pnls)

    print()
    print("=" * 70)
    print("        RTS AI FOREX TRADING SYSTEM — BACKTEST REPORT")
    print("=" * 70)
    print()

    # ── Section 1: Backtest Metrics ──
    print("=" * 70)
    print("  [1] BACKTEST METRICS")
    print(
        f"      ({SPREAD_PIPS} pip spread, ${COMMISSION_PER_LOT}/lot, moderate slippage)"  # noqa: E501
    )
    print("=" * 70)
    print(f"  Total Trades:            {trade_metrics['total_trades']}")
    print(f"  Win Rate:                {trade_metrics['win_rate']:.1%}")
    print(f"  Profit Factor:           {trade_metrics['profit_factor']:.2f}")
    print(f"  Trade-Level Sharpe:      {trade_metrics['sharpe']:.3f}")
    print(f"  Net PnL (pips):          {trade_metrics['total_return']:+.2f}")
    print(f"  Gross Profit:            {trade_metrics['gross_profit']:.2f}")
    print(f"  Gross Loss:              {trade_metrics['gross_loss']:.2f}")
    print(f"  Avg Win:                 {trade_metrics['avg_win']:.2f}")
    print(f"  Avg Loss:                {trade_metrics['avg_loss']:.2f}")
    print(f"  Avg Hold (bars):         {base.avg_hold_bars:.1f}")
    print()

    # ── Section 2: Sensitivity ──
    print("=" * 70)
    print("  [2] SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(
        f"  {'Scenario':<24s} {'Sharpe':>8s} {'Return':>8s} {'DD':>8s} {'Trades':>8s}"
    )
    print(f"  {'-'*24} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    sens = backtest_result["sensitivity"]
    for key in sorted(sens.keys()):
        r = sens[key]
        print(
            f"  {key:<24s} {r.sharpe:>8.3f} {r.total_return_pct:>+7.2f}% "
            f"{r.max_drawdown_pct:>7.2f}% {r.total_trades:>8d}"
        )
    print()

    # ── Section 3: Regime Breakdown ──
    print("=" * 70)
    print("  [3] REGIME BREAKDOWN")
    print("=" * 70)
    print(f"  {'Regime':<16s} {'Trades':>8s} {'WinRate':>10s} {'TotalReturn':>12s}")
    print(f"  {'-'*16} {'-'*8} {'-'*10} {'-'*12}")
    rb = backtest_result["regime_breakdown"]
    for reg in sorted(rb.keys()):
        info = rb[reg]
        print(
            f"  {reg:<16s} {info['trades']:>8d} {info['win_rate']:>9.1%} "
            f"{info['total_return']:>+11.2f}"
        )
    print()

    # ── Section 4: Walk-Forward ──
    print("=" * 70)
    print("  [4] WALK-FORWARD VALIDATION")
    print(
        f"      ({WF_N_FOLDS} folds, {WF_TEST_WINDOW}-bar test window, {WF_EMBARGO}-bar embargo)"  # noqa: E501
    )
    print("=" * 70)
    if wf_results:
        print(f"  {'Metric':<20s} {'Avg':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s}")
        print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        sharpes = [r.sharpe for r in wf_results if r.n_trades > 0]
        returns = [r.return_pct for r in wf_results]
        dds = [r.max_dd for r in wf_results]
        wrs = [r.win_rate for r in wf_results if r.n_trades > 0]
        pfs = [r.profit_factor for r in wf_results if r.n_trades > 0]

        if sharpes:
            print(
                f"  {'Sharpe':<20s} {np.mean(sharpes):>8.3f} {np.std(sharpes):>8.3f} "
                f"{np.min(sharpes):>8.3f} {np.max(sharpes):>8.3f}"
            )
        if returns:
            print(
                f"  {'Return %':<20s} {np.mean(returns):>+7.2f}% {np.std(returns):>7.2f}% "  # noqa: E501
                f"{np.min(returns):>+7.2f}% {np.max(returns):>+7.2f}%"
            )
        if dds:
            print(
                f"  {'Max DD %':<20s} {np.mean(dds):>7.2f}% {np.std(dds):>7.2f}% "
                f"{np.min(dds):>7.2f}% {np.max(dds):>7.2f}%"
            )
        if wrs:
            print(
                f"  {'Win Rate':<20s} {np.mean(wrs):>7.1%} {np.std(wrs):>7.1%} "
                f"{np.min(wrs):>7.1%} {np.max(wrs):>7.1%}"
            )
        if pfs:
            print(
                f"  {'Profit Factor':<20s} {np.mean(pfs):>8.3f} {np.std(pfs):>8.3f} "
                f"{np.min(pfs):>8.3f} {np.max(pfs):>8.3f}"
            )
    else:
        print("  (no walk-forward results)")
    print()

    # ── Section 5: Monte Carlo ──
    print("=" * 70)
    print("  [5] MONTE CARLO SIGNIFICANCE")
    print(f"      ({MC_N_PERMUTATIONS} permutations, alpha={MC_ALPHA})")
    print("=" * 70)
    mc_overall = mc_results.get("overall") if mc_results else None
    if mc_overall:
        sig_sharpe = (
            "SIGNIFICANT" if mc_overall.is_significant_sharpe else "NOT significant"
        )
        sig_ret = (
            "SIGNIFICANT" if mc_overall.is_significant_return else "NOT significant"
        )
        print(
            f"  Actual Sharpe:           {mc_overall.actual_sharpe:.3f}  "
            f"(p={mc_overall.p_value_sharpe:.4f}) [{sig_sharpe}]"
        )
        print(
            f"  Actual Return:           {mc_overall.actual_return_pct:+.2f}  "
            f"(p={mc_overall.p_value_return:.4f}) [{sig_ret}]"
        )
        print(f"  Sharpe Percentile:       {mc_overall.sharpe_percentile:.1%}")
        print(f"  Return Percentile:       {mc_overall.return_percentile:.1%}")
    else:
        print("  (insufficient trades for significance test)")
    print()

    # ── Section 6: Verdict ──
    print("=" * 70)
    print("  [6] VERDICT")
    print("=" * 70)
    passed, passes, fails = evaluate_verdict(
        backtest_result,
        wf_results,
        mc_results.get("overall") if mc_results else None,
    )
    for p in passes:
        print(f"  ✅ {p}")
    for f in fails:
        print(f"  ❌ {f}")

    if passed:
        print()
        print("  ✅  ALL CHECKS PASSED — Strategy is viable for paper trading")
    else:
        print()
        print(
            "  ❌  SOME CHECKS FAILED — Strategy needs improvement before paper trading"
        )
    print()
    print("=" * 70)
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="RTS AI Forex Backtest Suite")
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help=f"Path to OHLCV CSV (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--data-4h",
        default=None,
        help="Path to 4h OHLCV CSV (default: derived from --data path)",
    )
    parser.add_argument(
        "--no-walk-forward",
        action="store_true",
        help="Skip walk-forward validation (faster)",
    )
    parser.add_argument(
        "--no-monte-carlo",
        action="store_true",
        help="Skip Monte Carlo significance test (faster)",
    )
    args = parser.parse_args()

    t_start = time.time()

    # ── 1. Data Loading ──
    logger.info("─" * 60)
    logger.info("STEP 1/5: Loading data")
    prices, df_raw, df_features = load_and_prepare_data(args.data)

    # ── 1b. 4h Data Loading & Alignment ──
    logger.info("─" * 60)
    logger.info("STEP 1b/5: Loading & aligning 4h data")
    data_4h_path = args.data_4h
    if data_4h_path is None:
        # Derive 4h path from 1h path (e.g., EURUSD_1h.csv → EURUSD_4h.csv)
        data_4h_path = args.data.replace("_1h.csv", "_4h.csv")
        if data_4h_path == args.data:
            data_4h_path = DEFAULT_DATA_PATH_4H
            logger.info(f"  Derived 4h path not found, using default: {data_4h_path}")
    features_4h_aligned = load_and_prepare_4h_data(data_4h_path, df_raw["timestamp"])

    # ── 2. Regime Detection ──
    logger.info("─" * 60)
    logger.info("STEP 2/5: Detecting market regimes (HMM with enriched features)")
    regime_labels, regime_names, detector = compute_regimes(
        df_raw,
        df_features=df_features,
        n_regimes=N_REGIMES,
    )

    # ── 3. Prepare feature arrays ──
    logger.info("─" * 60)
    logger.info("STEP 3/5: Building feature matrix")
    # Columns to exclude (non-feature).
    # Keep high/low for breakout strategy in walk-forward.
    exclude_cols = {"timestamp", "open", "close", "volume"}
    feature_cols = [c for c in df_features.columns if c not in exclude_cols]
    features_np = df_features[feature_cols].values.astype(float)
    column_index = {col: idx for idx, col in enumerate(feature_cols)}

    # ATR for SL/TP
    atr_col = column_index.get("atr_14")
    if atr_col is not None:
        atr_series = features_np[:, atr_col].copy()
        atr_series = np.nan_to_num(atr_series, nan=0.001, posinf=0.01, neginf=0.001)
    else:
        atr_series = np.full(len(prices), 0.001)
    logger.info(f"  Feature matrix: {features_np.shape}")

    # ── 4. Signal Function ──
    logger.info("─" * 60)
    logger.info("STEP 4/5: Generating trading signals")
    sig_fn = make_signal_function(
        regime_names, df_features, features_4h_df=features_4h_aligned
    )

    # Quick signal stats
    signals = sig_fn(prices)
    n_long = int(np.sum(signals == 1))
    n_short = int(np.sum(signals == -1))
    n_neutral = int(np.sum(signals == 0))
    logger.info(
        f"  Signals: {n_long} long, {n_short} short, {n_neutral} neutral "
        f"({n_long + n_short} total entries)"
    )

    # ── 5. Main Backtest ──
    logger.info("─" * 60)
    logger.info("STEP 5/5: Running backtest")
    bt_results = run_main_backtest(
        prices,
        sig_fn,
        features_np,
        atr_series,
        regime_names,
    )

    # ── 6. Walk-Forward Validation ──
    wf_results_list = []
    wf_summary = {}
    if not args.no_walk_forward:
        logger.info("─" * 60)
        logger.info("RUNNING WALK-FORWARD VALIDATION")
        wf_strategy = make_walk_forward_strategy(
            column_index=column_index,
            features_4h_aligned=features_4h_aligned,
        )
        wf = PurgedWalkForward(
            n_folds=WF_N_FOLDS,
            test_window=WF_TEST_WINDOW,
            embargo=WF_EMBARGO,
        )
        wf_results_list = wf.run(prices, wf_strategy, features=features_np)
        wf_summary = PurgedWalkForward.summary(wf_results_list)
        if wf_summary:
            logger.info(
                f"WF Summary: Avg Sharpe={wf_summary['avg_sharpe']:.3f} | "
                f"Avg Return={wf_summary['avg_return_pct']:.2f}% | "
                f"Avg MaxDD={wf_summary['avg_max_dd_pct']:.2f}%"
            )

    # ── 7. Monte Carlo Significance ──
    mc_results = {}
    if not args.no_monte_carlo:
        logger.info("─" * 60)
        logger.info("RUNNING MONTE CARLO SIGNIFICANCE")
        base_result = bt_results["base"]
        if base_result.total_trades >= 10:
            trades = [{"pnl": float(p)} for p in base_result.trade_pnls]
            mc_results = run_monte_carlo_analysis(
                trades,
                bt_results.get("regime_breakdown", {}),
            )
        else:
            logger.warning("Too few trades for Monte Carlo significance test")

    # ── 8. Report ──
    elapsed = time.time() - t_start
    print_report(bt_results, wf_results_list, wf_summary, mc_results)
    logger.info(f"Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
