"""
Tick-Level Market Microstructure Analysis & Backtest.

Loads Dukascopy BI5 tick data, computes microstructure features,
tests predictive power, and runs event-driven backtest.

Usage:
    python -m src.scripts.tick_microstructure --symbol EURUSD --date 2026-01-02
    python -m src.scripts.tick_microstructure --symbol EURUSD --analyze-all
"""

import os
import sys
import struct
import lzma
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

_src = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _src not in sys.path:
    sys.path.insert(0, _src)

from loguru import logger  # noqa: E402

CACHE_DIR = Path("data/dukascopy_cache")
TICK_RECORD_SIZE = 20

# Dukascopy scaling: price = raw / 100000
PRICE_SCALE = 100000.0


def load_bi5_file(path: str) -> List[Tuple[float, float, float, float, float]]:
    """Load a BI5 tick file. Returns [(seconds_in_hour, bid, ask, bid_vol, ask_vol), ...]  # noqa: E501

    Each tick record: 20 bytes
      - 4 bytes: millisecond offset within hour (big-endian uint)
      - 4 bytes: ask price * 100000 (big-endian uint)
      - 4 bytes: bid price * 100000 (big-endian uint)
      - 4 bytes: ask volume (big-endian float)
      - 4 bytes: bid volume (big-endian float)
    """
    with open(path, "rb") as f:
        data = f.read()
    try:
        decompressed = lzma.decompress(data)
    except lzma.LZMAError:
        decompressed = data

    ticks = []
    for i in range(0, len(decompressed), TICK_RECORD_SIZE):
        chunk = decompressed[i : i + TICK_RECORD_SIZE]
        if len(chunk) < TICK_RECORD_SIZE:
            break
        ms_offset = struct.unpack(">I", chunk[0:4])[0]
        ask_raw = struct.unpack(">I", chunk[4:8])[0]
        bid_raw = struct.unpack(">I", chunk[8:12])[0]
        ask_vol = struct.unpack(">f", chunk[12:16])[0]
        bid_vol = struct.unpack(">f", chunk[16:20])[0]

        ticks.append(
            (
                ms_offset / 1000.0,  # seconds within hour
                bid_raw / PRICE_SCALE,  # bid price
                ask_raw / PRICE_SCALE,  # ask price
                bid_vol,  # bid volume
                ask_vol,  # ask volume
            )
        )
    return ticks


def load_tick_day(symbol: str, date: str) -> pd.DataFrame:
    """Load all BI5 files for a symbol/date into a DataFrame.

    Args:
        symbol: Currency pair (EURUSD, GBPUSD, etc.)
        date: Date string (2026-01-02)

    Returns:
        DataFrame with columns: timestamp, bid, ask, mid, spread, bid_vol, ask_vol
        Indexed by datetime
    """
    date_compact = date.replace("-", "")
    pattern = f"{CACHE_DIR}/{symbol}_{date_compact}_*.bi5"
    files = sorted(glob.glob(pattern))

    if not files:
        logger.warning(f"No tick files found: {pattern}")
        return pd.DataFrame()

    all_ticks = []
    for fpath in files:
        # Extract hour from filename: SYMBOL_YYYYMMDD_HH.bi5
        hour = int(os.path.basename(fpath).split("_")[2].split(".")[0])
        ticks = load_bi5_file(fpath)
        base_ts = datetime.strptime(f"{date} {hour:02d}:00:00", "%Y-%m-%d %H:%M:%S")
        for sec_offset, bid, ask, bv, av in ticks:
            ts = base_ts + timedelta(seconds=sec_offset)
            all_ticks.append((ts, bid, ask, bv, av))

    if not all_ticks:
        return pd.DataFrame()

    df = pd.DataFrame(
        all_ticks, columns=["timestamp", "bid", "ask", "bid_vol", "ask_vol"]
    )
    df = df.set_index("timestamp").sort_index()
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["spread"] = df["ask"] - df["bid"]
    df["spread_bps"] = df["spread"] / df["mid"] * 10000  # spread in basis points
    df["total_vol"] = df["bid_vol"] + df["ask_vol"]

    logger.info(f"Loaded {len(df):,} ticks for {symbol} on {date}")
    logger.info(f"  Price range: {df['mid'].min():.5f} - {df['mid'].max():.5f}")
    logger.info(f"  Avg spread: {df['spread_bps'].mean():.2f} bps")

    return df


def compute_microstructure_features(
    df: pd.DataFrame, tick_window: int = 100, second_window: int = 60
) -> pd.DataFrame:
    """Compute market microstructure features from tick data.

    Features computed:
      - OFI: Order Flow Imbalance = (ask_vol - bid_vol) / (ask_vol + bid_vol)
      - Microprice: (bid * ask_vol + ask * bid_vol) / (bid_vol + ask_vol)
      - Tick intensity: ticks per second
      - Volume intensity: volume per second
      - Spread volatility: rolling std of spread
      - Price impact: |mid_change| / volume_traded
      - Cumulative delta: rolling sum of signed volume
      - Microprice vs mid: microprice - mid (order book pressure)
      - Bid-ask bounce intensity
    """
    ft = df.copy()

    # Order flow imbalance (OFI) — tick level
    ft["ofi"] = (ft["ask_vol"] - ft["bid_vol"]) / (ft["total_vol"] + 1e-10)

    # Microprice: volume-weighted mid (more weight to the side with more volume)
    ft["microprice"] = (ft["bid"] * ft["ask_vol"] + ft["ask"] * ft["bid_vol"]) / (
        ft["total_vol"] + 1e-10
    )

    # Order book pressure: microprice vs mid (positive = selling pressure, negative = buying)  # noqa: E501
    ft["order_pressure"] = ft["microprice"] - ft["mid"]

    # Tick-level mid change
    ft["mid_change"] = ft["mid"].diff()
    ft["mid_change_next"] = ft["mid"].shift(-1) - ft["mid"]

    # Rolling features (tick window)
    ft["ofi_ma"] = ft["ofi"].rolling(tick_window).mean()
    ft["ofi_std"] = ft["ofi"].rolling(tick_window).std()
    ft["spread_ma"] = ft["spread_bps"].rolling(tick_window).mean()
    ft["spread_std"] = ft["spread_bps"].rolling(tick_window).std()

    # Cumulative delta (rolling sum of signed volume)
    ft["cum_delta"] = ft["ofi"].rolling(tick_window).sum()

    # Volume-weighted OFI (larger trades matter more)
    ft["vol_weighted_ofi"] = (
        (ft["ask_vol"] - ft["bid_vol"]) * ft["mid"] / (ft["total_vol"] + 1e-10)
    )

    # Resample to 1-second bars for second-level features
    second_bars = (
        ft.resample("1S")
        .agg(
            {
                "mid": "last",
                "bid": "last",
                "ask": "last",
                "spread_bps": "mean",
                "total_vol": "sum",
                "bid_vol": "sum",
                "ask_vol": "sum",
                "ofi": "mean",
                "microprice": "last",
                "order_pressure": "mean",
            }
        )
        .dropna()
    )

    # Second-level features
    second_bars["tick_count"] = ft.resample("1S").size()
    second_bars["mid_return"] = second_bars["mid"].pct_change()
    second_bars["ofi_1s"] = (second_bars["ask_vol"] - second_bars["bid_vol"]) / (
        second_bars["total_vol"] + 1e-10
    )
    second_bars["spread_change"] = second_bars["spread_bps"].diff()

    # Microprice analysis on second bars
    second_bars["microprice_1s"] = (
        second_bars["bid"] * second_bars["ask_vol"]
        + second_bars["ask"] * second_bars["bid_vol"]
    ) / (second_bars["total_vol"] + 1e-10)
    second_bars["pressure_1s"] = second_bars["microprice_1s"] - second_bars["mid"]
    second_bars["next_mid_return"] = second_bars["mid"].shift(-1).pct_change()

    # Rolling window on second bars
    for w in [5, 15, 60]:
        second_bars[f"ofi_{w}s"] = second_bars["ofi_1s"].rolling(w).mean()
        second_bars[f"spread_{w}s"] = second_bars["spread_bps"].rolling(w).mean()
        second_bars[f"tick_rate_{w}s"] = second_bars["tick_count"].rolling(w).mean()
        second_bars[f"vol_rate_{w}s"] = second_bars["total_vol"].rolling(w).mean()
        second_bars[f"pressure_{w}s"] = second_bars["pressure_1s"].rolling(w).mean()

    second_bars = second_bars.dropna()

    logger.info(
        f"Computed microstructure features: {len(second_bars):,} second-bars x {len(second_bars.columns)} features"  # noqa: E501
    )

    return second_bars


def test_predictive_power(features: pd.DataFrame) -> Dict:
    """Test which microstructure features predict the next-second return.

    Uses linear regression to measure predictive power of each feature
    for the next-second mid return.
    """
    from scipy import stats

    # Features to test (exclude price and return columns)
    exclude = {
        "mid",
        "bid",
        "ask",
        "mid_return",
        "microprice",
        "microprice_1s",
        "next_mid_return",
        "mid_change",
        "mid_change_next",
        "timestamp",
    }
    feature_cols = [
        c
        for c in features.columns
        if c not in exclude and features[c].dtype in (np.float64, np.float32, np.int64)
    ]

    results = []
    target = features["next_mid_return"].values
    target_binary = np.sign(target)  # direction prediction

    for col in feature_cols:
        x = features[col].values
        mask = ~(np.isnan(x) | np.isnan(target) | np.isinf(x) | np.isinf(target))
        x_clean = x[mask]
        y_clean = target[mask]
        y_bin = target_binary[mask]

        if len(x_clean) < 100:
            continue

        # Pearson correlation for magnitude
        r, p = stats.pearsonr(x_clean, y_clean)

        # Direction accuracy: does sign(feature - mean) predict sign(next_return)?
        x_bin = np.sign(x_clean - np.mean(x_clean))
        dir_acc = np.mean(x_bin == y_bin) if np.any(x_bin != 0) else 0.5

        # Simple trading signal test
        # Long when feature > 0, short when < 0
        trade_return = np.mean(np.abs(y_clean)) * (dir_acc - 0.5) * 2

        results.append(
            {
                "feature": col,
                "corr": r,
                "p_value": p,
                "dir_acc": dir_acc,
                "trade_return": trade_return,
                "n_samples": len(x_clean),
            }
        )

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("corr", key=abs, ascending=False)

    return df_results


def microprice_signal(
    features: pd.DataFrame,
    ofi_threshold: float = 0.3,
    pressure_threshold: float = 0.00001,
) -> np.ndarray:
    """Generate trading signals from microstructure features.

    Logic:
      - When order flow imbalance is strongly positive (more buying than selling)
        AND order book pressure is positive (microprice above mid) → LONG
      - When OFI is strongly negative AND pressure is negative → SHORT
      - Otherwise → HOLD

    Returns:
        signals array: 1 = long, -1 = short, 0 = hold
    """
    signals = np.zeros(len(features), dtype=int)

    ofi = features.get(
        "ofi_60s", features.get("ofi_1s", pd.Series([0] * len(features)))
    ).values
    pressure = features.get(
        "pressure_15s", features.get("pressure_1s", pd.Series([0] * len(features)))
    ).values

    for i in range(1, len(signals)):
        # Strong buying pressure
        if ofi[i] > ofi_threshold and pressure[i] > pressure_threshold:
            signals[i] = 1
        # Strong selling pressure
        elif ofi[i] < -ofi_threshold and pressure[i] < -pressure_threshold:
            signals[i] = -1
        else:
            signals[i] = 0

    return signals


def backtest_microstructure(features: pd.DataFrame, signals: np.ndarray) -> Dict:
    """Event-driven backtest of microstructure signals.

    Trades on 1-second bars with spread costs.
    Returns performance metrics.
    """
    prices = features["mid"].values
    spread = features["spread_bps"].values / 10000  # convert bps to price fraction
    n = min(len(prices), len(signals))

    trade_pnls = []
    pos = 0
    entry_p = 0
    entry_idx = 0
    trade_durations = []

    for i in range(1, n):
        sig = signals[i]

        if pos == 0 and sig != 0:
            pos = sig
            entry_p = prices[i]
            entry_idx = i
            continue

        if pos != 0:
            # Exit when signal reverses (or after 10 seconds max)
            time_up = (i - entry_idx) >= 10
            exit_sig = sig != 0 and sig != pos

            if exit_sig or time_up or i == n - 1:
                # Apply spread cost on entry and exit
                cost = spread[entry_idx] * entry_p + spread[i] * prices[i]
                pnl = (prices[i] - entry_p) * pos - cost
                trade_pnls.append(pnl)
                trade_durations.append(i - entry_idx)
                pos = 0

    tp = np.array(trade_pnls)
    if len(tp) == 0:
        return {"sharpe": 0, "pf": 0, "wr": 0, "trades": 0, "pnl": 0}

    wins = tp[tp > 0]
    losses = tp[tp < 0]
    sharpe = (
        np.mean(tp) / np.std(tp) * np.sqrt(252 * 6.5 * 3600)
        if np.std(tp) > 1e-10
        else 0
    )
    pf = np.sum(wins) / abs(np.sum(losses)) if np.sum(losses) != 0 else float("inf")

    return {
        "sharpe": sharpe,
        "pf": pf,
        "wr": np.mean(tp > 0),
        "trades": len(tp),
        "pnl": float(np.sum(tp)),
        "avg_hold_sec": np.mean(trade_durations) if trade_durations else 0,
        "avg_win": float(np.mean(wins)) if len(wins) > 0 else 0,
        "avg_loss": float(np.mean(losses)) if len(losses) > 0 else 0,
    }


def analyze_all(symbol: str):
    """Analyze all available tick data for a symbol."""
    # Find all dates
    pattern = f"{CACHE_DIR}/{symbol}_*.bi5"
    files = sorted(glob.glob(pattern))

    dates = set()
    for f in files:
        parts = os.path.basename(f).split("_")
        date_str = parts[1]
        date_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        dates.add(date_fmt)

    dates: list = sorted(dates)
    logger.info(f"Found {len(dates)} days of tick data for {symbol}")

    all_predictions = []
    all_trades = []

    for date in dates:
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing {symbol} on {date}")

        df_ticks = load_tick_day(symbol, date)
        if len(df_ticks) < 1000:
            continue

        features = compute_microstructure_features(df_ticks)

        # Predictive power test
        pred_results = test_predictive_power(features)
        top_features = (
            pred_results.head(10) if hasattr(pred_results, "head") else pred_results
        )
        all_predictions.append(pred_results)

        # Backtest
        signals = microprice_signal(features)
        bt_result = backtest_microstructure(features, signals)
        all_trades.append(bt_result)

        if isinstance(top_features, pd.DataFrame):
            logger.info(
                f"  Top feature: {top_features.iloc[0]['feature']} "
                f"corr={top_features.iloc[0]['corr']:.4f} "
                f"dir_acc={top_features.iloc[0]['dir_acc']:.1%}"
            )
        logger.info(
            f"  Backtest: Sharpe={bt_result['sharpe']:.3f} "
            f"PF={bt_result['pf']:.2f} "
            f"Trades={bt_result['trades']} "
            f"PnL={bt_result['pnl']:+.4f}"
        )

    # Combined results
    if all_predictions:
        combined_pred = pd.concat(all_predictions)
        top_overall = (
            combined_pred.groupby("feature")["corr"]
            .mean()
            .sort_values(key=abs, ascending=False)
            .head(10)
        )
        logger.info(f"\n{'='*60}")
        logger.info("TOP PREDICTIVE FEATURES across all dates:")
        for feat, corr in top_overall.items():
            logger.info(f"  {feat}: avg corr={corr:.4f}")

    if all_trades:
        avg_sharpe = np.mean([t["sharpe"] for t in all_trades])
        avg_pf = np.mean([t["pf"] for t in all_trades])
        total_trades = sum(t["trades"] for t in all_trades)
        total_pnl = sum(t["pnl"] for t in all_trades)
        logger.info(f"\n{'='*60}")
        logger.info(f"COMBINED BACKTEST: {len(all_trades)} days")
        logger.info(f"  Avg Sharpe: {avg_sharpe:.3f}")
        logger.info(f"  Avg PF:     {avg_pf:.2f}")
        logger.info(f"  Total Trades: {total_trades}")
        logger.info(f"  Total PnL:   {total_pnl:+.4f}")

        if avg_sharpe > 0.5:
            logger.info("  ✅ Microstructure signal is VIABLE!")
        elif avg_sharpe > 0:
            logger.info("  ⚠️  Marginal edge detected")
        else:
            logger.info("  ❌ No microstructure edge found")

    return all_predictions, all_trades


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Tick-Level Microstructure Analysis")
    parser.add_argument("--symbol", default="EURUSD", help="Symbol to analyze")
    parser.add_argument("--date", default=None, help="Specific date (YYYY-MM-DD)")
    parser.add_argument(
        "--analyze-all", action="store_true", help="Analyze all available dates"
    )
    args = parser.parse_args()

    if args.analyze_all or not args.date:
        analyze_all(args.symbol)
    else:
        logger.info(f"Loading {args.symbol} tick data for {args.date}")
        df = load_tick_day(args.symbol, args.date)
        if len(df) == 0:
            logger.error("No tick data found")
            return

        logger.info("\nComputing microstructure features...")
        features = compute_microstructure_features(df)

        logger.info("\nTesting predictive power...")
        pred_results = test_predictive_power(features)
        print("\n=== TOP PREDICTIVE FEATURES (next-second return) ===")
        print(f"{'Feature':<30s} {'Corr':>8s} {'p-val':>8s} {'Dir%':>8s}")
        print("-" * 54)
        for _, row in pred_results.head(15).iterrows():
            print(
                f"{row['feature']:<30s} {row['corr']:>+8.4f} {row['p_value']:>8.4f} {row['dir_acc']:>7.1%}"  # noqa: E501
            )

        print("\n=== MICROSTRUCTURE BACKTEST ===")
        signals = microprice_signal(features)
        result = backtest_microstructure(features, signals)
        for k, v in result.items():
            print(f"  {k}: {v}")

        if result["sharpe"] > 1.0:
            print("\n  ✅ Strong microstructure edge!")
        elif result["sharpe"] > 0.5:
            print("\n  ⚠️  Moderate microstructure edge")
        else:
            print("\n  ❌ No significant edge")


if __name__ == "__main__":
    main()
