"""
Multi-Asset Backtest — Tests all FX pairs with cross-asset features.
Finds the pair(s) with the strongest predictive signal.

Usage:
    python -m src.scripts.multi_asset_backtest
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from dataclasses import dataclass

_src = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _src not in sys.path:
    sys.path.insert(0, _src)

from loguru import logger  # noqa: E402
from rts_ai_fx.features_unified import compute_features  # noqa: E402
from rts_ai_fx.regime_detector import HMMRegimeDetector  # noqa: E402
from backtest.vectorized_backtester import VectorizedBacktester  # noqa: E402

FX_SYMBOLS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "USDCHF",
    "NZDUSD",
    "XAUUSD",
]
REGIME_NAMES = ["crisis", "ranging", "volatile", "trending"]
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "historical")


@dataclass
class PairResult:
    symbol: str
    sharpe: float
    win_rate: float
    profit_factor: float
    net_pnl: float
    total_trades: int
    regime_dist: Dict[str, int]
    lag1_corr: float  # predictive power of lag-1 return


def load_pair_data(symbol: str) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Load OHLCV data for a symbol, return (prices, df_raw, df_features)."""
    path = os.path.join(DATA_DIR, f"{symbol}_1h.csv")
    if not os.path.exists(path):
        logger.warning(f"Data not found: {path}")
        return np.array([]), pd.DataFrame(), pd.DataFrame()
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.sort_values("timestamp").reset_index(drop=True)
    prices = df["close"].values.astype(float)
    df_raw = df.copy()
    df_features = compute_features(df)
    return prices, df_raw, df_features


def add_cross_asset_features(
    target_symbol: str,
    all_data: Dict[str, Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]],
    lookback: int = 5,
) -> pd.DataFrame:
    """Add correlated pair returns as features for the target symbol.

    For each other symbol, adds:
    - lagged_returns_X: 1..lookback hour returns of the other symbol
    - lagged_vol_X: rolling 24h volatility of the other symbol
    """
    if target_symbol not in all_data:
        return all_data[target_symbol][2]

    _, _, target_features = all_data[target_symbol]
    target_prices = all_data[target_symbol][0]
    df_out = target_features.copy()

    for other_sym, (other_prices, _, _) in all_data.items():
        if other_sym == target_symbol:
            continue

        # Align by timestamp — use raw index alignment since both are hourly
        min_len = min(len(target_prices), len(other_prices))
        other_returns = np.diff(other_prices[:min_len]) / (
            other_prices[: min_len - 1] + 1e-10
        )

        # Add lagged returns as features (shifted by 1 to avoid look-ahead)
        for lag in range(1, lookback + 1):
            col_name = f"lag_ret_{other_sym}_{lag}h"
            lagged = np.concatenate(
                [[0.0] * lag, other_returns[: -lag + 1] if lag > 1 else other_returns]
            )
            # Pad or truncate to match df_out length
            if len(lagged) < len(df_out):
                lagged = np.concatenate([np.zeros(len(df_out) - len(lagged)), lagged])
            lagged = lagged[-len(df_out) :]
            df_out[col_name] = lagged

        # Add rolling volatility of other pair
        other_vol = pd.Series(other_prices).rolling(24).std().values
        if len(other_vol) < len(df_out):
            other_vol = np.concatenate(
                [np.zeros(len(df_out) - len(other_vol)), other_vol]
            )
        other_vol = other_vol[-len(df_out) :]
        df_out[f"vol_{other_sym}_24h"] = other_vol

    return df_out


def simple_momentum_signal(
    prices: np.ndarray,
    features_df: pd.DataFrame,
    regime_names: np.ndarray,
    burn_in: int = 200,
) -> np.ndarray:
    """Generate signals using lag-1 momentum + cross-asset confirmation.

    Core logic:
    - If price went up last hour AND cross-asset trend confirms → long
    - If price went down last hour AND cross-asset trend confirms → short
    - Cross-asset confirmation: majority of correlated pairs moving same direction
    """
    n = len(prices)
    signals = np.zeros(n, dtype=int)
    returns = np.diff(prices) / (prices[:-1] + 1e-10)
    returns = np.concatenate([[0.0], returns])  # pad first bar

    for i in range(burn_in, n):
        regime = regime_names[i]
        if regime == "crisis":
            continue

        # Momentum: direction of last price change
        mom = np.sign(prices[i] - prices[i - 1]) if i > 0 else 0

        # Cross-asset confirmation: check how many other pairs moved same direction
        cross_cols = [
            c
            for c in features_df.columns
            if c.startswith("lag_ret_") and c.endswith("_1h")
        ]
        if cross_cols:
            cross_signs = [
                np.sign(features_df[c].iloc[i])
                for c in cross_cols
                if not np.isnan(features_df[c].iloc[i])
            ]
            confirm_pct = (
                np.mean([s == mom for s in cross_signs])
                if cross_signs and mom != 0
                else 0.5
            )
        else:
            confirm_pct = 0.5

        # Trade only when momentum + cross-asset confirmation
        if mom > 0 and confirm_pct > 0.5:
            signals[i] = 1
        elif mom < 0 and confirm_pct > 0.5:
            signals[i] = -1
        else:
            signals[i] = 0

    return signals


def compute_trade_metrics(trade_pnls: np.ndarray) -> Dict:
    """Compute proper metrics from trade PnLs."""
    if len(trade_pnls) == 0:
        return {
            "total_return": 0.0,
            "sharpe": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_trades": 0,
        }
    wins = trade_pnls[trade_pnls > 0]
    losses = trade_pnls[trade_pnls < 0]
    n = len(trade_pnls)
    win_rate = len(wins) / n if n > 0 else 0.0
    sharpe = (
        float(np.mean(trade_pnls) / np.std(trade_pnls) * np.sqrt(252))
        if np.std(trade_pnls) > 1e-10
        else 0.0
    )
    gross_win = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 1.0
    pf = gross_win / max(gross_loss, 1e-8)
    return {
        "total_return": float(np.sum(trade_pnls)),
        "sharpe": sharpe,
        "win_rate": win_rate,
        "profit_factor": pf,
        "total_trades": n,
        "avg_win": float(np.mean(wins)) if len(wins) > 0 else 0.0,
        "avg_loss": float(np.mean(losses)) if len(losses) > 0 else 0.0,
    }


def compute_regimes_hmm(df_features: pd.DataFrame, df_raw: pd.DataFrame) -> np.ndarray:
    """Compute per-bar regime names using HMM."""
    try:
        detector = HMMRegimeDetector(n_regimes=4)
        detector.fit(df_features)
        hmm_feat = detector._extract_features(df_features)
        if len(hmm_feat) < 10:
            return np.full(len(df_raw), "ranging")
        hidden = detector.model.predict(hmm_feat)
        mean_ret = detector.model.means_[:, 0]
        order = np.argsort(mean_ret)
        label_to_name = {i: REGIME_NAMES[order[i]] for i in range(4)}
        names = np.array([label_to_name[s] for s in hidden])
        return np.concatenate([[names[0]], names])
    except Exception as e:
        logger.warning(f"HMM failed: {e}")
        return np.full(len(df_raw), "ranging")


def run_pair_backtest(symbol: str, all_data: Dict) -> PairResult:
    """Run full backtest for one pair with cross-asset features."""
    prices, df_raw, df_features_base = all_data[symbol]
    if len(prices) < 500:
        return PairResult(
            symbol=symbol,
            sharpe=0,
            win_rate=0,
            pf=0,
            net_pnl=0,
            trades=0,
            rd={},
            lag1_corr=0,
        )

    # Add cross-asset features
    df_features = add_cross_asset_features(symbol, all_data)

    # Regime detection
    regime_names = compute_regimes_hmm(df_features, df_raw)

    # Generate signals
    signals = simple_momentum_signal(prices, df_features, regime_names)

    # Feature matrix for backtester
    exclude_cols = {"timestamp", "open", "high", "low", "close", "volume"}
    feature_cols = [c for c in df_features.columns if c not in exclude_cols]
    features_np = df_features[feature_cols].values.astype(float)

    # ATR
    atr_col = [c for c in feature_cols if "atr_14" in c]
    if atr_col:
        atr_series = features_np[:, feature_cols.index(atr_col[0])].copy()
        atr_series = np.nan_to_num(atr_series, nan=0.001)
    else:
        atr_series = np.full(len(prices), 0.001)

    # Run backtest
    bt = VectorizedBacktester(
        spread_pips=0.5, commission_per_lot=7.0, slippage_model="moderate"
    )
    result = bt.run(
        prices,
        lambda p, f: signals,
        features=features_np,
        atr=atr_series,
        regimes=regime_names,
    )

    # Compute metrics
    tm = compute_trade_metrics(result.trade_pnls)

    # Lag-1 correlation
    returns = np.diff(prices) / (prices[:-1] + 1e-10)
    lag1_corr = (
        float(np.corrcoef(returns[1:], returns[:-1])[0, 1]) if len(returns) > 2 else 0.0
    )

    # Regime distribution
    rd = pd.Series(regime_names).value_counts().to_dict()

    logger.info(
        f"{symbol:8s} | Sharpe={tm['sharpe']:.3f} | WR={tm['win_rate']:.1%} | "
        f"PF={tm['profit_factor']:.2f} | PnL={tm['total_return']:+.0f} | "
        f"Trades={tm['total_trades']:4d} | Lag1={lag1_corr:.4f}"
    )

    return PairResult(
        symbol=symbol,
        sharpe=tm["sharpe"],
        win_rate=tm["win_rate"],
        profit_factor=tm["profit_factor"],
        net_pnl=tm["total_return"],
        total_trades=tm["total_trades"],
        regime_dist=rd,
        lag1_corr=lag1_corr,
    )


def main():
    print("=" * 70)
    print("  MULTI-ASSET BACKTEST — Cross-Asset Feature Analysis")
    print("=" * 70)
    print()

    # Load all pair data
    logger.info("Loading data for all symbols...")
    all_data: Dict[str, Tuple] = {}
    for sym in FX_SYMBOLS:
        prices, df_raw, df_features = load_pair_data(sym)
        if len(prices) > 0:
            all_data[sym] = (prices, df_raw, df_features)
            logger.info(f"  {sym}: {len(prices)} bars")

    print()
    print("─" * 70)
    print("  BACKTEST RESULTS (sorted by Sharpe)")
    print("─" * 70)
    print()

    # Run backtest for each pair
    results = []
    for sym in all_data:
        try:
            result = run_pair_backtest(sym, all_data)
            results.append(result)
        except Exception as e:
            logger.error(f"{sym} failed: {e}")

    # Sort by Sharpe
    results.sort(key=lambda r: r.sharpe, reverse=True)

    print()
    print("─" * 100)
    print(
        f"  {'Pair':<10s} {'Sharpe':>8s} {'WinRate':>8s} {'ProfFact':>8s} {'NetPnL':>10s} {'Trades':>8s} {'Lag1Corr':>10s}"  # noqa: E501
    )
    print("─" * 100)
    for r in results:
        print(
            f"  {r.symbol:<10s} {r.sharpe:>8.3f} {r.win_rate:>7.1%} {r.profit_factor:>8.2f} "  # noqa: E501
            f"{r.net_pnl:>+9.0f} {r.total_trades:>8d} {r.lag1_corr:>+10.4f}"
        )
    print("─" * 100)

    # Summary
    print()
    best = results[0] if results else None
    if best and best.sharpe > 0:
        print(
            f"  ✅ BEST PAIR: {best.symbol} (Sharpe={best.sharpe:.3f}, PF={best.profit_factor:.2f})"  # noqa: E501
        )
    else:
        print("  ❌ NO PROFITABLE PAIR FOUND")
        print(
            f"     Best was {results[0].symbol if results else 'N/A'} with Sharpe={results[0].sharpe:.3f}"  # noqa: E501
        )

    print()
    print("─" * 70)
    print("  LAG-1 AUTOCORRELATION BY PAIR (predictive power)")
    print("─" * 70)
    for r in sorted(results, key=lambda r: abs(r.lag1_corr), reverse=True):
        stars = (
            "***"
            if abs(r.lag1_corr) > 0.05
            else "**" if abs(r.lag1_corr) > 0.02 else "*"
        )
        print(f"  {r.symbol:<10s} Lag-1 r={r.lag1_corr:+.4f} {stars}")

    print()
    if best:
        print(
            f"  Recommendation: Trade {best.symbol} with cross-asset momentum strategy"
        )
        print(f"                  Sharpe={best.sharpe:.2f} after costs")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
