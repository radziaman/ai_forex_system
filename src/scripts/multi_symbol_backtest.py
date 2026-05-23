"""
multi_symbol_backtest.py — Multi-symbol backtest runner for RTS AI Forex Trading System.

Runs the same regime-adaptive breakout strategy across all major FX pairs,
commodities, indices, and crypto. Aggregates trades across symbols to improve
statistical significance for Monte Carlo testing.

Usage:
    python -m src.scripts.multi_symbol_backtest
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable

_src = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _src not in sys.path:
    sys.path.insert(0, _src)

from dataclasses import dataclass  # noqa: E402
from loguru import logger  # noqa: E402
from rts_ai_fx.features_unified import compute_features  # noqa: E402
from rts_ai_fx.regime_detector import SimpleRegimeDetector  # noqa: E402
from backtest.vectorized_backtester import VectorizedBacktester  # noqa: E402
from validation.monte_carlo import MonteCarloSigTest  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

REGIME_NAMES = ["crisis", "ranging", "volatile", "trending"]

# Backtest defaults (override for specific symbols in SYMBOL_CONFIG)
SL_ATR = 2.0
TP_ATR = 5.0
BREAKOUT_LOOKBACK = 5
BREAKEVEN_ATR = 1.0
MAX_HOLD_BARS = 48
COMMISSION_PER_LOT = 7.0
MIN_BARS = 500
BURN_IN = 100

# Monte Carlo defaults
MC_N_PERMUTATIONS = 10_000
MC_ALPHA = 0.05

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "historical")

# ─── Symbol Configuration ─────────────────────────────────────────────────────

SYMBOLS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "USDCHF",
    "NZDUSD",
    "XAUUSD",
    "XTIUSD",
    "US500",
    "BTCUSD",
]

SYMBOL_CONFIG: Dict[str, Dict] = {
    # Forex majors
    "EURUSD": {"pip_size": 0.0001, "spread_pips": 0.5},
    "GBPUSD": {"pip_size": 0.0001, "spread_pips": 0.5},
    "USDJPY": {"pip_size": 0.01, "spread_pips": 0.5},
    "AUDUSD": {"pip_size": 0.0001, "spread_pips": 0.5},
    "USDCAD": {"pip_size": 0.0001, "spread_pips": 0.5},
    "USDCHF": {"pip_size": 0.0001, "spread_pips": 0.5},
    "NZDUSD": {"pip_size": 0.0001, "spread_pips": 0.5},
    # Commodities
    "XAUUSD": {"pip_size": 0.01, "spread_pips": 5.0},
    "XTIUSD": {"pip_size": 0.01, "spread_pips": 5.0},
    # Indices
    "US500": {"pip_size": 0.1, "spread_pips": 5.0},
    # Crypto
    "BTCUSD": {"pip_size": 1.0, "spread_pips": 5.0},
}

# ═══════════════════════════════════════════════════════════════════════════════
# Section A: Data Loading
# ═══════════════════════════════════════════════════════════════════════════════


def load_symbol_data(
    symbol: str,
) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load OHLCV data for a single symbol.

    Returns:
        (prices, df_raw, df_features) or (None, None, None) if data is
        missing or has insufficient bars.
    """
    path = os.path.join(DATA_DIR, f"{symbol}_1h.csv")
    if not os.path.exists(path):
        logger.warning(f"Data not found: {path}")
        return None, None, None

    try:
        df = pd.read_csv(path)
    except Exception as e:
        logger.error(f"Failed to read {path}: {e}")
        return None, None, None

    if len(df) < MIN_BARS:
        logger.warning(f"{symbol}: only {len(df)} bars (< {MIN_BARS}), skipping")
        return None, None, None

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df_raw = df.copy()
    prices = df["close"].values.astype(float)

    logger.info(f"{symbol}: loaded {len(prices)} bars from {path}")
    logger.info(
        f"  Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}"
    )

    df_features = compute_features(df)
    logger.info(f"  Features computed: {df_features.shape[1]} columns")

    return prices, df_raw, df_features


# ═══════════════════════════════════════════════════════════════════════════════
# Section B: Regime Detection
# ═══════════════════════════════════════════════════════════════════════════════


def compute_regimes(
    df_features: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """ADX-based regime detection via SimpleRegimeDetector.

    Returns:
        regime_labels: numpy int array of shape (n,), values 0..3
        regime_names:  numpy str array of shape (n,)
    """
    detector = SimpleRegimeDetector(adx_threshold=20.0)
    regime_labels, regime_names = detector.detect(df_features)

    dist = pd.Series(regime_names).value_counts()
    logger.info(f"  Regime distribution: {dist.to_dict()}")

    return regime_labels, regime_names


# ═══════════════════════════════════════════════════════════════════════════════
# Section C: Signal Function Factory (parameterized for each symbol)
# ═══════════════════════════════════════════════════════════════════════════════


def make_symbol_signal_function(  # noqa: C901
    regime_names: np.ndarray,
    features_df: pd.DataFrame,
    pip_size: float,
    spread_pips: float,
    tp_atr: float = TP_ATR,
    commission_per_lot: float = COMMISSION_PER_LOT,
    breakout_lookback: int = BREAKOUT_LOOKBACK,
    burn_in: int = BURN_IN,
) -> Callable:
    """Build a signal function for a specific symbol's pip size and costs.

    Strategy logic (same as run_backtest.make_signal_function but with
    configurable pip_size and spread_pips):

      **Trending** (ADX >= 20):
        Breakout following:
        - Long: Close breaks ABOVE highest high of last N bars
        - Short: Close breaks BELOW lowest low of last N bars

      **Ranging / Volatile** (ADX < 20):
        No trades (breakout-only strategy).

      **Crisis** (ATR/price > 2%):
        Always neutral.

      **Filters**:
        - Volatility filter: Skip if ATR/close > 0.015 or < 0.0005
        - Minimum expected move: ATR * TP_ATR must exceed 2 * round_trip_cost
    """
    # Pre-compute breakout lookback series
    high_max = pd.Series(features_df["high"]).rolling(breakout_lookback).max().shift(1)
    low_min = pd.Series(features_df["low"]).rolling(breakout_lookback).min().shift(1)

    has_atr = "atr_14" in features_df.columns
    has_close = "close" in features_df.columns

    lot_size = 100000
    cost_per_side_pips = spread_pips + commission_per_lot / (pip_size * lot_size)
    round_trip_cost_pips = 2.0 * cost_per_side_pips

    VOLATILITY_MIN = 0.0005
    VOLATILITY_MAX = 0.015

    def signal_fn(
        prices_arg: np.ndarray,
        features_arg: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        n = len(regime_names)
        signals = np.zeros(n, dtype=int)

        for i in range(burn_in, n):
            regime = regime_names[i]

            # ── Crisis: no trades ──
            if regime == "crisis":
                signals[i] = 0
                continue

            # ── Required: ATR and close ──
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
            expected_move_pips = atr_val * tp_atr / pip_size
            if expected_move_pips <= 2.0 * round_trip_cost_pips:
                signals[i] = 0
                continue

            # ═══════════════════════════════════════════════════════════
            # Trending regime — Breakout following
            # ═══════════════════════════════════════════════════════════
            if regime == "trending":
                hh = high_max.iloc[i]
                ll = low_min.iloc[i]

                if pd.isna(hh) or pd.isna(ll):
                    continue

                if close > hh:
                    signals[i] = 1
                elif close < ll:
                    signals[i] = -1

            # Ranging / Volatile — skip (breakout-only)
            elif regime in ("ranging", "volatile"):
                continue

        return signals

    return signal_fn


# ═══════════════════════════════════════════════════════════════════════════════
# Section D: Per-Symbol Backtest
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SymbolResult:
    """Results from a single symbol's backtest."""

    symbol: str
    total_trades: int
    win_rate: float
    profit_factor: float
    sharpe: float
    total_pnl: float
    avg_hold_bars: float
    trade_pnls: np.ndarray
    skipped: bool = False
    skip_reason: str = ""


def run_symbol_backtest(
    symbol: str,
    prices: np.ndarray,
    df_features: pd.DataFrame,
    config: Dict,
) -> SymbolResult:
    """Run the full backtest pipeline for one symbol.

    Handles regime detection, signal generation, and vectorized backtest
    with the correct pip_size and spread for this symbol.
    """
    pip_size = config["pip_size"]
    spread_pips = config["spread_pips"]

    # 1. Regime detection
    _, regime_names = compute_regimes(df_features)

    # 2. Signal function
    sig_fn = make_symbol_signal_function(
        regime_names=regime_names,
        features_df=df_features,
        pip_size=pip_size,
        spread_pips=spread_pips,
        tp_atr=TP_ATR,
        commission_per_lot=COMMISSION_PER_LOT,
        breakout_lookback=BREAKOUT_LOOKBACK,
        burn_in=BURN_IN,
    )

    # 3. Quick signal stats
    signals = sig_fn(prices)
    n_long = int(np.sum(signals == 1))
    n_short = int(np.sum(signals == -1))
    logger.info(
        f"  Signals: {n_long} long, {n_short} short " f"({n_long + n_short} entries)"
    )

    # 4. Prepare feature array and ATR for backtester
    exclude_cols = {"timestamp", "open", "close", "volume"}
    feature_cols = [c for c in df_features.columns if c not in exclude_cols]
    features_np = df_features[feature_cols].values.astype(float)

    atr_col_idx = feature_cols.index("atr_14") if "atr_14" in feature_cols else None
    if atr_col_idx is not None:
        atr_series = features_np[:, atr_col_idx].copy()
        atr_series = np.nan_to_num(atr_series, nan=0.001, posinf=0.01, neginf=0.001)
    else:
        atr_series = np.full(len(prices), 0.001)

    # 5. Run backtest
    bt = VectorizedBacktester(
        spread_pips=spread_pips,
        commission_per_lot=COMMISSION_PER_LOT,
        slippage_model="moderate",
        pip_size=pip_size,
    )

    result = bt.run(
        prices,
        sig_fn,
        features=features_np,
        atr=atr_series,
        regimes=regime_names,
        sl_atr=SL_ATR,
        tp_atr=TP_ATR,
        breakeven_atr=BREAKEVEN_ATR,
        max_hold_bars=MAX_HOLD_BARS,
    )

    # 6. Compute trade-level metrics (avoids equity curve unitization)
    trade_pnls = result.trade_pnls
    n_trades = len(trade_pnls)

    if n_trades == 0:
        return SymbolResult(
            symbol=symbol,
            total_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            sharpe=0.0,
            total_pnl=0.0,
            avg_hold_bars=0.0,
            trade_pnls=np.array([]),
        )

    wins = trade_pnls[trade_pnls > 0]
    losses = trade_pnls[trade_pnls < 0]
    win_rate = len(wins) / n_trades

    sharpe = (
        float(np.mean(trade_pnls) / np.std(trade_pnls) * np.sqrt(252))
        if np.std(trade_pnls) > 1e-10
        else 0.0
    )

    gross_win = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 1.0
    profit_factor = gross_win / max(gross_loss, 1e-8)

    total_pnl = float(np.sum(trade_pnls))

    logger.info(
        f"  Result: Sharpe={sharpe:.3f} | WR={win_rate:.1%} | "
        f"PF={profit_factor:.2f} | PnL={total_pnl:+.1f} | "
        f"Trades={n_trades}"
    )

    return SymbolResult(
        symbol=symbol,
        total_trades=n_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        sharpe=sharpe,
        total_pnl=total_pnl,
        avg_hold_bars=result.avg_hold_bars,
        trade_pnls=trade_pnls,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Section E: Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():  # noqa: C901
    t_start = time.time()

    print()
    print("=" * 72)
    print("    RTS AI FOREX TRADING SYSTEM — MULTI-SYMBOL BACKTEST")
    print("=" * 72)
    print()

    # ── 1. Load data for all symbols ──
    logger.info("═" * 60)
    logger.info("STEP 1/4: Loading data for all symbols")
    logger.info("═" * 60)

    all_data: Dict[
        str,
        Tuple[np.ndarray, pd.DataFrame, pd.DataFrame],
    ] = {}
    skipped: List[Tuple[str, str]] = []

    for symbol in SYMBOLS:
        prices, df_raw, df_features = load_symbol_data(symbol)
        if prices is not None and df_features is not None:
            all_data[symbol] = (prices, df_raw, df_features)
        else:
            skipped.append((symbol, "data missing or insufficient"))

    if not all_data:
        logger.error("No symbols with valid data. Aborting.")
        sys.exit(1)

    logger.info(f"Loaded {len(all_data)}/{len(SYMBOLS)} symbols successfully")

    # ── 2. Run backtest for each symbol ──
    logger.info("═" * 60)
    logger.info("STEP 2/4: Running backtests")
    logger.info("═" * 60)

    results: List[SymbolResult] = []
    all_trade_pnls: List[float] = []

    for symbol, (prices, _df_raw, df_features) in all_data.items():
        logger.info("─" * 60)
        logger.info(f"Backtesting {symbol}...")
        logger.info("─" * 60)

        try:
            config = SYMBOL_CONFIG.get(
                symbol,
                {"pip_size": 0.0001, "spread_pips": 0.5},
            )
            sym_result = run_symbol_backtest(symbol, prices, df_features, config)
            results.append(sym_result)
            all_trade_pnls.extend(sym_result.trade_pnls.tolist())
        except Exception as e:
            logger.error(f"{symbol} backtest failed: {e}")
            import traceback

            traceback.print_exc()
            results.append(
                SymbolResult(
                    symbol=symbol,
                    total_trades=0,
                    win_rate=0.0,
                    profit_factor=0.0,
                    sharpe=0.0,
                    total_pnl=0.0,
                    avg_hold_bars=0.0,
                    trade_pnls=np.array([]),
                    skipped=True,
                    skip_reason=str(e),
                )
            )

    # ── 3. Monte Carlo Significance ──
    logger.info("═" * 60)
    logger.info("STEP 3/4: Monte Carlo significance test")
    logger.info("═" * 60)

    mc_result = None
    all_pnls_array = np.array(all_trade_pnls)
    if len(all_pnls_array) >= 10:
        trades_list = [{"pnl": float(p)} for p in all_pnls_array]
        mc = MonteCarloSigTest(n_permutations=MC_N_PERMUTATIONS, alpha=MC_ALPHA)
        mc_result = mc.test(trades_list)
    else:
        logger.warning(
            f"Too few total trades ({len(all_pnls_array)}) for "
            f"Monte Carlo significance test"
        )

    # ── 4. Print Results ──
    logger.info("═" * 60)
    logger.info("STEP 4/4: Results summary")
    logger.info("═" * 60)

    print()
    print("=" * 72)
    print("  RESULTS TABLE")
    print("=" * 72)
    print(
        f"  {'Symbol':<10s} {'Trades':>7s} {'WinRate':>8s} "
        f"{'PF':>8s} {'Sharpe':>8s} {'PnL':>12s} {'AvgHold':>8s}"
    )
    print(f"  {'-'*10} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*12} {'-'*8}")

    active_results = [r for r in results if not r.skipped]
    total_trades = sum(r.total_trades for r in active_results)
    total_pnl = sum(r.total_pnl for r in active_results)

    for r in sorted(active_results, key=lambda x: x.symbol):
        pnl_str = f"{r.total_pnl:+.1f}" if r.total_pnl >= 0 else f"{r.total_pnl:.1f}"
        print(
            f"  {r.symbol:<10s} {r.total_trades:>7d} {r.win_rate:>7.1%} "
            f"{r.profit_factor:>8.2f} {r.sharpe:>8.3f} "
            f"{pnl_str:>12s} {r.avg_hold_bars:>7.1f}"
        )

    print(f"  {'─' * 10} {'─' * 7} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 12} {'─' * 8}")

    # Aggregate metrics
    avg_sharpe = np.mean([r.sharpe for r in active_results]) if active_results else 0.0
    avg_pf = (
        np.mean([r.profit_factor for r in active_results]) if active_results else 0.0
    )

    print(
        f"  {'TOTAL':<10s} {total_trades:>7d} {'':>8s} {avg_pf:>8.2f} "
        f"{avg_sharpe:>8.3f} {total_pnl:>+12.1f} {'':>8s}"
    )

    # Combined metrics (trade-level from all PnLs)
    if len(all_pnls_array) > 0:
        wins = all_pnls_array[all_pnls_array > 0]
        losses = all_pnls_array[all_pnls_array < 0]
        comb_win_rate = len(wins) / len(all_pnls_array)
        comb_sharpe = (
            float(np.mean(all_pnls_array) / np.std(all_pnls_array) * np.sqrt(252))
            if np.std(all_pnls_array) > 1e-10
            else 0.0
        )
        gross_win = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 1.0
        comb_pf = gross_win / max(gross_loss, 1e-8)

        print()
        print("  COMBINED (pooled trades):")
        print(f"    Total trades:  {len(all_pnls_array)}")
        print(f"    Win rate:      {comb_win_rate:.1%}")
        print(f"    Profit factor: {comb_pf:.2f}")
        print(f"    Sharpe:        {comb_sharpe:.3f}")
        print(f"    Net PnL:       {float(np.sum(all_pnls_array)):+.1f}")

    # Monte Carlo
    print()
    print("  MONTE CARLO:")
    if mc_result is not None:
        sig_sharpe = (
            "SIGNIFICANT ✅"
            if mc_result.is_significant_sharpe
            else "NOT significant ❌"
        )
        sig_return = (
            "SIGNIFICANT ✅"
            if mc_result.is_significant_return
            else "NOT significant ❌"
        )
        print(f"    n_permutations: {mc_result.n_permutations:,}")
        print(
            f"    Actual Sharpe:  {mc_result.actual_sharpe:.4f}  "
            f"(p={mc_result.p_value_sharpe:.4f}) — {sig_sharpe}"
        )
        print(
            f"    Actual Return:  {mc_result.actual_return_pct:+.2f}  "
            f"(p={mc_result.p_value_return:.4f}) — {sig_return}"
        )
        print(f"    Sharpe pctile:  {mc_result.sharpe_percentile:.1%}")
        print(f"    Return pctile:  {mc_result.return_percentile:.1%}")
    else:
        print("    (insufficient trades for significance test)")

    # Skipped symbols
    if skipped:
        print()
        print("  SKIPPED SYMBOLS:")
        for sym, reason in skipped:
            print(f"    {sym:<10s} — {reason}")

    failed_results = [r for r in results if r.skipped]
    if failed_results:
        print()
        print("  FAILED SYMBOLS:")
        for r in failed_results:
            print(f"    {r.symbol:<10s} — {r.skip_reason or 'unknown error'}")

    # Runtime
    elapsed = time.time() - t_start
    print()
    print(f"  Runtime: {elapsed:.1f}s")
    print()
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()
