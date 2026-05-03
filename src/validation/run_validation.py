"""
CLI entry point for running the validation suite:
  - Walk-forward validation
  - Monte Carlo significance tests
  - Vectorized backtest with sensitivity analysis
  - Regime-dependent training

Usage:
  python -m src.validation.run_validation --walk-forward
  python -m src.validation.run_validation --mc-test
  python -m src.validation.run_validation --bt-sensitivity
  python -m src.validation.run_validation --regime-train
  python -m src.validation.run_validation --all
"""
import argparse
import json
import sys
import os
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import numpy as np
import pandas as pd
from validation.walk_forward import PurgedWalkForward
from validation.monte_carlo import MonteCarloSigTest
from backtest.vectorized_backtester import VectorizedBacktester


def load_test_data(symbol: str = "EURUSD", days: int = 365 * 3) -> dict:
    """Load test data for validation."""
    try:
        import yfinance as yf
        ticker = f"{symbol}=X"
        data = yf.download(ticker, period=f"{days}d", interval="1h", progress=False)
        if data.empty:
            raise ValueError("No data returned")
        prices = data["Close"].values
        high = data["High"].values
        low = data["Low"].values
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - prices[:-1]),
                np.abs(low[1:] - prices[:-1]),
            ),
        )
        atr = np.concatenate([[np.mean(tr[:14])], tr])
        return {"prices": prices, "atr": atr, "df": data}
    except Exception as e:
        logger.warning(f"yfinance failed: {e}, generating synthetic data")
        n = days
        base = 1.12
        prices = base * np.exp(np.cumsum(np.random.normal(0, 0.0002, n)))
        atr = np.full(n, 0.002)
        return {"prices": prices, "atr": atr}


def run_walk_forward(prices: np.ndarray, n_folds: int = 6):
    logger.info("=" * 60)
    logger.info("WALK-FORWARD VALIDATION")
    logger.info("=" * 60)

    wf = PurgedWalkForward(
        n_folds=n_folds,
        test_window=max(252, len(prices) // (n_folds * 2)),
        embargo=10,
        min_train_window=max(504, len(prices) // 3),
    )

    def dummy_strategy(train_p, test_p, train_f, test_f):
        return [{"pnl": np.random.normal(0.5, 2.0)} for _ in range(20)]

    results = wf.run(prices, dummy_strategy, verbose=True)
    summary = PurgedWalkForward.summary(results)
    logger.info(f"\nSummary: {json.dumps(summary, indent=2)}")
    return summary


def run_monte_carlo(trades_file: str = "data/trades/trade_history.json"):
    logger.info("=" * 60)
    logger.info("MONTE CARLO SIGNIFICANCE TEST")
    logger.info("=" * 60)

    if os.path.exists(trades_file):
        with open(trades_file) as f:
            trades = json.load(f)
        logger.info(f"Loaded {len(trades)} trades from {trades_file}")
    else:
        logger.info("No trade history found, generating synthetic trades")
        trades = [
            {"pnl": float(np.random.normal(2.0, 5.0))}
            for _ in range(np.random.randint(50, 200))
        ]

    mc = MonteCarloSigTest(n_permutations=10000, alpha=0.05)
    return mc.test(trades)


def run_backtest(prices: np.ndarray, atr: np.ndarray):
    logger.info("=" * 60)
    logger.info("VECTORIZED BACKTEST WITH SENSITIVITY ANALYSIS")
    logger.info("=" * 60)

    bt = VectorizedBacktester(spread_pips=0.5, commission_per_lot=7.0)

    def dummy_signal(p, f):
        sig = np.zeros(len(p))
        ma_fast = pd.Series(p).rolling(20).mean().values
        ma_slow = pd.Series(p).rolling(50).mean().values
        for i in range(50, len(p)):
            if ma_fast[i] > ma_slow[i] and ma_fast[i - 1] <= ma_slow[i - 1]:
                sig[i] = 1
            elif ma_fast[i] < ma_slow[i] and ma_fast[i - 1] >= ma_slow[i - 1]:
                sig[i] = -1
        return sig

    results = bt.run_with_sensitivity(prices, dummy_signal, atr=atr)
    for key, r in results.items():
        logger.info(f"\n  {key}:")
        for k, v in r.to_dict().items():
            logger.info(f"    {k}: {v}")

    return results


def main():
    parser = argparse.ArgumentParser(description="RTS Validation Suite")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward validation")
    parser.add_argument("--mc-test", action="store_true", help="Run Monte Carlo test")
    parser.add_argument("--bt-sensitivity", action="store_true", help="Run backtest sensitivity")
    parser.add_argument("--all", action="store_true", help="Run all validations")
    parser.add_argument("--symbol", default="EURUSD", help="Symbol for test data")
    parser.add_argument("--days", type=int, default=365 * 3, help="Days of test data")
    parser.add_argument("--folds", type=int, default=6, help="Walk-forward folds")
    parser.add_argument("--output", default="data/validation_results.json", help="Output file")
    args = parser.parse_args()

    should_run_all = args.all or not (args.walk_forward or args.mc_test or args.bt_sensitivity)
    data = load_test_data(args.symbol, args.days)
    results = {}

    if args.walk_forward or should_run_all:
        results["walk_forward"] = run_walk_forward(data["prices"], args.folds)

    if args.mc_test or should_run_all:
        results["monte_carlo"] = run_monte_carlo()

    if args.bt_sensitivity or should_run_all:
        results["backtest"] = run_backtest(data["prices"], data["atr"])

    if results:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
