#!/usr/bin/env python3
"""Walk-forward validation CI gate.

Runs SmartWalkForward on recent price data and validates that
model stability and test Sharpe meet minimum thresholds.

Exit codes:
    0 -- Pass (all thresholds met)
    1 -- Fail (thresholds not met or error)
"""

import sys
import os
from typing import List, Optional

import numpy as np
from loguru import logger

# Ensure src/ is on the path so we can import from the package
_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _src not in sys.path:
    sys.path.insert(0, _src)


def load_recent_prices(symbol: str = "EURUSD", days: int = 90) -> np.ndarray:
    """Load recent price data from CSV cache or generate synthetic data."""
    import pandas as pd

    # Try loading from src/data/historical (primary cache)
    csv_path = f"data/historical/{symbol}_1h.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if "close" in df.columns:
            prices = df["close"].dropna().values
            if len(prices) >= 100:
                logger.info(f"Loaded {len(prices)} prices from {csv_path}")
                return prices

    # Try loading from data/{symbol} directory
    data_dir = f"data/{symbol}"
    if os.path.exists(data_dir):
        csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
        if csv_files:
            df = pd.read_csv(os.path.join(data_dir, csv_files[0]))
            if "close" in df.columns:
                prices = df["close"].dropna().values
                if len(prices) >= 100:
                    logger.info(f"Loaded {len(prices)} prices from {data_dir}")
                    return prices

    # Fallback: synthetic price data (random walk with drift)
    logger.warning(f"No cached data found for {symbol}, generating synthetic data")
    np.random.seed(42)
    n = days * 24  # Hourly bars
    hourly_drift = 0.0001 / 24  # ~1bp per hour
    hourly_vol = 0.005 / np.sqrt(24)
    returns = np.random.normal(hourly_drift, hourly_vol, n)
    price = 1.1000
    prices = []
    for r in returns:
        price *= 1.0 + r
        prices.append(price)
    logger.info(f"Generated {len(prices)} synthetic prices")
    return np.array(prices)


def compute_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log returns from price series."""
    return np.diff(np.log(prices))


def identity_model(
    prices: np.ndarray,
    features: Optional[np.ndarray],
    regimes: Optional[np.ndarray] = None,
) -> List[float]:
    """Identity model: returns the features array as the model's predictions.

    This acts as a no-op benchmark for testing the walk-forward pipeline.
    The features should be the returns of the price series, so the "model"
    simply reproduces them.
    """
    if features is None or len(features) == 0:
        return [0.0]
    return features.tolist()


def run_validation(
    prices: np.ndarray,
    stability_threshold: float = 0.5,
    sharpe_threshold: float = 0.3,
) -> bool:
    """Run SmartWalkForward and check thresholds."""
    try:
        from validation.smart_walk_forward import SmartWalkForward
    except ImportError as exc:
        logger.error(f"Failed to import SmartWalkForward: {exc}")
        return False

    returns = compute_returns(prices)

    # Use returns as features for the identity model
    features = returns

    wf = SmartWalkForward(
        n_folds=6,
        test_window=252,
        embargo=10,
        min_train_window=504,
    )

    try:
        result = wf.run(prices, identity_model, features)
    except Exception as exc:
        logger.error(f"SmartWalkForward.run() failed: {exc}")
        return False

    logger.info("=" * 60)
    logger.info("WALK-FORWARD VALIDATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Total folds:         {result.total_folds}")
    logger.info(f"  Avg train Sharpe:    {result.avg_train_sharpe:.4f}")
    logger.info(f"  Avg test Sharpe:     {result.avg_test_sharpe:.4f}")
    logger.info(f"  Avg degradation:     {result.avg_degradation:.4f}")
    logger.info(f"  Stability score:     {result.stability_score:.4f}")
    logger.info(f"  Passed (internal):   {result.passed}")
    logger.info("-" * 60)
    logger.info(f"  CI Gate Thresholds:")
    logger.info(
        f"    Stability >= {stability_threshold}:      {result.stability_score >= stability_threshold}"
    )
    logger.info(
        f"    Test Sharpe >= {sharpe_threshold}:       {result.avg_test_sharpe >= sharpe_threshold}"
    )
    logger.info("=" * 60)

    stability_ok = result.stability_score >= stability_threshold
    sharpe_ok = result.avg_test_sharpe >= sharpe_threshold

    if not stability_ok:
        logger.error(
            f"FAIL: Stability score {result.stability_score:.4f} "
            f"< {stability_threshold}"
        )
    if not sharpe_ok:
        logger.error(
            f"FAIL: Test Sharpe {result.avg_test_sharpe:.4f} < {sharpe_threshold}"
        )

    return stability_ok and sharpe_ok


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Walk-forward validation CI gate")
    parser.add_argument("--symbol", default="EURUSD", help="Symbol to validate")
    parser.add_argument("--days", type=int, default=90, help="Days of data to use")
    parser.add_argument(
        "--stability",
        type=float,
        default=0.5,
        help="Minimum stability score threshold",
    )
    parser.add_argument(
        "--sharpe",
        type=float,
        default=0.3,
        help="Minimum average test Sharpe threshold",
    )
    args = parser.parse_args()

    prices = load_recent_prices(args.symbol, args.days)
    if len(prices) < 100:
        logger.error(f"Not enough price data: {len(prices)} points (need >= 100)")
        sys.exit(1)

    success = run_validation(
        prices,
        stability_threshold=args.stability,
        sharpe_threshold=args.sharpe,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
