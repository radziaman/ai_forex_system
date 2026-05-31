#!/usr/bin/env python3
"""Resave feature normalization with microstructure (49-dim canonical).

Loads historical data for all symbols, creates a FeaturePipeline with
use_microstructure=True, fits on each symbol, saves normalization to
models/feature_norm.npz, and verifies the saved feature count is 49.
"""

import sys
from pathlib import Path

# Run with: pip install -e . && python -m src.scripts.resave_feature_norm


import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402
from rts_ai_fx.features_unified import (  # noqa: E402
    FeaturePipeline,
    EXPECTED_FEATURE_DIM,
)

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
DATA_DIR = Path("data/historical")


def main():
    fp = FeaturePipeline(lookback=30, timeframes=["1h"], use_microstructure=True)

    n_fitted = 0
    for symbol in SYMBOLS:
        csv_path = DATA_DIR / f"{symbol}_1h.csv"
        if not csv_path.exists():
            logger.warning(f"Missing {csv_path}, skipping {symbol}")
            continue
        df = pd.read_csv(csv_path)
        dfs = {"1h": df}
        fp.fit(dfs, symbol=symbol)
        n_features = len(fp._feature_cols) if fp._feature_cols else 0
        logger.info(f"Fitted {symbol}: {n_features} features")
        n_fitted += 1

    if n_fitted == 0:
        logger.error("No symbols fitted — nothing to save")
        return False

    fp.save_normalization()

    # Verify
    fp2 = FeaturePipeline(lookback=30, timeframes=["1h"])
    fp2.load_normalization()
    n_feats = len(fp2._feature_cols) if fp2._feature_cols else 0
    logger.info(
        f"Saved normalization: {n_feats} features " f"(expected {EXPECTED_FEATURE_DIM})"
    )

    # Check feature names from last fit
    if fp._feature_cols:
        logger.info(f"Feature columns ({len(fp._feature_cols)}):")
        for i, col in enumerate(fp._feature_cols, 1):
            logger.info(f"  {i:2d}. {col}")

    if n_feats < EXPECTED_FEATURE_DIM:
        logger.warning(
            f"Missing {EXPECTED_FEATURE_DIM - n_feats} features — "
            "check that microstructure features are being generated "
            "(OHLCV data required)"
        )

    if n_feats == EXPECTED_FEATURE_DIM:
        logger.info("✓ Feature normalization is 49-dim canonical — all good")
    else:
        logger.warning(
            f"Feature count is {n_feats}, " f"expected {EXPECTED_FEATURE_DIM}"
        )

    return n_feats == EXPECTED_FEATURE_DIM


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
