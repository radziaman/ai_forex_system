#!/usr/bin/env python3
"""Train XGBoost models per symbol for ensemble diversity.

Chen & Guestrin (2016) — XGBoost: A Scalable Tree Boosting System.
Combining XGBoost with neural networks in an ensemble consistently
outperforms either model class alone (Brown 2005, ensemble diversity).
"""
import os
import sys
import warnings
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from loguru import logger  # noqa: E402
from data.data_manager import DataManager, SYMBOLS  # noqa: E402
from rts_ai_fx.features_unified import FeaturePipeline  # noqa: E402

LOOKBACK = 30
TIMEFRAMES = ["1h"]  # Single timeframe for training simplicity
HISTORICAL_PATH = "data/historical"
MODELS_PATH = "models"


def train_xgboost(symbol: str) -> bool:
    """Train a gradient boosting model for a symbol.

    Uses XGBoost if available (fast), falls back to sklearn's
    GradientBoostingRegressor (pure Python, no native deps).
    """
    try:
        import joblib

        try:
            from xgboost import XGBRegressor as GBMRegressor

            use_xgb = True
        except (ImportError, Exception):
            from sklearn.ensemble import GradientBoostingRegressor as GBMRegressor

            use_xgb = False

        dm = DataManager(historical_path=HISTORICAL_PATH)
        dm.load_historical(symbol, "1h", days=365)

        fp = FeaturePipeline(
            lookback=LOOKBACK,
            timeframes=TIMEFRAMES,
            use_microstructure=False,
            use_cross_asset=False,
        )

        sequences, targets = fp.create_sequences(dm.ohlcv, symbol, flatten=True)
        if len(sequences) < 100:
            logger.warning(f"{symbol}: only {len(sequences)} sequences, skipping")
            return False

        # Train/test split
        split = int(len(sequences) * 0.8)
        X_train, X_test = sequences[:split], sequences[split:]
        y_train, y_test = targets[:split], targets[split:]

        # Target: next period return
        y_train = np.diff(y_train) / y_train[:-1] if len(y_train) > 1 else y_train
        y_test = np.diff(y_test) / y_test[:-1] if len(y_test) > 1 else y_test
        # Align lengths
        min_len = min(len(X_train), len(y_train), len(X_test), len(y_test))
        X_train, y_train = X_train[:min_len], y_train[:min_len]
        X_test, y_test = X_test[:min_len], y_test[:min_len]

        if len(X_train) < 50:
            logger.warning(f"{symbol}: insufficient training data")
            return False

        # Train gradient boosting regressor
        model = GBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            max_features=0.8,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # Save model
        os.makedirs(MODELS_PATH, exist_ok=True)
        model_path = f"{MODELS_PATH}/xgboost_{symbol}.pkl"
        joblib.dump(model, model_path)

        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        n_features = X_train.shape[1]
        algo = "XGBoost" if use_xgb else "sklearn-GBR"
        logger.info(
            f"{symbol}: {algo} trained ({n_features} features, "
            f"R² train={train_score:.4f}, test={test_score:.4f})"
        )
        return True

    except Exception as e:
        logger.warning(f"{symbol}: Gradient boosting failed — {e}")
        return False


def main():
    logger.info("=" * 60)
    logger.info("XGBoost Training — Ensemble Diversity Enhancement")
    logger.info("=" * 60)

    results = {}
    for symbol in SYMBOLS:
        ok = train_xgboost(symbol)
        results[symbol] = "✅" if ok else "❌"
        print(f"  {results[symbol]} {symbol}")

    successes = sum(1 for v in results.values() if v == "✅")
    logger.info(f"XGBoost training complete: {successes}/{len(SYMBOLS)} models trained")


if __name__ == "__main__":
    main()
