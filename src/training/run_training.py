"""
Training pipeline runner — trains models with or without distributed compute.
Usage:
  python -m src.training.run_training --mode local
  python -m src.training.run_training --mode distributed --num-workers 4
  python -m src.training.run_training --sweep --trials 50
  python -m src.training.run_training --regime-train
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

# Run with: pip install -e . && python -m src.training.run_training

import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402
from rts_ai_fx.model import LSTMCNNHybrid, ProfitabilityClassifier  # noqa: E402
from rts_ai_fx.features_unified import FeaturePipeline  # noqa: E402
from training.regime_trainer import RegimeTrainer  # noqa: E402
from training.distributed_trainer import DistributedTrainer, TrialConfig  # noqa: E402


async def load_training_data(
    symbol: str = "EURUSD",
    years: int = 5,
    lookback: int = 30,
    timeframes: Optional[list] = None,
    frac: float = 1.0,
):
    timeframes = timeframes or ["1h", "4h"]
    logger.info(f"Loading {years} years of {symbol} data from Dukascopy...")

    try:
        from datetime import datetime, timezone, timedelta
        from data.dukascopy_provider import DukascopyDataProvider

        provider = DukascopyDataProvider(cache=True)
        dfs = {}
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start = (
            datetime.now(timezone.utc) - timedelta(days=int(years * 365))
        ).strftime("%Y-%m-%d")
        for tf_name in timeframes:
            interval_map = {"1h": "1h", "4h": "4h", "1d": "1d"}
            interval = interval_map.get(tf_name, "1h")
            ohlcv = await provider.fetch_ohlcv(symbol, interval, start=start, end=end)
            if not ohlcv or len(ohlcv) < 100:
                raise ValueError(f"Insufficient Dukascopy data for {symbol} {tf_name}")
            df = pd.DataFrame(
                [
                    {
                        "timestamp": o.timestamp,
                        "open": o.open,
                        "high": o.high,
                        "low": o.low,
                        "close": o.close,
                        "volume": o.volume,
                    }
                    for o in ohlcv
                ]
            )
            if frac < 1.0:
                cut = int(len(df) * frac)
                df = df.iloc[-cut:].reset_index(drop=True)
            dfs[tf] = df
            logger.info(f"  {tf}: {len(df)} bars")

        fp = FeaturePipeline(lookback=lookback, timeframes=timeframes)
        X, y = fp.fit_transform(dfs)
        logger.info(f"Generated {len(X)} training sequences (n_features={X.shape[1]})")
        return X, y, fp, dfs
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        raise


def train_model(X_train, y_train, X_val, y_val, config: TrialConfig) -> dict:
    n_features = X_train.shape[-1]
    model = LSTMCNNHybrid(
        lookback=config.lookback or 30,
        n_features=n_features,
        lstm_units=config.lstm_units,
        cnn_filters=config.cnn_filters,
    )
    model.build()
    hist = model.model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.early_stop_patience,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
            ),
        ],
        verbose=0,
    )
    return {
        "val_loss": float(min(hist.history.get("val_loss", [999]))),
        "val_mae": float(min(hist.history.get("val_mae", [999]))),
        "val_accuracy": float(max(hist.history.get("val_accuracy", [0]))),
        "params": model.model.count_params(),
    }


async def main():
    parser = argparse.ArgumentParser(
        description="RTS: Agentic FX System Elite - Training Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["local", "distributed", "sweep", "regime"],
        default="local",
    )
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--trials", type=int, default=50, help="HP sweep trials")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--multi-asset",
        action="store_true",
        help="Use global multi-asset model instead of per-symbol models",
    )
    parser.add_argument(
        "--adversarial",
        action="store_true",
        help="Apply adversarial training augmentation",
    )
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()

    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")

    X, y, fp, dfs = await load_training_data(args.symbol, args.years)
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    if args.mode == "regime":
        logger.info("Training regime-dependent models...")
        trainer = RegimeTrainer(
            n_regimes=4, model_dir=f"{args.output_dir}/regime_models"
        )
        results = trainer.train_regime_models(dfs, epochs=args.epochs)
        with open(f"{args.output_dir}/regime_results.json", "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Regime results: {json.dumps(results, indent=2)}")

    elif args.mode == "sweep":
        logger.info(f"Running hyperparameter sweep ({args.trials} trials)...")
        dt = DistributedTrainer(
            num_workers=args.num_workers,
            use_ray=args.mode == "distributed",
            results_dir=f"{args.output_dir}/hp_sweeps",
        )
        param_grid = {
            "lstm_units": [64, 128, 256],
            "cnn_filters": [64, 128, 256],
            "learning_rate": [0.0001, 0.001, 0.01],
            "dropout": [0.1, 0.2, 0.3],
            "batch_size": [16, 32, 64],
        }
        results = dt.hyperparameter_sweep(
            param_grid,
            train_model,
            X_train,
            y_train,
            X_val,
            y_val,
            n_trials=args.trials,
        )
        logger.info(f"Best config: {results[0].config}")

    elif args.multi_asset:
        from rts_ai_fx.multi_asset_model import GlobalMultiAssetModel, SYMBOL_NAMES

        logger.info("Training Global Multi-Asset Model...")

        # Load data for each symbol and combine
        all_X_train, all_y_multi, all_symbol_ids = [], [], []
        all_X_val, all_y_val_multi, all_symbol_ids_val = [], [], []

        for sym_id, sym_name in enumerate(SYMBOL_NAMES):
            try:
                X_sym, y_sym, _, _ = await load_training_data(sym_name, args.years)
                split = int(len(X_sym) * 0.8)

                # Create multi-target y with one column per symbol
                y_multi = np.zeros((len(y_sym), len(SYMBOL_NAMES)), dtype=np.float32)
                y_target = y_sym.flatten() if y_sym.ndim > 1 else y_sym
                y_multi[:, sym_id] = y_target

                sid = np.full((len(y_sym), 1), sym_id, dtype=np.int32)

                all_X_train.append(X_sym[:split])
                all_y_multi.append(y_multi[:split])
                all_symbol_ids.append(sid[:split])

                all_X_val.append(X_sym[split:])
                all_y_val_multi.append(y_multi[split:])
                all_symbol_ids_val.append(sid[split:])

                logger.info(
                    f"  {sym_name}: {split} train, " f"{len(X_sym) - split} val samples"
                )
            except Exception as e:
                logger.warning(f"Skipping {sym_name}: {e}")

        if not all_X_train:
            raise ValueError("No training data available for any symbol")

        X_train = np.concatenate(all_X_train, axis=0)
        y_train = np.concatenate(all_y_multi, axis=0)
        symbol_ids_train = np.concatenate(all_symbol_ids, axis=0)

        X_val = np.concatenate(all_X_val, axis=0)
        y_val = np.concatenate(all_y_val_multi, axis=0)
        symbol_ids_val = np.concatenate(all_symbol_ids_val, axis=0)

        # Shuffle training data
        idx = np.random.permutation(len(X_train))
        X_train = X_train[idx]
        y_train = y_train[idx]
        symbol_ids_train = symbol_ids_train[idx]

        n_features = X_train.shape[-1]
        model = GlobalMultiAssetModel(
            n_symbols=len(SYMBOL_NAMES),
            lookback=30,
            n_features=n_features,
        )
        model.build()
        model.train(
            X_train,
            y_train,
            symbol_ids_train,
            X_val,
            y_val,
            symbol_ids_val,
            epochs=args.epochs,
        )
        model.save(f"{args.output_dir}/global_multi_asset.keras")
        logger.info(
            f"Global Multi-Asset Model saved to "
            f"{args.output_dir}/global_multi_asset.keras"
        )

    else:
        logger.info("Training single model...")
        config = TrialConfig(epochs=args.epochs, lookback=30)
        metrics = train_model(X_train, y_train, X_val, y_val, config)
        logger.info(f"Training complete: {metrics}")

        n_features = X.shape[-1]
        model = LSTMCNNHybrid(lookback=30, n_features=n_features)
        model.build()

        # Adversarial augmentation (if enabled)
        if args.adversarial:
            from rts_ai_fx.adversarial import (
                AdversarialTrainer,
                PGDAdversarial,
            )

            if hasattr(model, "model") and model.model is not None:
                trainer = AdversarialTrainer(PGDAdversarial(), adversarial_ratio=0.3)
                X_train, y_train = trainer.augment_batch(
                    model.model.predict, X_train, y_train
                )
                logger.info(f"Adversarial augmentation: {len(X_train)} total samples")

        model.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=config.epochs,
            batch_size=config.batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    restore_best_weights=True,
                ),
            ],
            verbose=1,
        )
        model.save(f"{args.output_dir}/lstm_cnn_{args.symbol}.keras")

        classifier = ProfitabilityClassifier(lookback=30, n_features=n_features)
        classifier.build()
        classifier.train(
            X_train, y_train, X_val, y_val, epochs=args.epochs, batch_size=32
        )
        classifier.save(f"{args.output_dir}/classifier_{args.symbol}.keras")

        logger.info(f"Models saved to {args.output_dir}/")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
