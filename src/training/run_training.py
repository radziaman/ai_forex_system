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
import os
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import numpy as np
import pandas as pd
from rts_ai_fx.model import LSTMCNNHybrid, ProfitabilityClassifier
from rts_ai_fx.features_unified import FeaturePipeline
from training.regime_trainer import RegimeTrainer
from training.distributed_trainer import DistributedTrainer, TrialConfig


def load_training_data(
    symbol: str = "EURUSD",
    years: int = 5,
    lookback: int = 30,
    timeframes: list = None,
    frac: float = 1.0,
):
    timeframes = timeframes or ["1h", "4h"]
    logger.info(f"Loading {years} years of {symbol} data...")

    try:
        import yfinance as yf
        dfs = {}
        for tf in timeframes:
            interval = "1h" if tf == "1h" else "4h" if tf == "4h" else "1d"
            period = f"{max(years, 2)}y" if interval == "1h" else f"{max(years, 5)}y"
            ticker = f"{symbol}=X"
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            if data.empty:
                raise ValueError(f"No data for {ticker}")
            df = data.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            })
            df["timestamp"] = df.index.astype(np.int64) // 10**9
            df = df.reset_index(drop=True)
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
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=config.early_stop_patience,
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


def main():
    parser = argparse.ArgumentParser(description="RTS Training Pipeline")
    parser.add_argument("--mode", choices=["local", "distributed", "sweep", "regime"], default="local")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--trials", type=int, default=50, help="HP sweep trials")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()

    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    X, y, fp, dfs = load_training_data(args.symbol, args.years)
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    if args.mode == "regime":
        logger.info("Training regime-dependent models...")
        trainer = RegimeTrainer(n_regimes=4, model_dir=f"{args.output_dir}/regime_models")
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
            param_grid, train_model, X_train, y_train, X_val, y_val,
            n_trials=args.trials,
        )
        logger.info(f"Best config: {results[0].config}")

    else:
        logger.info("Training single model...")
        config = TrialConfig(epochs=args.epochs, lookback=30)
        metrics = train_model(X_train, y_train, X_val, y_val, config)
        logger.info(f"Training complete: {metrics}")

        n_features = X.shape[-1]
        model = LSTMCNNHybrid(lookback=30, n_features=n_features)
        model.build()
        model.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config.epochs, batch_size=config.batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=10, restore_best_weights=True,
                ),
            ],
            verbose=1,
        )
        model.save(f"{args.output_dir}/lstm_cnn_{args.symbol}.keras")

        classifier = ProfitabilityClassifier(lookback=30, n_features=n_features)
        classifier.build()
        classifier.train(X_train, y_train, X_val, y_val,
                         epochs=args.epochs, batch_size=32)
        classifier.save(f"{args.output_dir}/classifier_{args.symbol}.keras")

        logger.info(f"Models saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
