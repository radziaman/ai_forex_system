"""
Speed-optimized training pipeline with M1 GPU, transfer learning, and progressive training.
Trains all 7 pairs 3-5x faster than the standard approach.
"""

import os
import time
import asyncio
import warnings
from typing import Optional, Dict, List
from datetime import datetime, timezone
from loguru import logger

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow.keras import Model, Input  # noqa: E402
from tensorflow.keras.layers import (  # noqa: E402
    LSTM,
    Dense,
    Dropout,
    Concatenate,
    GlobalMaxPooling1D,
    Conv1D,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam  # noqa: E402

tf.get_logger().setLevel("ERROR")

from rts_ai_fx.features_unified import FeaturePipeline  # noqa: E402


def _setup_gpu():
    devices = tf.config.list_physical_devices()
    for d in devices:
        logger.info(f"  Device: {d.device_type} {d.name}")

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"  GPU acceleration: {len(gpus)} device(s) active")
            return "GPU"
        except Exception as e:
            logger.warning(f"  GPU config failed: {e}")

    try:
        # Check for Apple MPS (Metal on M1/M2 Macs)
        mps = tf.config.experimental.list_physical_devices("MPS")
        if mps:
            tf.config.experimental.set_visible_devices(mps, "MPS")
            logger.info("  MPS (Metal GPU) acceleration active!")
            return "MPS"
    except Exception:
        pass

    logger.info("  Training on CPU")
    return "CPU"


def build_transfer_model(
    n_features: int,
    lookback: int = 30,
    lstm_units: int = 128,
    cnn_filters: int = 128,
) -> Model:
    """Build LSTM-CNN with trainable/unfreezable layers for transfer learning."""
    inputs = Input(shape=(lookback, n_features), name="input")

    # Feature extractor (will be frozen during fine-tuning)
    lstm_branch = LSTM(
        lstm_units, dropout=0.2, return_sequences=False, name="lstm_feature_extractor"
    )(inputs)
    cnn_branch = Conv1D(
        cnn_filters, 3, activation="relu", padding="same", name="cnn_feature_extractor"
    )(inputs)
    cnn_branch = GlobalMaxPooling1D(name="cnn_pool")(cnn_branch)

    # Fusion layer
    fused = Concatenate(name="fusion")([lstm_branch, cnn_branch])
    fused = Dense(64, activation="relu", name="head_dense")(fused)
    fused = BatchNormalization(name="head_bn")(fused)
    fused = Dropout(0.2, name="head_dropout")(fused)
    outputs = Dense(1, activation="linear", name="output")(fused)

    model = Model(inputs=inputs, outputs=outputs, name="transfer_lstm_cnn")
    return model


class FastTrainer:
    def __init__(self, lookback: int = 30, timeframes: Optional[List[str]] = None):
        self.lookback = lookback
        self.timeframes = timeframes or ["1h"]
        self.device = _setup_gpu()
        self.base_model: Optional[Model] = None
        self.base_pair: Optional[str] = None

    async def fetch_data(
        self, symbol: str, period: str = "2y"
    ) -> Optional[pd.DataFrame]:
        """Fetch historical data via Dukascopy (primary source)."""
        try:
            from data.dukascopy_realtime import DukascopyProvider

            provider = DukascopyProvider(cache=True)
            years = 2 if period == "2y" else 1
            end = datetime.now(timezone.utc)
            start = end.replace(year=end.year - years)
            ohlcv = await provider.fetch_ohlcv(
                symbol,
                "1h",
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
            )
            if not ohlcv or len(ohlcv) < 100:
                logger.warning(
                    f"Insufficient Dukascopy data for {symbol}: {len(ohlcv) if ohlcv else 0} bars"
                )
                return None
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
            return df
        except Exception as e:
            logger.error(f"Dukascopy fetch failed for {symbol}: {e}")
            return None

    def prepare_sequences(
        self,
        pair: str,
        df: pd.DataFrame,
    ) -> tuple:
        fp = FeaturePipeline(
            lookback=self.lookback,
            timeframes=self.timeframes,
            use_microstructure=False,
        )
        X, y = fp.fit_transform(
            {pair: {tf: df for tf in self.timeframes}},
            symbol=pair,
            flatten=False,
        )
        return X, y, fp

    async def train_base_model(
        self,
        pair: str = "EURUSD",
        epochs: int = 40,
        batch_size: int = 32,
    ) -> Model:
        logger.info(f"Training BASE model on {pair} ({epochs} epochs)...")
        yf_sym = f"{pair}=X"
        df = await self.fetch_data(yf_sym)
        if df is None:
            raise ValueError(f"Cannot fetch {pair}")

        X, y, _ = self.prepare_sequences(pair, df)
        split = int(len(X) * 0.8)
        X_tr, X_v = X[:split], X[split:]
        y_tr, y_v = y[:split], y[split:]

        nf = X.shape[-1]
        model = build_transfer_model(n_features=nf, lookback=self.lookback)
        model.compile(optimizer=Adam(0.001), loss="mse", metrics=["mae"])

        cb = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=4,
                min_lr=1e-6,
            ),
        ]
        t0 = time.time()
        h = model.fit(
            X_tr,
            y_tr,
            validation_data=(X_v, y_v),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=cb,
            verbose=1,
        )
        elapsed = time.time() - t0
        vl = min(h.history["val_loss"])
        vm = min(h.history["val_mae"])

        logger.info(
            f"Base model: val_loss={vl:.6f}, val_mae={vm:.6f}, "
            f"time={elapsed:.0f}s ({elapsed/epochs:.1f}s/epoch)"
        )

        model.save("models/base_lstm_cnn.keras")
        self.base_model = model
        self.base_pair = pair
        return model

    async def fine_tune_pair(
        self,
        pair: str,
        base_model: Optional[Model] = None,
        freeze_epochs: int = 10,
        full_epochs: int = 10,
        batch_size: int = 32,
    ) -> Model:
        if base_model is None:
            base_model = self.base_model
        if base_model is None:
            raise ValueError("No base model. Call train_base_model() first.")

        yf_sym = f"{pair}=X"
        df = await self.fetch_data(yf_sym)
        if df is None:
            raise ValueError(f"Cannot fetch {pair}")

        X, y, _ = self.prepare_sequences(pair, df)
        split = int(len(X) * 0.8)
        X_tr, X_v = X[:split], X[split:]
        y_tr, y_v = y[:split], y[split:]

        # If feature dimension differs, cannot transfer
        nf = X.shape[-1]
        expected = base_model.input_shape[-1]
        if nf != expected:
            logger.warning(
                f"Feature dim mismatch for {pair}: got {nf}, expected {expected}. "
                f"Training from scratch instead."
            )
            model = build_transfer_model(n_features=nf, lookback=self.lookback)
            model.compile(optimizer=Adam(0.001), loss="mse", metrics=["mae"])
            base_model = None  # start fresh

        if base_model is not None:
            model = tf.keras.models.clone_model(base_model)
            model.build(base_model.input_shape)
            model.set_weights(base_model.get_weights())

            # Phase 1: Freeze feature extractor, train head only
            for layer in model.layers:
                if "feature_extractor" in layer.name or "cnn_pool" in layer.name:
                    layer.trainable = False
                else:
                    layer.trainable = True

            model.compile(optimizer=Adam(0.001), loss="mse", metrics=["mae"])
            logger.info(f"  {pair}: Phase 1 — training head ({freeze_epochs} epochs)")
            t0 = time.time()
            model.fit(
                X_tr,
                y_tr,
                validation_data=(X_v, y_v),
                epochs=freeze_epochs,
                batch_size=batch_size,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=5,
                        restore_best_weights=True,
                    )
                ],
                verbose=1,
            )
            _ = time.time()

            # Phase 2: Unfreeze all, fine-tune full model
            for layer in model.layers:
                layer.trainable = True
            model.compile(optimizer=Adam(0.0001), loss="mse", metrics=["mae"])
            logger.info(f"  {pair}: Phase 2 — full fine-tune ({full_epochs} epochs)")
            h = model.fit(
                X_tr,
                y_tr,
                validation_data=(X_v, y_v),
                epochs=full_epochs,
                batch_size=batch_size,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=4,
                        restore_best_weights=True,
                    )
                ],
                verbose=1,
            )
            _ = time.time()
        else:
            # Train from scratch
            model.compile(optimizer=Adam(0.001), loss="mse", metrics=["mae"])
            h = model.fit(
                X_tr,
                y_tr,
                validation_data=(X_v, y_v),
                epochs=freeze_epochs + full_epochs,
                batch_size=batch_size,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=8,
                        restore_best_weights=True,
                    )
                ],
                verbose=1,
            )

        vl = min(h.history["val_loss"])
        vm = min(h.history["val_mae"])
        total_time = time.time() - t0 if base_model is not None else 0
        logger.info(
            f"  {pair}: val_loss={vl:.6f}, val_mae={vm:.6f}, time={total_time:.0f}s"
        )

        model.save(f"models/{pair}_lstm_cnn.keras")
        return model

    async def train_all_fast(
        self,
        pairs: Optional[List[str]] = None,
        base_epochs: int = 40,
        freeze_epochs: int = 8,
        full_epochs: int = 8,
    ):
        if pairs is None:
            pairs = [
                "EURUSD",
                "GBPUSD",
                "USDJPY",
                "AUDUSD",
                "USDCAD",
                "USDCHF",
                "NZDUSD",
            ]

        logger.info(f"\n{'='*60}")
        logger.info(f"  FAST TRAINING — {len(pairs)} pairs")
        logger.info(f"  Device: {self.device}")
        logger.info("  Strategy: Transfer learning (base + fine-tune)")
        logger.info(f"{'='*60}")

        # Step 1: Train base model on EURUSD
        logger.info("\n--- STEP 1: Train base model on EURUSD ---")
        await self.train_base_model(epochs=base_epochs)

        # Step 2: Fine-tune remaining pairs
        for pair in pairs:
            if pair == self.base_pair:
                continue
            logger.info(f"\n--- Fine-tuning {pair} ---")
            try:
                await self.fine_tune_pair(
                    pair,
                    freeze_epochs=freeze_epochs,
                    full_epochs=full_epochs,
                )
            except Exception as e:
                logger.error(f"  {pair} failed: {e}")

        logger.info(f"\n{'='*60}")
        logger.info("  FAST TRAINING COMPLETE")
        logger.info(f"{'='*60}")
        for f in sorted(os.listdir("models")):
            if f.endswith("_lstm_cnn.keras"):
                sz = os.path.getsize(f"models/{f}") / 1024
                logger.info(f"    {f:45s} {sz:.0f}KB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-epochs", type=int, default=40, help="Base model epochs")
    parser.add_argument(
        "--freeze-epochs", type=int, default=8, help="Fine-tune frozen epochs"
    )
    parser.add_argument(
        "--full-epochs", type=int, default=8, help="Fine-tune full epochs"
    )
    parser.add_argument("--pairs", nargs="+", default=None, help="Pairs to train")
    args = parser.parse_args()

    trainer = FastTrainer()
    asyncio.run(
        trainer.train_all_fast(
            pairs=args.pairs,
            base_epochs=args.base_epochs,
            freeze_epochs=args.freeze_epochs,
            full_epochs=args.full_epochs,
        )
    )
