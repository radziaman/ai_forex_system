"""
Global Multi-Asset Model — single model with symbol embeddings.
Shares learning across all 11 FX pairs via learned symbol representations.

Architecture:
  - Embedding layer: maps symbol ID (0-10) to 16-dim vector
  - Shared LSTM(256): processes concatenated features + embedding
  - Per-symbol heads: 11 separate linear output layers
  - Training: batches contain (features, symbol_ids, targets)

Benefits over per-symbol models:
  - Transfer learning: EUR/USD patterns help predict GBP/USD
  - Fewer total parameters (~300K vs ~1.65M)
  - Cross-asset learning: model learns which features generalize
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from tensorflow.keras import Model

try:
    import tensorflow as tf
    from tensorflow.keras import Input
    from tensorflow.keras.layers import (
        LSTM,
        Dense,
        Dropout,
        Embedding,
        Concatenate,
        Reshape,
        RepeatVector,
        BatchNormalization,
    )

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


SYMBOL_NAMES = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "USDCHF",
    "NZDUSD",
    "XAUUSD",
    "BTCUSD",
    "XTIUSD",
    "US500",
]


class GlobalMultiAssetModel:
    """Single multi-symbol model with learned symbol embeddings.

    Usage:
        model = GlobalMultiAssetModel(n_symbols=11, lookback=30, n_features=49)
        model.build()
        # Training: provide symbol_ids alongside features
        model.train(X, y, symbol_ids, X_val, y_val, symbol_ids_val)
        # Inference: predict for specific symbol
        pred = model.predict(X, symbol_id=0)  # EURUSD
    """

    def __init__(
        self,
        n_symbols: int = 11,
        lookback: int = 30,
        n_features: int = 49,
        embedding_dim: int = 16,
        lstm_units: int = 256,
    ):
        self.n_symbols = n_symbols
        self.lookback = lookback
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.model: Optional[Model] = None

    def build(self):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow required for GlobalMultiAssetModel")

        # Inputs
        features_input = Input(shape=(self.lookback, self.n_features), name="features")
        symbol_input = Input(shape=(1,), dtype=tf.int32, name="symbol_id")

        # Embedding: (batch, 1) -> (batch, 1, embedding_dim)
        embedding = Embedding(
            input_dim=self.n_symbols,
            output_dim=self.embedding_dim,
            name="symbol_embedding",
        )(
            symbol_input
        )  # (batch, 1, embedding_dim)

        # Tile embedding to match lookback dimension
        # Flatten (batch, 1, embedding_dim) -> (batch, embedding_dim), then repeat
        embedding_flat = Reshape((self.embedding_dim,), name="embedding_flat")(
            embedding
        )
        embedding_tiled = RepeatVector(self.lookback, name="repeat_embedding")(
            embedding_flat
        )
        # (batch, lookback, embedding_dim)

        # Concatenate features with tiled embedding
        x = Concatenate(axis=-1, name="feature_embedding_fusion")(
            [features_input, embedding_tiled]
        )  # (batch, lookback, n_features + embedding_dim)

        # Shared LSTM
        x = LSTM(
            self.lstm_units, dropout=0.2, return_sequences=False, name="shared_lstm"
        )(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation="relu", name="shared_dense")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        # Per-symbol output heads
        symbol_outputs = []
        for i in range(self.n_symbols):
            head = Dense(1, name=f"output_{SYMBOL_NAMES[i]}")(x)
            symbol_outputs.append(head)

        # Concatenate all symbol outputs
        if len(symbol_outputs) > 1:
            output = Concatenate(axis=-1, name="symbol_outputs")(symbol_outputs)
        else:
            output = symbol_outputs[0]

        self.model = tf.keras.Model(
            inputs=[features_input, symbol_input],
            outputs=output,
            name="global_multi_asset",
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"],
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        symbol_ids: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        symbol_ids_val: np.ndarray,
        epochs: int = 30,
        batch_size: int = 32,
        callbacks: Optional[list] = None,
    ) -> tf.keras.callbacks.History:
        """Train the multi-asset model.

        Args:
            X: feature tensor (n_samples, lookback, n_features)
            y: target tensor (n_samples, n_symbols) — each column is one symbol's target
            symbol_ids: symbol indices (n_samples, 1)
            X_val, y_val, symbol_ids_val: validation data
        """
        if self.model is None:
            self.build()

        default_callbacks = callbacks or [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
        ]

        history = self.model.fit(
            [X, symbol_ids],
            y,
            validation_data=([X_val, symbol_ids_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=default_callbacks,
            verbose=1,
        )
        return history

    def predict(self, X: np.ndarray, symbol_id: int = 0) -> np.ndarray:
        """Predict for a specific symbol.

        Args:
            X: feature tensor (batch, lookback, n_features)
            symbol_id: integer ID for the symbol (0 = EURUSD, etc.)

        Returns:
            predictions: (batch,) array of price predictions
        """
        if self.model is None:
            raise ValueError("Model not built")

        batch_size = X.shape[0]
        symbol_ids = np.full((batch_size, 1), symbol_id, dtype=np.int32)
        all_preds = self.model.predict([X, symbol_ids], verbose=0)

        if all_preds.ndim == 2:
            return all_preds[:, symbol_id]
        return all_preds.flatten()

    def save(self, path: str):
        if self.model is not None:
            self.model.save(path)

    def load(self, path: str):
        self.model = tf.keras.models.load_model(path)

    @staticmethod
    def symbol_to_id(symbol: str) -> int:
        """Convert symbol name to integer ID."""
        sym = symbol.upper()
        if sym in SYMBOL_NAMES:
            return SYMBOL_NAMES.index(sym)
        raise ValueError(f"Unknown symbol: {symbol}")

    def get_embeddings(self) -> Dict[str, np.ndarray]:
        """Extract learned symbol embeddings.

        Returns:
            dict mapping symbol name -> embedding vector
        """
        if self.model is None:
            return {}
        embedding_layer = self.model.get_layer("symbol_embedding")
        weights = embedding_layer.get_weights()[0]  # (n_symbols, embedding_dim)
        return {name: weights[i] for i, name in enumerate(SYMBOL_NAMES[: len(weights)])}
