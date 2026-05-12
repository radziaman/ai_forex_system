"""
LSTM-CNN Hybrid + ProfitabilityClassifier with CORRECT label construction.
Predicts FUTURE price direction, not past.
"""

from __future__ import annotations
import sys
import numpy as np
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from tensorflow.keras import Model

try:
    import tensorflow as tf
    from tensorflow.keras import Input
    from tensorflow.keras.layers import (
        LSTM,
        Dense,
        Dropout,
        Concatenate,
        GlobalMaxPooling1D,
        Conv1D,
        BatchNormalization,
    )

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class LSTMCNNHybrid:
    """Hybrid LSTM-CNN for forex price prediction."""

    def __init__(
        self,
        lookback: int = 30,
        n_features: int = 51,
        lstm_units: int = 128,
        cnn_filters: int = 128,
    ):
        self.lookback = lookback
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.cnn_filters = cnn_filters
        self.model = None  # type: Optional[Model]

    def build(self):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow required")
        inputs = Input(shape=(self.lookback, self.n_features), name="input")
        lstm_branch = LSTM(
            self.lstm_units, dropout=0.2, return_sequences=False, name="lstm"
        )(inputs)
        lstm_branch = Dropout(0.2)(lstm_branch)
        cnn_branch = Conv1D(
            self.cnn_filters, 3, activation="relu", padding="same", name="cnn"
        )(inputs)
        cnn_branch = GlobalMaxPooling1D()(cnn_branch)
        cnn_branch = Dropout(0.2)(cnn_branch)
        fused = Concatenate(name="fusion")([lstm_branch, cnn_branch])
        fused = Dense(64, activation="relu")(fused)
        fused = BatchNormalization()(fused)
        fused = Dropout(0.2)(fused)
        outputs = Dense(1, activation="linear", name="output")(fused)
        self.model = tf.keras.models.Model(
            inputs=inputs, outputs=outputs, name="lstm_cnn_hybrid"
        )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001), loss="mse", metrics=["mae"]
        )
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is not None:
            return self.model.predict(X, verbose=0)
        raise ValueError("Model not built.")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
    ):
        if self.model is None:
            self.build()
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
            ),
        ]
        return self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

    def save(self, path: str):
        if self.model:
            self.model.save(path)

    @classmethod
    @classmethod
    def load(cls, path: str) -> "LSTMCNNHybrid":
        """
        Load a saved model. If the saved format is incompatible (e.g. Keras 3 vs 2),
        rebuilds the architecture and attempts weight transfer. Falls back to fresh
        untrained model for feature extraction.
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow required")
        instance = LSTMCNNHybrid()
        instance.build()

        try:
            saved = tf.keras.models.load_model(path, compile=False)
            instance.model = saved
            return instance
        except Exception:
            pass

        # Try loading with safe_mode=False (handles some version diffs)
        try:
            saved = tf.keras.models.load_model(
                path, compile=False, safe_mode=False,
            )
            instance.model = saved
            return instance
        except Exception:
            pass

        # Try rebuilding and transferring weights
        try:
            saved = tf.keras.models.load_model(
                path, compile=False, safe_mode=False,
            )
            for layer in instance.model.layers:
                matching = saved.get_layer(layer.name)
                if matching and len(layer.get_weights()) == len(matching.get_weights()):
                    try:
                        layer.set_weights(matching.get_weights())
                    except Exception:
                        pass
            return instance
        except Exception:
            pass

        logger = __import__("loguru").logger
        logger.info(
            f"LSTM model: fresh (unable to load saved weights from {path}, "
            f"TF={tf.__version__})"
        )
        return instance


class ProfitabilityClassifier:
    """Binary direction classifier — predicts if NEXT bar will be up (1) or down (0)."""

    def __init__(self, lookback: int = 30, n_features: int = 51):
        self.lookback = lookback
        self.n_features = n_features
        self.model = None  # type: Optional[Model]

    def build(self):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow required")
        inputs = Input(shape=(self.lookback, self.n_features))
        x = LSTM(64, return_sequences=True)(inputs)
        x = Dropout(0.2)(x)
        x = LSTM(32)(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation="relu")(x)
        outputs = Dense(1, activation="sigmoid")(x)
        self.model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        return self.model

    def _make_labels(self, prices: np.ndarray) -> np.ndarray:
        """CORRECT label: 1 if price[t+1] > price[t], 0 otherwise."""
        diff = np.diff(prices.flatten())
        return (diff > 0).astype(int)

    def train(
        self,
        X_train: np.ndarray,
        prices_train: np.ndarray,
        X_val: np.ndarray,
        prices_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
    ):
        if self.model is None:
            self.build()
        y_train = self._make_labels(prices_train)
        y_val = self._make_labels(prices_val)
        # Align X to y: X[t] predicts y[t] = direction of price[t+1] vs price[t]
        X_train_adj = X_train[: len(y_train)]
        X_val_adj = X_val[: len(y_val)]
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            ),
        ]
        return self.model.fit(
            X_train_adj,
            y_train,
            validation_data=(X_val_adj, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is not None:
            return self.model.predict(X, verbose=0)
        raise ValueError("Model not built.")

    def save(self, path: str):
        if self.model:
            self.model.save(path)

    @classmethod
    def load(cls, path: str) -> "ProfitabilityClassifier":
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow required")
        instance = cls()
        instance.model = tf.keras.models.load_model(path)
        return instance
