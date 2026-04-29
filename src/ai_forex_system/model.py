"""LSTM-CNN Hybrid Neural Network Architecture"""

import numpy as np
from typing import Optional

try:
    import tensorflow as tf
    from tensorflow.keras import Model, Input
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
    """Hybrid LSTM-CNN model for forex price prediction (Aurum AI architecture)"""

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
        self.model: Optional["Model"] = None

    def build(self) -> "Model":
        """Build LSTM-CNN hybrid architecture"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow required. Install: pip install tensorflow")

        inputs = Input(shape=(self.lookback, self.n_features), name="input")

        lstm_branch = LSTM(
            units=self.lstm_units,
            dropout=0.2,
            return_sequences=False,
            name="lstm_layer",
        )(inputs)
        lstm_branch = Dropout(0.2)(lstm_branch)

        cnn_branch = Conv1D(
            filters=self.cnn_filters,
            kernel_size=3,
            activation="relu",
            padding="same",
            name="cnn_layer",
        )(inputs)
        cnn_branch = GlobalMaxPooling1D()(cnn_branch)
        cnn_branch = Dropout(0.2)(cnn_branch)

        fused = Concatenate(name="fusion_layer")([lstm_branch, cnn_branch])
        fused = Dense(64, activation="relu")(fused)
        fused = BatchNormalization()(fused)
        fused = Dropout(0.2)(fused)

        outputs = Dense(1, activation="linear", name="output")(fused)

        if TENSORFLOW_AVAILABLE:
            self.model = Model(inputs=inputs, outputs=outputs, name="lstm_cnn_hybrid")
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss="mse",
                metrics=["mae", tf.keras.metrics.RootMeanSquaredError()],
            )
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is not None:
            return self.model.predict(X, verbose=0)
        raise ValueError("Model not built. Call build() first.")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
    ) -> "tf.keras.callbacks.History":
        if self.model is not None:
            return self.model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
            )
        raise ValueError("Model not built. Call build() first.")

    def save(self, path: str):
        if self.model:
            self.model.save(path)

    @classmethod
    def load(cls, path: str) -> "LSTMCNNHybrid":
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow required")
        model = tf.keras.models.load_model(path)
        instance = cls()
        instance.model = model
        return instance


class ProfitabilityClassifier:
    """Secondary network for trade confidence scoring"""

    def __init__(self, lookback: int = 30, n_features: int = 51):
        self.lookback = lookback
        self.n_features = n_features
        self.model: Optional["Model"] = None

    def build(self) -> "Model":
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow required")

        inputs = Input(shape=(self.lookback, self.n_features))
        x = LSTM(64, return_sequences=True)(inputs)
        x = Dropout(0.2)(x)
        x = LSTM(32)(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation="relu")(x)
        outputs = Dense(1, activation="sigmoid")(x)

        if TENSORFLOW_AVAILABLE:
            self.model = Model(inputs=inputs, outputs=outputs)
            self.model.compile(
                optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
            )
        return self.model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is not None:
            return self.model.predict(X, verbose=0)
        raise ValueError("Model not built. Call build() first.")
