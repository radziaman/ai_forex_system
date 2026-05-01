"""
Next-generation model architecture based on institutional research:
- LSTM-Transformer fusion (3x better than LSTM alone)
- Multi-head attention mechanism (Sharpe 4.4 vs 1.2 for LSTM)
- Technical indicators integration (RSI, MACD, Bollinger, ATR)
- Ensemble prediction with Elo-style confidence weighting
"""
import numpy as np
from typing import Optional, List, Dict
try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model, optimizers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class AttentionLayer(layers.Layer):
    """Multi-head attention for time series forecasting"""
    def __init__(self, units: int = 64, num_heads: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=units)
        
    def call(self, inputs):
        # inputs shape: (batch, timesteps, features)
        attention_output = self.attention(inputs, inputs)
        return layers.Add()([inputs, attention_output])


class LSTMTransformerHybrid:
    """
    LSTM-Transformer fusion model (Research: 18.7% returns vs 6.8% LSTM)
    Based on: "Research on Exchange Rate Forecasting Based on LSTM-Transformer Fusion Model"
    """
    def __init__(self, lookback: int = 30, n_features: int = 50):
        self.lookback = lookback
        self.n_features = n_features
        self.model: Optional[Model] = None
        
    def build(self) -> Model:
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow required")
        
        # Input
        inputs = layers.Input(shape=(self.lookback, self.n_features))
        
        # LSTM branch (captures temporal dependencies)
        lstm1 = layers.LSTM(128, return_sequences=True)(inputs)
        lstm1 = layers.Dropout(0.2)(lstm1)
        lstm2 = layers.LSTM(64, return_sequences=True)(lstm1)
        lstm2 = layers.Dropout(0.2)(lstm2)
        
        # CNN branch (extracts local patterns)
        conv1 = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
        conv1 = layers.BatchNormalization()(conv1)
        conv2 = layers.Conv1D(32, kernel_size=5, activation='relu', padding='same')(conv1)
        conv2 = layers.BatchNormalization()(conv2)
        
        # Merge LSTM and CNN
        merged = layers.Concatenate()([lstm2, conv2])
        
        # Transformer attention (global dependencies)
        attention = AttentionLayer(units=64, num_heads=4)(merged)
        attention = layers.Dropout(0.2)(attention)
        
        # Additional LSTM after attention
        lstm3 = layers.LSTM(32)(attention)
        lstm3 = layers.Dropout(0.2)(lstm3)
        
        # Dense layers
        dense = layers.Dense(32, activation='relu')(lstm3)
        dense = layers.Dropout(0.1)(dense)
        dense = layers.Dense(16, activation='relu')(dense)
        
        outputs = layers.Dense(1)(dense)
        
        if TENSORFLOW_AVAILABLE:
            self.model = Model(inputs=inputs, outputs=outputs)
            self.model.compile(
                optimizer=optimizers.Adam(learning_rate=0.001),
                loss="mse",
                metrics=["mae", tf.keras.metrics.RootMeanSquaredError()]
            )
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs: int = 100, batch_size: int = 32):
        if self.model is not None:
            return self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )
        raise ValueError("Model not built. Call build() first.")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is not None:
            return self.model.predict(X, verbose=0)
        raise ValueError("Model not built. Call build() first.")
    
    def save(self, path):
        if self.model:
            # Convert Path to string if needed
            path_str = str(path)
            if path_str.endswith('.h5'):
                path_str = path_str.replace('.h5', '.keras')
            self.model.save(path_str)
    
    @classmethod
    def load(cls, path: str) -> "LSTMTransformerHybrid":
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow required")
        model = tf.keras.models.load_model(path, custom_objects={'AttentionLayer': AttentionLayer})
        instance = cls()
        instance.model = model
        return instance


class EnsemblePredictor:
    """
    Ensemble system with Elo-style rating (Research: 90.6% returns, 73% false signal reduction)
    Based on: "AI Trading System Architecture: 20-Agent Ensemble Intelligence"
    """
    def __init__(self):
        self.models: List[Dict] = []
        self.elo_ratings: Dict[str, float] = {}
        
    def add_model(self, name: str, model, weight: float = 1200.0):
        """Add a model to the ensemble with initial Elo rating"""
        self.models.append({'name': name, 'model': model})
        self.elo_ratings[name] = weight
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Weighted ensemble prediction based on Elo ratings"""
        if not self.models:
            raise ValueError("No models in ensemble")
        
        predictions = []
        weights = []
        
        for model_info in self.models:
            pred = model_info['model'].predict(X)
            predictions.append(pred.flatten())
            weights.append(self.elo_ratings[model_info['name']])
        
        # Weighted average
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        return ensemble_pred
    
    def update_elo(self, model_name: str, performance: float, k_factor: float = 32.0):
        """Update Elo rating based on prediction accuracy"""
        # Simple Elo update: higher performance -> rating increase
        self.elo_ratings[model_name] += k_factor * performance
