"""
Uncertainty Quantification via Monte Carlo Dropout.
Runs forward pass N times with dropout enabled, returns mean + variance.
"""
import numpy as np
from typing import Optional, Tuple


def monte_carlo_dropout(model, X: np.ndarray, n_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    MC Dropout inference: runs forward pass n_samples times.
    Returns (mean_prediction, prediction_variance).
    High variance = low confidence → filter or reduce position size.
    """
    predictions = []
    for _ in range(n_samples):
        pred = model(X, training=True) if hasattr(model, "__call__") else model.predict(X, verbose=0)
        predictions.append(pred)
    preds = np.array(predictions)
    mean = np.mean(preds, axis=0)
    variance = np.var(preds, axis=0)
    return mean, variance


def get_confidence(mean: np.ndarray, variance: np.ndarray, direction_threshold: float = 0.0005) -> float:
    """
    Compute confidence score from MC Dropout output.
    Confidence = 1 - normalized_variance, with direction check.
    Returns value in [0, 1].
    """
    # Normalize variance relative to mean magnitude
    rel_std = np.sqrt(variance) / (np.abs(mean) + 1e-8)
    # Direction certainty: how consistently the prediction is above/below threshold
    if len(mean) > 0:
        consistency = np.mean(np.abs(mean) > direction_threshold)
    else:
        consistency = 0.0
    # Combine: low uncertainty + high direction consistency
    uncertainty_score = np.clip(1.0 - np.mean(rel_std), 0, 1)
    return float(uncertainty_score * 0.7 + consistency * 0.3)
