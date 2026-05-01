"""
Online Concept Drift Detection using ADWIN (Adaptive Windowing).
Monitors prediction error and triggers retraining when distribution shifts.
"""
import numpy as np
from typing import Optional


class ADWIN:
    """
    ADWIN (Adaptive Windowing) drift detector.
    Maintains two windows: if their means differ significantly, drift is signaled.
    """

    def __init__(self, delta: float = 0.05, min_window: int = 30):
        self.delta = delta
        self.min_window = min_window
        self.window: list = []
        self.drift_count = 0
        self._total = 0.0

    def update(self, value: float) -> bool:
        """Add a new error value. Returns True if drift detected."""
        self.window.append(value)
        self._total += value
        if len(self.window) < self.min_window:
            return False
        # Check for drift by splitting window at every possible point
        n = len(self.window)
        for i in range(self.min_window, n - self.min_window + 1):
            left = self.window[:i]
            right = self.window[i:]
            mu_left = np.mean(left)
            mu_right = np.mean(right)
            diff = abs(mu_left - mu_right)
            # Threshold based on Hoeffding bound
            m = 1 / (1 / len(left) + 1 / len(right))
            epsilon = np.sqrt(1 / (2 * m) * np.log(4 * n / self.delta))
            if diff > epsilon:
                self.drift_count += 1
                # Reset window to the right half
                self.window = right
                self._total = sum(right)
                return True
        return False

    @property
    def mean(self) -> float:
        return self._total / len(self.window) if self.window else 0.0


class DriftMonitor:
    """
    Monitors multiple drift detectors for model retraining signals.
    """

    def __init__(self, error_threshold: float = 0.02):
        self.error_detector = ADWIN()
        self.feature_detector = ADWIN(delta=0.01, min_window=50)
        self.error_threshold = error_threshold
        self.retrain_triggered = False

    def update(self, prediction: float, actual: float) -> bool:
        """Update monitors. Returns True if retraining is recommended."""
        error = abs(prediction - actual)
        error_drift = self.error_detector.update(error)
        # Check if sustained error exceeds threshold
        sustained_bad = self.error_detector.mean > self.error_threshold
        self.retrain_triggered = error_drift or sustained_bad
        return self.retrain_triggered

    def reset(self):
        self.error_detector = ADWIN()
        self.feature_detector = ADWIN(delta=0.01, min_window=50)
        self.retrain_triggered = False
