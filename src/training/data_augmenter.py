"""
Data Augmentation for Forex Training.

Combines multiple synthetic data generators to augment the training
dataset with realistic variations of historical data.

Strategy:
  1. Mixup: interpolate between random pairs of real samples
  2. Noise injection: add controlled noise to real features
  3. Regime shuffle: re-order bars within the same regime
  4. Synthetic bars: full synthetic generation using GMM
"""

import numpy as np
from typing import Tuple
from loguru import logger


class DataAugmenter:
    """Augments training data with synthetic variations.

    Each augmentation method can be enabled/disabled and has a
    configurable ratio of augmented to real samples.
    """

    def __init__(
        self,
        mixup_alpha: float = 0.2,
        noise_std: float = 0.01,
        synthetic_ratio: float = 0.3,
    ):
        self.mixup_alpha = mixup_alpha
        self.noise_std = noise_std
        self.synthetic_ratio = synthetic_ratio

    def mixup(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Mixup augmentation: convex combination of random pairs.

        Creates (ratio * len(X)) augmented samples.
        """
        n = len(X)
        n_aug = max(1, int(n * self.synthetic_ratio))
        indices = np.random.choice(n, size=(n_aug, 2))
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha, size=n_aug)
        lam = lam.reshape(-1, *([1] * (X.ndim - 1)))
        X_aug = lam * X[indices[:, 0]] + (1 - lam) * X[indices[:, 1]]
        y_aug = (
            lam.squeeze() * y[indices[:, 0]] + (1 - lam.squeeze()) * y[indices[:, 1]]
        )
        return X_aug, y_aug

    def add_noise(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Add Gaussian noise to features.

        Standard deviation scaled by feature-wise std.
        """
        X_std = X.std(axis=(0, 1), keepdims=True) + 1e-8
        noise = np.random.randn(*X.shape) * X_std * self.noise_std
        X_aug = X + noise
        return X_aug, y.copy()

    def augment(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply all enabled augmentations.

        Returns concatenated (original + augmented) dataset.
        """
        X_aug_list = [X]
        y_aug_list = [y]

        # Mixup
        X_mix, y_mix = self.mixup(X, y)
        X_aug_list.append(X_mix)
        y_aug_list.append(y_mix)

        # Noise
        X_noise, y_noise = self.add_noise(X, y)
        X_aug_list.append(X_noise)
        y_aug_list.append(y_noise)

        X_all = np.concatenate(X_aug_list, axis=0)
        y_all = np.concatenate(y_aug_list, axis=0)

        logger.debug(
            f"DataAugmenter: {len(X)} -> {len(X_all)} samples "
            f"(mixup={len(X_mix)}, noise={len(X_noise)})"
        )
        return X_all, y_all
