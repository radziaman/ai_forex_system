"""
Adversarial Training for Financial Models.
Implements PGD (Projected Gradient Descent) and FGSM (Fast Gradient Sign Method)
adversarial perturbations for robust training against market noise and manipulation.

Adversarial training forces the model to learn features that are invariant to
small perturbations — making it resistant to stop hunts, liquidity grabs,
and outlier events that plague financial time series.
"""

import numpy as np
from typing import Optional, Callable, Tuple
from dataclasses import dataclass


@dataclass
class AdversarialConfig:
    epsilon: float = 0.01  # Max perturbation magnitude (fraction of feature range)
    alpha: float = 0.002  # Step size for PGD
    steps: int = 7  # Number of PGD iterations
    random_start: bool = True  # Random initialization within epsilon ball
    epsilon_clip: bool = True  # Clip perturbations to feature bounds


class PGDAdversarial:
    """Projected Gradient Descent adversarial attack.

    Generates adversarial examples by taking multiple small steps in the
    direction that maximizes loss, projected back to within epsilon of
    the original input.

    This is TensorFlow-optional: works with any model that has a
    callable (model, x) -> predictions interface.

    Usage:
        attacker = PGDAdversarial()
        X_adv = attacker.generate(model, X, y)
        # Train on both clean and adversarial examples
        model.train(X, y)
        model.train(X_adv, y)
    """

    def __init__(self, config: Optional[AdversarialConfig] = None):
        self.config = config or AdversarialConfig()

    def generate(
        self,
        model_fn: Callable,
        X: np.ndarray,
        y: np.ndarray,
        loss_fn: Optional[Callable] = None,
    ) -> np.ndarray:
        """Generate PGD adversarial examples.

        Args:
            model_fn: callable model(X) -> predictions
            X: clean input batch (batch, lookback, n_features)
            y: target values
            loss_fn: callable loss(y_true, y_pred) -> scalar

        Returns:
            X_adv: adversarial examples with same shape as X
        """
        if loss_fn is None:

            def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
                return float(np.mean((y_true - y_pred) ** 2))

            loss_fn = _mse

        X_adv = X.copy().astype(np.float64)

        # Random initialization within epsilon ball
        if self.config.random_start:
            noise = np.random.uniform(
                -self.config.epsilon, self.config.epsilon, size=X.shape
            ).astype(np.float64)
            X_adv = X_adv + noise
            if self.config.epsilon_clip:
                X_adv = np.clip(X_adv, X - self.config.epsilon, X + self.config.epsilon)

        # PGD iterations
        for step in range(self.config.steps):
            X_adv_tensor = X_adv.astype(np.float32)
            y_tensor = y.astype(np.float32)

            # Numerical gradient approximation
            grad = self._compute_numerical_gradient(
                model_fn, loss_fn, X_adv_tensor, y_tensor
            )

            if grad is None or np.max(np.abs(grad)) < 1e-12:
                break

            # Gradient sign step
            X_adv = X_adv.astype(np.float64) + self.config.alpha * np.sign(grad)

            # Project back to epsilon ball
            if self.config.epsilon_clip:
                perturbation = X_adv - X
                perturbation = np.clip(
                    perturbation, -self.config.epsilon, self.config.epsilon
                )
                X_adv = X + perturbation

        return X_adv.astype(np.float32)

    def _compute_numerical_gradient(
        self,
        model_fn: Callable,
        loss_fn: Callable,
        X: np.ndarray,
        y: np.ndarray,
        h: float = 1e-4,
    ) -> Optional[np.ndarray]:
        """Compute gradient of loss w.r.t. input using central difference.

        This is framework-agnostic — works with TensorFlow, PyTorch, or
        any model that acts as a callable.
        """
        grad = np.zeros_like(X, dtype=np.float64)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    X_plus = X.copy()
                    X_minus = X.copy()
                    X_plus[i, j, k] += h
                    X_minus[i, j, k] -= h

                    try:
                        loss_plus = loss_fn(y[i : i + 1], model_fn(X_plus[i : i + 1]))
                        loss_minus = loss_fn(y[i : i + 1], model_fn(X_minus[i : i + 1]))
                        grad[i, j, k] = (loss_plus - loss_minus) / (2 * h)
                    except Exception:
                        grad[i, j, k] = 0.0

        return grad

    def generate_fgsm(
        self,
        model_fn: Callable,
        X: np.ndarray,
        y: np.ndarray,
        loss_fn: Optional[Callable] = None,
    ) -> np.ndarray:
        """Fast Gradient Sign Method — single-step attack.

        Faster than PGD but less effective. Good for quick adversarial
        augmentation during training.
        """
        config = AdversarialConfig(
            epsilon=self.config.epsilon,
            alpha=self.config.epsilon,  # Single step
            steps=1,
            random_start=False,
            epsilon_clip=True,
        )
        single_step = PGDAdversarial(config)
        return single_step.generate(model_fn, X, y, loss_fn)


class AdversarialTrainer:
    """Wraps a model training loop with adversarial augmentation.

    At each training step, generates adversarial examples and adds them
    to the training batch. This forces the model to learn robust features.

    Usage:
        trainer = AdversarialTrainer(adversary, ratio=0.5)
        X_aug, y_aug = trainer.augment_batch(model_fn, X, y)
        # Train on X_aug, y_aug
    """

    def __init__(
        self,
        adversary: Optional[PGDAdversarial] = None,
        adversarial_ratio: float = 0.5,
    ):
        self.adversary = adversary or PGDAdversarial()
        self.adversarial_ratio = np.clip(adversarial_ratio, 0.0, 1.0)

    def augment_batch(
        self,
        model_fn: Callable,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate adversarial examples and augment the training batch.

        Returns:
            (X_augmented, y_augmented) — concatenation of clean and adversarial data
        """
        if self.adversarial_ratio <= 0.0:
            return X, y

        n_adv = max(1, int(len(X) * self.adversarial_ratio))
        indices = np.random.choice(len(X), n_adv, replace=False)

        X_adv = self.adversary.generate(model_fn, X[indices], y[indices])
        y_adv = y[indices].copy()

        # Concatenate
        X_aug = np.concatenate([X, X_adv], axis=0)
        y_aug = np.concatenate([y, y_adv], axis=0)

        return X_aug, y_aug
