import numpy as np
from rts_ai_fx.adversarial import (
    PGDAdversarial,
    AdversarialConfig,
    AdversarialTrainer,
)


def _dummy_model(X: np.ndarray) -> np.ndarray:
    """Simple linear model for testing."""
    return np.mean(X, axis=(1, 2), keepdims=True)


def _dummy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


class TestPGDAdversarial:
    def test_generate_returns_same_shape(self):
        X = np.random.randn(4, 10, 5).astype(np.float32)
        y = np.random.randn(4, 1).astype(np.float32)
        attacker = PGDAdversarial()
        X_adv = attacker.generate(_dummy_model, X, y)
        assert X_adv.shape == X.shape

    def test_generate_differs_from_input(self):
        X = np.random.randn(4, 10, 5).astype(np.float32)
        y = np.random.randn(4, 1).astype(np.float32)
        attacker = PGDAdversarial(AdversarialConfig(epsilon=0.1, steps=10))
        X_adv = attacker.generate(_dummy_model, X, y)
        assert not np.allclose(X, X_adv, atol=1e-6)

    def test_generate_stays_within_epsilon(self):
        X = np.random.randn(4, 10, 5).astype(np.float32)
        y = np.random.randn(4, 1).astype(np.float32)
        eps = 0.05
        attacker = PGDAdversarial(AdversarialConfig(epsilon=eps, steps=5))
        X_adv = attacker.generate(_dummy_model, X, y)
        perturbation = X_adv - X
        assert np.all(np.abs(perturbation) <= eps + 1e-6)

    def test_fgsm_returns_same_shape(self):
        X = np.random.randn(3, 10, 5).astype(np.float32)
        y = np.random.randn(3, 1).astype(np.float32)
        attacker = PGDAdversarial()
        X_adv = attacker.generate_fgsm(_dummy_model, X, y)
        assert X_adv.shape == X.shape

    def test_adversarial_trainer_augments_batch(self):
        X = np.random.randn(10, 10, 5).astype(np.float32)
        y = np.random.randn(10, 1).astype(np.float32)
        trainer = AdversarialTrainer(adversarial_ratio=0.5)
        X_aug, y_aug = trainer.augment_batch(_dummy_model, X, y)
        assert len(X_aug) == len(X) + len(X) // 2  # 10 + 5 = 15
        assert len(y_aug) == len(y) + len(y) // 2

    def test_adversarial_trainer_zero_ratio_no_change(self):
        X = np.random.randn(10, 10, 5).astype(np.float32)
        y = np.random.randn(10, 1).astype(np.float32)
        trainer = AdversarialTrainer(adversarial_ratio=0.0)
        X_aug, y_aug = trainer.augment_batch(_dummy_model, X, y)
        assert len(X_aug) == len(X)
        np.testing.assert_array_equal(X, X_aug)

    def test_config_defaults(self):
        config = AdversarialConfig()
        assert config.epsilon == 0.01
        assert config.alpha == 0.002
        assert config.steps == 7
        assert config.random_start is True
