"""Tests for synthetic data generation."""

import numpy as np
import pytest
from training.synthetic_data import (
    GMMSampler,
    CrisisAugmenter,
    VolatilityClusterSampler,
)
from training.data_augmenter import DataAugmenter


class TestGMMSampler:
    def test_fit_and_sample(self):
        returns = np.random.randn(500) * 0.01
        sampler = GMMSampler(n_components=2)
        sampler.fit(returns)
        prices = sampler.sample(100, start_price=1.0)
        assert len(prices) == 100
        assert prices[0] > 0

    def test_sample_requires_fit(self):
        sampler = GMMSampler()
        with pytest.raises(ValueError, match="not fitted"):
            sampler.sample(10)

    def test_regime_conditional(self):
        returns = np.random.randn(500, 2) * 0.01
        sampler = GMMSampler(n_components=3)
        sampler.fit(returns)
        prices = sampler.sample_regime_conditional(
            50, regime_component=0, start_price=1.0
        )
        assert len(prices) == 50


class TestCrisisAugmenter:
    def test_generate_flash_crash(self):
        aug = CrisisAugmenter(random_state=42)
        prices = aug.generate_crisis(100, "flash_crash", 0.10, 1.0)
        assert len(prices) == 100
        assert prices[-1] > 0

    def test_generate_all_types(self):
        aug = CrisisAugmenter(random_state=42)
        scenarios = aug.generate_all_crisis_types(100, 0.10, 1.0)
        assert len(scenarios) == 4
        for name, prices in scenarios.items():
            assert len(prices) == 100
            assert name in CrisisAugmenter.CRISIS_TYPES

    def test_crash_depth_approx(self):
        aug = CrisisAugmenter(random_state=42)
        prices = aug.generate_crisis(50, "flash_crash", 0.20, 100.0)
        max_dd = (prices.max() - prices.min()) / prices.max()
        assert max_dd > 0.05  # Should have significant drawdown

    def test_crisis_types_enum(self):
        assert len(CrisisAugmenter.CRISIS_TYPES) == 4


class TestVolatilityClusterSampler:
    def test_sample(self):
        sampler = VolatilityClusterSampler(p_high=0.1, p_low=0.2)
        prices = sampler.sample(200, low_vol=0.002, high_vol=0.02, start_price=1.0)
        assert len(prices) == 200
        assert prices[-1] > 0
        # Check that volatility varies (not constant)
        returns = np.diff(np.log(prices))
        assert np.std(returns) > 0


class TestDataAugmenter:
    def test_mixup_shape(self):
        aug = DataAugmenter(mixup_alpha=0.2, synthetic_ratio=0.5)
        X = np.random.randn(100, 30, 49)
        y = np.random.randn(100)
        X_aug, y_aug = aug.mixup(X, y)
        assert len(X_aug) == 50  # 50% of 100
        assert X_aug.shape[1:] == (30, 49)

    def test_add_noise_shape(self):
        aug = DataAugmenter(noise_std=0.01)
        X = np.random.randn(50, 30, 49)
        y = np.random.randn(50)
        X_aug, y_aug = aug.add_noise(X, y)
        assert X_aug.shape == X.shape
        assert not np.allclose(X, X_aug)  # Should differ due to noise

    def test_augment_increases_size(self):
        aug = DataAugmenter(synthetic_ratio=0.3)
        X = np.random.randn(50, 30, 49)
        y = np.random.randn(50)
        X_all, y_all = aug.augment(X, y)
        assert len(X_all) > len(X)
