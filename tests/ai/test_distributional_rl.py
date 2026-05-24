"""Tests for Distributional RL components."""

import numpy as np
import torch
import pytest
from ai.distributional_rl import (
    QuantileValueHead,
    QuantilePPOCritic,
    IQNEmbedding,
    quantile_huber_loss,
)


class TestQuantileHuberLoss:
    def test_loss_is_scalar(self):
        pred = torch.randn(4, 64)
        target = torch.randn(4, 64)
        tau = torch.rand(4, 64)
        loss = quantile_huber_loss(pred, target, tau)
        assert loss.ndim == 0
        assert loss > 0

    def test_zero_loss_when_perfect(self):
        pred = torch.ones(4, 64) * 0.5
        target = torch.ones(4, 64) * 0.5
        tau = torch.rand(4, 64)
        loss = quantile_huber_loss(pred, target, tau)
        assert loss < 1e-6

    def test_loss_penalizes_overestimate(self):
        """Overestimates (pred > target) should be penalized according to tau."""
        tau = torch.full((1, 1), 0.1)  # Low quantile
        pred = torch.ones(1, 1) * 1.0
        target = torch.zeros(1, 1)
        loss_high_tau = quantile_huber_loss(pred, target, tau)
        tau2 = torch.full((1, 1), 0.9)  # High quantile
        loss_low_tau = quantile_huber_loss(pred, target, tau2)
        # For overestimate, lower tau means lower weight -> lower loss
        assert loss_high_tau > loss_low_tau


class TestQuantileValueHead:
    def test_output_shape(self):
        head = QuantileValueHead(input_dim=128, n_quantiles=64)
        features = torch.randn(16, 128)
        out = head(features)
        assert out.shape == (16, 64)

    def test_output_dim_property(self):
        head = QuantileValueHead(input_dim=64, n_quantiles=32)
        assert head.output_dim == 32

    def test_empty_batch(self):
        head = QuantileValueHead(input_dim=64, n_quantiles=16)
        features = torch.randn(0, 64)
        out = head(features)
        assert out.shape == (0, 16)


class TestQuantilePPOCritic:
    def test_forward_shape(self):
        critic = QuantilePPOCritic(state_dim=49, hidden_dims=[256, 128], n_quantiles=64)
        state = torch.randn(8, 49)
        quantiles = critic(state)
        assert quantiles.shape == (8, 64)

    def test_quantiles_sorted(self):
        critic = QuantilePPOCritic(state_dim=10, hidden_dims=[32], n_quantiles=16)
        state = torch.randn(4, 10)
        quantiles = critic(state)
        for i in range(quantiles.shape[1] - 1):
            assert (quantiles[:, i] <= quantiles[:, i + 1]).all()

    def test_expected_value_shape(self):
        critic = QuantilePPOCritic(state_dim=10, hidden_dims=[32], n_quantiles=64)
        state = torch.randn(4, 10)
        ev = critic.expected_value(state)
        assert ev.shape == (4, 1)

    def test_var_shape(self):
        critic = QuantilePPOCritic(state_dim=10, hidden_dims=[32], n_quantiles=64)
        state = torch.randn(4, 10)
        var = critic.value_at_risk(state, alpha=0.05)
        assert var.shape == (4, 1)

    def test_cvar_less_than_var(self):
        """CVaR (mean of worst quantiles) should be <= VaR (single quantile)."""
        critic = QuantilePPOCritic(state_dim=10, hidden_dims=[32], n_quantiles=64)
        state = torch.randn(4, 10)
        var = critic.value_at_risk(state, alpha=0.05)
        cvar = critic.cvar(state, alpha=0.05)
        assert (cvar <= var).all()


class TestIQNEmbedding:
    def test_embedding_shape(self):
        embed = IQNEmbedding(embedding_dim=64)
        tau = torch.rand(4, 8)
        emb = embed(tau)
        assert emb.shape == (4, 8, 64)

    def test_embedding_bounds(self):
        """cos(pi * i * tau) for tau in [0,1] should be in [-1, 1]."""
        embed = IQNEmbedding(embedding_dim=16)
        tau = torch.tensor([[0.0, 0.5, 1.0]])
        emb = embed(tau)
        assert emb.min() >= -1.0 - 1e-6
        assert emb.max() <= 1.0 + 1e-6
