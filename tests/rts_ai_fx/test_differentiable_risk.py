"""Tests for differentiable risk-constrained optimization."""

import pytest
from rts_ai_fx.differentiable_risk import (
    SoftSigmoidProjection,
    SoftLeverageConstraint,
    SoftDiversificationConstraint,
    DifferentiableRiskLayer,
    RiskConstrainedPolicy,
)
from rts_ai_fx.constrained_trainer import (
    ConstrainedPPOTrainer,
    ConstrainedTrainerConfig,
)

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestSoftSigmoidProjection:
    def test_output_range(self):
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        proj = SoftSigmoidProjection(max_position=2.0)
        x = torch.tensor([[-10.0, 0.0, 10.0]])
        out = proj(x)
        assert out.min() >= 0.0
        assert out.max() <= 2.0


class TestSoftLeverageConstraint:
    def test_enforces_leverage(self):
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        constraint = SoftLeverageConstraint(max_leverage=1.0)
        positions = torch.tensor([[0.8, 0.8]])  # total = 1.6 > 1.0
        out = constraint(positions)
        assert out.sum().item() <= 1.0 + 1e-6

    def test_passthrough_when_under_limit(self):
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        constraint = SoftLeverageConstraint(max_leverage=2.0)
        positions = torch.tensor([[0.5, 0.5]])  # total = 1.0 < 2.0
        out = constraint(positions)
        assert abs(out.sum().item() - 1.0) < 1e-6


class TestSoftDiversificationConstraint:
    def test_penalty_for_concentration(self):
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        constraint = SoftDiversificationConstraint(max_hhi=0.3)
        # Concentrated: all in one asset
        positions = torch.tensor([[1.0, 0.0, 0.0]])
        _, penalty = constraint(positions)
        assert penalty.item() > 0

    def test_no_penalty_for_diversified(self):
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        constraint = SoftDiversificationConstraint(max_hhi=0.5)
        # Evenly distributed across 3 assets
        positions = torch.tensor([[1.0, 1.0, 1.0]])
        _, penalty = constraint(positions)
        assert penalty.item() < 1e-6


class TestDifferentiableRiskLayer:
    def test_forward_shape(self):
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        layer = DifferentiableRiskLayer(max_position=1.0, max_leverage=2.0, n_assets=5)
        raw = torch.randn(4, 5)
        constrained, loss = layer(raw)
        assert constrained.shape == (4, 5)
        assert loss.item() >= 0

    def test_enforces_all_constraints(self):
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        layer = DifferentiableRiskLayer(max_position=0.5, max_leverage=1.0, n_assets=3)
        raw = torch.tensor([[100.0, 100.0, 100.0]])  # Extreme inputs
        constrained, _ = layer(raw)
        assert constrained.max() <= 0.5 + 1e-4
        assert constrained.sum() <= 1.0 + 1e-4


class TestRiskConstrainedPolicy:
    def test_forward_shape(self):
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        base_policy = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 5))
        policy = RiskConstrainedPolicy(
            base_policy, n_actions=5, max_position=1.0, max_leverage=2.0
        )
        state = torch.randn(4, 10)
        actions, loss = policy(state)
        assert actions.shape == (4, 5)
        assert loss.item() >= 0

    def test_all_actions_within_bounds(self):
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        base_policy = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 3))
        policy = RiskConstrainedPolicy(
            base_policy, n_actions=3, max_position=0.5, max_leverage=1.0
        )
        state = torch.randn(10, 5)
        actions, _ = policy(state)
        assert actions.max() <= 0.5 + 1e-4
        assert actions.sum(dim=1).max() <= 1.0 + 1e-4


class TestConstrainedPPOTrainer:
    def test_compute_loss(self):
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        trainer = ConstrainedPPOTrainer(
            base_trainer=None,
            config=ConstrainedTrainerConfig(initial_lagrange_mult=0.1),
        )
        actor_loss = torch.tensor(0.5)
        critic_loss = torch.tensor(0.3)
        constraint_loss = torch.tensor(0.1)
        total = trainer.compute_loss(actor_loss, critic_loss, constraint_loss)
        assert total.item() > 0
        assert trainer.lagrange_mult > 0

    def test_lagrange_adapts(self):
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        trainer = ConstrainedPPOTrainer(
            base_trainer=None,
            config=ConstrainedTrainerConfig(initial_lagrange_mult=0.1, lagrange_lr=0.1),
        )
        # High violation -> multiplier should increase
        for _ in range(10):
            trainer.compute_loss(
                torch.tensor(0.5), torch.tensor(0.3), torch.tensor(1.0)
            )
        assert trainer.lagrange_mult > 0.1

    def test_get_stats(self):
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        trainer = ConstrainedPPOTrainer(base_trainer=None)
        stats = trainer.get_stats()
        assert "lagrange_multiplier" in stats
        assert "avg_constraint_loss" in stats

    def test_config_defaults(self):
        config = ConstrainedTrainerConfig()
        assert config.initial_lagrange_mult == 0.1
        assert config.lagrange_lr == 0.01
        assert config.max_lagrange_mult == 10.0
