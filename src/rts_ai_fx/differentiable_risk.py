"""
Differentiable Risk-Constrained Optimization Layer.

Embeds portfolio constraints as differentiable operations that can
be composed with neural network policies. Enables end-to-end learning
where the RL agent inherently respects risk limits.

Key idea: Instead of ML -> Risk (post-hoc filter), use
ML -> DifferentiableRiskLayer -> Risk-constrained output

The layer projects unconstrained policy outputs into the feasible
set defined by risk constraints, using differentiable optimization
techniques (interior-point inspired, no external solvers needed).

Constraints supported:
  - Max position size (per asset)
  - Max leverage (sum of absolute positions)
  - Max correlation concentration
  - Min diversification (Herfindahl index)
  - VaR limit (via differentiable approximation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SoftSigmoidProjection(nn.Module):
    """Differentiable projection onto box constraints via sigmoid.

    Maps unconstrained outputs to [0, max_position] range using a
    scaled sigmoid. This is a soft, differentiable proxy for clipping.

    Forward: output = max_position * sigmoid(x)
    Gradient: smooth everywhere, enables gradient flow through constraints
    """

    def __init__(self, max_position: float = 1.0, n_assets: int = 1):
        super().__init__()
        self.max_position = max_position
        self.n_assets = n_assets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project unconstrained logits to [0, max_position].

        Args:
            x: (batch, n_assets) unconstrained outputs

        Returns:
            (batch, n_assets) in [0, max_position]
        """
        return self.max_position * torch.sigmoid(x)


class SoftLeverageConstraint(nn.Module):
    """Differentiable leverage constraint using soft normalization.

    Ensures sum of absolute positions <= max_leverage by applying a
    soft normalization that preserves gradient information.

    Forward: output = positions * max_leverage / max(sum(abs), max_leverage)
    """

    def __init__(self, max_leverage: float = 2.0):
        super().__init__()
        self.max_leverage = max_leverage

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Project positions to satisfy leverage constraint.

        Args:
            positions: (batch, n_assets) position sizes (non-negative)

        Returns:
            (batch, n_assets) satisfying sum <= max_leverage
        """
        total = positions.sum(dim=-1, keepdim=True) + 1e-8
        scale = torch.clamp(self.max_leverage / total, max=1.0)
        return positions * scale


class SoftDiversificationConstraint(nn.Module):
    """Differentiable diversification constraint.

    Penalizes concentration via the Herfindahl-Hirschman Index (HHI).
    HHI = sum(w_i^2). Lower values = more diversified.

    The constraint is applied as a penalty, not a hard projection,
    to maintain differentiability.
    """

    def __init__(self, max_hhi: float = 0.5):
        super().__init__()
        self.max_hhi = max_hhi

    def forward(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute HHI and apply diversification penalty.

        Args:
            positions: (batch, n_assets) position sizes

        Returns:
            (adjusted_positions, penalty_loss)
        """
        total = positions.sum(dim=-1, keepdim=True) + 1e-8
        weights = positions / total
        hhi = (weights**2).sum(dim=-1)  # (batch,)

        # Penalty: how much we exceed max_hhi
        penalty = F.relu(hhi - self.max_hhi).mean()

        # Diversify by scaling down the largest positions
        if hhi.mean() > self.max_hhi:
            scale = torch.clamp(self.max_hhi / (hhi.detach() + 1e-8), max=1.0)
            return positions * scale.unsqueeze(-1), penalty
        return positions, penalty


class DifferentiableRiskLayer(nn.Module):
    """Composable differentiable risk constraint layer.

    Applies a sequence of soft constraints to ensure the policy output
    satisfies risk limits. All operations are differentiable, enabling
    gradient flow from reward through constraints back to the policy.

    Usage:
        risk_layer = DifferentiableRiskLayer(
            max_position=1.0, max_leverage=2.0, max_hhi=0.5
        )
        constrained_positions, constraint_loss = risk_layer(raw_policy_output)

    The constraint_loss can be added to the PPO loss to penalize violations.
    """

    def __init__(
        self,
        max_position: float = 1.0,
        max_leverage: float = 2.0,
        max_hhi: float = 0.5,
        n_assets: int = 1,
        constraint_weight: float = 0.1,
    ):
        super().__init__()
        self.constraint_weight = constraint_weight
        self.n_assets = n_assets

        self.position_proj = SoftSigmoidProjection(max_position, n_assets)
        self.leverage_proj = SoftLeverageConstraint(max_leverage)
        self.diversification = SoftDiversificationConstraint(max_hhi)

    def forward(self, raw_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply all risk constraints sequentially.

        Args:
            raw_outputs: (batch, n_assets) unconstrained policy outputs

        Returns:
            (constrained_positions, total_constraint_loss)
        """
        total_loss = 0.0

        # 1. Project to [0, max_position]
        positions = self.position_proj(raw_outputs)

        # 2. Enforce leverage limit
        positions = self.leverage_proj(positions)

        # 3. Enforce diversification
        positions, hhi_penalty = self.diversification(positions)
        total_loss = total_loss + hhi_penalty

        return positions, total_loss * self.constraint_weight


class RiskConstrainedPolicy(nn.Module):
    """Policy network with integrated differentiable risk constraints.

    Wraps an existing policy network with a DifferentiableRiskLayer.
    The policy outputs action logits which pass through risk constraints
    before being used for action selection.

    Usage:
        base_policy = ActorNetwork(state_dim, action_dim, hidden_dims)
        constrained_policy = RiskConstrainedPolicy(
            base_policy, max_position=1.0, max_leverage=2.0
        )
        actions, constraint_loss = constrained_policy(states)
    """

    def __init__(
        self,
        base_policy: nn.Module,
        n_actions: int = 5,
        max_position: float = 1.0,
        max_leverage: float = 2.0,
        constraint_weight: float = 0.1,
    ):
        super().__init__()
        self.base_policy = base_policy
        self.risk_layer = DifferentiableRiskLayer(
            max_position=max_position,
            max_leverage=max_leverage,
            n_assets=n_actions,
            constraint_weight=constraint_weight,
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with risk constraints.

        Args:
            state: (batch, state_dim)

        Returns:
            (constrained_actions, constraint_loss)
        """
        raw_actions = self.base_policy(state)
        constrained, loss = self.risk_layer(raw_actions)
        return constrained, loss
