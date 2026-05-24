"""
Constrained Training -- wraps PPO training with differentiable risk constraints.

Ensures the policy learns to satisfy risk constraints during training
by adding a constraint violation penalty to the PPO loss.

This implements a simplified Augmented Lagrangian approach:
  L_total = L_PPO + lambda * L_constraint

Where lambda adapts over time: if constraints are violated, lambda increases;
if constraints are satisfied, lambda decreases.
"""

import torch
from typing import Dict, Optional, Callable
from dataclasses import dataclass


@dataclass
class ConstrainedTrainerConfig:
    initial_lagrange_mult: float = 0.1
    lagrange_lr: float = 0.01  # How fast to adjust multiplier
    max_lagrange_mult: float = 10.0
    constraint_target: float = 0.01  # Target constraint violation level
    constraint_smoothing: float = 0.99  # EMA smoothing for constraint violation


class ConstrainedPPOTrainer:
    """PPO trainer with adaptive risk constraint enforcement.

    Wraps standard PPO loss with a constraint penalty:
      L_total = L_PPO(clip) + lambda * L_constraint(violations)

    lambda (lagrange multiplier) adapts:
      - If violations > target -> lambda increases (harder constraint)
      - If violations < target -> lambda decreases (softer constraint)
    """

    def __init__(
        self,
        base_trainer: object,
        risk_layer: Optional[Callable] = None,
        config: Optional[ConstrainedTrainerConfig] = None,
    ):
        self.base_trainer = base_trainer
        self.risk_layer = risk_layer
        self.config = config or ConstrainedTrainerConfig()
        self.lagrange_mult = self.config.initial_lagrange_mult
        self._smoothed_violation = 0.0
        self._total_constraint_loss = 0.0
        self._n_updates = 0

    def compute_loss(
        self,
        actor_loss: torch.Tensor,
        critic_loss: torch.Tensor,
        constraint_loss: torch.Tensor,
    ) -> torch.Tensor:
        """Compute total loss with adaptive constraint weighting.

        Args:
            actor_loss: standard PPO clipped surrogate loss
            critic_loss: value function MSE loss
            constraint_loss: differentiable constraint violation loss

        Returns:
            total_loss: weighted combination with adaptive lambda
        """
        ppo_loss = actor_loss + 0.5 * critic_loss

        # Adaptive Lagrange multiplier for constraints
        violation = constraint_loss.detach().item()
        self._smoothed_violation = (
            self.config.constraint_smoothing * self._smoothed_violation
            + (1 - self.config.constraint_smoothing) * violation
        )

        # Adjust Lagrange multiplier
        if self._smoothed_violation > self.config.constraint_target:
            self.lagrange_mult = min(
                self.config.max_lagrange_mult,
                self.lagrange_mult * (1 + self.config.lagrange_lr),
            )
        else:
            self.lagrange_mult = max(
                0.0,
                self.lagrange_mult * (1 - self.config.lagrange_lr * 0.5),
            )

        total_loss = ppo_loss + self.lagrange_mult * constraint_loss
        self._total_constraint_loss += float(constraint_loss.item())
        self._n_updates += 1

        return total_loss

    def get_stats(self) -> Dict:
        return {
            "lagrange_multiplier": round(self.lagrange_mult, 4),
            "smoothed_violation": round(self._smoothed_violation, 6),
            "avg_constraint_loss": round(
                self._total_constraint_loss / max(self._n_updates, 1), 6
            ),
        }
