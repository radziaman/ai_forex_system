"""
Meta-Learning for Fast Adaptation (MAML).
Model-Agnostic Meta-Learning allows rapid adaptation to new market regimes.
Adapts to new conditions in 5-10 samples instead of thousands.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Callable
from copy import deepcopy
from loguru import logger


class MAMLModel(nn.Module):
    """Base model wrapper for MAML."""

    def __init__(self, input_dim: int, hidden_dims: list = [256, 128, 64], output_dim: int = 1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MAMLAgent:
    """
    Model-Agnostic Meta-Learning Agent.
    Learns how to adapt quickly to new market regimes.
    """

    def __init__(
        self,
        base_model: Optional[nn.Module] = None,
        input_dim: int = 55 * 30,  # Default feature size
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        inner_steps: int = 5,
    ):
        self.input_dim = input_dim
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps

        # Base model to be cloned
        self.base_model = base_model or MAMLModel(input_dim)

        # Meta-optimizer
        self.meta_optimizer = optim.Adam(self.base_model.parameters(), lr=meta_lr)

        # Track meta-performance
        self.meta_loss_history: List[float] = []
        self.adaptation_losses: Dict[str, List[float]] = {}

    def adapt(
        self,
        support_set: Tuple[torch.Tensor, torch.Tensor],
        steps: Optional[int] = None,
    ) -> nn.Module:
        """
        Quickly adapt to new regime with few gradient steps.
        Returns adapted model.
        """
        adapted_model = deepcopy(self.base_model)
        steps = steps or self.inner_steps

        support_x, support_y = support_set
        if len(support_x) == 0:
            return adapted_model

        optimizer = optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        criterion = nn.MSELoss()

        for step in range(steps):
            optimizer.zero_grad()
            predictions = adapted_model(support_x)
            loss = criterion(predictions, support_y.unsqueeze(1) if support_y.dim() == 1 else support_y)
            loss.backward()
            optimizer.step()

        return adapted_model

    def meta_train(
        self,
        tasks: List[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]],
        epochs: int = 10,
    ) -> float:
        """
        Train meta-learning across multiple tasks.
        Each task is a (support_set, query_set) pair.
        """
        total_meta_loss = 0.0

        for epoch in range(epochs):
            epoch_loss = 0.0

            for (support_x, support_y), (query_x, query_y) in tasks:
                # Adapt model to this task
                adapted = self.adapt((support_x, support_y), self.inner_steps)

                # Compute loss on query set
                predictions = adapted(query_x)
                loss = nn.MSELoss()(
                    predictions, query_y.unsqueeze(1) if query_y.dim() == 1 else query_y
                )

                epoch_loss += loss.item()

                # Meta-gradient step
                self.meta_optimizer.zero_grad()
                loss.backward()
                self.meta_optimizer.step()

            avg_loss = epoch_loss / len(tasks) if tasks else 0.0
            total_meta_loss += avg_loss
            self.meta_loss_history.append(avg_loss)

            if (epoch + 1) % 5 == 0:
                logger.info(f"MAML Epoch {epoch+1}/{epochs}, Meta-Loss: {avg_loss:.6f}")

        return total_meta_loss / epochs if epochs > 0 else 0.0

    def predict_with_adaptation(
        self,
        new_data_x: torch.Tensor,
        new_data_y: torch.Tensor,
        adapt_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Adapt and predict in one call."""
        adapted = self.adapt((new_data_x, new_data_y), adapt_steps)
        with torch.no_grad():
            return adapted(new_data_x)

    def create_task(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        regime: str,
        k_shot: int = 10,
    ) -> Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Create a meta-learning task from regime data.
        Uses k-shot support set and remaining as query set.
        """
        if len(features) < k_shot + 5:
            return None

        # Split into support (k-shot) and query
        support_x = torch.FloatTensor(features[:k_shot])
        support_y = torch.FloatTensor(targets[:k_shot])
        query_x = torch.FloatTensor(features[k_shot:])
        query_y = torch.FloatTensor(targets[k_shot:])

        return (support_x, support_y), (query_x, query_y)

    def save(self, path: str):
        """Save meta-learned parameters."""
        torch.save({
            'model_state_dict': self.base_model.state_dict(),
            'optimizer_state_dict': self.meta_optimizer.state_dict(),
            'meta_loss_history': self.meta_loss_history,
        }, path)
        logger.info(f"MAML agent saved to {path}")

    def load(self, path: str):
        """Load meta-learned parameters."""
        try:
            checkpoint = torch.load(path)
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
            self.meta_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.meta_loss_history = checkpoint.get('meta_loss_history', [])
            logger.info(f"MAML agent loaded from {path}")
        except Exception as e:
            logger.warning(f"MAML load failed: {e}")

    def get_meta_summary(self) -> Dict:
        """Get summary of meta-learning performance."""
        return {
            "avg_meta_loss": float(np.mean(self.meta_loss_history)) if self.meta_loss_history else 0.0,
            "n_meta_updates": len(self.meta_loss_history),
            "inner_lr": self.inner_lr,
            "meta_lr": self.meta_lr,
            "inner_steps": self.inner_steps,
        }
