"""
Meta-Learning for Fast Adaptation (MAML).
Model-Agnostic Meta-Learning allows rapid adaptation to new market regimes.
Adapts to new conditions in 5-10 samples instead of thousands.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Callable, Iterator
from loguru import logger


class MAMLModel(nn.Module):
    """Base model wrapper for MAML."""

    def __init__(
        self, input_dim: int, hidden_dims: list = [256, 128, 64], output_dim: int = 1
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class FastParameterCopy(nn.Module):
    """
    Wrapper that uses functional parameters instead of owning them.
    Enables gradient flow through inner-loop updates for MAML.
    """

    def __init__(self, model: nn.Module, params: Dict[str, torch.Tensor]):
        super().__init__()
        self.model_template = model
        self._params = params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            nn.functional.linear(x, self._params.get(list(self._params.keys())[-1], x))
            if False
            else self._manual_forward(x)
        )

    def _manual_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using functional parameter dict."""
        h = x
        for name, param in self._params.items():
            if "weight" in name:
                if "network.0" in name or ".".join(name.split(".")[:-1]) in (
                    f"network.{i}" for i in range(0, 7, 2)
                ):
                    h = nn.functional.linear(h, param)
            elif "bias" in name:
                h = h + param
                h = nn.functional.relu(h)
        return h


class MAMLAgent:
    """
    Model-Agnostic Meta-Learning Agent with correct gradient flow.
    Learns how to adapt quickly to new market regimes via inner-loop SGD
    that propagates gradients back to the meta-parameters.
    """

    def __init__(
        self,
        base_model: Optional[nn.Module] = None,
        input_dim: int = 55 * 30,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        inner_steps: int = 5,
    ):
        self.input_dim = input_dim
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps

        self.base_model = base_model or MAMLModel(input_dim)
        self.meta_optimizer = optim.Adam(self.base_model.parameters(), lr=meta_lr)

        self.meta_loss_history: List[float] = []
        self.adaptation_losses: Dict[str, List[float]] = {}

    def _inner_loop(
        self, params: Dict[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Perform inner-loop gradient descent on parameter dict.
        Uses torch.autograd.grad with create_graph=True to maintain
        gradient flow back to original parameters.
        """
        criterion = nn.MSELoss()
        for _ in range(self.inner_steps):
            pred = self._forward_from_params(params, x)
            loss = criterion(pred, y.unsqueeze(1) if y.dim() == 1 else y)
            grads = torch.autograd.grad(loss, params.values(), create_graph=True)
            grad_dict = {name: g for name, g in zip(params.keys(), grads)}
            params = {
                name: p - self.inner_lr * grad_dict[name] for name, p in params.items()
            }
        return params

    @staticmethod
    def _forward_from_params(
        params: Dict[str, torch.Tensor], x: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the network using parameter dict."""
        h = x
        weight_keys = sorted([k for k in params.keys() if "weight" in k])
        bias_keys = sorted([k for k in params.keys() if "bias" in k])
        for i in range(len(weight_keys)):
            h = nn.functional.linear(
                h, params[weight_keys[i]], params.get(bias_keys[i])
            )
            if i < len(weight_keys) - 1:
                h = nn.functional.relu(h)
        return h

    def adapt(
        self,
        support_set: Tuple[torch.Tensor, torch.Tensor],
        steps: Optional[int] = None,
    ) -> nn.Module:
        """Quickly adapt to new regime with few gradient steps. Returns adapted model."""
        orig_steps = self.inner_steps
        if steps is not None:
            self.inner_steps = steps

        adapted = self.base_model
        support_x, support_y = support_set
        if len(support_x) == 0:
            return adapted

        criterion = nn.MSELoss()
        optimizer = optim.SGD(adapted.parameters(), lr=self.inner_lr)
        actual_steps = steps or self.inner_steps
        for _ in range(actual_steps):
            optimizer.zero_grad()
            pred = adapted(support_x)
            loss = criterion(
                pred, support_y.unsqueeze(1) if support_y.dim() == 1 else support_y
            )
            loss.backward()
            optimizer.step()

        self.inner_steps = orig_steps
        return adapted

    def meta_train(
        self,
        tasks: List[
            Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        ],
        epochs: int = 10,
    ) -> float:
        """
        Train meta-learning across multiple tasks.
        Uses gradient-through-gradient for proper MAML inner-loop gradient flow.
        """
        total_meta_loss = 0.0

        for epoch in range(epochs):
            epoch_meta_loss = 0.0
            task_count = 0

            for (support_x, support_y), (query_x, query_y) in tasks:
                meta_loss = self._meta_task_loss(support_x, support_y, query_x, query_y)
                if meta_loss is None:
                    continue
                epoch_meta_loss += meta_loss.item()
                task_count += 1

            if task_count == 0:
                continue

            avg_loss = epoch_meta_loss / task_count
            self.meta_loss_history.append(avg_loss)
            total_meta_loss += avg_loss

            if (epoch + 1) % 5 == 0:
                logger.info(f"MAML Epoch {epoch+1}/{epochs}, Meta-Loss: {avg_loss:.6f}")

        return total_meta_loss / max(epochs, 1)

    def _meta_task_loss(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Compute meta-loss for a single task with correct gradient flow.
        Uses functional inner-loop via torch.autograd.grad(create_graph=True).
        """
        try:
            params = {name: p.clone() for name, p in self.base_model.named_parameters()}
            for p in params.values():
                p.requires_grad_(True)

            for _ in range(self.inner_steps):
                pred = self._forward_from_params(params, support_x)
                loss = nn.MSELoss()(
                    pred, support_y.unsqueeze(1) if support_y.dim() == 1 else support_y
                )

                grads = torch.autograd.grad(
                    loss,
                    params.values(),
                    create_graph=True,
                    allow_unused=True,
                )
                grad_map = {}
                for (name, _), g in zip(params.items(), grads):
                    if g is not None:
                        grad_map[name] = g
                    else:
                        grad_map[name] = torch.zeros_like(params[name])

                params = {
                    name: p - self.inner_lr * grad_map[name]
                    for name, p in params.items()
                }

            meta_pred = self._forward_from_params(params, query_x)
            meta_loss = nn.MSELoss()(
                meta_pred, query_y.unsqueeze(1) if query_y.dim() == 1 else query_y
            )

            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), max_norm=1.0)
            self.meta_optimizer.step()

            return meta_loss.detach()
        except Exception as e:
            logger.debug(f"MAML task error: {e}")
            return None

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
    ) -> Optional[
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
    ]:
        """Create a meta-learning task from regime data."""
        if len(features) < k_shot + 5:
            return None
        support_x = torch.FloatTensor(features[:k_shot])
        support_y = torch.FloatTensor(targets[:k_shot])
        query_x = torch.FloatTensor(features[k_shot:])
        query_y = torch.FloatTensor(targets[k_shot:])
        return (support_x, support_y), (query_x, query_y)

    def save(self, path: str):
        torch.save(
            {
                "model_state_dict": self.base_model.state_dict(),
                "optimizer_state_dict": self.meta_optimizer.state_dict(),
                "meta_loss_history": self.meta_loss_history,
            },
            path,
        )
        logger.info(f"MAML agent saved to {path}")

    def load(self, path: str):
        try:
            checkpoint = torch.load(path)
            self.base_model.load_state_dict(checkpoint["model_state_dict"])
            self.meta_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.meta_loss_history = checkpoint.get("meta_loss_history", [])
            logger.info(f"MAML agent loaded from {path}")
        except Exception as e:
            logger.warning(f"MAML load failed: {e}")

    def get_meta_summary(self) -> Dict:
        return {
            "avg_meta_loss": (
                float(np.mean(self.meta_loss_history))
                if self.meta_loss_history
                else 0.0
            ),
            "n_meta_updates": len(self.meta_loss_history),
            "inner_lr": self.inner_lr,
            "meta_lr": self.meta_lr,
            "inner_steps": self.inner_steps,
        }
