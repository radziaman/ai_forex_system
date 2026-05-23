"""
Online PPO Trainer — Real-time PPO updates with reward shaping.
"""

import numpy as np
from collections import deque
import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Tuple
from loguru import logger


class OnlinePPOTrainer:
    """Online PPO trainer with circular experience buffer and
    real-time reward shaping."""

    def __init__(self, agent, buffer_size: int = 1000, update_freq: int = 50):
        self.agent = agent
        self.buffer_size = buffer_size
        self.update_freq = update_freq
        self.buffer: deque = deque(maxlen=buffer_size)
        self._new_experiences: int = 0
        self.consecutive_wins: int = 0

    def record_experience(self, state, action, reward, next_state, done):
        """Store a transition in the circular buffer."""
        self.buffer.append((state, action, reward, next_state, done))
        self._new_experiences += 1

    def should_update(self) -> bool:
        """Return True when buffer has enough new experiences for an update."""
        return self._new_experiences >= self.update_freq

    def shape_reward(self, pnl: float, holding_time: float, drawdown: float) -> float:
        """Shape reward with opportunity cost, drawdown penalty, and momentum bonus."""
        # Base reward = PnL in R-multiples
        base = pnl

        # Penalty for long holding time (opportunity cost)
        holding_penalty = -0.02 * max(0, holding_time - 5.0)

        # Severe penalty for hitting max drawdown
        dd_penalty = 0.0
        if drawdown >= 0.20:
            dd_penalty = -50.0 * drawdown
        elif drawdown >= 0.10:
            dd_penalty = -20.0 * drawdown
        elif drawdown >= 0.05:
            dd_penalty = -10.0 * drawdown

        # Bonus for consecutive wins (momentum bonus)
        if pnl > 0:
            self.consecutive_wins += 1
        else:
            self.consecutive_wins = 0
        momentum_bonus = 0.5 * self.consecutive_wins

        return float(base + holding_penalty + dd_penalty + momentum_bonus)

    def update_policy(self, n_epochs: int = 4, batch_size: int = 64) -> dict:
        """Run PPO update on the circular buffer."""
        if len(self.buffer) < 2:
            return {}

        states = np.array([e[0] for e in self.buffer])
        actions = np.array([e[1] for e in self.buffer])
        rewards = np.array([e[2] for e in self.buffer])
        next_states = np.array([e[3] for e in self.buffer])
        dones = np.array([e[4] for e in self.buffer])

        device = self.agent.device

        # Compute values
        with torch.no_grad():
            states_t = torch.FloatTensor(states).to(device)
            _, _, _, _, values = self.agent.actor(states_t)
            values_np = values.cpu().numpy()
            next_states_t = torch.FloatTensor(next_states).to(device)
            _, _, _, _, next_values = self.agent.actor(next_states_t)
            next_values_np = next_values.cpu().numpy()

        # Compute GAE
        advantages, returns = self._compute_gae(
            rewards, dones, values_np, next_values_np
        )

        states_t = torch.FloatTensor(states).to(device)
        actions_t = torch.LongTensor(actions).to(device)
        advantages_t = torch.FloatTensor(advantages).to(device)
        returns_t = torch.FloatTensor(returns).to(device)

        advantages_t = (advantages_t - advantages_t.mean()) / (
            advantages_t.std() + 1e-8
        )

        # Old log probs
        with torch.no_grad():
            act_logits, _, _, _, _ = self.agent.actor(states_t)
            probs = torch.softmax(act_logits, dim=1)
            dist = Categorical(probs)
            old_log_probs = dist.log_prob(actions_t)

        losses = {"policy_loss": [], "value_loss": [], "entropy": []}

        for _ in range(n_epochs):
            idx = np.random.permutation(len(states_t))
            for start in range(0, len(states_t), batch_size):
                b_idx = idx[start : start + batch_size]
                act_logits, _, _, _, values_new = self.agent.actor(states_t[b_idx])
                probs = torch.softmax(act_logits, dim=1)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(actions_t[b_idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs[b_idx])
                s1 = ratio * advantages_t[b_idx]
                s2 = (
                    torch.clamp(
                        ratio,
                        1 - self.agent.clip_range,
                        1 + self.agent.clip_range,
                    )
                    * advantages_t[b_idx]
                )
                pol_loss = -torch.min(s1, s2).mean()
                val_loss = ((values_new - returns_t[b_idx]) ** 2).mean()

                loss = (
                    pol_loss
                    + self.agent.vf_coef * val_loss
                    - self.agent.ent_coef * entropy
                )

                self.agent.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.agent.actor.parameters(), self.agent.max_grad_norm
                )
                self.agent.optimizer.step()

                losses["policy_loss"].append(pol_loss.item())
                losses["value_loss"].append(val_loss.item())
                losses["entropy"].append(entropy.item())

        self._new_experiences = 0
        result = {k: np.mean(v) for k, v in losses.items()}
        logger.info(f"Online PPO update: {result}")
        return result

    def _compute_gae(
        self, rewards, dones, values, next_values
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation."""
        advantages = np.zeros_like(rewards, dtype=np.float64)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_v = next_values[t] if not dones[t] else 0.0
            delta = rewards[t] + self.agent.gamma * next_v - values[t]
            gae = delta + self.agent.gamma * self.agent.gae_lambda * gae
            advantages[t] = gae
        return advantages, advantages + values
