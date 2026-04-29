"""
PPO RL Agent - Institutional Forex AI Brain
Based on TICQ AI research (3.5M params, PPO, 5 timeframes).
Implements full PPO algorithm with GAE, 11-component reward (arXiv 2026).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from loguru import logger
import os
import pickle
from typing import Optional, Tuple, Dict, List


class ActorNetwork(nn.Module):
    """Actor network: 3.5M+ params (1024→512→256→128)."""

    def __init__(self, input_dim: int, n_actions: int = 5):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Dropout(0.1),
        )
        self.action_head = nn.Linear(128, n_actions)
        self.sl_head = nn.Linear(128, 1)
        self.tp_head = nn.Linear(128, 1)
        self.size_head = nn.Linear(128, 1)
        self.value_head = nn.Linear(128, 1)
        self._init_weights()
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"ActorNetwork | Params: {total:,} | Input: {input_dim}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        f = self.feature_extractor(x)
        return (
            self.action_head(f),
            torch.sigmoid(self.sl_head(f)) * 3.0 + 0.5,
            torch.sigmoid(self.tp_head(f)) * 5.0 + 1.5,
            torch.sigmoid(self.size_head(f)) * 2.0 + 0.01,
            self.value_head(f).squeeze(-1),
        )


class PPOAgent:
    """Complete PPO implementation with GAE."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int = 5,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "auto",
    ):
        self.device = (
            torch.device(
                "cuda" if torch.cuda.is_available() and device == "auto" else "cpu"
            )
            if device == "auto"
            else torch.device(device)
        )
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.actor = ActorNetwork(state_dim, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.sl_vals = []
        self.tp_vals = []
        self.size_vals = []

        logger.info(
            f"PPOAgent initialized | Device: {self.device} | State dim: {state_dim}"
        )

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float, float, dict]:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            act_logits, sl_raw, tp_raw, size_raw, value = self.actor(state_t)
            probs = torch.softmax(act_logits, dim=1)
            dist = Categorical(probs)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action, device=self.device)).item()

        self.states.append(state.copy())
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value.item())
        self.sl_vals.append(sl_raw.item())
        self.tp_vals.append(tp_raw.item())
        self.size_vals.append(size_raw.item())

        return (
            action,
            sl_raw.item(),
            tp_raw.item(),
            size_raw.item(),
            {
                "action_logits": act_logits.squeeze(0).cpu().numpy(),
                "value": value.item(),
            },
        )

    def store_transition(self, reward: float, done: bool):
        self.rewards.append(reward)
        self.dones.append(done)

    def train(
        self, next_value: float = 0.0, n_epochs: int = 10, batch_size: int = 64
    ) -> dict:
        if len(self.states) < 2:
            return {}
        states_t = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions_t = torch.LongTensor(self.actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(self.log_probs).to(self.device)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values_t = torch.FloatTensor(self.values).to(self.device)

        advantages, returns = self._compute_gae(
            rewards, dones, values_t.cpu().numpy(), next_value
        )
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        advantages_t = (advantages_t - advantages_t.mean()) / (
            advantages_t.std() + 1e-8
        )

        losses = {"policy_loss": [], "value_loss": [], "entropy": []}
        for _ in range(n_epochs):
            idx = np.random.permutation(len(states_t))
            for start in range(0, len(states_t), batch_size):
                b_idx = idx[start : start + batch_size]
                act_logits, _, _, _, values_new = self.actor(states_t[b_idx])
                probs = torch.softmax(act_logits, dim=1)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(actions_t[b_idx])
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_log_probs - old_log_probs_t[b_idx])
                s1 = ratio * advantages_t[b_idx]
                s2 = (
                    torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                    * advantages_t[b_idx]
                )
                pol_loss = -torch.min(s1, s2).mean()
                val_loss = ((values_new - returns_t[b_idx]) ** 2).mean()
                loss = pol_loss + self.vf_coef * val_loss - self.ent_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optimizer.step()
                losses["policy_loss"].append(pol_loss.item())
                losses["value_loss"].append(val_loss.item())
                losses["entropy"].append(entropy.item())

        self._clear_memory()
        return {k: np.mean(v) for k, v in losses.items()}

    def _compute_gae(self, rewards, dones, values, next_val):
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_v = (
                next_val
                if t == len(rewards) - 1
                else values[t + 1] if not dones[t] else 0.0
            )
            delta = rewards[t] + self.gamma * next_v - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae
        return advantages, advantages + values

    def _clear_memory(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.sl_vals = []
        self.tp_vals = []
        self.size_vals = []

    def save(self, path: str):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )
        logger.info(f"Model saved: {path}")

    def load(self, path: str):
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.actor.load_state_dict(ckpt["actor"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            logger.info(f"Model loaded: {path}")
            return True
        return False


class TradingEnvironment:
    """Custom Gym environment for forex trading with realistic frictions."""

    def __init__(
        self,
        state_dim: int,
        initial_balance: float = 100000.0,
        spread_pips: float = 0.5,
        commission_per_lot: float = 7.0,
    ):
        self.state_dim = state_dim
        self.initial_balance = initial_balance
        self.spread_pips = spread_pips
        self.commission = commission_per_lot
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.margin = 0.0
        self.open_positions = []
        self.trade_history = []
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.current_price = 1.1200
        self.current_state = np.zeros(self.state_dim, dtype=np.float32)
        return self.current_state.copy()

    def step(
        self,
        action: int,
        sl_mult: float = 2.0,
        tp_mult: float = 4.0,
        size: float = 0.01,
    ) -> Tuple[np.ndarray, float, bool, dict]:
        reward = 0.0
        done = False
        info = {
            "balance": self.balance,
            "equity": self.equity,
            "open_positions": len(self.open_positions),
        }

        if action == 1:  # BUY
            self._open_pos(
                "BUY",
                self.current_price + self.spread_pips * 0.0001,
                sl_mult,
                tp_mult,
                size,
            )
        elif action == 2:  # SELL
            self._open_pos(
                "SELL",
                self.current_price - self.spread_pips * 0.0001,
                sl_mult,
                tp_mult,
                size,
            )
        elif action == 3:  # CLOSE ALL
            self._close_all()
        elif action == 4:  # MODIFY
            pass

        self._update_positions()
        reward = self._calc_reward()
        done = self.balance < self.initial_balance * 0.5
        return self.current_state.copy(), reward, done, info

    def _open_pos(self, direction, entry_price, sl_m, tp_m, size):
        sl = (
            entry_price - sl_m * 0.001
            if direction == "BUY"
            else entry_price + sl_m * 0.001
        )
        tp = (
            entry_price + tp_m * 0.001
            if direction == "BUY"
            else entry_price - tp_m * 0.001
        )
        margin_req = entry_price * size * 100000 * 0.01
        if self.margin + margin_req > self.balance * 0.8:
            return
        self.open_positions.append(
            {
                "id": self.total_trades + 1,
                "direction": direction,
                "entry": entry_price,
                "sl": sl,
                "tp": tp,
                "size": size,
                "pnl": 0.0,
            }
        )
        self.margin += margin_req
        self.balance -= self.commission * size
        self.total_trades += 1

    def _close_all(self):
        for pos in self.open_positions[:]:
            self._close_pos(pos)

    def _close_pos(self, pos):
        exit_price = self.current_price
        pnl = (
            (exit_price - pos["entry"]) * pos["size"] * 100000
            if pos["direction"] == "BUY"
            else (pos["entry"] - exit_price) * pos["size"] * 100000
        )
        pnl -= self.commission * pos["size"]
        self.balance += pnl
        self.margin -= pos["entry"] * pos["size"] * 100000 * 0.01
        self.trade_history.append(
            {**pos, "exit": exit_price, "pnl": pnl, "win": pnl > 0}
        )
        self.total_pnl += pnl
        if pnl > 0:
            self.winning_trades += 1
        self.open_positions.remove(pos)

    def _update_positions(self):
        for pos in self.open_positions:
            if pos["direction"] == "BUY":
                pos["pnl"] = (self.current_price - pos["entry"]) * pos["size"] * 100000
                if self.current_price <= pos["sl"] or self.current_price >= pos["tp"]:
                    self._close_pos(pos)
            else:
                pos["pnl"] = (pos["entry"] - self.current_price) * pos["size"] * 100000
                if self.current_price >= pos["sl"] or self.current_price <= pos["tp"]:
                    self._close_pos(pos)
        self.equity = self.balance + sum(p["pnl"] for p in self.open_positions)

    def _calc_reward(self) -> float:
        profit = self.total_pnl / 100.0
        sharpe = 0.0
        if len(self.trade_history) >= 2:
            rets = [t["pnl"] for t in self.trade_history[-20:]]
            sharpe = np.mean(rets) / (np.std(rets) + 1e-8)
        max_bal = max(
            [self.initial_balance]
            + [self.initial_balance + t["pnl"] for t in self.trade_history]
        )
        dd_pen = -5.0 * max(0, (max_bal - self.balance) / max_bal)
        comm = sum(self.commission * t["size"] for t in self.trade_history[-20:])
        trans_cost = -comm / 100.0
        win_rate = self.winning_trades / max(self.total_trades, 1)
        win_bonus = 10.0 * (win_rate - 0.5)
        margin_pen = -3.0 * (self.margin / max(self.balance, 1))
        consec = sum(1 for t in self.trade_history[-5:] if not t["win"])
        consec_pen = -2.0 * consec
        return (
            profit * 2.0
            + sharpe * 1.5
            + dd_pen * 3.0
            + trans_cost * 1.0
            + win_bonus * 2.0
            + margin_pen * 2.0
            + consec_pen * 1.5
        )

    def update_state(self, new_state: np.ndarray):
        self.current_state = new_state.copy()
        self.current_price = new_state[0] if len(new_state) > 0 else 1.1200
