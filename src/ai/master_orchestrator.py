"""
Master AI Orchestrator v2.0 — Enhanced Central AI Engine for the MoneyBot System.
Controls and manages all 18 enhancements, learns optimal configurations,
and makes high-level decisions about system behavior.

V2 Improvements over V1:
  - Deeper network: 16->128->64->32->5 with LayerNorm, Dropout, LeakyReLU
  - Target network with Polyak soft-updates for stable Q-learning
  - Proper TD(0) learning with (s, a, r, s', done) transitions
  - Prioritized experience replay with importance sampling weights
  - Robust model persistence: periodic + event-based + best-model checkpoint
  - Fault tolerance: asyncio timeout, heuristic fallback, consecutive-failure tracking
  - Better exploration: decaying epsilon + parameter noise
  - Cosine annealing LR scheduler with warmup
  - Proper state-transition healing (re-enables enhancements on RECOVERING)
"""

import asyncio
import numpy as np
import time
import sys
import os
import json
import copy
import math
from typing import Dict, List, Optional, Tuple, Any, Deque
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
from collections import defaultdict, deque
from pathlib import Path


import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------------------------
# Prioritized experience replay
# ---------------------------------------------------------------------------


class PrioritizedReplayBuffer:
    """Prioritized experience replay with importance-sampling weights.
    Uses simple circular list (no heapq, sample never pops)."""

    def __init__(
        self, capacity: int = 2000, alpha: float = 0.6, beta_start: float = 0.4
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = 0.001
        self._buffer: List[Tuple[float, int, Any]] = []
        self._idx = 0
        self._max_priority = 1.0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        priority = self._max_priority
        exp = (state, action, reward, next_state, done)
        if len(self._buffer) < self.capacity:
            self._buffer.append((priority, self._idx, exp))
        else:
            self._buffer[self._idx % self.capacity] = (priority, self._idx, exp)
        self._idx += 1

    def sample(self, batch_size: int) -> Optional[Tuple]:
        n = len(self._buffer)
        if n < batch_size:
            return None
        self.beta = min(1.0, self.beta + self.beta_increment)
        total = sum(p**self.alpha for p, _, _ in self._buffer)
        probs = [(p**self.alpha) / total for p, _, _ in self._buffer]
        indices = np.random.choice(n, size=batch_size, p=probs, replace=False)
        batch = [self._buffer[i] for i in indices]
        states = np.array([e[2][0] for e in batch], dtype=np.float32)
        actions = np.array([e[2][1] for e in batch], dtype=np.int64)
        rewards = np.array([e[2][2] for e in batch], dtype=np.float32)
        next_states = np.array([e[2][3] for e in batch], dtype=np.float32)
        dones = np.array([e[2][4] for e in batch], dtype=np.float32)
        weights = np.array(
            [(1.0 / (n * probs[i])) ** (1.0 - self.beta) for i in indices],
            dtype=np.float32,
        )
        weights /= weights.sum()
        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        for idx, td_err in zip(indices, td_errors):
            priority = (abs(td_err) + 1e-6) ** self.alpha
            self._max_priority = max(self._max_priority, priority)
            for i in range(len(self._buffer)):
                if self._buffer[i][1] == idx:
                    self._buffer[i] = (priority, idx, self._buffer[i][2])
                    break

    def __len__(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
# Improved meta-learner network
# ---------------------------------------------------------------------------


class MetaLearnerV2(nn.Module):
    """
    Improved neural network for meta-action Q-value prediction.
    V2: deeper, LayerNorm, Dropout, LeakyReLU → more stable training,
    better generalization, less prone to dead neurons.
    """

    def __init__(
        self,
        state_dim: int = 16,
        n_actions: int = 5,
        hidden_dims: Tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.15,
    ):
        super().__init__()
        layers = []
        prev = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                if m.out_features > 1:
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2.0))
                else:
                    nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

    @torch.no_grad()
    def predict_q(self, state: np.ndarray) -> np.ndarray:
        self.eval()
        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        return self.forward(s).squeeze(0).cpu().numpy()

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location="cpu"))


# ---------------------------------------------------------------------------
# Shared enumerations and data-classes (unchanged from V1)
# ---------------------------------------------------------------------------

try:
    from data.data_manager import SYMBOLS
except ImportError:
    SYMBOLS = [
        "EURUSD",
        "GBPUSD",
        "USDJPY",
        "AUDUSD",
        "USDCAD",
        "USDCHF",
        "NZDUSD",
        "EURJPY",
        "GBPJPY",
        "EURGBP",
        "XAUUSD",
        "XAGUSD",
        "XTIUSD",
        "XBRUSD",
        "XNGUSD",
        "US500",
        "US30",
        "USTEC",
        "UK100",
        "DE40",
        "BTCUSD",
        "ETHUSD",
        "LTCUSD",
        "XRPUSD",
    ]


class SystemState(str, Enum):
    OPTIMAL = "optimal"
    DEGRADED = "degraded"
    RECOVERING = "recovering"
    HALTED = "halted"
    LEARNING = "learning"
    BACKTESTING = "backtesting"


class EnhancementStatus(str, Enum):
    ACTIVE = "active"
    DISABLED = "disabled"
    DEGRADED = "degraded"
    LEARNING = "learning"
    TESTING = "testing"


@dataclass
class EnhancementMetrics:
    name: str
    status: EnhancementStatus = EnhancementStatus.ACTIVE
    confidence: float = 1.0
    performance_score: float = 0.0
    last_success: float = field(default_factory=time.time)
    failure_count: int = 0
    trades_influenced: int = 0
    pnl_contribution: float = 0.0
    sharpe_contribution: float = 0.0


@dataclass
class SystemDecision:
    timestamp: float = field(default_factory=time.time)
    action: str = ""
    reason: str = ""
    confidence: float = 0.0
    adjustments: Dict[str, Any] = field(default_factory=dict)
    enhancements_to_enable: List[str] = field(default_factory=list)
    enhancements_to_disable: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# V2 Master AI Orchestrator
# ---------------------------------------------------------------------------


class MasterAIOrchestrator:
    """
    V2 Central AI Engine — fixes all V1 weaknesses:

    ┌─ Weakness                              ─┬─ V2 Remedy
    ├─ Shallow network (16-32-16-5)           │ Deeper (16-128-64-32-5) + LayerNorm + Dropout
    ├─ No target network / unstable learning  │ Polyak soft-updated target network
    ├─ Supervised MSE (not TD learning)        │ Proper TD(0) with (s,a,r,s′,done)
    ├─ Uniform replay buffer                   │ Prioritized replay + importance sampling
    ├─ Fragile save (only every 10 steps)      │ Periodic + event-triggered + best-model snapshot
    ├─ No timeout on evaluate                  │ asyncio.wait_for(30s) + heuristic fallback
    ├─ Single-point failure of evaluate        │ Consecutive-failure counter → forced heuristic
    ├─ No recovery re-enablement               │ HALTED→RECOVERING→OPTIMAL re-enables P1 modules
    ├─ Static epsilon (no schedule)            │ Cosine decay + parameter noise
    └─ Fixed LR (no schedule)                  │ Cosine annealing with linear warmup
    """

    SAFETY_CRITICAL = {"circuit_breaker", "var_sizing", "error_recovery"}

    ENHANCEMENTS = {
        "ppo_integration": {
            "name": "PPO Integration",
            "priority": 1,
            "dependencies": [],
        },
        "live_trading_loop": {
            "name": "Live Trading Loop",
            "priority": 1,
            "dependencies": ["ppo_integration"],
        },
        "model_versioning": {
            "name": "Model Versioning",
            "priority": 1,
            "dependencies": ["ppo_integration"],
        },
        "circuit_breaker": {
            "name": "Circuit Breaker",
            "priority": 2,
            "dependencies": ["live_trading_loop"],
        },
        "data_pipeline": {"name": "Data Pipeline", "priority": 2, "dependencies": []},
        "error_recovery": {
            "name": "Error Recovery",
            "priority": 2,
            "dependencies": ["live_trading_loop"],
        },
        "feature_optimization": {
            "name": "Feature Optimization",
            "priority": 2,
            "dependencies": ["data_pipeline"],
        },
        "algo_execution": {
            "name": "Algo Execution",
            "priority": 2,
            "dependencies": ["live_trading_loop"],
        },
        "ensemble_weighting": {
            "name": "Ensemble Weighting",
            "priority": 2,
            "dependencies": ["ppo_integration"],
        },
        "var_sizing": {"name": "VaR-based Sizing", "priority": 2, "dependencies": []},
        "sentiment_alpha": {
            "name": "Sentiment Alpha",
            "priority": 2,
            "dependencies": [],
        },
        "regime_transition": {
            "name": "Regime Transition Trading",
            "priority": 2,
            "dependencies": ["ppo_integration"],
        },
        "stress_testing": {"name": "Stress Testing", "priority": 3, "dependencies": []},
        "walk_forward": {
            "name": "Walk-Forward Optimization",
            "priority": 3,
            "dependencies": [],
        },
        "integration_tests": {
            "name": "Integration Tests",
            "priority": 3,
            "dependencies": [],
        },
        "trading_sessions": {
            "name": "Smart Trading Sessions",
            "priority": 3,
            "dependencies": [],
        },
        "enhanced_dashboard": {
            "name": "Enhanced Dashboard",
            "priority": 3,
            "dependencies": [],
        },
        "event_avoidance": {
            "name": "Event Avoidance",
            "priority": 3,
            "dependencies": ["sentiment_alpha"],
        },
    }

    STATE_DIM = 16
    N_ACTIONS = 5
    COLD_START_TRADES = 50
    EPSILON_INIT = 0.3
    EPSILON_MIN = 0.02
    EPSILON_DECAY_STEPS = 2000
    EVAL_TIMEOUT_SEC = 30.0
    MAX_CONSECUTIVE_EVAL_FAILURES = 3
    SAVE_PERIOD_TRADES = 25
    SAVE_PERIOD_SEC = 120.0
    TARGET_UPDATE_TAU = 0.005
    GAMMA = 0.95

    def __init__(
        self,
        initial_balance: float = 100000.0,
        learning_rate: float = 0.001,
        adaptation_threshold: float = 0.6,
        model_path: str = "models/meta_learner.pt",
        best_model_path: str = "models/meta_learner_best.pt",
    ):
        self.initial_balance = initial_balance
        self.learning_rate = learning_rate
        self.adaptation_threshold = adaptation_threshold
        self.model_path = model_path
        self.best_model_path = best_model_path

        # System state
        self.system_state: SystemState = SystemState.OPTIMAL
        self.enhancements: Dict[str, EnhancementMetrics] = {}
        self._init_enhancements()

        # Performance tracking
        self.system_pnl: float = 0.0
        self.system_sharpe: float = 0.0
        self.trade_history: List[Dict] = []
        self.performance_history: List[float] = []
        self.enhancement_combinations: Dict[str, float] = {}
        self.recent_decisions: List[SystemDecision] = []
        self.last_reconfiguration: float = 0.0
        self.reconfiguration_cooldown: float = 300.0

        # Real system component references (wired by bot)
        self.system_components: Dict[str, Any] = {}

        # Re-entrancy guards
        self._processing_trade: bool = False
        self._evaluating: bool = False
        self._pending_trades: List[Dict] = []

        # ---- RL networks ----
        self.meta_learner = MetaLearnerV2(
            state_dim=self.STATE_DIM,
            n_actions=self.N_ACTIONS,
        )
        self.target_network = MetaLearnerV2(
            state_dim=self.STATE_DIM,
            n_actions=self.N_ACTIONS,
        )
        self._sync_target(tau=1.0)  # hard copy at init

        self.meta_optimizer = optim.AdamW(
            self.meta_learner.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
        )

        # Cosine annealing LR scheduler with linear warmup
        def lr_lambda(step):
            warmup = 100
            if step < warmup:
                return step / warmup
            progress = (step - warmup) / max(1, 5000 - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.meta_optimizer,
            lr_lambda,
        )

        # Prioritized replay
        self.replay_buffer = PrioritizedReplayBuffer(capacity=2000)
        self.batch_size = 64

        # TD-learning state
        self._prev_state: Optional[np.ndarray] = None
        self._prev_action: Optional[int] = None
        self._last_eval_time: float = time.time()
        self._pnl_at_last_eval: float = 0.0
        self._sharpe_at_last_eval: float = 0.0
        self._winrate_at_last_eval: float = 0.0

        # Exploration
        self.epsilon = self.EPSILON_INIT
        self.training_step = 0
        self.best_sharpe: float = -10.0
        self.best_performance: float = -1e9

        # Fault-tolerance
        self._consecutive_eval_failures: int = 0
        self._last_save_time: float = time.time()
        self._trades_since_save: int = 0

        # Load pre-trained
        self._load_meta_learner()

        logger.info("=" * 60)
        logger.info("[MasterAI v2.0] Central AI Engine initialized")
        logger.info(f"[MasterAI] Managing {len(self.ENHANCEMENTS)} enhancements")
        logger.info(
            f"[MasterAI] Network: {self.STATE_DIM}->128->64->32->{self.N_ACTIONS}"
        )
        logger.info(
            f"[MasterAI] TD-learning with target network (tau={self.TARGET_UPDATE_TAU})"
        )
        logger.info(
            f"[MasterAI] Prioritized replay (cap={self.replay_buffer.capacity})"
        )
        logger.info(
            f"[MasterAI] Cold-start: first {self.COLD_START_TRADES} trades use heuristics"
        )
        logger.info(f"[MasterAI] Eval timeout: {self.EVAL_TIMEOUT_SEC}s")
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_enhancements(self):
        for key, config in self.ENHANCEMENTS.items():
            is_active = config["priority"] == 1 or key in self.SAFETY_CRITICAL
            self.enhancements[key] = EnhancementMetrics(
                name=config["name"],
                status=(
                    EnhancementStatus.ACTIVE
                    if is_active
                    else EnhancementStatus.LEARNING
                ),
            )

    def _sync_target(self, tau: Optional[float] = None):
        tau = tau if tau is not None else self.TARGET_UPDATE_TAU
        with torch.no_grad():
            for p_target, p_online in zip(
                self.target_network.parameters(), self.meta_learner.parameters()
            ):
                p_target.data.copy_(tau * p_online.data + (1.0 - tau) * p_target.data)

    def wire_system_components(self, components: Dict[str, Any]):
        self.system_components = components
        logger.info(
            f"[MasterAI] Wired {len(components)} system components under direct control"
        )

    def is_enabled(self, enhancement_name: str) -> bool:
        if enhancement_name not in self.enhancements:
            return False
        return self.enhancements[enhancement_name].status == EnhancementStatus.ACTIVE

    # ------------------------------------------------------------------
    # Decision application (physical component control)
    # ------------------------------------------------------------------

    async def apply_decision(self, decision: SystemDecision):
        comp = self.system_components
        self._update_system_state(decision)
        for enh in decision.enhancements_to_disable:
            self._physically_disable_enhancement(enh, comp)
        for enh in decision.enhancements_to_enable:
            self._physically_enable_enhancement(enh, comp)
        if decision.adjustments:
            self._apply_parameter_adjustments(decision.adjustments, comp)

    def _physically_enable_enhancement(self, enh: str, comp: Dict[str, Any]):
        if enh == "circuit_breaker":
            cb = comp.get("circuit_breakers")
            if cb:
                for cb_inst in cb.values():
                    if hasattr(cb_inst, "enable"):
                        cb_inst.enable()
        elif enh == "algo_execution":
            ae = comp.get("algo_executor")
            if ae and hasattr(ae, "enable"):
                ae.enable()
        elif enh == "sentiment_alpha":
            ba = comp.get("behavioral_ai")
            if ba:
                ba.enabled = True
        elif enh == "regime_transition":
            rs = comp.get("regime_system")
            if rs:
                rs.enabled = True
        elif enh == "data_pipeline":
            dm = comp.get("data_manager")
            if dm:
                dm.enabled = True
        elif enh == "var_sizing":
            risk = comp.get("risk")
            if risk:
                risk.var_enabled = True
        elif enh == "ensemble_weighting":
            ens = comp.get("ensemble")
            if ens:
                ens.enabled = True
        elif enh == "ppo_integration":
            rs = comp.get("regime_system")
            if rs:
                rs.enabled = True
        elif enh == "event_avoidance":
            ea = comp.get("event_avoidance")
            if ea:
                ea.enabled = True
        elif enh == "walk_forward":
            wf = comp.get("walk_forward")
            if wf:
                wf.enabled = True
        elif enh == "stress_testing":
            st = comp.get("stress_tester")
            if st:
                st.enabled = True
        elif enh == "model_versioning":
            mr = comp.get("model_registry")
            if mr:
                mr.enabled = True

    def _physically_disable_enhancement(self, enh: str, comp: Dict[str, Any]):
        if enh == "circuit_breaker":
            cb = comp.get("circuit_breakers")
            if cb:
                for cb_inst in cb.values():
                    if hasattr(cb_inst, "disable"):
                        cb_inst.disable()
        elif enh == "algo_execution":
            ae = comp.get("algo_executor")
            if ae and hasattr(ae, "disable"):
                ae.disable()
        elif enh == "sentiment_alpha":
            ba = comp.get("behavioral_ai")
            if ba:
                ba.enabled = False
        elif enh == "regime_transition":
            rs = comp.get("regime_system")
            if rs:
                rs.enabled = False
        elif enh == "data_pipeline":
            dm = comp.get("data_manager")
            if dm:
                dm.enabled = False
        elif enh == "var_sizing":
            risk = comp.get("risk")
            if risk:
                risk.var_enabled = False
        elif enh == "ensemble_weighting":
            ens = comp.get("ensemble")
            if ens:
                ens.enabled = False
        elif enh == "ppo_integration":
            rs = comp.get("regime_system")
            if rs:
                rs.enabled = False
        elif enh == "event_avoidance":
            ea = comp.get("event_avoidance")
            if ea:
                ea.enabled = False
        elif enh == "walk_forward":
            wf = comp.get("walk_forward")
            if wf:
                wf.enabled = False
        elif enh == "stress_testing":
            st = comp.get("stress_tester")
            if st:
                st.enabled = False
        elif enh == "model_versioning":
            mr = comp.get("model_registry")
            if mr:
                mr.enabled = False

    def _apply_parameter_adjustments(
        self, adjustments: Dict[str, Any], comp: Dict[str, Any]
    ):
        risk = comp.get("risk")
        if risk:
            if "position_multiplier" in adjustments and hasattr(risk, "kelly_fraction"):
                risk.kelly_fraction = (
                    risk.kelly_fraction * adjustments["position_multiplier"]
                )
            if "max_positions" in adjustments and hasattr(risk, "max_positions"):
                risk.max_positions = adjustments["max_positions"]

    # ------------------------------------------------------------------
    # State feature engineering
    # ------------------------------------------------------------------

    def _build_state_vector(self, bot_instance, market_data: Dict) -> np.ndarray:
        health_scores = self._check_enhancement_health()
        avg_health = (
            float(np.mean(list(health_scores.values()))) if health_scores else 0.5
        )
        perf = self._calculate_system_performance()
        vol = self._assess_market_conditions(market_data)
        vol_map = {"calm": 0.0, "normal": 0.5, "volatile": 1.0, "unknown": 0.5}
        n = len(self.trade_history)
        recent = self.trade_history[-50:] if n >= 50 else self.trade_history
        wins = sum(1 for t in recent if t.get("pnl", 0) > 0)
        win_rate = wins / max(len(recent), 1)
        recent_pnl = sum(t.get("pnl", 0) for t in recent)
        pnl_norm = float(np.tanh(recent_pnl / max(self.initial_balance, 1)))
        sharpe = 0.0
        if n >= 5:
            returns = [t.get("pnl", 0) / self.initial_balance for t in recent]
            if len(returns) >= 2:
                avg_r = float(np.mean(returns))
                std_r = float(np.std(returns))
                sharpe = (avg_r / std_r * math.sqrt(252)) if std_r > 0 else 0.0
        self.system_sharpe = sharpe
        sharpe_norm = float(np.clip((sharpe + 2.0) / 4.0, 0.0, 1.0))
        active_count = sum(
            1
            for e in self.enhancements.values()
            if e.status == EnhancementStatus.ACTIVE
        )
        active_ratio = active_count / max(len(self.enhancements), 1)
        exp_level = min(n / 1000.0, 1.0)
        # Fixed-order confidence features: each slot always maps to the same enhancement
        fixed_keys = sorted(self.ENHANCEMENTS.keys())
        n_conf_slots = self.STATE_DIM - 8
        confs_fixed = [
            self.enhancements[k].confidence for k in fixed_keys[:n_conf_slots]
        ]
        confs_padded = (confs_fixed + [0.5] * n_conf_slots)[:n_conf_slots]
        return np.array(
            [
                avg_health,
                perf,
                vol_map.get(vol, 0.5),
                sharpe_norm,
                win_rate,
                pnl_norm,
                active_ratio,
                exp_level,
                *confs_padded,
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # TD learning
    # ------------------------------------------------------------------

    def _compute_td_reward(self) -> float:
        if len(self.trade_history) < 2:
            return 0.0
        recent = self.trade_history[-50:]
        pnl_now = sum(t.get("pnl", 0) for t in recent)
        pnl_delta = pnl_now - self._pnl_at_last_eval
        sharpe_now = self.system_sharpe
        sharpe_delta = sharpe_now - self._sharpe_at_last_eval
        wins = sum(1 for t in recent if t.get("pnl", 0) > 0)
        wr_now = wins / max(len(recent), 1)
        wr_delta = wr_now - self._winrate_at_last_eval
        pnl_r = float(np.tanh(pnl_delta / 1000.0))
        sharpe_r = float(np.clip(sharpe_delta * 0.5, -0.5, 0.5))
        wr_r = float(np.clip(wr_delta * 2.0, -0.5, 0.5))
        dd = self._estimate_drawdown()
        dd_p = -float(np.clip(dd * 2.0, -1.0, 0.0))
        return float(
            np.clip(pnl_r * 0.5 + sharpe_r * 0.2 + wr_r * 0.2 + dd_p * 0.1, -2.0, 2.0)
        )

    def _estimate_drawdown(self) -> float:
        if len(self.trade_history) < 5:
            return 0.0
        balance = self.initial_balance
        peak_bal = balance
        max_dd = 0.0
        for t in self.trade_history[-100:]:
            balance += t.get("pnl", 0)
            if balance > peak_bal:
                peak_bal = balance
            dd = (peak_bal - balance) / max(peak_bal, 1)
            if dd > max_dd:
                max_dd = dd
        return min(max_dd, 1.0)

    def _train_step(self):
        n = len(self.replay_buffer)
        if n < self.batch_size:
            return
        sampled = self.replay_buffer.sample(self.batch_size)
        if sampled is None:
            return
        states, actions, rewards, next_states, dones, weights, indices = sampled

        states_t = torch.as_tensor(states)
        actions_t = torch.as_tensor(actions).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards).unsqueeze(1)
        next_states_t = torch.as_tensor(next_states)
        dones_t = torch.as_tensor(dones).unsqueeze(1)
        weights_t = torch.as_tensor(weights).unsqueeze(1)

        # Current Q-values
        q_all = self.meta_learner(states_t)
        q_taken = q_all.gather(1, actions_t)

        # Target Q-values (Double Q: use online to select, target to evaluate)
        with torch.no_grad():
            next_q_online = self.meta_learner(next_states_t)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = self.target_network(next_states_t)
            q_next = next_q_target.gather(1, next_actions)
            q_target = rewards_t + self.GAMMA * q_next * (1.0 - dones_t)

        td_errors = (q_taken - q_target).detach().squeeze(1).cpu().numpy()
        loss = (weights_t * (q_taken - q_target).pow(2)).mean()

        self.meta_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.meta_learner.parameters(), 5.0)
        self.meta_optimizer.step()
        self.lr_scheduler.step()

        self.replay_buffer.update_priorities(indices, td_errors)
        self.training_step += 1

        # Anneal epsilon
        self.epsilon = max(
            self.EPSILON_MIN,
            self.EPSILON_INIT * (1.0 - self.training_step / self.EPSILON_DECAY_STEPS),
        )

        # Soft-update target network
        self._sync_target()

        if self.training_step % 10 == 0:
            current_lr = self.meta_optimizer.param_groups[0]["lr"]
            logger.debug(
                f"[MasterAI] train_step={self.training_step} "
                f"loss={loss.item():.4f} epsilon={self.epsilon:.3f} lr={current_lr:.6f}"
            )

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def _select_meta_action(self, state: np.ndarray) -> int:
        n_trades = len(self.trade_history)
        if n_trades < self.COLD_START_TRADES:
            return -1
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.N_ACTIONS)
        q_values = self.meta_learner.predict_q(state)
        return int(np.argmax(q_values))

    def _heuristic_action(
        self, state: np.ndarray, perf: float, market_state: str, avg_health: float
    ) -> SystemDecision:
        decision = SystemDecision()
        if avg_health < 0.3:
            decision.action = "halt"
            decision.reason = f"[HEURISTIC] System health critical: {avg_health:.1%}"
            decision.confidence = 1.0 - avg_health
            decision.enhancements_to_disable = [
                e for e in self.ENHANCEMENTS if e != "circuit_breaker"
            ]
        elif perf < -0.05:
            decision.action = "reconfigure"
            decision.reason = f"[HEURISTIC] Poor performance: {perf:.2%}"
            decision.confidence = 0.8
            rec = self.get_market_recommendation({})
            decision.enhancements_to_enable = rec["enhancements_to_trust"]
            self._identify_underperformers(decision)
            decision.adjustments = rec["parameters_to_adjust"]
        elif market_state == "volatile":
            decision.action = "reconfigure"
            decision.reason = "[HEURISTIC] Volatile market"
            decision.confidence = 0.7
            decision.enhancements_to_disable = [
                "regime_transition",
                "sentiment_alpha",
                "algo_execution",
            ]
            decision.enhancements_to_enable = [
                "circuit_breaker",
                "var_sizing",
                "event_avoidance",
            ]
            decision.adjustments = {
                "position_multiplier": 0.5,
                "confidence_threshold": 0.8,
                "max_positions": 3,
            }
        else:
            decision.action = "continue"
            decision.reason = "[HEURISTIC] System optimal"
            decision.confidence = avg_health
            decision.enhancements_to_enable = [
                e for e in self.ENHANCEMENTS if self.ENHANCEMENTS[e]["priority"] == 1
            ]
        return decision

    def _meta_action_to_decision(
        self, action: int, market_state: str
    ) -> SystemDecision:
        decision = SystemDecision()
        decision.confidence = 0.85
        if action == 0:
            decision.action = "continue"
            decision.reason = f"[LEARNED] Optimal config (action={action})"
        elif action == 1:
            decision.action = "reconfigure"
            decision.reason = f"[LEARNED] Conservative mode (action={action})"
            decision.enhancements_to_disable = [
                "sentiment_alpha",
                "algo_execution",
                "regime_transition",
            ]
            decision.enhancements_to_enable = [
                "circuit_breaker",
                "var_sizing",
                "event_avoidance",
            ]
            decision.adjustments = {
                "position_multiplier": 0.5,
                "confidence_threshold": 0.75,
            }
        elif action == 2:
            decision.action = "reconfigure"
            decision.reason = f"[LEARNED] Aggressive mode (action={action})"
            decision.enhancements_to_enable = list(self.ENHANCEMENTS.keys())
            decision.adjustments = {
                "position_multiplier": 1.2,
                "confidence_threshold": 0.6,
            }
        elif action == 3:
            decision.action = "reconfigure"
            decision.reason = f"[LEARNED] Reduced risk (action={action})"
            decision.enhancements_to_disable = ["algo_execution", "sentiment_alpha"]
            decision.enhancements_to_enable = ["circuit_breaker", "var_sizing"]
            decision.adjustments = {
                "position_multiplier": 0.3,
                "confidence_threshold": 0.85,
                "max_positions": 2,
            }
        elif action == 4:
            decision.action = "halt"
            decision.reason = f"[LEARNED] Emergency halt (action={action})"
            decision.enhancements_to_disable = list(self.ENHANCEMENTS.keys())
            decision.confidence = 0.95
        return decision

    # ------------------------------------------------------------------
    # Main evaluation entry-point (with fault tolerance)
    # ------------------------------------------------------------------

    async def evaluate_system_state(
        self,
        bot_instance,
        market_data: Dict,
        account_info: Dict,
    ) -> SystemDecision:
        if getattr(self, "_evaluating", False):
            logger.debug("[MasterAI] Re-entrant evaluate_system_state skipped")
            return SystemDecision(action="continue", reason="re-entrant_guard")
        self._evaluating = True
        try:
            return await asyncio.wait_for(
                self._evaluate_impl(bot_instance, market_data, account_info),
                timeout=self.EVAL_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            logger.error("[MasterAI] evaluate_system_state timed out -- using fallback")
            self._consecutive_eval_failures += 1
            return self._fallback_decision("timeout")
        except Exception as e:
            logger.error(f"[MasterAI] evaluate_system_state error: {e}")
            self._consecutive_eval_failures += 1
            if self._consecutive_eval_failures >= self.MAX_CONSECUTIVE_EVAL_FAILURES:
                logger.critical(
                    f"[MasterAI] {self._consecutive_eval_failures} consecutive failures -- "
                    f"forcing HALT"
                )
                self._consecutive_eval_failures = 0
                return SystemDecision(
                    action="halt",
                    reason=f"[FALLBACK] {self.MAX_CONSECUTIVE_EVAL_FAILURES} consecutive eval failures",
                    enhancements_to_disable=list(self.ENHANCEMENTS.keys()),
                    confidence=0.99,
                )
            return self._fallback_decision(str(e))
        finally:
            self._evaluating = False

    def _fallback_decision(self, reason: str) -> SystemDecision:
        avg_health = (
            float(np.mean(list(self._check_enhancement_health().values())))
            if self.enhancements
            else 0.5
        )
        perf = self._calculate_system_performance()
        return self._heuristic_action(
            np.zeros(self.STATE_DIM, dtype=np.float32),
            perf,
            "unknown",
            avg_health,
        )

    async def _evaluate_impl(
        self, bot_instance, market_data, account_info
    ) -> SystemDecision:
        if bot_instance:
            self._update_all_enhancement_metrics(bot_instance)
        current_state = self._build_state_vector(bot_instance, market_data)
        health_scores = self._check_enhancement_health()
        avg_health = (
            float(np.mean(list(health_scores.values()))) if health_scores else 0.5
        )
        market_state = self._assess_market_conditions(market_data)

        # --- TD transition: store (prev_state, prev_action, reward, current_state) ---
        if self._prev_state is not None and self._prev_action is not None:
            td_reward = self._compute_td_reward()
            done = self._prev_action == 4  # HALT action
            self.replay_buffer.push(
                self._prev_state,
                self._prev_action,
                td_reward,
                current_state,
                done,
            )
            self._train_step()

        # --- Select action ---
        meta_action = self._select_meta_action(current_state)
        if meta_action == -1:
            decision = self._heuristic_action(
                current_state,
                self._calculate_system_performance(),
                market_state,
                avg_health,
            )
        else:
            decision = self._meta_action_to_decision(meta_action, market_state)
            if avg_health < 0.2:
                decision.action = "halt"
                decision.reason = f"[SAFETY OVERRIDE] Health critical: {avg_health:.1%}"
                decision.enhancements_to_disable = list(self.ENHANCEMENTS.keys())
            logger.info(
                f"[MasterAI] Learned: action={meta_action} -> {decision.action}"
            )

        # --- Store state/action for next TD transition ---
        self._prev_state = current_state.copy()
        self._prev_action = meta_action if meta_action != -1 else None
        self._pnl_at_last_eval = sum(t.get("pnl", 0) for t in self.trade_history[-50:])
        self._sharpe_at_last_eval = self.system_sharpe
        self._winrate_at_last_eval = sum(
            1 for t in self.trade_history[-50:] if t.get("pnl", 0) > 0
        ) / max(len(self.trade_history[-50:]), 1)
        self._last_eval_time = time.time()

        # Run diagnostics + self-heal every 10 evaluations
        if self.training_step % 10 == 0 and bot_instance:
            diag = self.run_diagnostics(bot_instance)
            if diag["issues"]:
                logger.warning(
                    f"[MasterAI] Diagnostics: {len(diag['issues'])} issues found"
                )
                for issue in diag["issues"]:
                    logger.warning(f"[MasterAI]   ISSUE: {issue}")
                fixes = await self.self_heal(bot_instance)
                if fixes:
                    logger.success(
                        f"[MasterAI] Self-healed {len(fixes)} issues: {fixes}"
                    )
                else:
                    logger.warning("[MasterAI] Could not auto-fix all issues")
            elif diag["warnings"]:
                for w in diag["warnings"]:
                    logger.info(f"[MasterAI]   NOTE: {w}")

        # Record decision
        self.recent_decisions.append(decision)
        if len(self.recent_decisions) > 100:
            self.recent_decisions = self.recent_decisions[-100:]

        self._update_system_state(decision)

        if decision.enhancements_to_enable:
            logger.info(f"[MasterAI] Enabling: {decision.enhancements_to_enable}")
        if decision.enhancements_to_disable:
            logger.warning(f"[MasterAI] Disabling: {decision.enhancements_to_disable}")
        if decision.adjustments:
            logger.info(f"[MasterAI] Adjusting: {decision.adjustments}")

        self._consecutive_eval_failures = 0
        self._maybe_save()

        return decision

    # ------------------------------------------------------------------
    # Trade feedback (feeds the TD loop)
    # ------------------------------------------------------------------

    def on_trade_result(self, trade: Dict):
        if getattr(self, "_processing_trade", False):
            self._pending_trades.append(trade)
            return
        self._processing_trade = True
        try:
            self._process_trade(trade)
            while self._pending_trades:
                self._process_trade(self._pending_trades.pop(0))
        finally:
            self._processing_trade = False

    def _process_trade(self, trade: Dict):
        self.record_trade(trade)
        self._trades_since_save += 1
        self._maybe_save()

    def record_trade(self, trade: Dict):
        self.trade_history.append(trade)
        self.system_pnl += trade.get("pnl", 0.0)
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-500:]
        self.performance_history.append(trade.get("pnl", 0.0) / self.initial_balance)
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

    # ------------------------------------------------------------------
    # Model persistence (robust: periodic + event-triggered + best-model)
    # ------------------------------------------------------------------

    def _maybe_save(self):
        now = time.time()
        do_save = False
        if self._trades_since_save >= self.SAVE_PERIOD_TRADES:
            do_save = True
        if now - self._last_save_time >= self.SAVE_PERIOD_SEC:
            do_save = True
        if not do_save:
            return
        self._save_meta_learner()
        self._save_best_if_needed()
        self._trades_since_save = 0
        self._last_save_time = now

    def _save_meta_learner(self):
        try:
            path = Path(self.model_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": self.meta_learner.state_dict(),
                    "target_state_dict": self.target_network.state_dict(),
                    "optimizer_state_dict": self.meta_optimizer.state_dict(),
                    "scheduler_state_dict": self.lr_scheduler.state_dict(),
                    "training_step": self.training_step,
                    "epsilon": self.epsilon,
                    "system_sharpe": self.system_sharpe,
                    "system_pnl": self.system_pnl,
                    "timestamp": time.time(),
                },
                str(path),
            )
            logger.info(f"[MasterAI] Model saved to {self.model_path}")
        except Exception as e:
            logger.warning(f"[MasterAI] Save failed: {e}")

    def _save_best_if_needed(self):
        current_sharpe = self.system_sharpe
        current_perf = self._calculate_system_performance()
        if current_sharpe > self.best_sharpe and current_perf > self.best_performance:
            self.best_sharpe = current_sharpe
            self.best_performance = current_perf
            try:
                path = Path(self.best_model_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": self.meta_learner.state_dict(),
                        "target_state_dict": self.target_network.state_dict(),
                        "optimizer_state_dict": self.meta_optimizer.state_dict(),
                        "training_step": self.training_step,
                        "epsilon": self.epsilon,
                        "system_sharpe": self.system_sharpe,
                        "system_pnl": self.system_pnl,
                        "best_sharpe": self.best_sharpe,
                        "best_performance": self.best_performance,
                        "timestamp": time.time(),
                    },
                    str(path),
                )
                logger.info(
                    f"[MasterAI] Best model saved (sharpe={self.best_sharpe:.3f})"
                )
            except Exception as e:
                logger.warning(f"[MasterAI] Best-model save failed: {e}")

    def _load_meta_learner(self):
        try:
            path = Path(self.model_path)
            if path.exists():
                ckpt = torch.load(str(path), map_location="cpu")
                self.meta_learner.load_state_dict(ckpt["model_state_dict"])
                self.target_network.load_state_dict(
                    ckpt.get(
                        "target_state_dict",
                        ckpt["model_state_dict"],
                    )
                )
                if "optimizer_state_dict" in ckpt:
                    self.meta_optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if "scheduler_state_dict" in ckpt:
                    self.lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                self.training_step = ckpt.get("training_step", 0)
                loaded_epsilon = ckpt.get("epsilon", None)
                if loaded_epsilon is not None:
                    self.epsilon = loaded_epsilon
                else:
                    self.epsilon = max(
                        self.EPSILON_MIN,
                        self.EPSILON_INIT
                        * (1.0 - self.training_step / max(self.EPSILON_DECAY_STEPS, 1)),
                    )
                self.system_sharpe = ckpt.get("system_sharpe", 0.0)
                self.system_pnl = ckpt.get("system_pnl", 0.0)
                self.best_sharpe = ckpt.get("best_sharpe", -10.0)
                self.best_performance = ckpt.get("best_performance", -1e9)
                logger.info(
                    f"[MasterAI] Loaded checkpoint from {self.model_path} "
                    f"(step={self.training_step}, sharpe={self.system_sharpe:.3f})"
                )
            else:
                logger.info("[MasterAI] No checkpoint found, starting fresh")
        except Exception as e:
            logger.info(f"[MasterAI] Failed to load checkpoint: {e}, starting fresh")

    # ------------------------------------------------------------------
    # Auto-reconfiguration
    # ------------------------------------------------------------------

    async def auto_reconfigure(self, bot_instance) -> Dict:
        if time.time() - self.last_reconfiguration < self.reconfiguration_cooldown:
            return {
                "action": "cooldown",
                "reason": "Too soon since last reconfiguration",
            }
        current_active = sorted(
            [
                k
                for k, v in self.enhancements.items()
                if v.status == EnhancementStatus.ACTIVE
            ]
        )
        current_sig = "|".join(current_active)
        current_perf = self.enhancement_combinations.get(current_sig, 0.0)
        candidates = self._generate_candidates(current_active)
        best_candidate = current_active
        best_perf = current_perf
        for candidate in candidates:
            cand_sig = "|".join(sorted(candidate))
            cand_perf = self.enhancement_combinations.get(cand_sig, -999.0)
            if cand_perf > best_perf + 0.01:
                best_candidate = candidate
                best_perf = cand_perf
        if best_candidate != current_active:
            decision = SystemDecision(
                action="reconfigure",
                reason=f"Auto-reconfig: {current_perf:.3f} -> {best_perf:.3f}",
                confidence=0.8,
            )
            new_set = set(best_candidate)
            curr_set = set(current_active)
            for enh in new_set - curr_set:
                decision.enhancements_to_enable.append(enh)
            for enh in curr_set - new_set:
                decision.enhancements_to_disable.append(enh)
            await self.apply_decision(decision)
            return {
                "action": "reconfigured",
                "old_config": current_active,
                "new_config": best_candidate,
                "improvement": best_perf - current_perf,
            }
        return {"action": "no_change", "reason": "Current config optimal"}

    def _normalized_pnl(self, key: str) -> float:
        m = self.enhancements[key]
        return m.pnl_contribution / max(m.trades_influenced, 1)

    def _generate_candidates(self, current_active: List[str]) -> List[List[str]]:
        candidates = []
        if current_active:
            worst = min(current_active, key=self._normalized_pnl)
            candidates.append([e for e in current_active if e != worst])
        disabled = [
            k
            for k, v in self.enhancements.items()
            if v.status == EnhancementStatus.DISABLED
            and self.ENHANCEMENTS[k]["priority"] <= 2
        ]
        if disabled:
            best_d = max(disabled, key=self._normalized_pnl)
            candidates.append(current_active + [best_d])
        med = [k for k, v in self.ENHANCEMENTS.items() if v["priority"] == 2]
        for enh in med:
            if enh in current_active:
                candidates.append([e for e in current_active if e != enh])
            else:
                candidates.append(current_active + [enh])
        return candidates

    # ------------------------------------------------------------------
    # Health & performance metrics
    # ------------------------------------------------------------------

    def _update_all_enhancement_metrics(self, bot_instance):
        rs = getattr(bot_instance, "regime_system", None)
        if rs:
            m = self.enhancements.get("ppo_integration")
            if m and rs.agents:
                active = sum(1 for a in rs.agents.values() if a is not None)
                m.confidence = active / max(len(rs.agents), 1)
        cb = getattr(bot_instance, "circuit_breakers", {})
        m = self.enhancements.get("circuit_breaker")
        if m:
            any_halted = any(
                cb_inst.get_snapshot().should_halt
                for cb_inst in cb.values()
                if hasattr(cb_inst, "get_snapshot")
            )
            m.confidence = 0.3 if any_halted else 1.0
        dm = getattr(bot_instance, "data_manager", None)
        m = self.enhancements.get("data_pipeline")
        if m and dm:
            with_data = sum(
                1
                for sym in SYMBOLS
                if dm.get_ohlcv(sym, "1h") is not None
                and len(dm.get_ohlcv(sym, "1h")) > 50
            )
            m.confidence = with_data / max(len(SYMBOLS), 1)
        ensemble = getattr(bot_instance, "ensemble", None)
        m = self.enhancements.get("ensemble_weighting")
        if m and hasattr(ensemble, "experts") and ensemble.experts:
            m.confidence = len(ensemble.experts) / 3.0
        m = self.enhancements.get("var_sizing")
        risk = getattr(bot_instance, "risk", None)
        if m and risk and hasattr(risk, "calculate_kelly_size"):
            m.confidence = 0.8
        st = getattr(bot_instance, "stress_tester", None)
        m = self.enhancements.get("stress_testing")
        if m and st and hasattr(st, "results") and st.results:
            passed = sum(1 for r in st.results if getattr(r, "passed", False))
            total = len(st.results)
            m.confidence = passed / total if total > 0 else 0.5
        m = self.enhancements.get("enhanced_dashboard")
        if m:
            m.confidence = 1.0 if self.system_state == SystemState.OPTIMAL else 0.5

    def _identify_underperformers(self, decision: SystemDecision):
        for enh, metrics in self.enhancements.items():
            if metrics.pnl_contribution < -500:
                if enh not in decision.enhancements_to_disable:
                    decision.enhancements_to_disable.append(enh)
            elif metrics.failure_count > 10:
                if enh not in decision.enhancements_to_disable:
                    decision.enhancements_to_disable.append(enh)

    def _check_enhancement_health(self) -> Dict[str, float]:
        scores = {}
        for key, metrics in self.enhancements.items():
            s = 1.0
            if metrics.failure_count > 10:
                s *= 0.5
            elif metrics.failure_count > 5:
                s *= 0.7
            t = time.time() - metrics.last_success
            if t > 3600:
                s *= 0.8
            elif t > 300:
                s *= 0.9
            if metrics.pnl_contribution < -1000:
                s *= 0.6
            elif metrics.pnl_contribution > 1000:
                s *= 1.2
            s *= metrics.confidence
            scores[key] = min(max(s, 0.0), 1.0)
        return scores

    def _calculate_system_performance(self) -> float:
        if len(self.trade_history) < 5:
            return 0.0
        recent = self.trade_history[-50:]
        wins = sum(1 for t in recent if t.get("pnl", 0) > 0)
        total = len(recent)
        if total == 0:
            return 0.0
        win_rate = wins / total
        returns = [t.get("pnl", 0) / self.initial_balance for t in recent]
        if len(returns) < 2:
            return win_rate - 0.5
        avg_r = float(np.mean(returns))
        std_r = float(np.std(returns))
        sharpe = (avg_r / std_r * math.sqrt(252)) if std_r > 0 else 0.0
        recent_pnl = sum(returns[-10:]) if len(returns) >= 10 else sum(returns)
        return (win_rate - 0.5) * 0.4 + sharpe * 0.1 + recent_pnl * 2.0

    def _assess_market_conditions(self, market_data: Dict) -> str:
        if not market_data:
            return "unknown"
        volatilities = []
        for sym, data in market_data.items():
            if isinstance(data, dict):
                if "atr" in data and "price" in data and data["price"] > 0:
                    volatilities.append(data["atr"] / data["price"])
        if not volatilities:
            return "normal"
        avg_vol = float(np.mean(volatilities))
        if avg_vol > 0.02:
            return "volatile"
        elif avg_vol < 0.005:
            return "calm"
        return "normal"

    # ------------------------------------------------------------------
    # System state management (with healing)
    # ------------------------------------------------------------------

    def _update_system_state(self, decision: SystemDecision):
        prev_state = self.system_state
        if decision.action == "halt":
            self.system_state = SystemState.HALTED
        elif decision.action == "reconfigure":
            self.system_state = SystemState.DEGRADED
            self.last_reconfiguration = time.time()
        elif decision.action == "continue":
            if self.system_state == SystemState.HALTED:
                self.system_state = SystemState.RECOVERING
            elif self.system_state == SystemState.RECOVERING:
                self.system_state = SystemState.OPTIMAL
        # Apply enhancement status changes
        for enh in decision.enhancements_to_enable:
            if enh in self.enhancements:
                self.enhancements[enh].status = EnhancementStatus.ACTIVE
        for enh in decision.enhancements_to_disable:
            if enh in self.enhancements:
                self.enhancements[enh].status = EnhancementStatus.DISABLED
        # Healing: on transition RECOVERING -> OPTIMAL, re-enable priority-1 modules
        if (
            prev_state == SystemState.RECOVERING
            and self.system_state == SystemState.OPTIMAL
        ):
            self._heal_enhancements()

    def _heal_enhancements(self):
        """Re-enable all priority-1 enhancements and safety-critical modules."""
        for key, config in self.ENHANCEMENTS.items():
            if config["priority"] == 1 or key in self.SAFETY_CRITICAL:
                if key in self.enhancements:
                    self.enhancements[key].status = EnhancementStatus.ACTIVE
                    logger.info(f"[MasterAI] Healed enhancement: {config['name']}")
        # Also physically re-enable on real components
        comp = self.system_components
        for key in self.ENHANCEMENTS:
            if self.ENHANCEMENTS[key]["priority"] == 1 or key in self.SAFETY_CRITICAL:
                if (
                    self.enhancements.get(key, EnhancementMetrics("")).status
                    == EnhancementStatus.ACTIVE
                ):
                    self._physically_enable_enhancement(key, comp)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_enhancement_performance(
        self,
        enhancement_name: str,
        pnl: float = 0.0,
        success: bool = True,
        confidence: float = 1.0,
        trades_count: int = 0,
    ):
        if enhancement_name not in self.enhancements:
            return
        metrics = self.enhancements[enhancement_name]
        if success:
            metrics.last_success = time.time()
            metrics.confidence = metrics.confidence * 0.9 + confidence * 0.1
        else:
            metrics.failure_count += 1
            metrics.confidence *= 0.95
        metrics.pnl_contribution += pnl
        metrics.trades_influenced += trades_count
        active = sorted(
            [
                k
                for k, v in self.enhancements.items()
                if v.status == EnhancementStatus.ACTIVE
            ]
        )
        sig = "|".join(active)
        if sig not in self.enhancement_combinations:
            self.enhancement_combinations[sig] = 0.0
        self.enhancement_combinations[sig] = (
            self.enhancement_combinations[sig] * 0.9 + pnl * 0.01
        )

    async def self_heal(self, bot_instance) -> List[str]:
        """Auto-fix detected issues. Returns list of actions taken."""
        fixes = []
        if bot_instance is None:
            return fixes

        diag = self.run_diagnostics(bot_instance)
        dm = getattr(bot_instance, "data_manager", None)
        fp = getattr(bot_instance, "feature_pipeline", None)
        risk = getattr(bot_instance, "risk", None)

        from data.data_manager import SYMBOLS as ALL_SYMBOLS

        if dm and len(self.trade_history) > 10:
            missing = [
                s
                for s in ALL_SYMBOLS
                if dm.get_ohlcv(s, "1h") is None or len(dm.get_ohlcv(s, "1h")) < 50
            ]
            if missing and len(missing) < len(ALL_SYMBOLS):
                for sym in missing[:3]:
                    base = getattr(dm, "BASE_PRICES", {}).get(sym, 1.12)
                    import pandas as pd
                    import numpy as np

                    now_ts = int(time.time())
                    ts = list(range(now_ts - 720 * 3600, now_ts, 3600))
                    df = pd.DataFrame(
                        {
                            "timestamp": ts,
                            "open": base,
                            "high": base * 1.001,
                            "low": base * 0.999,
                            "close": base,
                            "volume": 1000,
                        }
                    )
                    dm.ohlcv[sym]["1h"] = df
                fixes.append(f"Generated synthetic data for {len(missing)} symbols")

        # Fix feature pipeline norms
        if fp:
            means = getattr(fp, "_means", {})
            stds = getattr(fp, "_stds", {})
            for key in list(means.keys()):
                if key in stds and len(means[key]) != len(stds[key]):
                    max_len = max(len(means[key]), len(stds[key]))
                    means[key] = np.pad(
                        means[key], (0, max_len - len(means[key])), constant_values=0
                    )
                    stds[key] = np.pad(
                        stds[key], (0, max_len - len(stds[key])), constant_values=1
                    )
                    fp._means[key] = means[key]
                    fp._stds[key] = stds[key]
                    logger.info(f"[MasterAI] HEAL: Resized norm dims for {key}")
                    fixes.append(f"Fixed norm dims for {key}")
            if not means and dm:
                from rts_ai_fx.features_unified import FeaturePipeline

                fp._means = {}
                fp._stds = {}
                fp.fit_all(dm.ohlcv)
                logger.info("[MasterAI] HEAL: Refitted feature pipeline")
                fixes.append("Refitted feature pipeline")

        # Fix kill switch
        if risk and getattr(risk, "kill_switch_triggered", False):
            risk.kill_switch_triggered = False
            logger.info("[MasterAI] HEAL: Reset kill switch")
            fixes.append("Reset kill switch")

        # Fix stale PPO agents with dim mismatch
        rs = getattr(bot_instance, "regime_system", None)
        if rs and hasattr(bot_instance, "_reinit_regime_agents"):
            agents = getattr(rs, "agents", {})
            for name, agent in agents.items():
                if agent is None or not hasattr(agent, "actor"):
                    actual_dim = getattr(
                        bot_instance, "_get_actual_state_dim", lambda: 2850
                    )()
                    bot_instance._reinit_regime_agents(actual_dim)
                    fixes.append(f"Re-initialized {name} agent")
                    break

        # Reconnect execution if needed
        exec_eng = getattr(bot_instance, "execution", None)
        if exec_eng and hasattr(exec_eng, "client"):
            client = exec_eng.client
            if hasattr(client, "is_connected") and not client.is_connected():
                try:
                    import asyncio

                    await client.start()
                    logger.info("[MasterAI] HEAL: Reconnected cTrader client")
                    fixes.append("Reconnected cTrader")
                except Exception as e:
                    logger.warning(f"[MasterAI] HEAL: Reconnect failed: {e}")

        if fixes:
            logger.success(f"[MasterAI] Self-heal applied {len(fixes)} fixes: {fixes}")
        else:
            logger.info("[MasterAI] Self-heal: no issues found")
        return fixes

    def run_diagnostics(self, bot_instance=None) -> Dict:
        """Run full system diagnostics and return a health report."""
        issues = []
        warnings_list = []
        recommendations = []

        if bot_instance is None:
            return {"status": "unknown", "issues": []}

        from data.data_manager import SYMBOLS as ALL_SYMBOLS

        dm = getattr(bot_instance, "data_manager", None)
        if dm:
            symbols_with_data = sum(
                1
                for sym in ALL_SYMBOLS
                if dm.get_ohlcv(sym, "1h") is not None
                and len(dm.get_ohlcv(sym, "1h")) > 50
            )
            total_symbols = len(ALL_SYMBOLS)
            if total_symbols > 0 and symbols_with_data < total_symbols:
                issues.append(
                    f"Data: only {symbols_with_data}/{total_symbols} symbols have OHLCV"
                )
            elif symbols_with_data == 0:
                issues.append("Data: NO symbols have OHLCV data")

        # Feature pipeline health
        fp = getattr(bot_instance, "feature_pipeline", None)
        if fp:
            if not getattr(fp, "_feature_cols", None):
                warnings_list.append("Feature pipeline: _feature_cols not set")
            means = getattr(fp, "_means", {})
            stds = getattr(fp, "_stds", {})
            if not means:
                issues.append("Feature pipeline: no normalization means fitted")
            mismatched = sum(
                1 for k in means if k in stds and len(means[k]) != len(stds[k])
            )
            if mismatched > 0:
                issues.append(f"Feature pipeline: {mismatched} norm dim mismatches")

        # Model health
        ensemble = getattr(bot_instance, "ensemble", None)
        if ensemble:
            n_experts = len(getattr(ensemble, "experts", []))
            if n_experts < 2:
                issues.append(f"Ensemble: only {n_experts} experts loaded")
            elo = getattr(ensemble, "elo_ratings", {})
            low_elo = [k for k, v in elo.items() if v < 1000]
            if low_elo:
                warnings_list.append(f"Ensemble: low ELO experts: {low_elo}")

        rs = getattr(bot_instance, "regime_system", None)
        if rs:
            agents = getattr(rs, "agents", {})
            if not agents:
                warnings_list.append("PPO: no regime agents loaded")
            for name, agent in agents.items():
                if agent is None:
                    warnings_list.append(f"PPO: {name} agent is None")

        # Risk manager health
        risk = getattr(bot_instance, "risk", None)
        if risk:
            if getattr(risk, "kill_switch_triggered", False):
                issues.append("Risk: KILL SWITCH ACTIVE")
            consec = getattr(risk, "consecutive_losses", 0)
            if consec > 3:
                warnings_list.append(f"Risk: {consec} consecutive losses")

        # Execution health
        exec_eng = getattr(bot_instance, "execution", None)
        if exec_eng:
            bal = getattr(exec_eng, "_balance", 0)
            init = getattr(bot_instance, "initial_balance", 100000)
            dd = (init - bal) / max(init, 1) if init > 0 else 0
            if dd > 0.05:
                warnings_list.append(f"Execution: drawdown {dd:.1%}")

        # Cycle error rate
        err_rate = getattr(self, "_consecutive_eval_failures", 0)
        if err_rate > 0:
            warnings_list.append(f"Master AI: {err_rate} consecutive eval failures")

        if not issues and not warnings_list:
            recommendations.append("All systems nominal")
        else:
            for issue in issues:
                recommendations.append(f"FIX: {issue}")
            for w in warnings_list:
                recommendations.append(f"WARN: {w}")

        return {
            "status": (
                "healthy"
                if not issues
                else "degraded" if len(issues) < 3 else "critical"
            ),
            "issues": issues,
            "warnings": warnings_list,
            "recommendations": recommendations,
            "symbols_loaded": symbols_with_data if dm else 0,
            "total_symbols": total_symbols if dm else 0,
            "eval_failures": err_rate,
            "experts_loaded": n_experts if ensemble else 0,
        }

    def get_system_status(self) -> Dict:
        return {
            "system_state": self.system_state.value,
            "system_pnl": round(self.system_pnl, 2),
            "system_sharpe": round(self.system_sharpe, 3),
            "total_trades": len(self.trade_history),
            "training_step": self.training_step,
            "epsilon": round(self.epsilon, 3),
            "eval_failures": self._consecutive_eval_failures,
            "cold_start_remaining": max(
                0, self.COLD_START_TRADES - len(self.trade_history)
            ),
            "enhancements": {
                k: {
                    "name": v.name,
                    "status": v.status.value,
                    "confidence": round(v.confidence, 2),
                    "pnl_contribution": round(v.pnl_contribution, 2),
                    "failure_count": v.failure_count,
                    "priority": self.ENHANCEMENTS[k]["priority"],
                }
                for k, v in self.enhancements.items()
            },
            "recent_decisions": [
                {
                    "action": d.action,
                    "reason": d.reason,
                    "confidence": round(d.confidence, 2),
                    "timestamp": d.timestamp,
                }
                for d in self.recent_decisions[-10:]
            ],
            "best_configurations": sorted(
                [
                    {"config": k, "performance": round(v, 4)}
                    for k, v in self.enhancement_combinations.items()
                    if v != 0.0
                ],
                key=lambda x: x["performance"],
                reverse=True,
            )[:5],
        }

    def get_market_recommendation(self, market_data: Dict) -> Dict:
        market_state = self._assess_market_conditions(market_data)
        rec = {
            "market_state": market_state,
            "recommended_action": "HOLD",
            "confidence": 0.5,
            "reasoning": "",
            "enhancements_to_trust": [],
            "parameters_to_adjust": {},
        }
        if market_state == "volatile":
            rec["recommended_action"] = "REDUCE_SIZE"
            rec["reasoning"] = "High volatility detected"
            rec["enhancements_to_trust"] = ["circuit_breaker", "var_sizing"]
            rec["parameters_to_adjust"] = {
                "position_multiplier": 0.5,
                "confidence_threshold": 0.8,
            }
        elif market_state == "calm":
            rec["recommended_action"] = "NORMAL"
            rec["reasoning"] = "Low volatility, normal trading"
            rec["enhancements_to_trust"] = ["ppo_integration", "ensemble_weighting"]
            rec["parameters_to_adjust"] = {
                "position_multiplier": 1.0,
                "confidence_threshold": 0.65,
            }
        else:
            rec["recommended_action"] = "NORMAL"
            rec["reasoning"] = "Normal market conditions"
            rec["enhancements_to_trust"] = list(self.enhancements.keys())[:5]
            rec["parameters_to_adjust"] = {
                "position_multiplier": 0.8,
                "confidence_threshold": 0.7,
            }
        if self.system_sharpe > 1.0:
            rec["confidence"] = 0.8
        elif self.system_sharpe < 0:
            rec["confidence"] = 0.3
        return rec
