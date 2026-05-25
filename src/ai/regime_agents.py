"""
Multi-Agent Regime Specialist System.
Each regime has a dedicated PPO agent with specialized hyperparameters and SL/TP logic.
Gating network routes decisions to the correct specialist.
"""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from loguru import logger

if TYPE_CHECKING:
    from ai.rl_agent import PPOAgent


@dataclass
class RegimeAgentConfig:
    """Per-regime PPO hyperparameters."""

    name: str
    hidden_dims: list
    sl_atr_mult: float
    tp_atr_mult: float
    position_multiplier: float
    learning_rate: float = 3e-4
    clip_range: float = 0.2


# Default configurations per regime
REGIME_CONFIGS = {
    "trending": RegimeAgentConfig(
        name="trend_agent",
        hidden_dims=[1024, 512, 256],
        sl_atr_mult=2.0,
        tp_atr_mult=4.0,
        position_multiplier=1.0,
        clip_range=0.3,  # Wider exploration for trends
    ),
    "ranging": RegimeAgentConfig(
        name="range_agent",
        hidden_dims=[512, 256, 128],
        sl_atr_mult=1.5,
        tp_atr_mult=3.0,
        position_multiplier=1.0,
        clip_range=0.15,  # Tighter for mean reversion
    ),
    "volatile": RegimeAgentConfig(
        name="vol_agent",
        hidden_dims=[512, 256, 128],
        sl_atr_mult=2.5,
        tp_atr_mult=5.0,
        position_multiplier=0.5,
        learning_rate=1e-4,  # Slower learning in volatile regimes
        clip_range=0.1,
    ),
    "crisis": RegimeAgentConfig(
        name="crisis_agent",
        hidden_dims=[256, 128, 64],
        sl_atr_mult=1.0,
        tp_atr_mult=2.0,
        position_multiplier=0.0,  # No trading in crisis by default
        learning_rate=5e-5,
        clip_range=0.05,
    ),
}


class RegimeGatingNetwork(nn.Module):
    """Learned gating network that routes to regime specialist."""

    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # 4 regimes
            nn.Softmax(dim=-1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class RegimeSpecialistSystem:
    """
    Multi-agent system with regime-specific RL agents.
    Each agent specializes in one market regime (trending/ranging/volatile/crisis).
    """

    def __init__(self, state_dim: int, n_actions: int = 5):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.agents: Dict[str, "PPOAgent"] = {}
        self.enabled: bool = True
        self.gating_network = RegimeGatingNetwork(state_dim)
        self._init_agents()
        self.regime_performance: Dict[str, Dict] = {
            regime: {"trades": 0, "wins": 0, "pnl": 0.0}
            for regime in REGIME_CONFIGS.keys()
        }

    def _init_agents(self):
        """Initialize or load pre-trained agents for each regime."""
        for regime, config in REGIME_CONFIGS.items():
            try:
                from ai.rl_agent import PPOAgent

                # NOTE: hidden_dims NOT passed here — the training script
                # train_all_models.py uses the PPOAgent default
                # [1024, 512, 256, 128]. Passing regime-specific dims here
                # would mismatch the saved network architecture.
                agent = PPOAgent(
                    state_dim=self.state_dim,
                    n_actions=self.n_actions,
                    clip_range=config.clip_range,
                )
                agent.optimizer = optim.Adam(
                    agent.actor.parameters(), lr=config.learning_rate
                )
                agent_path = f"models/{regime}_agent.pth"
                # Load pre-trained if exists (graceful on shape mismatch)
                if os.path.exists(agent_path):
                    try:
                        agent.load(agent_path)
                        logger.info(f"Loaded {regime} agent from {agent_path}")
                    except Exception as load_err:
                        logger.info(
                            f"Fresh {regime} agent (saved dim differs from current {self.state_dim})"  # noqa: E501
                        )
                self.agents[regime] = agent
            except Exception as e:
                logger.warning(f"Failed to init {regime} agent: {e}")
                self.agents[regime] = None

    def select_action(
        self, state: np.ndarray, regime: str
    ) -> Tuple[int, float, float, float, Dict]:
        """
        Select action using regime-specific agent.
        Returns: (action, sl_raw, tp_raw, value, info_dict)
        """
        agent = self.agents.get(regime)
        if agent is None:
            # Fallback to trending agent or rule-based
            agent = self.agents.get("trending") or self.agents.get("ranging")
            if agent is None:
                return 0, 0.0, 0.0, 0.0, {"error": "no_agent_available"}

        try:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            action, sl_raw, tp_raw, size_raw, info = agent.select_action(state)
            info["regime"] = regime
            info["agent_name"] = REGIME_CONFIGS[regime].name
            return action, sl_raw, tp_raw, size_raw, info
        except Exception as e:
            logger.debug(f"Regime agent error ({regime}): {e}")
            return 0, 0.0, 0.0, 0.0, {"error": str(e)}

    def get_regime_params(self, regime: str) -> Dict:
        """Get SL/TP multipliers for the regime."""
        config = REGIME_CONFIGS.get(regime, REGIME_CONFIGS["ranging"])
        return {
            "sl_atr": config.sl_atr_mult,
            "tp_atr": config.tp_atr_mult,
            "pos_mult": config.position_multiplier,
        }

    def update_performance(self, regime: str, pnl: float):
        """Track per-regime agent performance."""
        if regime in self.regime_performance:
            perf = self.regime_performance[regime]
            perf["trades"] += 1
            perf["pnl"] += pnl
            if pnl > 0:
                perf["wins"] += 1

    def get_best_regime_agent(self) -> Optional[str]:
        """Return the regime with the best performing agent."""
        best_regime = None
        best_win_rate = -1.0
        for regime, perf in self.regime_performance.items():
            if perf["trades"] < 10:
                continue
            win_rate = perf["wins"] / perf["trades"] if perf["trades"] > 0 else 0
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_regime = regime
        return best_regime

    def should_trade_in_regime(self, regime: str) -> bool:
        """Check if we should trade given the current regime."""
        config = REGIME_CONFIGS.get(regime)
        if config is None:
            return True
        return config.position_multiplier > 0
