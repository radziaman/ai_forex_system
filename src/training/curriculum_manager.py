"""
Curriculum Learning Manager — Progressive difficulty for PPO training.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from loguru import logger


@dataclass
class LevelConfig:
    volatility_multiplier: float
    slippage_multiplier: float
    regime_change_frequency: int
    max_drawdown_penalty: str


LEVEL_CONFIGS = {
    "easy": LevelConfig(
        volatility_multiplier=0.5,
        slippage_multiplier=0.0,
        regime_change_frequency=100,
        max_drawdown_penalty="light",
    ),
    "medium": LevelConfig(
        volatility_multiplier=1.0,
        slippage_multiplier=0.5,
        regime_change_frequency=50,
        max_drawdown_penalty="medium",
    ),
    "hard": LevelConfig(
        volatility_multiplier=1.5,
        slippage_multiplier=1.0,
        regime_change_frequency=20,
        max_drawdown_penalty="severe",
    ),
    "extreme": LevelConfig(
        volatility_multiplier=2.0,
        slippage_multiplier=2.0,
        regime_change_frequency=10,
        max_drawdown_penalty="critical",
    ),
}


class CurriculumManager:
    """Manages curriculum learning for PPO agents."""

    def __init__(self, difficulty_levels: Optional[List[str]] = None):
        if difficulty_levels is None:
            difficulty_levels = ["easy", "medium", "hard", "extreme"]
        self.difficulty_levels = difficulty_levels
        self.current_level_idx = 0
        self.performance_history: List[Dict[str, Any]] = []

    def get_current_level(self) -> str:
        """Return the current difficulty level name."""
        return self.difficulty_levels[self.current_level_idx]

    def get_env_params(self) -> Dict[str, Any]:
        """Return environment parameters for the current difficulty level."""
        level = self.get_current_level()
        config = LEVEL_CONFIGS[level]
        return {
            "volatility_multiplier": config.volatility_multiplier,
            "slippage_multiplier": config.slippage_multiplier,
            "regime_change_frequency": config.regime_change_frequency,
            "max_drawdown_penalty": config.max_drawdown_penalty,
        }

    def evaluate_performance(
        self,
        agent,
        env,
        test_episodes: int = 20,
        max_steps_per_episode: int = 100,
    ) -> Dict[str, Any]:
        """Evaluate agent performance on the current difficulty level."""
        total_reward = 0.0
        wins = 0
        total_trades = 0
        all_returns = []

        for _ in range(test_episodes):
            state = env.reset()
            episode_reward = 0.0
            done = False
            step_count = 0
            while not done and step_count < max_steps_per_episode:
                action, sl_raw, tp_raw, size_raw, _ = agent.select_action(state)
                next_state, reward, done, _ = env.step(action, sl_raw, tp_raw, size_raw)
                state = next_state
                episode_reward += reward
                step_count += 1
                if done:
                    break
            total_reward += episode_reward
            all_returns.append(episode_reward)

            if episode_reward > 0:
                wins += 1
            total_trades += getattr(env, "total_trades", 0)

        win_rate = wins / test_episodes if test_episodes > 0 else 0.0
        sharpe = 0.0
        if len(all_returns) >= 2:
            mean_ret = np.mean(all_returns)
            std_ret = np.std(all_returns) + 1e-8
            sharpe = float(mean_ret / std_ret)

        result = {
            "level": self.get_current_level(),
            "win_rate": win_rate,
            "sharpe": sharpe,
            "avg_reward": total_reward / test_episodes if test_episodes > 0 else 0.0,
            "total_trades": total_trades,
        }
        self.performance_history.append(result)
        logger.info(
            f"Curriculum eval ({self.get_current_level()}): "
            f"win_rate={win_rate:.2%}, sharpe={sharpe:.3f}"
        )
        return result

    def should_advance(self) -> bool:
        """Return True if agent should advance to next difficulty."""
        if not self.performance_history:
            return False
        last_perf = self.performance_history[-1]
        win_rate = last_perf.get("win_rate", 0.0)
        sharpe = last_perf.get("sharpe", 0.0)
        return win_rate > 0.60 and sharpe > 0.5

    def advance(self) -> bool:
        """Advance to the next difficulty level. Return True if advanced."""
        if self.current_level_idx < len(self.difficulty_levels) - 1:
            self.current_level_idx += 1
            logger.info(f"Curriculum advanced to {self.get_current_level()}")
            return True
        logger.info("Curriculum at max difficulty")
        return False

    def is_complete(self) -> bool:
        """Return True if all difficulty levels are completed."""
        return (
            self.current_level_idx >= len(self.difficulty_levels) - 1
            and self.should_advance()
        )
