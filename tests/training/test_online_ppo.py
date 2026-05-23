"""Tests for Online PPO Trainer."""

import numpy as np

from ai.rl_agent import PPOAgent, TradingEnvironment
from training.online_ppo_trainer import OnlinePPOTrainer


class TestOnlinePPOTrainer:
    """Comprehensive tests for OnlinePPOTrainer."""

    def test_record_experience(self):
        agent = PPOAgent(state_dim=4)
        trainer = OnlinePPOTrainer(agent, buffer_size=10, update_freq=5)
        trainer.record_experience(np.zeros(4), 0, 1.0, np.zeros(4), False)
        assert len(trainer.buffer) == 1

    def test_circular_buffer_eviction(self):
        agent = PPOAgent(state_dim=4)
        trainer = OnlinePPOTrainer(agent, buffer_size=3, update_freq=10)
        for i in range(5):
            trainer.record_experience(np.zeros(4), 0, float(i), np.zeros(4), False)
        assert len(trainer.buffer) == 3
        rewards = [e[2] for e in trainer.buffer]
        assert rewards == [2.0, 3.0, 4.0]

    def test_should_update(self):
        agent = PPOAgent(state_dim=4)
        trainer = OnlinePPOTrainer(agent, buffer_size=100, update_freq=5)
        for i in range(4):
            trainer.record_experience(np.zeros(4), 0, float(i), np.zeros(4), False)
        assert not trainer.should_update()
        trainer.record_experience(np.zeros(4), 0, 4.0, np.zeros(4), False)
        assert trainer.should_update()

    def test_shape_reward_components(self):
        agent = PPOAgent(state_dim=4)
        trainer = OnlinePPOTrainer(agent, buffer_size=10, update_freq=5)

        # Basic PnL
        r1 = trainer.shape_reward(pnl=10.0, holding_time=2, drawdown=0.01)
        assert r1 >= 10.0

        # Long holding penalty
        r_long = trainer.shape_reward(pnl=10.0, holding_time=100, drawdown=0.01)
        assert r_long < r1

        # Drawdown penalty
        r_dd = trainer.shape_reward(pnl=10.0, holding_time=2, drawdown=0.15)
        assert r_dd < r1

        # Consecutive wins build momentum
        trainer.consecutive_wins = 0
        r_win1 = trainer.shape_reward(pnl=1.0, holding_time=0, drawdown=0.0)
        r_win2 = trainer.shape_reward(pnl=1.0, holding_time=0, drawdown=0.0)
        r_win3 = trainer.shape_reward(pnl=1.0, holding_time=0, drawdown=0.0)
        assert r_win3 > r_win2 > r_win1

    def test_update_policy_runs(self):
        agent = PPOAgent(state_dim=4)
        trainer = OnlinePPOTrainer(agent, buffer_size=100, update_freq=5)
        for i in range(10):
            state = np.random.randn(4).astype(np.float32)
            next_state = np.random.randn(4).astype(np.float32)
            trainer.record_experience(state, i % 5, float(i), next_state, False)
        metrics = trainer.update_policy(n_epochs=2, batch_size=4)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert trainer._new_experiences == 0

    def test_agent_online_integration(self):
        agent = PPOAgent(state_dim=4)
        trainer = OnlinePPOTrainer(agent, buffer_size=50, update_freq=5)
        agent.enable_online_learning(trainer)
        env = TradingEnvironment(state_dim=4)

        state = env.reset()
        action, sl, tp, size, info = agent.select_action(state)
        next_state, reward, done, _ = env.step(action, sl, tp, size)
        agent.store_transition(state, action, reward, next_state, done)

        for _ in range(5):
            state = env.reset() if done else next_state
            action, sl, tp, size, info = agent.select_action(state)
            next_state, reward, done, _ = env.step(action, sl, tp, size)
            agent.store_transition(state, action, reward, next_state, done)

        assert len(trainer.buffer) >= 5
