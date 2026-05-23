"""Tests for Curriculum Learning Manager."""

from ai.rl_agent import PPOAgent, TradingEnvironment
from training.curriculum_manager import CurriculumManager


class TestCurriculumManager:
    """Comprehensive tests for CurriculumManager."""

    def test_get_current_level_starts_easy(self):
        cm = CurriculumManager()
        assert cm.get_current_level() == "easy"

    def test_advance(self):
        cm = CurriculumManager()
        assert cm.advance()
        assert cm.get_current_level() == "medium"
        assert cm.advance()
        assert cm.get_current_level() == "hard"
        assert cm.advance()
        assert cm.get_current_level() == "extreme"
        assert not cm.advance()

    def test_should_advance_true(self):
        cm = CurriculumManager()
        cm.performance_history.append({"win_rate": 0.65, "sharpe": 0.6})
        assert cm.should_advance()

    def test_should_advance_false(self):
        cm = CurriculumManager()
        cm.performance_history.append({"win_rate": 0.50, "sharpe": 0.6})
        assert not cm.should_advance()

        cm2 = CurriculumManager()
        cm2.performance_history.append({"win_rate": 0.70, "sharpe": 0.3})
        assert not cm2.should_advance()

    def test_env_params_match_level(self):
        cm = CurriculumManager()
        easy = cm.get_env_params()
        assert easy["volatility_multiplier"] == 0.5
        assert easy["slippage_multiplier"] == 0.0
        assert easy["regime_change_frequency"] == 100
        assert easy["max_drawdown_penalty"] == "light"

        cm.advance()
        med = cm.get_env_params()
        assert med["volatility_multiplier"] == 1.0
        assert med["max_drawdown_penalty"] == "medium"

        cm.advance()
        hard = cm.get_env_params()
        assert hard["volatility_multiplier"] == 1.5
        assert hard["max_drawdown_penalty"] == "severe"

        cm.advance()
        extreme = cm.get_env_params()
        assert extreme["volatility_multiplier"] == 2.0
        assert extreme["max_drawdown_penalty"] == "critical"

    def test_evaluate_performance(self):
        cm = CurriculumManager()
        agent = PPOAgent(state_dim=4)
        env = TradingEnvironment(state_dim=4)

        result = cm.evaluate_performance(agent, env, test_episodes=5)
        assert "win_rate" in result
        assert "sharpe" in result
        assert "level" in result
        assert result["level"] == "easy"
        assert 0.0 <= result["win_rate"] <= 1.0
