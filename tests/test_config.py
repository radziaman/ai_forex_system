"""Tests for configuration loading and alignment."""
import pytest
import os
import tempfile
import yaml


@pytest.fixture
def sample_config():
    return {
        "trading": {
            "max_risk_per_trade": 0.02,
            "max_drawdown": 0.10,
            "max_margin_usage": 0.80,
            "max_positions": 10,
            "asset_classes": {
                "forex": {"max_positions": 5, "slippage_pips": 0.5},
                "crypto": {"max_positions": 3, "slippage_pips": 5.0},
            },
        },
        "ai": {
            "algorithm": "PPO-Regime-Specialist",
            "regime_agents": {
                "trending": {"hidden_dims": [1024, 512, 256], "learning_rate": 3e-4},
                "ranging": {"hidden_dims": [512, 256, 128], "learning_rate": 3e-4},
            },
            "model_versioning": {"enabled": True, "registry_path": "models/registry"},
            "ensemble": {"method": "moe", "experts": ["ppo_regime", "lstm_cnn"]},
            "training": {"total_timesteps": 1000000, "batch_size": 64},
        },
        "features": {"timeframes": ["15m", "1h", "4h"], "lookback": 30},
        "risk": {"kelly_fraction": 0.25, "sl_atr_multiplier": 2.0},
        "dashboard": {"host": "0.0.0.0", "port": 8000},
        "data": {"historical_path": "data/historical", "logs_path": "data/logs"},
        "logging": {"level": "INFO", "rotation": "100 MB"},
    }


def test_config_loads_yaml(sample_config):
    from infrastructure.config import Config

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(sample_config, f)
        config_path = f.name

    try:
        config = Config()
        config.config_path = config_path
        config._load_yaml()

        # Check trading params loaded
        assert config.trading.max_risk_per_trade == 0.02
        assert config.trading.max_drawdown == 0.10
        assert config.trading.max_positions == 10

        # Check risk section loaded
        assert config.trading.kelly_fraction == 0.25

        # Check raw data stored
        assert config.raw["ai"]["algorithm"] == "PPO-Regime-Specialist"
        assert "features" in config.raw
        assert "asset_classes" in config.raw["trading"]
    finally:
        os.unlink(config_path)


def test_config_env_overrides(sample_config):
    from infrastructure.config import Config

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(sample_config, f)
        config_path = f.name

    try:
        os.environ["LOG_LEVEL"] = "DEBUG"
        os.environ["DASHBOARD_PORT"] = "9090"

        config = Config()
        config.config_path = config_path
        config._load_yaml()
        config._apply_env_overrides()

        assert config.logging.level == "DEBUG"
        assert config.dashboard.port == 9090

        del os.environ["LOG_LEVEL"]
        del os.environ["DASHBOARD_PORT"]
    finally:
        os.unlink(config_path)


def test_config_defaults():
    from infrastructure.config import Config

    config = Config()
    config.config_path = "nonexistent.yaml"
    config._load_yaml()

    # Should use defaults
    assert config.trading.max_risk_per_trade == 0.02
    assert config.dashboard.port == 8000


def test_config_provider():
    from infrastructure.config import Config

    config = Config()
    assert hasattr(config, 'provider')


def test_risk_config_fields():
    from infrastructure.config import RiskConfig

    rc = RiskConfig()
    assert rc.max_risk_per_trade == 0.02
    assert rc.max_drawdown == 0.10
    assert rc.kelly_fraction == 0.25
    assert rc.sl_atr_multiplier == 2.0
    assert rc.tp_atr_multiplier == 4.0
    assert rc.max_consecutive_losses == 5
    assert rc.max_daily_loss == 0.05
    assert rc.var_confidence == 0.95


def test_ai_config_fields():
    from infrastructure.config import AIConfig

    ai = AIConfig()
    assert ai.algorithm == "PPO"
    assert ai.learning_rate == 3e-4
    assert ai.batch_size == 64
    assert ai.gamma == 0.99
