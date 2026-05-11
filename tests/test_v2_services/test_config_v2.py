"""Tests for typed configuration."""
import pytest
import os
import tempfile
import yaml

from infrastructure.config_v2 import AppConfig, SymbolsConfig


class TestAppConfig:
    def test_default_config(self):
        cfg = AppConfig()
        assert cfg.trading.max_risk_per_trade == 0.02
        assert cfg.trading.max_drawdown == 0.10
        assert cfg.features.lookback == 30
        assert cfg.features.timeframes == ["1h"]
        assert len(cfg.symbols.all) == 24

    def test_from_yaml_with_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({"trading": {"max_positions": 5, "max_risk_per_trade": 0.01}}, f)
            fpath = f.name
        try:
            cfg = AppConfig.from_yaml(fpath)
            assert cfg.trading.max_positions == 5
            assert cfg.trading.max_risk_per_trade == 0.01
            assert cfg.trading.max_drawdown == 0.10  # default
        finally:
            os.unlink(fpath)

    def test_from_yaml_missing_file(self):
        cfg = AppConfig.from_yaml("/nonexistent/path.yaml")
        assert cfg.trading.max_positions == 10  # defaults

    def test_from_yaml_reads_real_config(self):
        cfg = AppConfig.from_yaml("config.yaml")
        assert cfg.trading.max_positions == 10
        assert cfg.features.timeframes == ["15m", "1h", "4h"]
        assert cfg.trading.commission_per_lot == 7.0

    def test_validate_passes(self):
        cfg = AppConfig()
        assert cfg.validate() == []

    def test_validate_fails_bad_risk(self):
        cfg = AppConfig()
        cfg.trading.max_risk_per_trade = 0.0
        errors = cfg.validate()
        assert "max_risk_per_trade" in errors[0]

    def test_validate_fails_no_timeframes(self):
        cfg = AppConfig()
        cfg.features.timeframes = []
        errors = cfg.validate()
        assert any("timeframe" in e for e in errors)

    def test_validate_fails_short_lookback(self):
        cfg = AppConfig()
        cfg.features.lookback = 3
        errors = cfg.validate()
        assert any("lookback" in e for e in errors)

    def test_symbols_config(self):
        sym = SymbolsConfig()
        assert "EURUSD" in sym.forex
        assert "BTCUSD" in sym.crypto
        assert "XAUUSD" in sym.metals
        assert "US500" in sym.indices
        assert "XTIUSD" in sym.energy
        assert len(sym.all) == 24
        assert all(s in sym.all for s in ["EURUSD", "BTCUSD", "XAUUSD", "US500", "XTIUSD"])

    def test_env_override_port(self):
        os.environ["DASHBOARD_PORT"] = "9000"
        cfg = AppConfig.from_yaml("config.yaml")
        assert cfg.dashboard.port == 9000
        del os.environ["DASHBOARD_PORT"]

    def test_env_override_log_level(self):
        os.environ["LOG_LEVEL"] = "DEBUG"
        cfg = AppConfig.from_yaml("config.yaml")
        assert cfg.logging.level == "DEBUG"
        del os.environ["LOG_LEVEL"]
