"""
Typed configuration — single source of truth for all parameters.
Loads from config.yaml, then overrides with environment variables.
"""
import os
import yaml
from dataclasses import dataclass, field
from typing import List, Optional
from .secrets import Secrets


@dataclass
class AIConfig:
    algorithm: str = "PPO"
    device: str = "auto"
    policy: str = "MlpPolicy"
    net_arch: List[int] = field(default_factory=lambda: [1024, 512, 256, 128])
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    total_timesteps: int = 1_000_000
    eval_freq: int = 10000
    save_freq: int = 50000
    online_finetune: bool = True
    finetune_freq: int = 100


@dataclass
class RiskConfig:
    max_risk_per_trade: float = 0.02
    max_drawdown: float = 0.10
    max_margin_usage: float = 0.80
    min_win_rate_live: float = 0.55
    min_win_rate_paper: float = 0.45
    base_lot_size: float = 0.01
    max_positions: int = 5
    slippage_pips: float = 0.5
    commission_per_lot: float = 7.0
    kelly_fraction: float = 0.25
    confidence_multiplier: bool = True
    volatility_adjustment: bool = True
    sl_atr_multiplier: float = 2.0
    tp_atr_multiplier: float = 4.0
    trailing_activation_rr: float = 1.0
    trailing_distance_rr: float = 0.5
    max_consecutive_losses: int = 5
    max_daily_loss: float = 0.05
    max_position_hold_time: int = 86400
    var_confidence: float = 0.95
    var_lookback_days: int = 60


@dataclass
class DashboardConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    websocket_ping_interval: int = 20
    websocket_ping_timeout: int = 20
    update_interval_ms: int = 1000


@dataclass
class DataConfig:
    redis_url: str = ""
    historical_path: str = "data/historical"
    models_path: str = "models"
    logs_path: str = "data/logs"
    trades_path: str = "data/trades"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    rotation: str = "100 MB"
    retention: str = "30 days"


@dataclass
class Config:
    ai: AIConfig = field(default_factory=AIConfig)
    trading: RiskConfig = field(default_factory=RiskConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    secrets: Secrets = field(default_factory=Secrets)
    config_path: str = "config.yaml"

    def __post_init__(self):
        self._load_yaml()
        self._apply_env_overrides()

    def _load_yaml(self):
        if not os.path.exists(self.config_path):
            return
        with open(self.config_path) as f:
            raw = yaml.safe_load(f) or {}
        ai = raw.get("ai", {})
        for k, v in ai.items():
            if hasattr(self.ai, k) and v is not None:
                setattr(self.ai, k, v)
        trading = raw.get("trading", {})
        for k, v in trading.items():
            if hasattr(self.trading, k) and v is not None:
                setattr(self.trading, k, v)
        trading_r = raw.get("risk", {})
        for k, v in trading_r.items():
            if hasattr(self.trading, k) and v is not None:
                setattr(self.trading, k, v)
        db = raw.get("dashboard", {})
        for k, v in db.items():
            if hasattr(self.dashboard, k) and v is not None:
                setattr(self.dashboard, k, v)
        dt = raw.get("data", {})
        for k, v in dt.items():
            if hasattr(self.data, k) and v is not None:
                setattr(self.data, k, v)
        log = raw.get("logging", {})
        for k, v in log.items():
            if hasattr(self.logging, k) and v is not None:
                setattr(self.logging, k, v)

    def _apply_env_overrides(self):
        self.data.redis_url = os.getenv("REDIS_URL", self.data.redis_url)
        self.logging.level = os.getenv("LOG_LEVEL", self.logging.level)
        self.dashboard.port = int(os.getenv("DASHBOARD_PORT", str(self.dashboard.port)))
