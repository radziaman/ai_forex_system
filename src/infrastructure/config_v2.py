from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import os
import yaml


@dataclass
class TradingConfig:
    max_risk_per_trade: float = 0.02
    max_drawdown: float = 0.10
    max_margin_usage: float = 0.80
    base_lot_size: float = 0.01
    max_positions: int = 10
    slippage_pips: float = 0.5
    commission_per_lot: float = 7.0
    sl_atr_multiplier: float = 2.0
    tp_atr_multiplier: float = 4.0
    max_consecutive_losses: int = 5
    max_daily_loss: float = 0.05
    kelly_fraction: float = 0.25
    trailing_activation_rr: float = 1.0
    trailing_distance_rr: float = 0.5


@dataclass
class AIConfig:
    algorithm: str = "PPO-Regime-Specialist"
    device: str = "auto"
    ensemble_method: str = "moe"
    experts: List[str] = field(
        default_factory=lambda: ["ppo_regime", "lstm_cnn", "rule_based"]
    )
    use_elo_rating: bool = True
    use_sharpe_weighting: bool = True
    total_timesteps: int = 1_000_000
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    online_finetune: bool = True
    finetune_freq: int = 100
    registry_path: str = "models/registry"
    auto_promote: bool = True


@dataclass
class FeaturesConfig:
    timeframes: List[str] = field(default_factory=lambda: ["1h"])
    lookback: int = 30
    use_microstructure: bool = True
    use_cross_asset: bool = False


@dataclass
class RiskConfig:
    var_confidence: float = 0.95
    var_lookback_days: int = 60
    max_position_hold_time: int = 86400
    price_velocity_threshold: float = 0.005
    spread_multiplier_threshold: float = 5.0
    volume_spike_multiplier: float = 10.0


@dataclass
class DashboardConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    websocket_ping_interval: int = 20
    update_interval_ms: int = 1000


@dataclass
class DataConfig:
    historical_path: str = "data/historical"
    models_path: str = "models"
    logs_path: str = "data/logs"
    trades_path: str = "data/trades"
    max_bars_in_memory: int = 5000


@dataclass
class LoggingConfig:
    level: str = "INFO"
    rotation: str = "100 MB"
    retention: str = "30 days"
    path: str = "data/logs/moneybot.log"


@dataclass
class SymbolsConfig:
    forex: List[str] = field(
        default_factory=lambda: [
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
        ]
    )
    metals: List[str] = field(default_factory=lambda: ["XAUUSD", "XAGUSD"])
    energy: List[str] = field(default_factory=lambda: ["XTIUSD", "XBRUSD", "XNGUSD"])
    indices: List[str] = field(
        default_factory=lambda: ["US500", "US30", "USTEC", "UK100", "DE40"]
    )
    crypto: List[str] = field(
        default_factory=lambda: ["BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD"]
    )

    @property
    def all(self) -> List[str]:
        return self.forex + self.metals + self.energy + self.indices + self.crypto


@dataclass
class AppConfig:
    trading: TradingConfig = field(default_factory=TradingConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    symbols: SymbolsConfig = field(default_factory=SymbolsConfig)

    @classmethod
    def from_yaml(cls, path: str) -> AppConfig:
        if not os.path.exists(path):
            return cls()
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        cfg = cls()
        _merge(raw.get("trading", {}), cfg.trading)
        _merge(raw.get("ai", {}), cfg.ai)
        _merge(raw.get("features", {}), cfg.features)
        _merge(raw.get("risk", {}), cfg.risk)
        _merge(raw.get("dashboard", {}), cfg.dashboard)
        _merge(raw.get("data", {}), cfg.data)
        _merge(raw.get("logging", {}), cfg.logging)
        cfg.data.logs_path = os.getenv("LOG_PATH", cfg.data.logs_path)
        cfg.dashboard.port = int(os.getenv("DASHBOARD_PORT", str(cfg.dashboard.port)))
        cfg.logging.level = os.getenv("LOG_LEVEL", cfg.logging.level)
        return cfg

    def validate(self) -> List[str]:
        errors = []
        if self.trading.max_risk_per_trade <= 0 or self.trading.max_risk_per_trade > 1:
            errors.append("max_risk_per_trade must be in (0, 1]")
        if self.features.lookback < 5:
            errors.append("lookback must be >= 5")
        if not self.features.timeframes:
            errors.append("at least one timeframe required")
        return errors


def _merge(src: dict, dest) -> None:
    for k in dest.__dataclass_fields__:
        if k in src and src[k] is not None:
            setattr(dest, k, src[k])
