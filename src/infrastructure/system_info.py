from __future__ import annotations
import os
import sys
from typing import List, Tuple
from dataclasses import dataclass, field

VERSION = "4.0"

from infrastructure.config_v2 import AppConfig
from data.data_manager import SYMBOLS


@dataclass
class ComponentStatus:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class SystemReport:
    version: str = "4.0"
    mode: str = "paper"
    symbols: int = 0
    timeframes: List[str] = field(default_factory=list)
    lookback: int = 30
    initial_balance: float = 100_000.0
    components: List[ComponentStatus] = field(default_factory=list)
    model_files: List[str] = field(default_factory=list)
    alternative_data: List[str] = field(default_factory=list)

    def print_summary(self):
        lines = []
        w = 60
        lines.append("=" * w)
        lines.append(f"  RTS: AI Moneybot System Elite v{self.version}")
        lines.append(f"  Mode: {self.mode.upper()}")
        lines.append("=" * w)
        lines.append("")
        lines.append(f"  Symbols monitored:  {self.symbols}")
        lines.append(f"  Timeframes:         {', '.join(self.timeframes)}")
        lines.append(f"  Feature lookback:   {self.lookback} bars")
        lines.append(f"  Starting capital:   ${self.initial_balance:,.0f}")
        lines.append("")
        ok = sum(1 for c in self.components if c.ok)
        total = len(self.components)
        lines.append(f"  Components:  {ok}/{total} healthy")
        for c in self.components:
            icon = "+" if c.ok else "X"
            lines.append(f"    [{icon}] {c.name}: {c.detail}")
        lines.append("")
        if self.model_files:
            lines.append(f"  Trained models:  {len(self.model_files)}")
            for mf in self.model_files[:5]:
                size = os.path.getsize(mf) / 1024 / 1024
                lines.append(f"    {mf.rsplit('/', 1)[-1]}  ({size:.1f} MB)")
            if len(self.model_files) > 5:
                lines.append(f"    ... and {len(self.model_files) - 5} more")
        else:
            lines.append("  Trained models:  none (will use defaults)")
        lines.append("")
        if self.alternative_data:
            lines.append(f"  Alternative data sources: {len(self.alternative_data)}")
            for ad in self.alternative_data:
                lines.append(f"    {ad}")
        lines.append("")
        lines.append(f"  Log file: data/logs/moneybot_v2.log")
        lines.append(f"  Dashboard: http://0.0.0.0:8000")
        lines.append("=" * w)
        return "\n".join(lines)

    def print_check(self) -> Tuple[bool, str]:
        lines = []
        w = 60
        all_ok = True
        lines.append("=" * w)
        lines.append("  SYSTEM READINESS CHECK")
        lines.append("=" * w)
        lines.append("")
        for c in self.components:
            if c.ok:
                lines.append(f"  [+] {c.name}: {c.detail}")
            else:
                all_ok = False
                lines.append(f"  [X] {c.name}: {c.detail}")
        lines.append("")
        if all_ok:
            lines.append("  Result: ALL CHECKS PASSED")
        else:
            lines.append("  Result: SOME CHECKS FAILED — review above")
        lines.append("=" * w)
        return all_ok, "\n".join(lines)


def collect_report(config: AppConfig, mode: str = "paper", balance: float = 100_000.0) -> SystemReport:
    report = SystemReport(
        mode=mode,
        symbols=len(SYMBOLS),
        timeframes=config.features.timeframes,
        lookback=config.features.lookback,
        initial_balance=balance,
    )

    # Config check
    config_errors = config.validate()
    report.components.append(ComponentStatus(
        "Configuration", ok=len(config_errors) == 0,
        detail="validated" if not config_errors else f"errors: {config_errors}",
    ))

    # Symbols
    report.components.append(ComponentStatus(
        "Symbol registry", ok=len(SYMBOLS) > 0,
        detail=f"{len(SYMBOLS)} symbols loaded",
    ))

    # Feature pipeline
    report.components.append(ComponentStatus(
        "Feature pipeline", ok=True,
        detail=f"{len(config.features.timeframes)} timeframes, lookback={config.features.lookback}",
    ))

    # Risk limits
    report.components.append(ComponentStatus(
        "Risk limits", ok=True,
        detail=f"max_risk={config.trading.max_risk_per_trade:.0%}, "
               f"max_dd={config.trading.max_drawdown:.0%}, "
               f"Kelly={config.trading.kelly_fraction:.0%}",
    ))

    # Model files
    model_dir = "models"
    if os.path.isdir(model_dir):
        pth_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir)
                     if f.endswith(('.pth', '.pt', '.keras'))]
        report.model_files = pth_files

    has_models = len(report.model_files) > 0
    report.components.append(ComponentStatus(
        "AI models", ok=has_models,
        detail=f"{len(report.model_files)} trained models found" if has_models else "no pre-trained models",
    ))

    # Alternative data
    alt_dir = "data/alternative_data"
    if os.path.isdir(alt_dir):
        report.alternative_data = sorted(os.listdir(alt_dir))
        report.components.append(ComponentStatus(
            "Alternative data", ok=True,
            detail=f"{len(report.alternative_data)} cached datasets",
        ))
    else:
        report.components.append(ComponentStatus(
            "Alternative data", ok=False, detail="cache directory not found",
        ))

    # API credentials check
    from infrastructure.secrets import Secrets
    s = Secrets()
    missing = s.validate()
    report.components.append(ComponentStatus(
        "cTrader API credentials", ok=len(missing) == 0,
        detail="configured" if not missing else f"missing: {', '.join(missing)}",
    ))

    has_telegram = bool(s.telegram_bot_token and s.telegram_bot_token != "")
    report.components.append(ComponentStatus(
        "Telegram notifications", ok=has_telegram,
        detail="configured" if has_telegram else "not configured",
    ))

    # Cached data
    cache_dir = "data/dukascopy_cache"
    if os.path.isdir(cache_dir):
        bi5_count = len([f for f in os.listdir(cache_dir) if f.endswith('.bi5')])
        report.components.append(ComponentStatus(
            "Historical tick cache", ok=bi5_count > 0,
            detail=f"{bi5_count} BI5 files cached" if bi5_count > 0 else "empty cache",
        ))
    else:
        report.components.append(ComponentStatus(
            "Historical tick cache", ok=False, detail="cache directory not found",
        ))

    return report
