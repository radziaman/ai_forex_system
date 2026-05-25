"""
Structured Alpha Research Pipeline.
Uses Information Coefficient (IC) framework to discover and validate alpha signals.

Each candidate signal is evaluated against forward returns:
  - IC: Spearman rank correlation between signal and forward returns
  - IR: Information ratio (mean(IC) / std(IC))
  - Alpha decay: IC at multiple horizons (1h, 4h, 1d, 1w)
  - Regime consistency: IC stability across market regimes

Signals that pass validation (IC_mean > 0.02, IR > 0.5, t-stat > 2.0)
are promoted to the MoE ensemble.
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


class SignalStatus(Enum):
    CANDIDATE = "candidate"
    VALIDATING = "validating"
    ACTIVE = "active"
    REJECTED = "rejected"
    DECAYED = "decayed"


@dataclass
class SignalResult:
    name: str
    ic: float = 0.0
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ir: float = 0.0
    t_stat: float = 0.0
    p_value: float = 0.0
    pct_positive: float = 0.0
    is_significant: bool = False
    status: SignalStatus = SignalStatus.CANDIDATE
    horizon_best: int = 1
    ic_by_horizon: Dict[int, float] = field(default_factory=dict)
    regime_ics: Dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class AlphaPipelineConfig:
    min_ic: float = 0.02
    min_ir: float = 0.5
    min_t_stat: float = 2.0
    min_horizon_ic: float = 0.01
    validation_periods: int = 5
    decay_threshold: float = 0.5  # IC decay > 50% → demote


class AlphaResearchPipeline:
    """Systematic alpha signal research and validation pipeline.

    Usage:
        1. Register a signal function via register_signal()
        2. Call evaluate() with price/return data
        3. Check result.is_significant to determine if signal has edge
        4. Promoted signals can be added to the MoE ensemble
    """

    def __init__(
        self,
        config: Optional[AlphaPipelineConfig] = None,
        registry_path: str = "models/signal_registry.json",
    ):
        self.config = config or AlphaPipelineConfig()
        self.registry_path = Path(registry_path)
        self._signals: Dict[str, Callable] = {}
        self._results: Dict[str, SignalResult] = {}
        self._active_signals: List[str] = []
        self._load_registry()

    def register_signal(self, name: str, signal_fn: Callable):
        """Register a signal generation function.

        The function must accept (prices: pd.DataFrame) and return pd.Series.
        """
        self._signals[name] = signal_fn
        logger.info(f"Alpha Pipeline: registered signal '{name}'")

    def evaluate(
        self,
        name: str,
        prices: pd.DataFrame,
        forward_returns: Optional[pd.Series] = None,
        regimes: Optional[pd.Series] = None,
    ) -> SignalResult:
        """Evaluate a signal against forward returns across multiple horizons.

        If forward_returns is not provided, it computes from prices.
        """
        if name not in self._signals:
            raise ValueError(f"Unknown signal: {name}")

        signal = self._signals[name](prices)

        if forward_returns is None:
            returns = (
                prices.iloc[:, 0].pct_change()
                if isinstance(prices, pd.DataFrame) and prices.shape[1] > 0
                else prices.pct_change()
            )
        else:
            returns = forward_returns

        # IC at primary horizon (1 period)
        aligned = self._align(signal, returns.shift(-1))
        if len(aligned) < 30:
            return SignalResult(name=name, status=SignalStatus.REJECTED)

        ic, p_value = stats.spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
        ic = ic if not np.isnan(ic) else 0.0
        p_value = p_value if not np.isnan(p_value) else 1.0

        # IC by time period (monthly)
        try:
            _idx = pd.DatetimeIndex(aligned.index)
            monthly_ics = aligned.groupby(_idx.to_period("M")).apply(
                lambda g: (
                    stats.spearmanr(g.iloc[:, 0], g.iloc[:, 1])[0]
                    if len(g) > 10
                    else np.nan
                )
            )
            monthly_ics = monthly_ics.dropna()
        except Exception:
            monthly_ics = pd.Series(dtype=float)

        ic_mean = float(monthly_ics.mean()) if len(monthly_ics) > 0 else ic
        ic_std = float(monthly_ics.std()) if len(monthly_ics) > 1 else 0.1
        ir = ic_mean / max(ic_std, 1e-10)
        t_stat = ic_mean / max(ic_std / np.sqrt(max(len(monthly_ics), 1)), 1e-10)
        pct_positive = float((monthly_ics > 0).mean()) if len(monthly_ics) > 0 else 0.0

        # IC by horizon (alpha decay)
        ic_by_horizon = {}
        best_horizon = 1
        best_ic = abs(ic)
        for horizon in [1, 4, 24, 168]:  # 1h, 4h, 1d, 1w
            try:
                fwd = returns.rolling(horizon).sum().shift(-horizon)
                h_aligned = self._align(signal, fwd)
                if len(h_aligned) >= 30:
                    h_ic, _ = stats.spearmanr(
                        h_aligned.iloc[:, 0], h_aligned.iloc[:, 1]
                    )
                    h_ic = h_ic if not np.isnan(h_ic) else 0.0
                    ic_by_horizon[horizon] = h_ic
                    if abs(h_ic) > best_ic:
                        best_ic = abs(h_ic)
                        best_horizon = horizon
            except Exception:
                pass

        # Regime-specific IC
        regime_ics = {}
        if regimes is not None:
            combined = pd.concat([signal, returns.shift(-1), regimes], axis=1).dropna()
            combined.columns = ["signal", "returns", "regime"]
            for regime in combined["regime"].unique():
                subset = combined[combined["regime"] == regime]
                if len(subset) >= 20:
                    r_ic, _ = stats.spearmanr(subset["signal"], subset["returns"])
                    regime_ics[str(regime)] = r_ic if not np.isnan(r_ic) else 0.0

        is_significant = (
            abs(ic_mean) >= self.config.min_ic
            and abs(ir) >= self.config.min_ir
            and abs(t_stat) >= self.config.min_t_stat
        )

        status = SignalStatus.ACTIVE if is_significant else SignalStatus.REJECTED

        result = SignalResult(
            name=name,
            ic=ic,
            ic_mean=ic_mean,
            ic_std=ic_std,
            ir=ir,
            t_stat=t_stat,
            p_value=p_value,
            pct_positive=pct_positive,
            is_significant=is_significant,
            status=status,
            horizon_best=best_horizon,
            ic_by_horizon=ic_by_horizon,
            regime_ics=regime_ics,
        )

        self._results[name] = result
        if is_significant:
            self._active_signals.append(name)
            logger.success(
                f"Alpha Pipeline: signal '{name}' PASSED validation "
                f"(IC={ic_mean:.4f}, IR={ir:.2f}, t={t_stat:.2f})"
            )
        else:
            logger.info(
                f"Alpha Pipeline: signal '{name}' FAILED validation "
                f"(IC={ic_mean:.4f}, IR={ir:.2f}, t={t_stat:.2f})"
            )

        self._save_registry()
        return result

    def get_active_signals(self) -> List[str]:
        return list(set(self._active_signals))

    def get_result(self, name: str) -> Optional[SignalResult]:
        return self._results.get(name)

    def get_all_results(self) -> Dict[str, SignalResult]:
        return dict(self._results)

    def generate_signal_portfolio(self, prices: pd.DataFrame) -> Dict[str, float]:
        """Generate a portfolio of signals from all active signals.

        Returns composite z-scores per symbol.
        """
        if not self._active_signals:
            return {}
        all_signals = {}
        for name in self._active_signals:
            if name in self._signals:
                all_signals[name] = self._signals[name](prices)
        if not all_signals:
            return {}
        df = pd.DataFrame(all_signals)
        composite = df.mean(axis=1)
        # Z-score normalize
        z = (composite - composite.mean()) / max(composite.std(), 1e-10)
        symbol_scores = {}
        if isinstance(prices, pd.DataFrame) and prices.shape[1] > 0:
            for col in prices.columns:
                symbol_scores[col] = float(z.iloc[-1]) if len(z) > 0 else 0.0
        return symbol_scores

    def _align(self, signal: pd.Series, target: pd.Series) -> pd.DataFrame:
        """Align two series and drop NaN."""
        aligned = pd.concat([signal, target], axis=1).dropna()
        if aligned.shape[1] >= 2:
            return aligned
        return pd.DataFrame()

    def _load_registry(self):
        if self.registry_path.exists():
            try:
                with open(self.registry_path) as f:
                    data = json.load(f)
                self._active_signals = data.get("active", [])
                logger.info(
                    f"Loaded {len(self._active_signals)} active signals from registry"
                )
            except Exception as e:
                logger.warning(f"Failed to load signal registry: {e}")

    def _save_registry(self):
        try:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "active": self._active_signals,
                "results": {
                    name: {
                        "name": r.name,
                        "ic": r.ic,
                        "ic_mean": r.ic_mean,
                        "ir": r.ir,
                        "is_significant": r.is_significant,
                        "status": r.status.value,
                    }
                    for name, r in self._results.items()
                },
            }
            with open(self.registry_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save signal registry: {e}")
