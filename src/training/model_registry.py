"""
Model Versioning & A/B Testing Framework.
Tracks model versions, auto-promotes winners, runs A/B tests.
"""

import os
import json
import time
import shutil
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger


@dataclass
class ModelVersion:
    """Metadata for a model version."""

    name: str
    version: str
    path: str
    sharpe: float = 0.0
    return_pct: float = 0.0
    win_rate: float = 0.0
    max_dd: float = 0.0
    profit_factor: float = 0.0
    created_at: float = field(default_factory=time.time)
    is_active: bool = False
    is_champion: bool = False
    total_trades: int = 0
    backtest_results: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


@dataclass
class ABTestConfig:
    """Configuration for A/B test."""

    name: str
    model_a: str  # Control (champion)
    model_b: str  # Treatment (challenger)
    allocation: float = 0.5  # Fraction of trades to model B
    min_trades: int = 50
    confidence: float = 0.95
    auto_promote: bool = True
    promotion_threshold: float = 0.05  # Minimum Sharpe improvement


class ModelRegistry:
    """
    Track model versions and auto-promote winners.
    Implements MLOps-style model lifecycle management.
    """

    def __init__(self, registry_path: str = "models/registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self._models: Dict[str, List[ModelVersion]] = {}
        self._ab_tests: Dict[str, ABTestConfig] = {}
        self._ab_results: Dict[str, Dict] = {}
        self._load_registry()

    def register(
        self,
        name: str,
        path: str,
        performance: Dict,
        set_active: bool = False,
        set_champion: bool = False,
    ) -> ModelVersion:
        """Register a new model version."""
        version_str = f"v{int(time.time())}"
        model = ModelVersion(
            name=name,
            version=version_str,
            path=path,
            **{
                k: v
                for k, v in performance.items()
                if k
                in [
                    "sharpe",
                    "return_pct",
                    "win_rate",
                    "max_dd",
                    "profit_factor",
                    "total_trades",
                    "backtest_results",
                ]
            },
        )

        if name not in self._models:
            self._models[name] = []
        self._models[name].append(model)

        if set_active:
            self.set_active(name, version_str)
        if set_champion:
            self.set_champion(name, version_str)

        self._save_registry()
        logger.info(f"Registered {name} {version_str}: Sharpe={model.sharpe:.2f}")
        return model

    def get_active(self, name: str) -> Optional[ModelVersion]:
        """Get currently active model version."""
        if name not in self._models:
            return None
        for v in self._models[name]:
            if v.is_active:
                return v
        return None

    def get_champion(self, name: str) -> Optional[ModelVersion]:
        """Get champion (best performing) model."""
        if name not in self._models:
            return None
        for v in self._models[name]:
            if v.is_champion:
                return v
        # Fallback to best Sharpe
        best = max(self._models[name], key=lambda v: v.sharpe)
        return best

    def set_active(self, name: str, version: str):
        """Set a specific version as active."""
        if name not in self._models:
            return
        for v in self._models[name]:
            v.is_active = v.version == version
        logger.info(f"Set {name} {version} as active")

    def set_champion(self, name: str, version: str):
        """Set a specific version as champion."""
        if name not in self._models:
            return
        for v in self._models[name]:
            v.is_champion = v.version == version
        logger.info(f"Set {name} {version} as CHAMPION")

    def deploy_candidate(
        self, name: str, min_improvement: float = 0.05
    ) -> Tuple[bool, str]:
        """
        Deploy new model if it beats current champion.
        Returns (deployed, reason).
        """
        champion = self.get_champion(name)
        if not champion:
            return False, "no_champion_exists"

        candidates = sorted(
            [
                v
                for v in self._models.get(name, [])
                if not v.is_champion and v.sharpe > champion.sharpe
            ],
            key=lambda v: v.sharpe,
            reverse=True,
        )

        if not candidates:
            return False, "no_better_candidates"

        best_candidate = candidates[0]
        required_sharpe = champion.sharpe * (1 + min_improvement)

        if best_candidate.sharpe >= required_sharpe:
            self.set_champion(name, best_candidate.version)
            self.set_active(name, best_candidate.version)
            logger.success(
                f"PROMOTED {name} {best_candidate.version}: "
                f"Sharpe {champion.sharpe:.2f} → {best_candidate.sharpe:.2f}"
            )
            return True, f"promoted_{best_candidate.version}"

        return (
            False,
            f"insufficient_improvement_{best_candidate.sharpe:.2f}_vs_{required_sharpe:.2f}",
        )

    def start_ab_test(self, config: ABTestConfig) -> str:
        """Start an A/B test between two models."""
        test_id = f"ab_{config.name}_{int(time.time())}"
        self._ab_tests[test_id] = config
        self._ab_results[test_id] = {
            "model_a_trades": 0,
            "model_a_pnl": 0.0,
            "model_b_trades": 0,
            "model_b_pnl": 0.0,
            "start_time": time.time(),
            "status": "running",
        }
        logger.info(f"Started A/B test {test_id}: {config.model_a} vs {config.model_b}")
        return test_id

    def record_ab_trade(self, test_id: str, model: str, pnl: float):
        """Record a trade result in an A/B test."""
        if test_id not in self._ab_results:
            return
        results = self._ab_results[test_id]
        config = self._ab_tests.get(test_id)
        if not config:
            return

        if model == config.model_a:
            results["model_a_trades"] += 1
            results["model_a_pnl"] += pnl
        elif model == config.model_b:
            results["model_b_trades"] += 1
            results["model_b_pnl"] += pnl

        # Check if we can conclude
        if (
            results["model_a_trades"] >= config.min_trades
            and results["model_b_trades"] >= config.min_trades
        ):
            self._evaluate_ab_test(test_id)

    def _evaluate_ab_test(self, test_id: str):
        """Evaluate A/B test and auto-promote if configured."""
        results = self._ab_results[test_id]
        config = self._ab_tests.get(test_id)
        if not config or results.get("status") != "running":
            return

        # Calculate Sharpe for each model (simplified)
        pnl_a = results["model_a_pnl"]
        pnl_b = results["model_b_pnl"]
        trades_a = results["model_a_trades"]
        trades_b = results["model_b_trades"]

        sharpe_a = pnl_a / trades_a if trades_a > 0 else 0
        sharpe_b = pnl_b / trades_b if trades_b > 0 else 0

        results["sharpe_a"] = sharpe_a
        results["sharpe_b"] = sharpe_b
        results["status"] = "completed"
        results["end_time"] = time.time()

        improvement = (sharpe_b - sharpe_a) / max(abs(sharpe_a), 0.01)

        if config.auto_promote and improvement >= config.promotion_threshold:
            logger.success(
                f"A/B Test {test_id}: PROMOTING {config.model_b} "
                f"(Sharpe {sharpe_a:.2f} → {sharpe_b:.2f}, +{improvement:.1%})"
            )
            self.set_champion(config.name, config.model_b)
            self.set_active(config.name, config.model_b)
        else:
            logger.info(
                f"A/B Test {test_id} completed: "
                f"Model A (Sharpe {sharpe_a:.2f}) vs "
                f"Model B (Sharpe {sharpe_b:.2f}), "
                f"Improvement: {improvement:.1%}"
            )

    def get_model_history(self, name: str) -> List[ModelVersion]:
        """Get version history for a model."""
        return sorted(
            self._models.get(name, []),
            key=lambda v: v.created_at,
            reverse=True,
        )

    def plot_evolution(self, name: str) -> Dict:
        """Get model evolution metrics."""
        history = self.get_model_history(name)
        return {
            "versions": [v.version for v in history],
            "sharpes": [v.sharpe for v in history],
            "returns": [v.return_pct for v in history],
            "win_rates": [v.win_rate for v in history],
        }

    def _save_registry(self):
        """Save registry to disk."""
        try:
            data = {}
            for name, versions in self._models.items():
                data[name] = [
                    {
                        "version": v.version,
                        "path": v.path,
                        "sharpe": v.sharpe,
                        "return_pct": v.return_pct,
                        "win_rate": v.win_rate,
                        "max_dd": v.max_dd,
                        "profit_factor": v.profit_factor,
                        "is_active": v.is_active,
                        "is_champion": v.is_champion,
                        "total_trades": v.total_trades,
                        "created_at": v.created_at,
                        "metadata": v.metadata,
                    }
                    for v in versions
                ]
            with open(self.registry_path / "registry.json", "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")

    def _load_registry(self):
        """Load registry from disk."""
        try:
            registry_file = self.registry_path / "registry.json"
            if not registry_file.exists():
                return
            with open(registry_file) as f:
                data = json.load(f)
            for name, versions in data.items():
                self._models[name] = [ModelVersion(**v) for v in versions]
            logger.info(f"Loaded registry: {len(self._models)} model types")
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
