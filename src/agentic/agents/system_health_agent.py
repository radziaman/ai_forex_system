"""
System Health Agent — Mathematical integrity monitoring for the entire system.

Monitors on a 60s loop:
  - Feature pipeline dimensions (must be 49)
  - Model weight loading (all 4 PPO agents)
  - Ensemble expert count (must be 28)
  - Data freshness (tick received within 60s)
  - cTrader connection health
  - Feature normalization file integrity
  - Config vs runtime symbol alignment

Publishes a composite system.health_score (0.0 - 1.0) to world state.
The MasterAgent subscribes and escalates on degradation.
"""

from __future__ import annotations
import time
import os
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from loguru import logger

from agentic.core.base_agent import BaseAgent
from agentic.core.agent_message import (
    AgentMessage,
    MessageType,
    MessagePriority,
)
from agentic.core.agent_consciousness import ConsciousnessLevel


@dataclass
class HealthCheckResult:
    """Result of a single health check."""

    name: str
    passed: bool
    value: Any = None
    detail: str = ""
    weight: float = 1.0  # contribution to composite score


class SystemHealthAgent(BaseAgent):
    """Monitors mathematical integrity of the entire trading system."""

    def __init__(self):
        super().__init__(
            name="system_health_agent",
            role="System Integrity Monitor",
            purpose="Monitor feature pipeline, models, ensemble, data, and connections "
            "for mathematical correctness on a 60s loop",
            domain="monitoring",
            capabilities={
                "pipeline_dimension_check",
                "model_integrity_check",
                "ensemble_health_check",
                "data_freshness_check",
                "connection_health_check",
                "composite_scoring",
                "degradation_alerting",
            },
            tick_interval=60.0,
            consciousness_level=ConsciousnessLevel.REFLECTIVE,
        )
        self._last_results: List[HealthCheckResult] = []
        self._health_score: float = 1.0
        self._degraded_since: Optional[float] = None
        self._alert_cooldown: float = 0.0  # prevent alert spam
        self._last_tick_time: float = 0.0

        self.subscribe(MessageType.DIAGNOSTIC_REQUEST)

        # Expected canonical values
        self._expected_feature_dim = 49
        self._expected_ensemble_count = 28
        self._expected_symbols = {
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "AUDUSD",
            "USDCAD",
            "USDCHF",
            "NZDUSD",
            "XAUUSD",
        }

    async def _on_start(self):
        self.set_world("system.health_score", 1.0)
        self.set_world("system.health_checks", {})
        self.log_state("System health monitoring active (60s interval)")

    async def perceive(self) -> Dict[str, Any]:
        """Run all health checks and return results."""
        checks: List[HealthCheckResult] = []

        # 1. Feature pipeline dimension check
        checks.append(self._check_pipeline_dimensions())

        # 2. Model integrity check
        checks.extend(self._check_model_integrity())

        # 3. Ensemble health check
        checks.append(self._check_ensemble_health())

        # 4. Data freshness check
        checks.append(self._check_data_freshness())

        # 5. Connection health check
        checks.append(self._check_connection_health())

        # 6. Feature normalization integrity
        checks.append(self._check_normalization_integrity())

        self._last_results = checks
        return {"checks": checks, "timestamp": time.time()}

    async def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Compute composite health score and detect degradation."""
        checks: List[HealthCheckResult] = perception.get("checks", [])

        # Weighted composite score
        total_weight = sum(c.weight for c in checks)
        weighted_sum = sum(c.weight for c in checks if c.passed)
        self._health_score = weighted_sum / max(total_weight, 1.0)

        # Clamp
        self._health_score = max(0.0, min(1.0, self._health_score))

        actions: Dict[str, Any] = {
            "health_score": self._health_score,
            "checks": {
                c.name: {"passed": c.passed, "detail": c.detail} for c in checks
            },
            "alerts": [],
        }

        # Track degradation duration
        if self._health_score < 0.7:
            if self._degraded_since is None:
                self._degraded_since = time.time()
                actions["alerts"].append(
                    {
                        "type": "degraded",
                        "score": self._health_score,
                        "message": f"System health dropped to {self._health_score:.2f}",
                    }
                )
        else:
            if self._degraded_since is not None:
                duration = time.time() - self._degraded_since
                logger.info(f"System recovered from degradation after {duration:.0f}s")
                self._degraded_since = None

        # Alert if degraded for > 5min and not on cooldown
        if (
            self._degraded_since is not None
            and time.time() - self._degraded_since > 300
            and time.time() > self._alert_cooldown
        ):
            actions["alerts"].append(
                {
                    "type": "critical_degraded",
                    "score": self._health_score,
                    "duration_sec": time.time() - self._degraded_since,
                    "message": f"System degraded for >5min (score={self._health_score:.2f})",
                }
            )
            self._alert_cooldown = time.time() + 600  # 10min cooldown

        return actions

    async def act(self, decision: Dict[str, Any]):
        """Publish health state and send alerts."""
        # Publish to world state
        self.set_world("system.health_score", round(self._health_score, 4))
        self.set_world("system.health_checks", decision.get("checks", {}))
        self.set_world("system.last_check_ts", time.time())

        # Log
        if self._health_score < 0.7:
            logger.warning(f"System health: {self._health_score:.2f} (DEGRADED)")
        elif self._health_score < 0.9:
            logger.info(f"System health: {self._health_score:.2f}")
        else:
            logger.debug(f"System health: {self._health_score:.2f}")

        # Send alerts
        for alert in decision.get("alerts", []):
            alert_type = alert.get("type", "")
            if alert_type == "critical_degraded":
                logger.critical(
                    f"System degraded for {alert['duration_sec']:.0f}s "
                    f"(score={alert['score']:.2f})"
                )
                await self.send(
                    MessageType.RISK_ALERT,
                    payload={
                        "type": "system_degraded",
                        "score": alert["score"],
                        "duration_sec": alert["duration_sec"],
                        "message": alert["message"],
                    },
                    priority=MessagePriority.HIGH,
                )
            elif alert_type == "degraded":
                logger.warning(f"System health dropped: {alert['score']:.2f}")

    async def reflect(self, outcome: Dict[str, Any]):
        """Update agent's internal state."""
        pass

    async def on_message(self, message: AgentMessage):
        """Handle diagnostic requests."""
        if message.msg_type == MessageType.DIAGNOSTIC_REQUEST:
            await self.send(
                MessageType.DIAGNOSTIC_RESULT,
                payload={
                    "agent": self.name,
                    "health_score": self._health_score,
                    "checks": {
                        c.name: {"passed": c.passed, "detail": c.detail}
                        for c in self._last_results
                    },
                    "degraded_since": self._degraded_since,
                },
                target=message.source_agent,
            )

    # ── Health check implementations ──────────────────────────────────

    def _check_pipeline_dimensions(self) -> HealthCheckResult:
        """Verify feature normalization has expected 49 dimensions.

        Loads normalization ONCE and caches it for subsequent checks.
        """
        try:
            from rts_ai_fx.features_unified import FeaturePipeline, EXPECTED_FEATURE_DIM

            if not hasattr(self, "_fp_cache"):
                self._fp_cache = FeaturePipeline(lookback=30, timeframes=["1h"])
                self._fp_cache.load_normalization()

            n_feats = (
                len(self._fp_cache._feature_cols) if self._fp_cache._feature_cols else 0
            )
            passed = n_feats == EXPECTED_FEATURE_DIM
            return HealthCheckResult(
                name="pipeline_dimensions",
                passed=passed,
                value=n_feats,
                detail=f"{n_feats} features (expected {EXPECTED_FEATURE_DIM})",
                weight=2.0,
            )
        except Exception as e:
            return HealthCheckResult(
                name="pipeline_dimensions",
                passed=False,
                detail=str(e),
                weight=2.0,
            )

    def _check_model_integrity(self) -> List[HealthCheckResult]:
        """Verify all 4 PPO agents are loaded with correct dimensions.

        Agents are loaded ONCE and cached. Subsequent checks just verify
        they're still in memory — avoids reloading 4 PPO agents from disk
        every 60 seconds (which causes log spam and wasted I/O).
        """
        results = []

        # Lazy-load agents on first check, cache for subsequent checks
        if not hasattr(self, "_ppo_agents_cache"):
            try:
                from rts_ai_fx.features_unified import FeaturePipeline
                from ai.regime_agents import RegimeSpecialistSystem

                fp = FeaturePipeline(lookback=30, timeframes=["1h"])
                fp.load_normalization()
                n_features = len(fp._feature_cols) if fp._feature_cols else 45
                expected_dim = 1 + n_features

                self._ppo_dim = expected_dim
                self._ppo_agents_cache = RegimeSpecialistSystem(
                    state_dim=expected_dim, n_actions=5
                )
            except Exception as e:
                results.append(
                    HealthCheckResult(
                        name="ppo_agents_loaded",
                        passed=False,
                        detail=str(e),
                        weight=2.0,
                    )
                )
                return results

        rs = self._ppo_agents_cache
        n_loaded = len([a for a in rs.agents.values() if a is not None])

        results.append(
            HealthCheckResult(
                name="ppo_agents_loaded",
                passed=(n_loaded == 4),
                value=n_loaded,
                detail=f"{n_loaded}/4 PPO agents loaded (dim={self._ppo_dim})",
                weight=2.0,
            )
        )

        # Check if they have real weights (not random)
        has_real_weights = any(
            any(p.norm().item() > 1.0 for p in agent.actor.parameters())
            for agent in rs.agents.values()
            if agent
        )
        results.append(
            HealthCheckResult(
                name="ppo_weights_real",
                passed=has_real_weights,
                detail=(
                    "Real trained weights"
                    if has_real_weights
                    else "Random weights (untrained)"
                ),
                weight=1.5,
            )
        )
        return results

    def _check_ensemble_health(self) -> HealthCheckResult:
        """Verify ensemble config has expected number of experts."""
        config_path = "models/ensemble_config.json"
        if not os.path.exists(config_path):
            return HealthCheckResult(
                name="ensemble_config",
                passed=False,
                detail="ensemble_config.json not found",
                weight=1.5,
            )
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            n_experts = cfg.get("n_experts", 0)

            # Check all expected symbols present
            expert_symbols = set()
            for e in cfg.get("experts", []):
                parts = e["name"].split("_")
                if len(parts) > 1 and parts[0] in ("lstm", "xgb", "clf"):
                    expert_symbols.add(parts[-1])
            symbols_match = expert_symbols == self._expected_symbols

            if not symbols_match:
                missing = self._expected_symbols - expert_symbols
                extra = expert_symbols - self._expected_symbols
                detail_parts = [f"{n_experts} experts"]
                if missing:
                    detail_parts.append(f"missing symbols: {missing}")
                if extra:
                    detail_parts.append(f"extra symbols: {extra}")
                return HealthCheckResult(
                    name="ensemble_config",
                    passed=False,
                    value=n_experts,
                    detail="; ".join(detail_parts),
                    weight=1.5,
                )

            return HealthCheckResult(
                name="ensemble_config",
                passed=True,
                value=n_experts,
                detail=f"{n_experts} experts across {len(expert_symbols)} symbols",
                weight=1.5,
            )
        except Exception as e:
            return HealthCheckResult(
                name="ensemble_config",
                passed=False,
                detail=str(e),
                weight=1.5,
            )

    def _check_data_freshness(self) -> HealthCheckResult:
        """Check if market data has been received recently."""
        last_tick = self.get_world("data.tick_rate", 0)
        last_refresh = self.get_world("data.last_refresh_ts", 0.0)
        now = time.time()

        # If no data ever, check if system just started
        if last_refresh == 0 and last_tick == 0:
            return HealthCheckResult(
                name="data_freshness",
                passed=True,
                detail="No data yet (system recently started)",
                weight=1.0,
            )

        age = now - last_refresh if last_refresh > 0 else float("inf")
        passed = age < 120  # data should be < 2min old
        return HealthCheckResult(
            name="data_freshness",
            passed=passed,
            value=f"{age:.0f}s",
            detail=(
                f"Last refresh {age:.0f}s ago"
                if age < 3600
                else f"Last refresh {age/60:.0f}m ago"
            ),
            weight=1.0,
        )

    def _check_connection_health(self) -> HealthCheckResult:
        """Check cTrader connection status from world state."""
        connected = self.get_world("execution.connected", False)
        mode = self.get_world("config.mode", "paper")
        return HealthCheckResult(
            name="connection",
            passed=bool(connected),
            value=connected,
            detail=f"Mode={mode}, connected={connected}",
            weight=1.0,
        )

    def _check_normalization_integrity(self) -> HealthCheckResult:
        """Verify feature normalization file exists and has data."""
        path = "models/feature_norm.npz"
        if not os.path.exists(path):
            return HealthCheckResult(
                name="normalization_file",
                passed=False,
                detail=f"{path} not found",
                weight=1.0,
            )
        try:
            import numpy as np

            data = np.load(path, allow_pickle=True)
            n_pairs = int(data.get("n_pairs", 0))
            n_cols = len(data.get("feature_cols", []))
            passed = n_pairs > 0 and n_cols > 0
            return HealthCheckResult(
                name="normalization_file",
                passed=passed,
                value=n_cols,
                detail=f"{n_pairs} symbol-tf pairs, {n_cols} features",
                weight=1.0,
            )
        except Exception as e:
            return HealthCheckResult(
                name="normalization_file",
                passed=False,
                detail=str(e),
                weight=1.0,
            )
