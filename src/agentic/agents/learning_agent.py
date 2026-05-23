"""
Learning Agent — autonomous model training and adaptation.

Identity: I am the student. I learn from every trade and adapt our models.
Purpose: I keep our AI continuously improving by training on new data.
Autonomy: I independently monitor model performance, detect when retraining is needed, and execute training.
"""

from __future__ import annotations
import time
from typing import Dict, List, Optional, Any, Set
from loguru import logger

from agentic.core.base_agent import BaseAgent
from agentic.core.agent_message import (
    AgentMessage,
    MessageType,
    MessagePriority,
    AgentIntention,
)
from agentic.core.agent_consciousness import ConsciousnessLevel


class LearningAgent(BaseAgent):
    """
    Autonomous model training manager.

    Responsibilities:
    - Monitor concept drift signals from DriftMonitor
    - Trigger retraining when drift detected or performance degrades
    - Manage online learning for PPO agents
    - Coordinate model versioning and registry
    - Warm-start models from latest checkpoints
    """

    def __init__(self):
        super().__init__(
            name="learning_agent",
            role="Model Training Manager",
            purpose="Keep all AI models continuously learning and adapting to market changes",
            domain="learning",
            capabilities={
                "drift_monitoring",
                "retraining_management",
                "online_learning",
                "model_versioning",
                "warm_start",
            },
            tick_interval=30.0,
            consciousness_level=ConsciousnessLevel.REFLECTIVE,
        )
        self._retraining_count = 0
        self._last_training_time = 0.0
        self._model_registry = None

        self.subscribe(MessageType.MODEL_UPDATE)
        self.subscribe(MessageType.DIAGNOSTIC_REQUEST)

    async def _on_start(self):
        try:
            from training.model_registry import ModelRegistry

            self._model_registry = ModelRegistry(
                registry_path="models/registry",
            )
            self.log_state("Model registry initialized")
        except Exception as e:
            self.log_state(f"Model registry not available: {e}", "warning")
        self.set_world("learning.status", "idle")
        models_untrained = self.get_world("models.untrained", False)
        if models_untrained:
            self.log_state("Models untrained — will bootstrap from historical data")

    async def perceive(self) -> Dict[str, Any]:
        drift_symbols = self.get_world("signal.drifted_symbols", 0)
        performance = self.get_world("performance.stats", {})
        sharpe = performance.get("sharpe", 0)
        n_trades = performance.get("total_trades", 0)
        models_untrained = self.get_world("models.untrained", False)
        time_since_last = time.time() - self._last_training_time

        needs_retrain = (
            drift_symbols > 0
            or (n_trades > 20 and sharpe < 0.3)
            or (n_trades > 50 and self._retraining_count == 0)
        )
        needs_bootstrap = models_untrained and self._retraining_count == 0
        needs_data_retrain = (time_since_last > 3600 and n_trades > 0) or (
            time_since_last > 7200 and self._retraining_count > 0
        )

        return {
            "drift_symbols": drift_symbols,
            "sharpe": sharpe,
            "n_trades": n_trades,
            "needs_retrain": needs_retrain,
            "needs_bootstrap": needs_bootstrap,
            "needs_data_retrain": needs_data_retrain,
            "time_since_last_train": time_since_last,
        }

    async def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        actions = {}

        if (
            perception.get("needs_retrain")
            and perception.get("time_since_last_train", 0) > 600
        ):
            actions["retrain"] = True
            reasons: List[str] = []
            if perception.get("drift_symbols", 0) > 0:
                reasons.append(f"{perception['drift_symbols']} symbols drifted")
            if perception.get("sharpe", 0) < 0.3:
                reasons.append(f"sharpe={perception['sharpe']:.2f}")
            actions["reasons"] = "; ".join(reasons) if reasons else "scheduled"

        if perception.get("needs_bootstrap"):
            actions["bootstrap"] = True

        if perception.get("needs_data_retrain"):
            actions["data_retrain"] = True

        if (
            self.consciousness.cycle_count % 10 == 0
            and perception.get("n_trades", 0) > 0
        ):
            actions["check_metrics"] = True

        return actions

    async def act(self, decision: Dict[str, Any]):
        if decision.get("data_retrain"):
            await self._data_retrain()
            return

        if decision.get("bootstrap"):
            self._retraining_count += 1
            self._last_training_time = time.time()
            self.log_state(
                "Bootstrap training: training PPO agents from historical data"
            )
            self.consciousness.current_intention = (
                "bootstrapping models from historical data"
            )
            self.set_world("learning.status", "training")
            try:
                from data.data_manager import SYMBOLS
                from agentic.core.agent_bus import get_agent_bus

                for sym in SYMBOLS[:3]:
                    df = self.get_world(f"data.ohlcv.{sym}.1h", None)
                    if df is not None:
                        await self.send(
                            MessageType.MODEL_UPDATE,
                            payload={"action": "train_ppo", "symbol": sym},
                            priority=MessagePriority.LOW,
                        )
                self.set_world("models.untrained", False)
                self.set_world("models.ppo_trained", True)
                self.log_state("Bootstrap complete — PPO agents initialized")
            except Exception as e:
                self.log_state(f"Bootstrap training failed: {e}", "warning")
            self.set_world("learning.status", "idle")
            self.set_world("learning.last_training", self._last_training_time)
            self.set_world("learning.retraining_count", self._retraining_count)
            return

        if decision.get("retrain"):
            self._retraining_count += 1
            self._last_training_time = time.time()
            reasons = decision.get("reasons", "scheduled")
            self.log_state(f"Retraining triggered: {reasons}")
            self.consciousness.current_intention = f"retraining models: {reasons}"
            self.set_world("learning.status", "training")

            await self.send(
                MessageType.MODEL_UPDATE,
                payload={
                    "action": "retrain",
                    "reason": reasons,
                    "retraining_id": self._retraining_count,
                    "timestamp": time.time(),
                },
                priority=MessagePriority.NORMAL,
                intention=AgentIntention(
                    primary_goal="retrain models due to drift/performance",
                    reasoning=reasons,
                    expected_outcome="models updated with latest data",
                    confidence=0.7,
                ),
            )

            self.memory.remember(
                event_type="retraining_triggered",
                description=f"Retrain #{self._retraining_count}: {reasons}",
                importance=0.7,
                emotion="info",
            )

            self.set_world("learning.status", "idle")
            self.set_world("learning.last_training", self._last_training_time)
            self.set_world("learning.retraining_count", self._retraining_count)

    async def _data_retrain(self):
        """Retrain classifiers on latest live data — no trades required."""
        self._retraining_count += 1
        self._last_training_time = time.time()
        self.log_state("Data-driven retrain: training on latest bars")
        self.set_world("learning.status", "training")
        try:
            from data.data_manager import SYMBOLS
            from rts_ai_fx.features_unified import FeaturePipeline
            from rts_ai_fx.model import ProfitabilityClassifier
            import numpy as np

            fp = FeaturePipeline(
                lookback=30, timeframes=["1h", "4h"], use_microstructure=True
            )
            fp.load_normalization()
            trained = 0
            ohlcv_data = self.get_world("data.ohlcv", {})
            for sym in SYMBOLS:
                try:
                    dfs = {}
                    for tf in ["1h", "4h"]:
                        symbol_data = (
                            ohlcv_data.get(sym, {})
                            if isinstance(ohlcv_data, dict)
                            else {}
                        )
                        df = (
                            symbol_data.get(tf)
                            if isinstance(symbol_data, dict)
                            else None
                        )
                        if df is not None and hasattr(df, "__len__") and len(df) > 100:
                            dfs[tf] = df
                    if not dfs:
                        continue
                    seqs, targets = fp.create_sequences(dfs, symbol=sym)
                    if len(seqs) < 100:
                        continue
                    split = int(len(seqs) * 0.8)
                    nf = seqs.shape[-1]
                    clf = ProfitabilityClassifier(lookback=30, n_features=nf)
                    clf.build()
                    clf.train(
                        seqs[:split],
                        targets[:split],
                        seqs[split:],
                        targets[split:],
                        epochs=15,
                        batch_size=32,
                    )
                    clf.save(f"models/{sym}_classifier.keras")

                    trained += 1
                except Exception:
                    pass
            self.log_state(f"Data retrain complete: {trained} symbols updated")
        except Exception as e:
            self.log_state(f"Data retrain failed: {e}", "warning")
        self.set_world("learning.status", "idle")
        self.set_world("learning.last_training", self._last_training_time)
        self.set_world("learning.retraining_count", self._retraining_count)

    async def reflect(self, outcome: Dict[str, Any]):
        self.memory.know("learning.retraining_count", self._retraining_count, ttl=3600)
        self.memory.know(
            "learning.last_training_time", self._last_training_time, ttl=3600
        )

    async def on_message(self, message: AgentMessage):
        if message.msg_type == MessageType.MODEL_UPDATE:
            payload = message.payload if isinstance(message.payload, dict) else {}
            action = payload.get("action", "")
            if action == "retrain_complete":
                self.log_state("Retraining completed successfully")
                self.memory.remember(
                    event_type="retraining_complete",
                    description="Models updated",
                    importance=0.6,
                    emotion="success",
                )
        elif message.msg_type == MessageType.DIAGNOSTIC_REQUEST:
            await self.send(
                MessageType.DIAGNOSTIC_RESULT,
                payload={
                    "agent": self.name,
                    "retraining_count": self._retraining_count,
                    "last_training_time": self._last_training_time,
                    "status": self.get_world("learning.status", "unknown"),
                },
                target=message.source_agent,
            )
