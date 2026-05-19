"""
Feature Agent — autonomous feature engineering and computation.

Identity: I transform raw OHLCV data into mathematical features.
Purpose: I bridge data and intelligence — my features are what every model sees.
Autonomy: I independently compute, normalize, cache, and version all features.
"""

from __future__ import annotations
import time
import numpy as np
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


class FeatureAgent(BaseAgent):
    """
    Autonomous feature engineering pipeline.

    Responsibilities:
    - Compute 55+ technical features across multiple timeframes
    - Apply z-score normalization with persistent statistics
    - Manage feature cache with hash-based invalidation
    - Version feature schemas to detect dimensional drift
    - Publish feature vectors to signal agent
    """

    FEATURE_SCHEMA_VERSION = "4.0.0"

    def __init__(self, config):
        super().__init__(
            name="feature_agent",
            role="Feature Engineering Pipeline",
            purpose="Transform raw OHLCV into normalized multi-timeframe feature vectors",
            domain="features",
            capabilities={
                "technical_indicators",
                "multi_timeframe_fusion",
                "z_score_normalization",
                "feature_caching",
                "microstructure_features",
                "cross_asset_features",
                "cyclical_encoding",
                "hurst_exponent",
            },
            tick_interval=0.5,
            consciousness_level=ConsciousnessLevel.REFLECTIVE,
        )
        self.config = config
        self._feature_pipeline = None
        self._feature_cols: List[str] = []
        self._means: Dict[str, np.ndarray] = {}
        self._stds: Dict[str, np.ndarray] = {}

        self.subscribe(MessageType.FEATURES_READY)
        self.subscribe(MessageType.AGENT_DIRECTIVE)
        self.subscribe(MessageType.MODEL_UPDATE)
        self.subscribe(MessageType.DIAGNOSTIC_REQUEST)

    async def _on_start(self):
        self.consciousness.current_intention = "initializing feature pipeline"
        from rts_ai_fx.features_unified import FeaturePipeline

        self._feature_pipeline = FeaturePipeline(
            lookback=self.config.features.lookback,
            timeframes=self.config.features.timeframes,
            use_microstructure=self.config.features.use_microstructure,
            use_cross_asset=self.config.features.use_cross_asset,
        )
        loaded = self._feature_pipeline.load_normalization()
        if loaded:
            self.log_state(
                f"Loaded normalization: {len(self._feature_pipeline._means)} symbol-tf pairs"
            )
        else:
            self.log_state("No normalization found — will fit on first data")
        self.set_world("feature.schema_version", self.FEATURE_SCHEMA_VERSION)
        self.memory.know(
            "schema_version",
            self.FEATURE_SCHEMA_VERSION,
            confidence=1.0,
            ttl=86400 * 30,
        )

    async def perceive(self) -> Dict[str, Any]:
        dfs = self.get_world("data.ohlcv")
        if dfs is None:
            return {"skip": True}
        features_requests = self.get_world("feature.pending_requests", [])
        return {"dfs_available": dfs is not None, "pending": features_requests}

    async def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "should_fit": perception.get("dfs_available", False)
            and self._feature_pipeline is not None
        }

    async def act(self, decision: Dict[str, Any]):
        if decision.get("should_fit") and not self._feature_pipeline._means:
            dfs = self.get_world("data.ohlcv")
            if dfs:
                self._feature_pipeline.fit_all(dfs)
                self._feature_pipeline.save_normalization()
                self.log_state("Feature pipeline fitted and saved")
                self.set_world("feature.ready", True)

    async def reflect(self, outcome: Dict[str, Any]):
        if self.consciousness.cycle_count % 200 == 0:
            n_fitted = (
                len(self._feature_pipeline._means) if self._feature_pipeline else 0
            )
            self.memory.know("fitted_pairs", n_fitted, ttl=3600)

    async def on_message(self, message: AgentMessage):
        if message.msg_type == MessageType.FEATURES_READY:
            payload = message.payload if isinstance(message.payload, dict) else {}
            symbol = payload.get("symbol", "")
            if symbol and self._feature_pipeline:
                dfs = self.get_world("data.ohlcv")
                if dfs:
                    features = self._feature_pipeline.transform(
                        dfs,
                        symbol=symbol,
                    )
                    if features is not None:
                        await self.send(
                            MessageType.FEATURES_READY,
                            payload={
                                "symbol": symbol,
                                "features": features,
                                "timestamp": time.time(),
                            },
                        )
        elif message.msg_type == MessageType.MODEL_UPDATE:
            payload = message.payload or {}
            if isinstance(payload, dict) and payload.get("action") == "refit_features":
                self._feature_pipeline._means = {}
                self._feature_pipeline._stds = {}
                self.log_state("Feature normalization reset for refit")
        elif message.msg_type == MessageType.DIAGNOSTIC_REQUEST:
            n_cols = len(self._feature_cols)
            n_means = (
                len(self._feature_pipeline._means) if self._feature_pipeline else 0
            )
            await self.send(
                MessageType.DIAGNOSTIC_RESULT,
                payload={
                    "agent": self.name,
                    "n_features": n_cols,
                    "n_fitted": n_means,
                    "schema": self.FEATURE_SCHEMA_VERSION,
                },
                target=message.source_agent,
            )
