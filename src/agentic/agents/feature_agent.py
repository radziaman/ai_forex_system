"""
Feature Agent — autonomous feature engineering and computation.

Identity: I transform raw OHLCV data into mathematical features.
Purpose: I bridge data and intelligence — my features are what every model sees.
Autonomy: I independently compute, normalize, cache, and version all features.
"""

from __future__ import annotations
import time
from typing import Dict, Any, Optional, List
from loguru import logger

import numpy as np

from agentic.core.base_agent import BaseAgent
from agentic.core.agent_message import AgentMessage, MessageType
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
            purpose="Transform raw OHLCV into normalized multi-timeframe feature vectors",  # noqa: E501
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
                f"Loaded normalization: {len(self._feature_pipeline._means)} symbol-tf pairs"  # noqa: E501
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

    def _build_sentiment_scores(self) -> Optional[Dict[str, float]]:
        """Build sentiment_scores dict from world state and alternative data.

        Returns dict mapping currency code → sentiment score (-1 to +1), or
        None if no orthogonal data is available.

        Sources (in priority order):
          1. AlternativeDataAggregator composite signals (oil, gold, CB)
          2. FRED macro data (high-impact events)
        """
        scores: Dict[str, float] = {}

        # Source 1: AlternativeDataAggregator (commodity + CB signals)
        try:
            alts = self.get_world("alternative_data.snapshot", None)
            if alts and isinstance(alts, dict):
                composite = alts.get("composite", {})
                if isinstance(composite, dict):
                    for pair, signal in composite.items():
                        # Map pair-level composite signal to currency sentiment
                        if isinstance(signal, (int, float)):
                            if "USD" in pair:
                                scores["USD"] = scores.get("USD", 0) + signal * 0.3
                            if "EUR" in pair:
                                scores["EUR"] = scores.get("EUR", 0) + signal * 0.3
                            if "GBP" in pair:
                                scores["GBP"] = scores.get("GBP", 0) + signal * 0.3
                            if "JPY" in pair:
                                scores["JPY"] = scores.get("JPY", 0) + signal * 0.3
                            if "AUD" in pair:
                                scores["AUD"] = scores.get("AUD", 0) + signal * 0.3
                            if "CAD" in pair:
                                scores["CAD"] = scores.get("CAD", 0) + signal * 0.3
                            if "NZD" in pair:
                                scores["NZD"] = scores.get("NZD", 0) + signal * 0.3
                            if "CHF" in pair:
                                scores["CHF"] = scores.get("CHF", 0) + signal * 0.3
        except Exception:
            pass

        # Source 2: FRED macro calendar — high-impact events signal volatility
        try:
            macro = self.get_world("macro.data", None)
            if macro and isinstance(macro, dict):
                high_impact = macro.get("high_impact_count", 0)
                if high_impact > 0:
                    for currency in scores:
                        scores[currency] = scores.get(currency, 0) - 0.1 * min(
                            high_impact, 5
                        )
        except Exception:
            pass

        return scores if scores else None

    def _build_external_signals(self) -> Optional[np.ndarray]:
        """Build external_signals array from orthogonal non-price data sources.

        Sources:
          1. FRED macro calendar — high-impact event count
          2. Options market — 25d risk reversal sentiment per major pair
          3. NASA EONET — natural disaster impact score (safe-haven flows)
          4. NASA POWER — agricultural weather anomaly (commodity currencies)

        Returns a numpy array fused into the feature vector as additional
        dimensions — genuinely orthogonal to the 49 price-based features.
        """
        signals: List[float] = []

        # ── Source 1: FRED macro calendar ──
        macro = self.get_world("macro.data", None)
        if macro and isinstance(macro, dict):
            signals.append(macro.get("high_impact_count", 0) / 10.0)
            signals.append(min(macro.get("total_events", 0) / 50.0, 1.0))
        else:
            signals.extend([0.0, 0.0])

        # ── Source 2: Options market — institutional positioning ──
        try:
            from data.options_data import OptionsDataProvider

            opts = OptionsDataProvider()
            for pair in ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]:
                rr = opts.get_25d_risk_reversal(pair)  # bullish/bearish skew
                bf = opts.get_butterfly_spread(pair)  # crash protection
                sk = opts.get_skew_index(pair)  # downside fear
                signals.extend([rr * 10, bf * 10, sk])
        except Exception:
            signals.extend([0.0, 0.0, 0.0] * 4)

        # ── Source 3: NASA natural events & alternative data ──
        try:
            from data.alternative_data import AlternativeDataEngine  # type: ignore[attr-defined]

            alt = AlternativeDataEngine()
            signals.append(alt.get_natural_event_impact())  # 0-1
            signals.append(alt.get_agricultural_weather_score())  # 0-1
            signals.append(alt.get_oil_shipping_score())  # 0-1
        except Exception:
            signals.extend([0.0, 0.0, 0.0])

        # ── Total: 2 (FRED) + 12 (4 pairs × 3 options) + 3 (NASA) = 17 ──
        arr = np.array(signals, dtype=np.float32)
        return arr if arr.any() else None

    async def on_message(self, message: AgentMessage):
        if message.msg_type == MessageType.FEATURES_READY:
            payload = message.payload if isinstance(message.payload, dict) else {}
            symbol = payload.get("symbol", "")
            logger.debug(f"FEATURES_READY received for {symbol}")
            if symbol and self._feature_pipeline:
                dfs = self.get_world("data.ohlcv")
                logger.debug(
                    f"data.ohlcv type={type(dfs).__name__}, "
                    f"has_eurusd={bool(dfs and 'EURUSD' in dfs)}"
                )
                if dfs:
                    # Build orthogonal data signals to fuse into feature vector
                    sentiment_scores = self._build_sentiment_scores()
                    external_signals = self._build_external_signals()

                    features = self._feature_pipeline.transform(
                        dfs,
                        symbol=symbol,
                        # sentiment_scores and external_signals are NOT passed
                        # into transform() — they would change the feature
                        # dimension and break existing models. Instead, they
                        # are attached as separate metadata below.
                    )
                    if features is not None:
                        payload: Dict[str, Any] = {
                            "symbol": symbol,
                            "features": features,
                            "timestamp": time.time(),
                        }
                        # Orthogonal signals travel as metadata alongside
                        # the 49-dim feature vector. The signal agent / ensemble
                        # can consume them at the decision level without
                        # changing model input dimensions.
                        if sentiment_scores:
                            payload["sentiment_scores"] = sentiment_scores
                        if external_signals is not None:
                            payload["external_signals"] = external_signals.tolist()
                            payload["n_external"] = len(external_signals)

                        await self.send(
                            MessageType.FEATURES_READY,
                            payload=payload,
                        )
                        if sentiment_scores or external_signals is not None:
                            self.log_state(
                                f"Orthogonal data attached for {symbol}",
                                "debug",
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
