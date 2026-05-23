"""
Regime Agent — autonomous market regime detection and transition awareness.

Identity: I see the market's hidden state. I classify every moment into trending/ranging/volatile/crisis.
Purpose: I tell every other agent what kind of market we are in right now.
Autonomy: I independently learn regime transition probabilities from data and detect regime shifts.
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


class RegimeAgent(BaseAgent):
    """
    Autonomous HMM-based regime detector.

    Responsibilities:
    - Maintain 4-state HMM (trending/ranging/volatile/crisis)
    - Fit transition probabilities from historical data
    - Detect regime transitions in real-time
    - Publish REGIME_CHANGED when regime shifts
    - Provide regime-appropriate parameters to other agents
    """

    REGIME_NAMES = ["trending", "ranging", "volatile", "crisis"]

    async def _publish_cross_asset_regime(self):
        """Detect global risk-on/risk-off regime from cross-asset data.

        Combines signals from FX, equity, bond, and commodity markets to
        determine the global risk regime.  During risk-off (crisis),
        the system becomes defensive: reduces position sizes, tightens SLs.
        """
        try:
            # Get prices from world state
            us500 = self.get_world("data.price.US500", 0)
            xauusd = self.get_world("data.price.XAUUSD", 0)
            usdjpy = self.get_world("data.price.USDJPY", 0)
            eurusd = self.get_world("data.price.EURUSD", 0)

            if not all([us500, xauusd, usdjpy]):
                return

            # Simplified risk regime scoring:
            # - Equities up → risk-on
            # - Gold up → mixed (can be risk-on or fear)
            # - USD/JPY up → risk-on (carry trade)
            # - EUR/USD up → risk-on (anti-dollar)

            risk_score = 0.0
            # We store previous prices to compute direction
            prev = self.memory.recall("cross_asset.prev", {})

            if prev:
                if prev.get("us500", 0) > 0:
                    risk_score += 1.0 if us500 > prev["us500"] else -1.0
                if prev.get("xauusd", 0) > 0:
                    risk_score += 0.5 if xauusd > prev["xauusd"] else -0.5
                if prev.get("usdjpy", 0) > 0:
                    risk_score += 1.0 if usdjpy > prev["usdjpy"] else -1.0
                if prev.get("eurusd", 0) > 0:
                    risk_score += 0.5 if eurusd > prev["eurusd"] else -0.5

            # Normalize to -1 to 1
            risk_score = max(-1.0, min(1.0, risk_score / 4.0))

            regime = (
                "risk_on"
                if risk_score > 0.3
                else ("risk_off" if risk_score < -0.3 else "neutral")
            )
            self.set_world("regime.global", regime, ttl=60)
            self.set_world("regime.global_score", risk_score, ttl=60)

            # Store for next comparison
            self.memory.know(
                "cross_asset.prev",
                {
                    "us500": us500,
                    "xauusd": xauusd,
                    "usdjpy": usdjpy,
                    "eurusd": eurusd,
                },
                ttl=120,
            )

            logger.debug(f"Cross-asset regime: {regime} (score={risk_score:.2f})")
        except Exception:
            pass

    def __init__(self, config):
        super().__init__(
            name="regime_agent",
            role="Market Regime Detector",
            purpose="Detect and communicate the current market regime to all agents",
            domain="regime",
            capabilities={
                "hmm_regime_detection",
                "regime_transition_detection",
                "regime_parameter_provision",
                "regime_memory",
                "fallback_rule_based_regime",
            },
            tick_interval=5.0,
            consciousness_level=ConsciousnessLevel.REFLECTIVE,
        )
        self.config = config
        self.detector = None
        self._current_regime: str = "ranging"
        self._previous_regime: str = "ranging"
        self._regime_history: List[str] = []
        self._regime_confidence: float = 0.0
        self._transition_count = 0

        self.subscribe(MessageType.FEATURES_READY)
        self.subscribe(MessageType.DIAGNOSTIC_REQUEST)

    async def _on_start(self):
        self.consciousness.current_intention = "initializing HMM regime detector"
        from rts_ai_fx.regime_detector import HMMRegimeDetector

        self.detector = HMMRegimeDetector(n_regimes=4, lookback=60)
        self.set_world("regime.current", self._current_regime)
        self.set_world("regime.confidence", 0.0)
        self.set_world("regime.history", [])
        self.log_state(f"Initial regime: {self._current_regime}")

    async def perceive(self) -> Dict[str, Any]:
        dfs = self.get_world("data.ohlcv")
        if dfs is None:
            return {"skip": True}
        symbol = self.get_world("data.primary_symbol", "EURUSD")
        df = dfs.get(symbol, {}).get("1h") if isinstance(dfs, dict) else None
        if df is None:
            df = self.get_world(f"data.ohlcv.{symbol}.1h")

        return {
            "symbol": symbol,
            "df_available": df is not None and len(df) > 60,
            "df": df,
        }

    async def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        if not perception.get("df_available"):
            return {"skip": True}

        df = perception.get("df")
        if df is None:
            return {"skip": True, "reason": "no_df"}

        should_fit = self.detector and self.detector.model is None and len(df) >= 100

        regime = self.detector.detect_regime(df) if self.detector else "ranging"
        self._previous_regime = self._current_regime
        self._current_regime = regime
        self._regime_history.append(regime)
        if len(self._regime_history) > 100:
            self._regime_history = self._regime_history[-100:]

        is_transition = self._previous_regime != self._current_regime
        if is_transition:
            self._transition_count += 1
            self.memory.remember(
                event_type="regime_transition",
                description=f"{self._previous_regime} → {self._current_regime}",
                importance=0.8,
                emotion="warning",
                data={"from": self._previous_regime, "to": self._current_regime},
            )

        should_trade = self.detector.should_trade(regime) if self.detector else True
        params = self.detector.get_regime_params(regime) if self.detector else {}

        transition_info = {}
        if self.detector:
            transition_info = self.detector.detect_transition()

        return {
            "should_fit": should_fit,
            "regime": regime,
            "is_transition": is_transition,
            "should_trade": should_trade,
            "params": params,
            "transition_info": transition_info,
            "symbol": perception["symbol"],
        }

    async def act(self, decision: Dict[str, Any]):
        if decision.get("should_fit"):
            self.detector.fit(decision.get("df"))
            self.log_state("HMM fitted on historical data")

        self.set_world("regime.current", decision["regime"])
        self.set_world("regime.history", self._regime_history)
        self.set_world("regime.params", decision.get("params", {}))
        self.set_world(f"regime.{decision['symbol']}", decision["regime"])

        if decision["is_transition"]:
            await self.send(
                MessageType.REGIME_CHANGED,
                payload={
                    "from": self._previous_regime,
                    "to": decision["regime"],
                    "transition_info": decision.get("transition_info", {}),
                    "params": decision.get("params", {}),
                    "should_trade": decision["should_trade"],
                    "timestamp": time.time(),
                },
                priority=MessagePriority.HIGH,
                intention=AgentIntention(
                    primary_goal="notify all agents of regime change",
                    reasoning=f"market regime shifted from {self._previous_regime} to {decision['regime']}",
                    expected_outcome="agents adjust strategies for new regime",
                    confidence=0.85,
                ),
            )

        self._regime_confidence = 0.8 if not decision.get("is_transition") else 0.6
        self.set_world("regime.confidence", self._regime_confidence)

    async def reflect(self, outcome: Dict[str, Any]):
        if self.consciousness.cycle_count % 50 == 0:
            self.memory.know("regime.current", self._current_regime, ttl=300)
            self.memory.know(
                "regime.transition_count", self._transition_count, ttl=3600
            )
            n_transitions = sum(
                1
                for i in range(1, len(self._regime_history))
                if self._regime_history[i] != self._regime_history[i - 1]
            )
            stability = 1.0 - (n_transitions / max(len(self._regime_history) - 1, 1))
            self.set_world("regime.stability", round(stability, 3))

        # Publish cross-asset regime every cycle (~5s)
        await self._publish_cross_asset_regime()

    async def on_message(self, message: AgentMessage):
        if message.msg_type == MessageType.DIAGNOSTIC_REQUEST:
            await self.send(
                MessageType.DIAGNOSTIC_RESULT,
                payload={
                    "agent": self.name,
                    "current_regime": self._current_regime,
                    "history": self._regime_history[-20:],
                    "transitions": self._transition_count,
                    "hmm_fitted": self.detector is not None
                    and self.detector.model is not None,
                },
                target=message.source_agent,
            )
