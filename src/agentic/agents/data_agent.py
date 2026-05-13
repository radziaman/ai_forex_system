"""
Data Agent — G1: handles TICK_RECEIVED messages from execution_agent.
"""

from __future__ import annotations
import time
import asyncio
from typing import Dict, List, Optional, Any, Set
from loguru import logger

from agentic.core.base_agent import BaseAgent
from agentic.core.agent_message import MessageType, MessagePriority, AgentIntention
from agentic.core.agent_consciousness import ConsciousnessLevel, AgentState

from data.data_manager import DataManager, SYMBOLS, DUKASCOPE_SYMBOLS, BASE_PRICES


class DataAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(
            name="data_agent",
            role="Market Data Manager",
            purpose="Ingest, aggregate, and maintain fresh market data for all symbols",
            domain="data",
            capabilities={
                "tick_ingestion", "ohlcv_aggregation", "data_freshness_monitoring",
                "gap_detection", "gap_healing", "multi_source_fallback",
                "feature_caching", "cvd_tracking", "market_depth",
            },
            tick_interval=0.1,
            consciousness_level=ConsciousnessLevel.REFLECTIVE,
        )
        self.config = config
        self.dm = DataManager(historical_path=config.data.historical_path)
        self._last_bar_ts: Dict[str, float] = {}
        self._features_dirty: Dict[str, bool] = {}
        self.tick_counter = 0
        self._symbol: str = SYMBOLS[0] if SYMBOLS else "EURUSD"

        self.subscribe(MessageType.TICK_RECEIVED)
        self.subscribe(MessageType.AGENT_DIRECTIVE)
        self.subscribe(MessageType.DIAGNOSTIC_REQUEST)

    async def _on_start(self):
        self.consciousness.current_intention = "loading historical data"
        await self._load_historical_data()
        fresh = self._check_freshness()
        self.log_state(f"Started: {fresh['fresh']}/{fresh['total']} symbols fresh")
        self.set_world("data.freshness", fresh)
        self.set_world("data.symbols", SYMBOLS)
        self.set_world("data.status", "ready")

    async def perceive(self) -> Dict[str, Any]:
        result = {"skip": False, "new_bars": [], "stale_symbols": [],
                  "tick_rate": self.tick_counter}
        self.tick_counter = 0

        for sym in SYMBOLS:
            df = self.dm.get_ohlcv(sym, "1m")
            if df is not None and not df.empty:
                current_ts = float(df["timestamp"].iloc[-1])
                last_ts = self._last_bar_ts.get(sym, 0)
                if current_ts != last_ts:
                    self._last_bar_ts[sym] = current_ts
                    self._features_dirty[sym] = True
                    result["new_bars"].append(sym)
            last_tick = self._last_bar_ts.get(sym, 0)
            age = time.time() - last_tick if last_tick > 0 else float("inf")
            if age > 300 and last_tick > 0:  # Grace period: skip stale check on first cycle
                result["stale_symbols"].append(sym)

        self.set_world("data.tick_rate", self.tick_counter)
        self.set_world("data.stale_symbols", result["stale_symbols"])

        if not result["new_bars"] and not result["stale_symbols"]:
            result["skip"] = True
        return result

    async def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        decision = {"heal_symbols": [], "emit_features": [], "status_update": False}
        for sym in perception.get("stale_symbols", []):
            decision["heal_symbols"].append(sym)
        for sym in perception.get("new_bars", []):
            df = self.dm.get_ohlcv(sym, "1h")
            if df is not None and len(df) > self.config.features.lookback + 10:
                decision["emit_features"].append(sym)
        if self.consciousness.cycle_count % 50 == 0:
            decision["status_update"] = True
        return decision

    async def act(self, decision: Dict[str, Any]):
        self.consciousness.current_state = AgentState.ACTING
        self.consciousness.current_intention = f"processing {len(decision.get('emit_features', []))} symbols"

        for sym in decision.get("heal_symbols", []):
            gaps = self.dm.detect_gaps(sym)
            if gaps:
                healed = self.dm.heal_gaps(sym)
                if healed > 0:
                    self.memory.remember(event_type="gap_healed",
                        description=f"Healed {healed} gaps for {sym}", importance=0.5, emotion="success")

        for sym in decision.get("emit_features", []):
            features = self._get_features(sym)
            if features is not None:
                df = self.dm.get_ohlcv(sym, "1h")
                price = self.dm.get_price(sym, "1h")
                await self.send(MessageType.FEATURES_READY, payload={
                    "symbol": sym, "timeframe": "1h", "features": features,
                    "ohlcv": df, "price": price, "timestamp": time.time(),
                }, intention=AgentIntention(
                    primary_goal=f"emit features for {sym}",
                    reasoning="new bar closed, features ready for signal generation",
                    expected_outcome="signal agent processes features",
                    confidence=0.9,
                ))
                self._features_dirty[sym] = False

        if decision.get("status_update"):
            fresh = self._check_freshness()
            self.set_world("data.freshness", fresh)

    async def reflect(self, outcome: Dict[str, Any]):
        if self.consciousness.cycle_count % 100 == 0:
            self.memory.consolidate_semantic()
            healthy = sum(1 for sym in SYMBOLS
                         if self.dm.freshness.get(sym, type('',(),{'is_healthy':True})()).is_healthy)
            self.set_world("data.health_pct", healthy / max(len(SYMBOLS), 1))

    async def on_message(self, message: AgentMessage):
        # G1: Handle live ticks from execution_agent
        if message.msg_type == MessageType.TICK_RECEIVED:
            self._on_tick(message)
        elif message.msg_type == MessageType.DIAGNOSTIC_REQUEST:
            fresh = self._check_freshness()
            await self.send(MessageType.DIAGNOSTIC_RESULT, payload={
                "agent": self.name, "freshness": fresh, "tick_rate": self.tick_counter,
            }, target=message.source_agent)
        elif message.msg_type == MessageType.AGENT_DIRECTIVE:
            payload = message.payload or {}
            if isinstance(payload, dict) and payload.get("action") == "reload_data":
                await self._load_historical_data()

    # G1: Process incoming ticks
    def _on_tick(self, message: AgentMessage):
        payload = message.payload if isinstance(message.payload, dict) else {}
        symbol = payload.get("symbol", "")
        bid = payload.get("bid", 0)
        ask = payload.get("ask", 0)
        volume = payload.get("volume", 0)
        ts = payload.get("timestamp", time.time())
        if symbol and bid > 0 and ask > 0:
            self.dm.update_tick(symbol, bid, ask, volume, ts)
            self.tick_counter += 1

    def _get_features(self, symbol: str):
        cached = self.dm.get_cached_features(symbol, "1h")
        if cached is not None and not self._features_dirty.get(symbol, True):
            return cached
        df = self.dm.get_ohlcv(symbol, "1h")
        if df is None or len(df) < self.config.features.lookback + 10:
            return None
        try:
            from rts_ai_fx.features_unified import FeaturePipeline
            fp = FeaturePipeline(
                lookback=self.config.features.lookback,
                timeframes=self.config.features.timeframes,
                use_microstructure=self.config.features.use_microstructure,
            )
            features = fp.transform(self.dm.ohlcv, symbol=symbol)
            if features is not None:
                self.dm.set_cached_features(symbol, "1h", features)
            return features
        except Exception as e:
            logger.debug(f"Feature computation failed for {symbol}: {e}")
            return None

    def _check_freshness(self) -> Dict:
        fresh = sum(1 for sym in SYMBOLS if self._last_bar_ts.get(sym, 0) > 0)
        return {"fresh": fresh, "total": len(SYMBOLS), "stale": len(SYMBOLS) - fresh}

    async def _load_historical_data(self):
        self.consciousness.current_intention = "loading historical data"
        self.dm.load_from_dukascopy_cache(max_hours=168)
        for sym in SYMBOLS:
            df = self.dm.get_ohlcv(sym, "1h")
            if df is None or (hasattr(df, 'empty') and df.empty) or len(df) < 50:
                self.dm.try_alternative_source(sym, "1h", days=60)
        for tf in [tf for tf in self.config.features.timeframes if tf != "1h"]:
            for sym in SYMBOLS:
                df = self.dm.get_ohlcv(sym, "1h")
                if df is not None and hasattr(df, 'empty') and not df.empty and len(df) >= 30:
                    self._resample_timeframe(sym, tf, df)
        total_bars = 0
        for sym in SYMBOLS:
            df = self.dm.get_ohlcv(sym, "1h")
            if df is not None and hasattr(df, 'empty') and not df.empty:
                total_bars += len(df)
        self.log_state(f"Historical: {total_bars} bars across {len(SYMBOLS)} symbols")

    def _resample_timeframe(self, symbol, tf, df_1h):
        import numpy as np, pandas as pd
        if tf == "4h":
            res = df_1h.copy()
            res["timestamp"] = (res["timestamp"] // 14400) * 14400
            resampled = res.groupby("timestamp").agg(
                open=("open", "first"), high=("high", "max"),
                low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
            ).reset_index()
            self.dm.ohlcv[symbol][tf] = resampled
