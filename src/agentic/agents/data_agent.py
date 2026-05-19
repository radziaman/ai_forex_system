"""
Data Agent — G1: handles TICK_RECEIVED messages from execution_agent.
"""

from __future__ import annotations
import os
import time
import asyncio
from typing import Dict, List, Optional, Any, Set
from loguru import logger

from agentic.core.base_agent import BaseAgent
from agentic.core.agent_message import (
    MessageType,
    MessagePriority,
    AgentIntention,
    AgentMessage,
)
from agentic.core.agent_consciousness import ConsciousnessLevel, AgentState

from data.data_manager import DataManager, SYMBOLS, DUKASCOPE_SYMBOLS, BASE_PRICES

# The full screener universe — includes all instruments the system can potentially trade.
# The screener_agent tests all of these, and only tradeable ones flow through to processing.
# Merge with existing SYMBOLS to ensure backward compatibility.
try:
    from agentic.agents.screener_agent import INSTRUMENT_UNIVERSE as SCREENER_UNIVERSE

    SCREENER_SYMBOLS = list(set(SYMBOLS) | set(SCREENER_UNIVERSE.keys()))
except ImportError:
    SCREENER_SYMBOLS = SYMBOLS


class DataAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(
            name="data_agent",
            role="Market Data Manager",
            purpose="Ingest, aggregate, and maintain fresh market data for all symbols",
            domain="data",
            capabilities={
                "tick_ingestion",
                "ohlcv_aggregation",
                "data_freshness_monitoring",
                "gap_detection",
                "gap_healing",
                "multi_source_fallback",
                "feature_caching",
                "cvd_tracking",
                "market_depth",
            },
            tick_interval=0.1,
            consciousness_level=ConsciousnessLevel.REFLECTIVE,
        )
        self.config = config
        self.dm = DataManager(historical_path=config.data.historical_path)
        self._last_bar_ts: Dict[str, float] = {}
        self._features_dirty: Dict[str, bool] = {}
        self.tick_counter = 0
        self._symbol: str = SCREENER_SYMBOLS[0] if SCREENER_SYMBOLS else "EURUSD"
        self._heal_cooldown: Dict[str, float] = {}

        self.subscribe(MessageType.TICK_RECEIVED)
        self.subscribe(MessageType.AGENT_DIRECTIVE)
        self.subscribe(MessageType.DIAGNOSTIC_REQUEST)

    async def _on_start(self):
        self.consciousness.current_intention = "loading historical data"
        await self._load_historical_data()
        fresh = self._check_freshness()
        self.log_state(f"Started: {fresh['fresh']}/{fresh['total']} symbols fresh")
        self.set_world("data.freshness", fresh)
        self.set_world("data.symbols", SCREENER_SYMBOLS)
        self.set_world("data.status", "ready")

        # G18: In simulation mode, emit features immediately from historical data
        # so signal_agent can run ensemble inference without waiting for live bar closures.
        if self.consciousness.simulation_mode or self.get_world(
            "agentic.simulation_mode"
        ):
            self.log_state(
                "Simulation mode detected — emitting synthetic features from historical data"
            )
            await self._emit_simulation_features()

    async def perceive(self) -> Dict[str, Any]:
        result = {
            "skip": False,
            "new_bars": [],
            "stale_symbols": [],
            "tick_rate": self.tick_counter,
        }
        self.tick_counter = 0

        for sym in SCREENER_SYMBOLS:
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
            if (
                age > 300 and last_tick > 0
            ):  # Grace period: skip stale check on first cycle
                result["stale_symbols"].append(sym)

        self.set_world("data.tick_rate", self.tick_counter)
        self.set_world("data.stale_symbols", result["stale_symbols"])

        if not result["new_bars"] and not result["stale_symbols"]:
            result["skip"] = True
        return result

    async def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        decision: Dict[str, Any] = {
            "heal_symbols": [],
            "emit_features": [],
            "status_update": False,
        }
        for sym in perception.get("stale_symbols", []):
            decision["heal_symbols"].append(sym)  # type: ignore[attr-defined]
        for sym in perception.get("new_bars", []):
            df = self.dm.get_ohlcv(sym, "1h")
            if df is not None and len(df) > self.config.features.lookback + 10:
                decision["emit_features"].append(sym)  # type: ignore[attr-defined]
        if self.consciousness.cycle_count % 50 == 0:
            decision["status_update"] = True
        return decision

    async def act(self, decision: Dict[str, Any]):
        self.consciousness.current_state = AgentState.ACTING
        self.consciousness.current_intention = (
            f"processing {len(decision.get('emit_features', []))} symbols"
        )

        now = time.time()
        for sym in decision.get("heal_symbols", []):
            last_heal = self._heal_cooldown.get(sym, 0)
            if now - last_heal < 3600:
                continue
            healed = self.dm.heal_gaps(sym, max_gap_minutes=180)
            if healed > 0:
                self._heal_cooldown[sym] = now
                self.memory.remember(
                    event_type="gap_healed",
                    description=f"Healed {healed} gaps for {sym}",
                    importance=0.5,
                    emotion="success",
                )

        for sym in decision.get("emit_features", []):
            features = self._get_features(sym)
            if features is not None:
                df = self.dm.get_ohlcv(sym, "1h")
                price = self.dm.get_price(sym, "1h")
                await self.send(
                    MessageType.FEATURES_READY,
                    payload={
                        "symbol": sym,
                        "timeframe": "1h",
                        "features": features,
                        "ohlcv": df,
                        "price": price,
                        "timestamp": time.time(),
                    },
                    intention=AgentIntention(
                        primary_goal=f"emit features for {sym}",
                        reasoning="new bar closed, features ready for signal generation",
                        expected_outcome="signal agent processes features",
                        confidence=0.9,
                    ),
                )
                self._features_dirty[sym] = False

        if decision.get("status_update"):
            fresh = self._check_freshness()
            self.set_world("data.freshness", fresh)

    async def reflect(self, outcome: Dict[str, Any]):
        if self.consciousness.cycle_count % 100 == 0:
            self.memory.consolidate_semantic()
            healthy = sum(
                1
                for sym in SCREENER_SYMBOLS
                if self.dm.freshness.get(
                    sym, type("", (), {"is_healthy": True})()
                ).is_healthy
            )
            self.set_world("data.health_pct", healthy / max(len(SCREENER_SYMBOLS), 1))

            # Publish OHLCV reference for regime_agent/feature_agent (G1)
            self.set_world("data.ohlcv", self.dm.ohlcv, ttl=120)
            # Publish primary symbol for agents that need a default
            self.set_world(
                "data.primary_symbol",
                SCREENER_SYMBOLS[0] if SCREENER_SYMBOLS else "EURUSD",
                ttl=120,
            )

            # Publish ATR for all symbols (needed by risk_agent, position_agent)
            for sym in SCREENER_SYMBOLS:
                atr = self.dm.get_atr(sym, "1h", 14)
                self.set_world(f"data.atr.{sym}", atr, ttl=60)

        # G3: Periodically flush OHLCV bars to disk (every 500 cycles ≈ every 50s)
        if self.consciousness.cycle_count % 500 == 0:
            tfs_to_save = ["1m", "5m", "1h"]
            self.dm.save_all_ohlcv(timeframes=tfs_to_save)
            self.log_state(f"OHLCV flushed to disk ({len(tfs_to_save)} timeframes)")

    async def on_message(self, message: AgentMessage):
        # G1: Handle live ticks from execution_agent
        if message.msg_type == MessageType.TICK_RECEIVED:
            self._on_tick(message)
        elif message.msg_type == MessageType.DIAGNOSTIC_REQUEST:
            fresh = self._check_freshness()
            await self.send(
                MessageType.DIAGNOSTIC_RESULT,
                payload={
                    "agent": self.name,
                    "freshness": fresh,
                    "tick_rate": self.tick_counter,
                },
                target=message.source_agent,
            )
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

    async def _emit_simulation_features(self):
        """Emit FEATURES_READY for all symbols with loaded data.

        Called once during startup in simulation mode to kick-start the
        signal pipeline without waiting for live bar closures.
        """
        emitted = 0
        for sym in SCREENER_SYMBOLS:
            features = self._get_features(sym)
            if features is not None:
                df = self.dm.get_ohlcv(sym, "1h")
                price = self.dm.get_price(sym, "1h")
                await self.send(
                    MessageType.FEATURES_READY,
                    payload={
                        "symbol": sym,
                        "timeframe": "1h",
                        "features": features,
                        "ohlcv": df,
                        "price": price,
                        "timestamp": time.time(),
                    },
                    intention=AgentIntention(
                        primary_goal=f"emit synthetic features for {sym}",
                        reasoning="simulation mode: features generated from historical data",
                        expected_outcome="signal agent processes features",
                        confidence=0.9,
                    ),
                )
                self._features_dirty[sym] = False
                emitted += 1
        self.log_state(
            f"Simulation bootstrap: emitted features for {emitted}/{len(SCREENER_SYMBOLS)} symbols"
        )

    def _check_freshness(self) -> Dict:
        fresh = sum(1 for sym in SCREENER_SYMBOLS if self._last_bar_ts.get(sym, 0) > 0)
        return {
            "fresh": fresh,
            "total": len(SCREENER_SYMBOLS),
            "stale": len(SCREENER_SYMBOLS) - fresh,
        }

    async def _load_historical_data(self):
        self.consciousness.current_intention = "loading historical data"
        import pandas as pd  # noqa: F811

        # Ensure DataManager has entries for all SCREENER_SYMBOLS (including non-FX)
        for sym in SCREENER_SYMBOLS:
            if sym not in self.dm.ohlcv:
                self.dm.ohlcv[sym] = {}
                for tf in self.config.features.timeframes:
                    self.dm.ohlcv[sym][tf] = pd.DataFrame(
                        columns=[
                            "timestamp",
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                        ]
                    )

        # Phase 1: Try local CSV cache first (fastest)
        csv_loaded = 0
        for sym in SCREENER_SYMBOLS:
            csv_path = os.path.join(self.dm.historical_path, f"{sym}_1h.csv")
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    if df is not None and len(df) >= 50:
                        self.dm.ohlcv[sym]["1h"] = df
                        self.dm.freshness[sym].last_source = "csv"
                        self.dm.freshness[sym].bar_count["1h"] = len(df)
                        csv_loaded += 1
                        continue
                except Exception as e:
                    logger.debug(f"CSV load failed for {sym}: {e}")

        self.log_state(
            f"CSV cache: loaded {csv_loaded}/{len(SCREENER_SYMBOLS)} symbols"
        )

        # Phase 2: For symbols still without data, try yFinance
        for sym in SCREENER_SYMBOLS:
            df = self.dm.get_ohlcv(sym, "1h")
            if df is None or df.empty or len(df) < 50:
                if self.dm.try_alternative_source(sym, "1h", days=60):
                    logger.info(f"{sym}: loaded from yFinance")
                else:
                    logger.warning(f"No real data for {sym}")

        # Phase 3: Dukascopy BI5 cache (for FX symbols that still need data)
        self.dm.load_from_dukascopy_cache(max_hours=168)

        # Resample to additional timeframes from 1h base
        for tf in [tf for tf in self.config.features.timeframes if tf != "1h"]:
            for sym in SCREENER_SYMBOLS:
                df = self.dm.get_ohlcv(sym, "1h")
                if (
                    df is not None
                    and hasattr(df, "empty")
                    and not df.empty
                    and len(df) >= 30
                ):
                    self._resample_timeframe(sym, tf, df)

        # Persist any newly loaded data to CSV for future boots
        self.dm.save_all_ohlcv()
        total_bars = 0
        for sym in SCREENER_SYMBOLS:
            df = self.dm.get_ohlcv(sym, "1h")
            if df is not None and hasattr(df, "empty") and not df.empty:
                total_bars += len(df)
        self.log_state(
            f"Historical: {total_bars} bars across {len(SCREENER_SYMBOLS)} symbols (persisted to disk)"
        )

    def _resample_timeframe(self, symbol, tf, df_1h):
        import numpy as np
        import pandas as pd

        if tf == "4h":
            res = df_1h.copy()
            res["timestamp"] = (res["timestamp"] // 14400) * 14400
            resampled = (
                res.groupby("timestamp")
                .agg(
                    open=("open", "first"),
                    high=("high", "max"),
                    low=("low", "min"),
                    close=("close", "last"),
                    volume=("volume", "sum"),
                )
                .reset_index()
            )
            self.dm.ohlcv[symbol][tf] = resampled
