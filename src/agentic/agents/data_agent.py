"""
Data Agent — G1: handles TICK_RECEIVED messages from execution_agent.
"""

from __future__ import annotations
import os
import time
import asyncio
from typing import Dict, Optional, Any
from loguru import logger

from agentic.core.base_agent import BaseAgent
from agentic.core.agent_message import (
    MessageType,
    AgentIntention,
    AgentMessage,
)
from agentic.core.agent_consciousness import ConsciousnessLevel, AgentState

from data.data_manager import DataManager, SYMBOLS


# The full screener universe — includes all instruments the system can potentially trade.  # noqa: E501
# The screener_agent tests all of these, and only tradeable ones flow through to processing.  # noqa: E501
# Merge with existing SYMBOLS to ensure backward compatibility.
# NOTE: Screener universe uses Yahoo Finance symbols (EURUSD=X, GC=F, etc.).
# We normalize these to clean symbols for CSV path construction and data loading.
def _clean_symbol(sym: str) -> str:
    """Strip Yahoo Finance '=X' suffix from forex symbols.

    Yahoo-style: EURUSD=X → EURUSD
    Futures: GC=F → GC=F (no =X suffix, unchanged)
    ETFs: GLD → GLD (unchanged)
    """
    return sym.replace("=X", "")


try:
    from agentic.agents.screener_agent import INSTRUMENT_UNIVERSE as SCREENER_UNIVERSE

    SCREENER_SYMBOLS = list(
        set(SYMBOLS) | {_clean_symbol(s) for s in SCREENER_UNIVERSE.keys()}
    )
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

        # Periodic data refresh — runs in background during live mode
        self._refresh_task: Optional[asyncio.Task] = None
        self._refresh_interval: float = (
            max(getattr(config.data, "refresh_interval_minutes", 5), 1) * 60.0
        )  # Convert minutes → seconds, minimum 1 min

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
        # so signal_agent can run ensemble inference without waiting for live bar closures.  # noqa: E501
        if self.consciousness.simulation_mode or self.get_world(
            "agentic.simulation_mode"
        ):
            self.log_state(
                "Simulation mode detected — emitting synthetic features from historical data"  # noqa: E501
            )
            await self._emit_simulation_features()

        # Start background data refresh (runs in all modes)
        self._refresh_task = asyncio.create_task(self._periodic_data_refresh())

    async def _on_stop(self):
        """Cancel background refresh task on shutdown."""
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except (asyncio.CancelledError, Exception):
                pass
            self._refresh_task = None

    async def _periodic_data_refresh(self):  # noqa: C901
        """Background loop: monitor data freshness + periodic full refresh.

        Two-tier approach:
          1. Heartbeat (every 30s): checks if any symbol has gone stale
             (no new bar in >90s). If stale, triggers an immediate refresh
             for that symbol only.
          2. Full refresh (every N minutes): bulk refresh of all symbols ×
             timeframes from cTrader / yFinance.

        This catches data stalls quickly (within 30s) without hammering
        the broker APIs on every heartbeat.
        """
        heartbeat_interval = 30.0  # seconds
        heartbeats_per_full = max(int(self._refresh_interval / heartbeat_interval), 1)
        heartbeat_count = 0

        while True:
            try:
                await asyncio.sleep(heartbeat_interval)
                heartbeat_count += 1

                # --- Heartbeat: check for stale symbols ---
                stale = []
                now = time.time()
                for sym in SCREENER_SYMBOLS:
                    last_ts = self._last_bar_ts.get(sym, 0)
                    if last_ts > 0 and (now - last_ts) > 90:
                        stale.append(sym)

                if stale:
                    broker_ok = self.get_world("execution.connected", False)
                    # Suppress stale warnings when market is closed (weekend, after hours)  # noqa: E501
                    from data.market_session import MarketSession

                    market_open = MarketSession.is_market_open()
                    if broker_ok and market_open:
                        logger.warning(
                            f"Data heartbeat: {len(stale)} stale symbols: "
                            f"{stale[:5]}{'...' if len(stale)>5 else ''}"
                        )
                    # Try quick refresh for stale symbols only
                    # Skip cTrader if broker is disconnected — go straight
                    # to alternative sources (Dukascopy, Yahoo)
                    ctrader = self.get_world("execution.ctrader_client")
                    if ctrader is None:
                        ctrader = getattr(self, "_ctrader_client", None)
                    for sym in stale:
                        try:
                            if broker_ok and ctrader is not None:
                                await self.dm.load_from_ctrader(
                                    sym, "1h", days=1, client=ctrader
                                )
                            else:
                                self.dm.try_alternative_source(sym, "1h", days=1)
                        except Exception:
                            pass

                # --- Full refresh every N heartbeats ---
                if heartbeat_count >= heartbeats_per_full:
                    heartbeat_count = 0
                    await self._refresh_single_cycle()
                    # Also refresh macro/economic data
                    await self._refresh_macro_data()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Data refresh cycle error: {e}")

    async def _refresh_single_cycle(self):  # noqa: C901
        """Refresh OHLCV for all symbols × timeframes in parallel.

        Tries cTrader historical API first (fast, broker-authoritative)
        ONLY when broker is connected.  Falls back to Dukascopy/Yahoo
        when cTrader is unavailable or disconnected.
        All symbols and timeframes are refreshed concurrently via asyncio.gather.
        After refresh, saves to CSV and logs gap/integrity metrics.
        """
        broker_ok = self.get_world("execution.connected", False)
        ctrader = self.get_world("execution.ctrader_client")
        if ctrader is None:
            ctrader = getattr(self, "_ctrader_client", None)
        use_ctrader = broker_ok and ctrader is not None

        timeframes = list(
            dict.fromkeys(self.config.features.timeframes + ["1h"])
        )  # Deduplicate, ensure 1h is present

        async def _refresh_one(sym: str, tf: str) -> bool:
            """Try to refresh a single symbol+timeframe pair."""
            try:
                if use_ctrader:
                    if await self.dm.load_from_ctrader(sym, tf, days=1, client=ctrader):
                        return True
                return bool(self.dm.try_alternative_source(sym, tf, days=1))
            except Exception:
                return False

        # Build all (symbol, timeframe) tasks and run in parallel
        tasks = [_refresh_one(sym, tf) for sym in SCREENER_SYMBOLS for tf in timeframes]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        refreshed = sum(1 for r in results if r is True)

        if refreshed > 0:
            # Persist refreshed data to CSV
            self.dm.save_all_ohlcv(timeframes=timeframes)
            # Log integrity summary
            total_bars = 0
            gap_count = 0
            for sym in SCREENER_SYMBOLS:
                for tf in timeframes:
                    df = self.dm.get_ohlcv(sym, tf)
                    if df is not None and not df.empty:
                        total_bars += len(df)
                        gaps = self.dm.detect_gaps(sym, tf, max_gap_minutes=5)
                        gap_count += len(gaps)
            self.set_world("data.last_refresh_ts", time.time(), ttl=300)
            logger.info(
                f"Data refresh: {refreshed}/{len(tasks)} TF-pairs updated "
                f"({total_bars} bars, {gap_count} gaps)"
            )

    async def _refresh_macro_data(self):
        """Fetch macro/economic data from FRED API and publish to world state.

        Used by the macro_sentiment expert in SignalAgent to factor
        economic surprises into signal generation.
        """
        try:
            from data.economic_calendar import EconomicCalendar

            cal = EconomicCalendar()
            events = cal.fetch(days_forward=3)
            high_impact = [e for e in events if e.is_high_impact()]
            upcoming = cal.get_upcoming_events(hours=48)

            macro_state = {
                "total_events": len(events),
                "high_impact_count": len(high_impact),
                "upcoming_events": [
                    {
                        "currency": e.currency,
                        "event": e.event,
                        "impact": e.impact,
                        "timestamp": e.timestamp,
                    }
                    for e in upcoming[:10]
                ],
                "last_updated": time.time(),
            }
            self.set_world("macro.data", macro_state, ttl=600)
            logger.debug(
                f"Macro refresh: {len(events)} events, "
                f"{len(high_impact)} high impact, "
                f"{len(upcoming)} upcoming"
            )
        except Exception as e:
            logger.debug(f"Macro refresh failed: {e}")

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
        broker_ok = self.get_world("execution.connected", False)
        for sym in decision.get("heal_symbols", []):
            last_heal = self._heal_cooldown.get(sym, 0)
            if now - last_heal < 3600:
                continue
            if broker_ok:
                ctrader = self.get_world("execution.ctrader_client") or getattr(
                    self, "_ctrader_client", None
                )
                healed = self.dm.heal_gaps(
                    sym, max_gap_minutes=180, ctrader_client=ctrader
                )
            else:
                # Broker disconnected — try alternative source instead
                healed = 1 if self.dm.try_alternative_source(sym, "1h", days=1) else 0
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
                        reasoning="new bar closed, features ready for signal generation",  # noqa: E501
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

            # Publish order flow / CVD data for orderflow expert (Phase 7)
            for sym in SCREENER_SYMBOLS:
                of = self.dm.get_order_flow_metrics(sym)
                self.set_world(f"data.orderflow.{sym}", of, ttl=10)

        # G3: Periodically flush OHLCV bars to disk (every 500 cycles ≈ every 50s)
        if self.consciousness.cycle_count % 500 == 0:
            from data.data_manager import TIMEFRAMES as ALL_TFS

            self.dm.save_all_ohlcv(timeframes=ALL_TFS)
            self.log_state(f"OHLCV flushed to disk ({len(ALL_TFS)} timeframes)")

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
                        reasoning="simulation mode: features generated from historical data",  # noqa: E501
                        expected_outcome="signal agent processes features",
                        confidence=0.9,
                    ),
                )
                self._features_dirty[sym] = False
                emitted += 1
        self.log_state(
            f"Simulation bootstrap: emitted features for {emitted}/{len(SCREENER_SYMBOLS)} symbols"  # noqa: E501
        )

    def _check_freshness(self) -> Dict:
        fresh = sum(1 for sym in SCREENER_SYMBOLS if self._last_bar_ts.get(sym, 0) > 0)
        return {
            "fresh": fresh,
            "total": len(SCREENER_SYMBOLS),
            "stale": len(SCREENER_SYMBOLS) - fresh,
        }

    async def _load_historical_data(self):  # noqa: C901
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
            f"Historical: {total_bars} bars across {len(SCREENER_SYMBOLS)} symbols (persisted to disk)"  # noqa: E501
        )

    def _resample_timeframe(self, symbol, tf, df_1h):
        pass

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
