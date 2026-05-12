"""
Data Agent — autonomous data freshness monitoring, gap healing, and symbol prioritization.
"""

import time
from typing import Dict, List, Optional
from collections import defaultdict
from loguru import logger


class DataAgent:
    """
    Monitors data pipeline health:
    - Tracks tick freshness per symbol
    - Detects stale data and triggers gap healing
    - Prioritizes symbols with active positions for data quality
    - Reports data health metrics
    """

    def __init__(self, data_pipeline, symbols: List[str], stale_threshold: float = 120.0):
        self._dp = data_pipeline
        self._symbols = symbols
        self._stale_threshold = stale_threshold
        self._last_tick: Dict[str, float] = {}
        self._tick_rate: Dict[str, float] = defaultdict(float)
        self._gap_alerts: Dict[str, float] = {}
        self._fresh_count = 0
        self._stale_count = 0

    def record_tick(self, symbol: str):
        """Called on every tick. Updates freshness tracking."""
        now = time.time()
        self._last_tick[symbol] = now
        self._tick_rate[symbol] = self._tick_rate.get(symbol, 0) + 1

    def tick(self) -> Dict:
        """Called every cycle. Updates freshness from data pipeline prices."""
        now = time.time()
        fresh = []
        stale = []
        missing = []

        for sym in self._symbols:
            price = self._dp.get_live_price(sym)
            if price is not None and price > 0:
                self._last_tick[sym] = now
                self._tick_rate[sym] = self._tick_rate.get(sym, 0) + 1

            last = self._last_tick.get(sym, 0)
            if last == 0:
                missing.append(sym)
            elif now - last < self._stale_threshold:
                fresh.append(sym)
            else:
                stale.append(sym)

        self._fresh_count = len(fresh)
        self._stale_count = len(stale)

        report = {
            "total": len(self._symbols),
            "fresh": len(fresh),
            "stale": len(stale),
            "missing": len(missing),
            "fresh_symbols": fresh[:5],
            "stale_symbols": stale[:5],
        }

        if stale and self._should_heal(stale):
            self._request_heal(stale)

        return report

    def _should_heal(self, stale_symbols: List[str]) -> bool:
        """Only request healing once per symbol per 5 minutes."""
        now = time.time()
        for sym in stale_symbols:
            last = self._gap_alerts.get(sym, 0)
            if now - last > 300:
                return True
        return False

    def _request_heal(self, stale_symbols: List[str]):
        """Try to fill data gaps from alternative sources."""
        now = time.time()
        dm = self._dp.data_manager
        for sym in stale_symbols:
            self._gap_alerts[sym] = now
            try:
                if hasattr(dm, "detect_gaps") and hasattr(dm, "heal_gaps"):
                    gaps = dm.detect_gaps(sym)
                    if gaps:
                        dm.heal_gaps(sym, gaps)
                        logger.info(f"[DataAgent] Healed {len(gaps)} gaps for {sym}")
            except Exception as e:
                logger.debug(f"[DataAgent] Heal failed for {sym}: {e}")

    def get_priority_symbols(self, open_positions: List[Dict]) -> List[str]:
        """Return symbols sorted by data priority (active positions first)."""
        active = {p.get("symbol") for p in open_positions if p.get("symbol")}
        priority = []
        for sym in self._symbols:
            if sym in active:
                priority.insert(0, sym)
            else:
                priority.append(sym)
        return priority

    def get_status(self) -> Dict:
        return {
            "fresh_pct": round(
                self._fresh_count / max(len(self._symbols), 1) * 100, 1
            ),
            "stale_count": self._stale_count,
            "tick_rates": {
                sym: round(rate, 1)
                for sym, rate in sorted(
                    self._tick_rate.items(), key=lambda x: x[1], reverse=True
                )[:5]
            },
        }
