"""
Position Manager — autonomous oversight of all open positions.
Trailing stops, partial closes, correlation monitoring, concentration alerts.
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger


class PositionManager:
    """
    Manages the lifecycle of open positions:
    - Multi-tier trailing stops (3-tier Zenox-style)
    - Partial close suggestions at profit targets
    - Cross-symbol correlation monitoring
    - Concentration risk alerts
    """

    def __init__(self, engine, data_pipeline):
        self._engine = engine
        self._data = data_pipeline
        self._tp_hit: Dict[int, List[bool]] = {}
        self._last_check: Dict[str, float] = {}
        self._check_interval = 5.0

    async def tick(self):
        """Called every cycle. Evaluates all open positions."""
        now = time.time()
        for pid, trade in list(self._engine.open_positions.items()):
            if now - self._last_check.get(str(pid), 0) < self._check_interval:
                continue
            self._last_check[str(pid)] = now
            await self._evaluate_position(pid, trade)

    async def _evaluate_position(self, pid: int, trade):
        """Evaluate a single position for trail/partial/correlation."""
        symbol = trade.symbol
        direction = trade.direction
        entry = trade.entry_price
        current = self._get_current_price(symbol)
        atr = self._get_atr(symbol)

        if current <= 0 or atr <= 0:
            return

        # Multi-tier trailing stop
        new_sl = self._check_trailing_stop(pid, entry, current, atr, direction)
        if new_sl and new_sl != trade.sl:
            await self._update_sl(pid, new_sl)

        # Partial close suggestion at profit targets
        pnl_pct = self._pnl_pct(entry, current, direction)
        suggestion = self._partial_close_suggestion(pid, pnl_pct)
        if suggestion:
            logger.info(
                f"[PositionManager] {symbol} {direction} "
                f"pnl={pnl_pct:.1%} → {suggestion}"
            )

    def _check_trailing_stop(
        self, pid: int, entry: float, current: float,
        atr: float, direction: str,
    ) -> Optional[float]:
        """3-tier trailing stop activation."""
        if pid not in self._tp_hit:
            self._tp_hit[pid] = [False, False, False]

        pnl_pct = self._pnl_pct(entry, current, direction)
        hits = self._tp_hit[pid]
        mult = 1 if direction == "BUY" else -1

        # Tier 1: 1% profit → move SL to breakeven
        if pnl_pct >= 0.01 and not hits[0]:
            hits[0] = True
            logger.info(f"[Trail {pid}] Tier 1: breakeven")
            return entry

        # Tier 2: 2% profit → trail by 1 ATR
        if pnl_pct >= 0.02 and not hits[1]:
            hits[1] = True
            sl = entry + atr * 2 * mult if direction == "BUY" else entry - atr * 2 * mult
            logger.info(f"[Trail {pid}] Tier 2: trail={sl:.5f}")
            return sl

        # Tier 3: 3% profit → tight trail by 0.5 ATR
        if pnl_pct >= 0.03 and hits[0] and hits[1]:
            if not hits[2]:
                hits[2] = True
            sl = current - atr * 0.5 * mult if direction == "BUY" else current + atr * 0.5 * mult
            return sl

        return None

    def _partial_close_suggestion(self, pid: int, pnl_pct: float) -> Optional[str]:
        """Suggest partial close at profit milestones."""
        if pnl_pct >= 0.05:
            return "Close 30% at +5%"
        if pnl_pct >= 0.03:
            return "Close 20% at +3%"
        return None

    async def _update_sl(self, pid: int, new_sl: float):
        """Update stop loss on a position."""
        trade = self._engine.open_positions.get(pid)
        if trade:
            trade.sl = new_sl
            logger.info(f"[PositionManager] Updated SL for {trade.symbol}: {new_sl:.5f}")

    def get_concentration_risk(self) -> List[Dict]:
        """Alert if any symbol exceeds concentration threshold."""
        positions = self._engine.get_open_positions()
        if not positions:
            return []
        total_exposure = sum(abs(p.get("volume", 0) * self._get_current_price(p.get("symbol", "")))
                            for p in positions)
        alerts = []
        for p in positions:
            sym = p.get("symbol", "")
            exposure = abs(p.get("volume", 0) * self._get_current_price(sym))
            share = exposure / max(total_exposure, 1)
            if share > 0.4:
                alerts.append({
                    "symbol": sym,
                    "share_pct": round(share * 100, 1),
                    "action": "Reduce position — exceeds 40% of portfolio",
                })
        return alerts

    def get_correlation_warnings(self) -> List[str]:
        """Warn if correlated positions exist (e.g. EURUSD + GBPUSD both long)."""
        positions = self._engine.get_open_positions()
        if len(positions) < 2:
            return []
        CORRELATED_GROUPS = [
            {"EURUSD", "GBPUSD", "EURGBP"},
            {"USDJPY", "GBPJPY", "EURJPY"},
            {"XAUUSD", "XAGUSD"},
            {"BTCUSD", "ETHUSD"},
            {"US500", "US30", "USTEC"},
        ]
        active_symbols = {p["symbol"] for p in positions}
        directions = {p["symbol"]: p.get("direction", "BUY") for p in positions}
        warnings = []
        for group in CORRELATED_GROUPS:
            active_in_group = active_symbols & group
            if len(active_in_group) >= 2:
                dirs = {s: directions.get(s) for s in active_in_group}
                if len({d for d in dirs.values()}) == 1:
                    warnings.append(
                        f"Correlated long bias: {', '.join(active_in_group)}"
                    )
        return warnings

    def get_all_status(self) -> Dict:
        return {
            "correlation_warnings": self.get_correlation_warnings(),
            "concentration_alerts": self.get_concentration_risk(),
        }

    def _get_current_price(self, symbol: str) -> float:
        return self._data.get_live_price(symbol) or self._data.get_price(symbol)

    def _get_atr(self, symbol: str) -> float:
        return self._data.get_atr(symbol)

    @staticmethod
    def _pnl_pct(entry: float, current: float, direction: str) -> float:
        if entry == 0:
            return 0.0
        mult = 1 if direction == "BUY" else -1
        return (current - entry) / entry * mult
