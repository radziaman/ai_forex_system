"""SymbolDiscovery — auto-discovers tradeable forex symbols.

Periodically scans candidate symbols, evaluates them for tradeability
(liquidity, volatility, data availability), and emits symbol list updates.

Events consumed:
    tick -> {symbol, bid, ask, volume, timestamp}

Events emitted:
    symbols_updated -> {symbols: List[str], added: List[str], removed: List[str]}
    symbol_evaluated -> {symbol: str, score: float, spread: float, volatility: float}
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple
from loguru import logger


class SymbolDiscovery:
    """Auto-discovers tradeable forex symbols based on liquidity and volatility.

    Maintains a pool of candidate symbols, evaluates them on spread, volatility,
    and data availability, and emits 'symbols_updated' events when the set
    of active symbols changes.
    """

    # Candidate forex symbols ordered by liquidity (majors first)
    CANDIDATE_SYMBOLS: List[str] = [
        # Majors (highest liquidity)
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
        # Minors
        "EURGBP", "EURJPY", "GBPJPY", "EURAUD", "EURCHF",
        "GBPAUD", "GBPCAD", "GBPNZD", "AUDJPY", "CHFJPY",
        "NZDJPY", "EURNZD", "AUDCAD", "AUDCHF", "CADJPY",
        "NZDCAD", "NZDCHF",
        # Exotics (lower liquidity)
        "USDZAR", "USDTRY", "USDMXN", "USDSGD", "USDHKD",
        "EURTRY", "EURZAR",
    ]

    def __init__(
        self,
        event_bus,
        data_manager=None,
        max_symbols: int = 11,
        min_spread_score: float = 0.3,
        min_volatility_score: float = 0.3,
        scan_interval: float = 3600.0,
        warmup_ticks: int = 100,
    ):
        self._bus = event_bus
        self._data = data_manager
        self._max_symbols = max_symbols
        self._min_spread_score = min_spread_score
        self._min_volatility_score = min_volatility_score
        self._scan_interval = scan_interval
        self._warmup_ticks = warmup_ticks

        # Default active set (majors + popular minors)
        self._active_symbols: List[str] = [
            "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD",
            "AUDUSD", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY", "EURAUD",
        ]

        # Per-symbol statistics
        self._tick_counts: Dict[str, int] = {}
        self._spreads: Dict[str, List[float]] = {}
        self._prices: Dict[str, List[float]] = {}
        self._symbol_scores: Dict[str, float] = {}

        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Subscribe to ticks and start periodic scanning."""
        self._bus.on("tick", self._on_tick)
        self._bus.on("symbol_discovery_scan", self._on_manual_scan)
        self._running = True
        self._task = asyncio.create_task(self._scan_loop())
        logger.info(
            f"SymbolDiscovery started: {len(self.CANDIDATE_SYMBOLS)} candidates, "
            f"max {self._max_symbols} active, scan every {self._scan_interval}s"
        )

    async def stop(self):
        """Stop symbol discovery."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        self._bus.off("tick", self._on_tick)
        self._bus.off("symbol_discovery_scan", self._on_manual_scan)
        logger.info("SymbolDiscovery stopped")

    async def _on_manual_scan(self, **data):
        """Handle manual scan trigger from EventBus."""
        logger.info("SymbolDiscovery: manual scan triggered")
        await self._evaluate_and_update()

    async def _on_tick(self, **data):
        """Collect tick data for symbol evaluation."""
        symbol = data.get("symbol")
        if symbol not in self.CANDIDATE_SYMBOLS:
            return

        bid = data.get("bid", 0)
        ask = data.get("ask", 0)
        spread = ask - bid if ask > bid else 0.0

        self._tick_counts[symbol] = self._tick_counts.get(symbol, 0) + 1
        if symbol not in self._spreads:
            self._spreads[symbol] = []
        self._spreads[symbol].append(spread)
        if len(self._spreads[symbol]) > 200:
            self._spreads[symbol] = self._spreads[symbol][-200:]

        mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
        if mid > 0:
            if symbol not in self._prices:
                self._prices[symbol] = []
            self._prices[symbol].append(mid)
            if len(self._prices[symbol]) > 100:
                self._prices[symbol] = self._prices[symbol][-100:]

    async def _scan_loop(self):
        """Periodic scanning loop."""
        await asyncio.sleep(30)
        await self._evaluate_and_update()
        while self._running:
            await asyncio.sleep(self._scan_interval)
            await self._evaluate_and_update()

    async def _evaluate_and_update(self):
        """Evaluate all candidates and update active symbols."""
        scores = {}
        for symbol in self.CANDIDATE_SYMBOLS:
            score, details = self._evaluate_symbol(symbol)
            scores[symbol] = score
            self._symbol_scores[symbol] = score
            if score > 0:
                await self._bus.emit(
                    "symbol_evaluated",
                    symbol=symbol,
                    score=round(score, 3),
                    spread=details.get("avg_spread", 0),
                    volatility=details.get("volatility", 0),
                    tick_count=details.get("tick_count", 0),
                )

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        new_active = []
        for symbol, score in ranked:
            if len(new_active) >= self._max_symbols:
                break
            if score > 0:
                new_active.append(symbol)

        if not new_active:
            new_active = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD",
                          "AUDUSD", "NZDUSD"]

        old_set = set(self._active_symbols)
        new_set = set(new_active)
        added = list(new_set - old_set)
        removed = list(old_set - new_set)

        if added or removed:
            self._active_symbols = new_active
            await self._bus.emit(
                "symbols_updated",
                symbols=self._active_symbols,
                added=added,
                removed=removed,
            )
            if added:
                logger.info(f"SymbolDiscovery: added symbols {added}")
            if removed:
                logger.info(f"SymbolDiscovery: removed symbols {removed}")

    def _evaluate_symbol(self, symbol: str) -> Tuple[float, Dict]:
        """Score a symbol from 0.0 (untradeable) to 1.0 (ideal).

        Factors:
        - Tick count (data availability): need minimum warmup_ticks
        - Average spread: lower is better (inverted, normalized)
        - Volatility: moderate is better (too low = no movement, too high = risk)
        """
        tick_count = self._tick_counts.get(symbol, 0)
        if tick_count < self._warmup_ticks:
            return 0.0, {"tick_count": tick_count, "reason": "insufficient_data"}

        spreads = self._spreads.get(symbol, [])
        if not spreads:
            return 0.0, {"tick_count": tick_count, "reason": "no_spread_data"}
        avg_spread = sum(spreads) / len(spreads)

        # Spread score: 0.0001 spread (1 pip) = 1.0, 0.001 spread = 0.1
        spread_score = max(0.0, min(1.0, 0.0001 / (avg_spread + 1e-10)))

        # Volatility score with ideal band 0.0005-0.002
        prices = self._prices.get(symbol, [])
        volatility = 0.0
        if len(prices) >= 20:
            import numpy as np
            sample = prices[-50:] if len(prices) >= 50 else prices
            log_returns = np.diff(np.log(np.array(sample, dtype=float)))
            if len(log_returns) > 0:
                volatility = float(np.std(log_returns))

        vol_score = self._volatility_to_score(volatility)
        score = spread_score * 0.6 + vol_score * 0.4

        if score <= 0:
            return 0.0, {"tick_count": tick_count, "reason": "below_threshold",
                         "avg_spread": avg_spread, "volatility": volatility}

        return score, {
            "tick_count": tick_count,
            "avg_spread": avg_spread,
            "volatility": volatility,
            "spread_score": round(spread_score, 3),
            "volatility_score": round(vol_score, 3),
        }

    @staticmethod
    def _volatility_to_score(volatility: float) -> float:
        """Map volatility to a 0-1 score with ideal band 0.0005-0.002."""
        if volatility < 0.0001:
            return 0.0
        if volatility < 0.0005:
            return (volatility - 0.0001) / 0.0004 * 0.5
        if volatility < 0.002:
            return 0.5 + (volatility - 0.0005) / 0.0015 * 0.5
        if volatility < 0.005:
            return 1.0 - (volatility - 0.002) / 0.003 * 0.3
        return max(0.0, 0.7 - (volatility - 0.005) / 0.005 * 0.7)

    @property
    def active_symbols(self) -> List[str]:
        """Return the current list of active tradeable symbols."""
        return list(self._active_symbols)

    def get_symbol_report(self) -> Dict:
        """Return evaluation report for all symbols."""
        report = {}
        for symbol in self.CANDIDATE_SYMBOLS:
            report[symbol] = {
                "score": round(self._symbol_scores.get(symbol, 0.0), 3),
                "tick_count": self._tick_counts.get(symbol, 0),
                "active": symbol in self._active_symbols,
            }
        return report
