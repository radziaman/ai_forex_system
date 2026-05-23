"""Multi-Venue Data Framework — aggregates quotes from multiple FX venues.

Architecture:
  * Each venue registers a ``fetcher_fn(symbol) -> VenueQuote`` coroutine.
  * ``MultiVenueProvider`` fires all fetchers concurrently via ``asyncio.gather``.
  * Best-quote selection, arbitrage detection, and latency reporting are
    computed from the collected responses.

Out-of-the-box venues:
  * ``ctrader``  — via cTrader Open API depth subscription
  * ``dukascopy`` — via Dukascopy JForex feed

Extensible: call ``register_venue(name, fetcher_fn)`` to add LMAX, CMC, Saxo, etc.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any
from statistics import mean
from loguru import logger


@dataclass
class VenueQuote:
    """Single-quote snapshot from one venue."""

    venue_name: str
    bid: float
    ask: float
    mid: float
    timestamp: float
    latency_ms: float = 0.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid


class MultiVenueProvider:
    """Aggregates quotes from multiple trading venues concurrently.

    Usage:
        provider = MultiVenueProvider()
        provider.register_venue("lmax", lmax_fetcher)
        quotes = await provider.get_all_quotes("EURUSD")
        best = provider.get_best_quote("EURUSD")
    """

    # Minimum spread (in price units) that triggers an arbitrage flag.
    ARB_THRESHOLD: float = 0.0002  # 2 pips for majors

    def __init__(self, venues: Optional[List[str]] = None):
        self._venues: Dict[str, Callable[[str], Any]] = {}
        self._latest: Dict[str, List[VenueQuote]] = {}
        self._latency_history: Dict[str, List[float]] = {}

        default_venues = venues or ["ctrader", "dukascopy"]
        for v in default_venues:
            self._register_default(v)

    # -- registration -------------------------------------------------------

    def register_venue(self, name: str, fetcher_fn: Callable[[str], Any]) -> None:
        """Add a new venue fetcher.

        ``fetcher_fn`` must be an awaitable callable ``fetcher_fn(symbol)``
        that returns a ``VenueQuote`` (or raises on failure).
        """
        self._venues[name] = fetcher_fn
        self._latency_history[name] = []
        logger.info(f"MultiVenue: registered '{name}'")

    def _register_default(self, name: str) -> None:
        """Wire the built-in ctrader / dukascopy fetchers."""
        if name == "ctrader":
            self.register_venue(name, self._fetch_ctrader)
        elif name == "dukascopy":
            self.register_venue(name, self._fetch_dukascopy)
        else:
            logger.warning(f"MultiVenue: no default fetcher for '{name}'")

    # -- public API ----------------------------------------------------------

    async def get_all_quotes(self, symbol: str) -> List[VenueQuote]:
        """Fetch from every registered venue simultaneously."""
        if not self._venues:
            return []

        tasks = [
            self._timed_fetch(name, fn, symbol) for name, fn in self._venues.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        quotes: List[VenueQuote] = []
        for q in results:
            if isinstance(q, Exception):
                continue
            if q is not None:
                quotes.append(q)

        self._latest[symbol] = quotes
        return quotes

    def get_best_quote(self, symbol: str) -> Optional[VenueQuote]:
        """Return the venue with the tightest spread for *symbol*.

        If multiple venues share the same tightest spread, the one with
        the lowest latency wins.
        """
        quotes = self._latest.get(symbol, [])
        if not quotes:
            return None
        # Filter out stale quotes (>5s old)
        now = time.time()
        fresh = [q for q in quotes if now - q.timestamp < 5.0]
        if not fresh:
            fresh = quotes
        best = min(fresh, key=lambda q: (q.spread, q.latency_ms))
        return best

    def detect_arbitrage(self, symbol: str, threshold: Optional[float] = None) -> bool:
        """Return ``True`` if the max ask minus min bid exceeds *threshold*."""
        quotes = self._latest.get(symbol, [])
        if len(quotes) < 2:
            return False
        bids = [q.bid for q in quotes]
        asks = [q.ask for q in quotes]
        spread = max(asks) - min(bids)
        thr = threshold if threshold is not None else self.ARB_THRESHOLD
        return spread > thr

    def get_quote_latency_report(self) -> Dict[str, Dict[str, float]]:
        """Per-venue latency statistics (mean, min, max, count)."""
        report: Dict[str, Dict[str, float]] = {}
        for name, hist in self._latency_history.items():
            if not hist:
                report[name] = {"mean": 0.0, "min": 0.0, "max": 0.0, "count": 0.0}
                continue
            report[name] = {
                "mean": mean(hist),
                "min": min(hist),
                "max": max(hist),
                "count": float(len(hist)),
            }
        return report

    # -- internal helpers ----------------------------------------------------

    async def _timed_fetch(
        self, name: str, fn: Callable[[str], Any], symbol: str
    ) -> Optional[VenueQuote]:
        t0 = time.time()
        try:
            quote = await fn(symbol)
        except Exception as exc:
            logger.debug(f"MultiVenue '{name}' fetch failed: {exc}")
            return None
        latency_ms = (time.time() - t0) * 1000.0
        if quote is not None:
            quote.latency_ms = latency_ms
            self._latency_history.setdefault(name, []).append(latency_ms)
        return quote

    # -- built-in fetchers (framework stubs) ---------------------------------

    @staticmethod
    def _symbol_base(symbol: str) -> float:
        """Deterministic synthetic base price for a symbol."""
        h = abs(hash(symbol))
        return 1.0 + (h % 1000) / 10000.0

    @staticmethod
    async def _fetch_ctrader(symbol: str) -> Optional[VenueQuote]:
        """Placeholder for cTrader real-time depth fetch.

        In production this would call ``CtraderClient`` or subscribe via
        the Open API protobuf stream.  Returns a synthetic quote so the
        framework can be exercised in tests.
        """
        # Simulate network latency
        await asyncio.sleep(0.001)
        base = MultiVenueProvider._symbol_base(symbol)
        return VenueQuote(
            venue_name="ctrader",
            bid=base,
            ask=base + 0.0001,
            mid=base + 0.00005,
            timestamp=time.time(),
        )

    @staticmethod
    async def _fetch_dukascopy(symbol: str) -> Optional[VenueQuote]:
        """Placeholder for Dukascopy JForex tick fetch.

        In production this would call the Dukascopy REST endpoint.
        """
        await asyncio.sleep(0.002)
        base = MultiVenueProvider._symbol_base(symbol)
        return VenueQuote(
            venue_name="dukascopy",
            bid=base,
            ask=base + 0.00012,
            mid=base + 0.00006,
            timestamp=time.time(),
        )
