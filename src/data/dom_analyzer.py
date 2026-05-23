"""High-Frequency Depth-of-Market (DOM) Capture & Analytics.

Tracks Level II order-book snapshots and extracts:
  * depth imbalance, liquidity score, book pressure
  * spoofing detection, iceberg detection
  * support / resistance from volume clusters
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import numpy as np
from loguru import logger


@dataclass
class DOMSnapshot:
    """Immutable snapshot of the order book at a point in time."""

    timestamp: float
    bids: List[Tuple[float, float]] = field(default_factory=list)
    asks: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 0.0

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid if self.bids and self.asks else 0.0


class DOMAnalyzer:
    """Stateful DOM analyzer for a single symbol.

    Maintains a rolling window of ``DOMSnapshot`` objects and derives
    real-time metrics used by the feature engine.
    """

    def __init__(self, symbol: str, depth_levels: int = 5):
        self.symbol = symbol
        self.depth_levels = depth_levels
        self._snapshots: deque = deque(maxlen=1000)
        # For spoofing / iceberg detection we keep per-level size history
        self._bid_size_hist: Dict[float, deque] = defaultdict(lambda: deque(maxlen=50))
        self._ask_size_hist: Dict[float, deque] = defaultdict(lambda: deque(maxlen=50))

    # -- ingestion -----------------------------------------------------------

    def ingest_dom(
        self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]
    ):
        """Store a new DOM snapshot.

        Args:
            bids: List of ``(price, size)`` tuples, sorted descending by price.
            asks: List of ``(price, size)`` tuples, sorted ascending by price.
        """
        snap = DOMSnapshot(timestamp=time.time(), bids=list(bids), asks=list(asks))
        self._snapshots.append(snap)

        # Record size histories for spoofing / iceberg analysis
        for p, s in bids[: self.depth_levels]:
            self._bid_size_hist[p].append(s)
        for p, s in asks[: self.depth_levels]:
            self._ask_size_hist[p].append(s)

    # -- metrics -------------------------------------------------------------

    def get_depth_imbalance(self) -> float:
        """Bid volume / total volume ratio for the top *depth_levels*."""
        if not self._snapshots:
            return 0.5
        latest = self._snapshots[-1]
        bid_vol = sum(s for _, s in latest.bids[: self.depth_levels])
        ask_vol = sum(s for _, s in latest.asks[: self.depth_levels])
        total = bid_vol + ask_vol
        return bid_vol / total if total > 0 else 0.5

    def get_liquidity_score(self) -> float:
        """Total volume at best bid + best ask."""
        if not self._snapshots:
            return 0.0
        latest = self._snapshots[-1]
        best_bid_vol = latest.bids[0][1] if latest.bids else 0.0
        best_ask_vol = latest.asks[0][1] if latest.asks else 0.0
        return best_bid_vol + best_ask_vol

    def detect_spoofing(self, threshold_ratio: float = 3.0) -> bool:
        """Detect large orders that appear and disappear quickly.

        A level is flagged as spoofed when its max recorded size is
        *threshold_ratio* times larger than its most recent size.
        """
        if not self._snapshots:
            return False
        latest = self._snapshots[-1]
        for p, s in latest.bids[: self.depth_levels] + latest.asks[: self.depth_levels]:
            hist = self._bid_size_hist.get(p) or self._ask_size_hist.get(p)
            if hist is None or len(hist) < 3:
                continue
            max_size = max(hist)
            if max_size > 0 and s > 0 and max_size / s > threshold_ratio:
                return True
        return False

    def detect_iceberg(self, min_fills: int = 3, size_tolerance: float = 0.15) -> bool:
        """Detect repeated fills at the same price level (hidden size).

        Looks for price levels where the size changes by a similar
        fraction repeatedly, suggesting an iceberg order is refreshing.
        """
        if not self._snapshots:
            return False

        def _check_hist(hist: deque) -> bool:
            if len(hist) < min_fills + 1:
                return False
            diffs = []
            for i in range(1, len(hist)):
                prev = hist[i - 1]
                curr = hist[i]
                if prev > 0:
                    diffs.append(abs(curr - prev) / prev)
            if len(diffs) < min_fills:
                return False
            # Check if several consecutive diffs are within tolerance
            consistent = sum(1 for d in diffs if d <= size_tolerance)
            return consistent >= min_fills

        latest = self._snapshots[-1]
        for p, _ in latest.bids[: self.depth_levels]:
            if _check_hist(self._bid_size_hist[p]):
                return True
        for p, _ in latest.asks[: self.depth_levels]:
            if _check_hist(self._ask_size_hist[p]):
                return True
        return False

    def get_support_resistance(self) -> Tuple[Optional[float], Optional[float]]:
        """Price levels with the highest volume concentration.

        Returns:
            (support_price, resistance_price) where support is the
            bid level with max volume and resistance is the ask level
            with max volume across all stored snapshots.
        """
        if not self._snapshots:
            return None, None

        bid_vols: Dict[float, float] = defaultdict(float)
        ask_vols: Dict[float, float] = defaultdict(float)
        for snap in self._snapshots:
            for p, s in snap.bids:
                bid_vols[p] += s
            for p, s in snap.asks:
                ask_vols[p] += s

        support = max(bid_vols, key=bid_vols.get) if bid_vols else None
        resistance = max(ask_vols, key=ask_vols.get) if ask_vols else None
        return support, resistance

    def get_book_pressure(self) -> float:
        """Cumulative bid/ask ratio across *all* recorded levels.

        Returns a value in ``[-1, 1]`` where ``+1`` means 100 % bid
        dominance and ``-1`` means 100 % ask dominance.
        """
        if not self._snapshots:
            return 0.0
        total_bid = 0.0
        total_ask = 0.0
        for snap in self._snapshots:
            total_bid += sum(s for _, s in snap.bids)
            total_ask += sum(s for _, s in snap.asks)
        total = total_bid + total_ask
        if total == 0:
            return 0.0
        return (total_bid - total_ask) / total

    def latest_snapshot(self) -> Optional[DOMSnapshot]:
        """Return the most recent DOM snapshot, or ``None``."""
        return self._snapshots[-1] if self._snapshots else None
