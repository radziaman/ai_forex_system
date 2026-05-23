"""Microstructure Feature Engine for RTS AI Forex Trading System.

Implements real-time market microstructure analytics:
- Order Flow Imbalance (OFI)
- Cumulative Volume Delta (CVD)
- Spread percentiles & expansion
- Trade-size distribution (skew + whale ratio)
- VWAP / TWAP deviation
- DOM depth imbalance
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PriceTick:
    """Unified tick format extended with microstructure fields."""

    symbol: str
    bid: float
    ask: float
    mid: float
    timestamp: float
    volume: float = 0.0
    source: str = ""
    trade_volume: float = 0.0
    trade_side: Optional[str] = None
    dom_bids: List[Tuple[float, float]] = field(default_factory=list)
    dom_asks: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class MicrostructureSnapshot:
    """Current microstructure state for a symbol."""

    ofi: float = 0.0
    cvd: float = 0.0
    cvd_slope: float = 0.0
    spread_p50: float = 0.0
    spread_p95: float = 0.0
    spread_expansion: float = 0.0
    trade_size_skew: float = 0.0
    whale_ratio: float = 0.0
    vwap_deviation: float = 0.0
    twap_deviation: float = 0.0
    dom_depth_imbalance: float = 0.0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class MicrostructureEngine:
    """Stateful microstructure calculator per symbol.

    Buffers the last *maxlen* ticks per symbol and derives
    order-flow, spread, and depth features on demand.
    """

    def __init__(self, maxlen: int = 1000):
        self.maxlen = maxlen
        self._buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))
        self._cvd_series: Dict[str, List[float]] = defaultdict(list)
        self._last_tick: Dict[str, Optional[PriceTick]] = {}

    # -- ingestion ----------------------------------------------------------

    def ingest_tick(self, tick: PriceTick) -> None:
        """Add a tick to the symbol's rolling buffer."""
        symbol = tick.symbol
        prev = self._last_tick.get(symbol)
        signed_vol = self._signed_volume(tick, prev)

        self._buffers[symbol].append(tick)

        if self._cvd_series[symbol]:
            new_cvd = self._cvd_series[symbol][-1] + signed_vol
        else:
            new_cvd = signed_vol
        self._cvd_series[symbol].append(new_cvd)

        # bound history
        if len(self._cvd_series[symbol]) > self.maxlen:
            self._cvd_series[symbol] = self._cvd_series[symbol][-self.maxlen :]

        self._last_tick[symbol] = tick

    def _signed_volume(self, tick: PriceTick, prev: Optional[PriceTick]) -> float:
        """Return +volume for buyer-initiated, -volume for seller-initiated.

        Falls back to the Lee-Ready tick test when *trade_side* is absent.
        """
        vol = tick.trade_volume if tick.trade_volume > 0 else tick.volume
        if vol <= 0:
            return 0.0

        side = tick.trade_side
        if side == "buy":
            return vol
        if side == "sell":
            return -vol

        # Lee-Ready: compare mid to previous mid
        if prev is None:
            return 0.0
        mid = tick.mid
        prev_mid = prev.mid
        if mid > prev_mid:
            return vol
        if mid < prev_mid:
            return -vol
        return 0.0

    # -- helpers ------------------------------------------------------------

    def _buffer(self, symbol: str) -> deque:
        return self._buffers.get(symbol, deque())

    def _total_volume(self, symbol: str) -> float:
        total = 0.0
        for t in self._buffer(symbol):
            total += t.trade_volume if t.trade_volume > 0 else t.volume
        return total

    # -- feature accessors ------------------------------------------------

    def get_ofi(self, symbol: str) -> float:
        """Order Flow Imbalance = net signed volume / total volume."""
        buf = self._buffer(symbol)
        if not buf:
            return 0.0
        net = 0.0
        total = 0.0
        prev = None
        for tick in buf:
            sv = self._signed_volume(tick, prev)
            net += sv
            total += abs(sv)
            prev = tick
        if total == 0:
            return 0.0
        return net / total

    def get_cvd(self, symbol: str) -> float:
        """Cumulative Volume Delta (running net signed volume)."""
        series = self._cvd_series.get(symbol, [])
        return series[-1] if series else 0.0

    def get_cvd_slope(self, symbol: str, lookback: int = 20) -> float:
        """Per-tick CVD trend over the last *lookback* observations."""
        series = self._cvd_series.get(symbol, [])
        if len(series) < lookback:
            return 0.0
        # number of intervals between lookback points is lookback-1
        denom = max(1, lookback - 1)
        return (series[-1] - series[-lookback]) / denom

    def get_spread_percentiles(self, symbol: str) -> Tuple[float, float]:
        """Median (P50) and 95th percentile of the quoted spread."""
        buf = self._buffer(symbol)
        if not buf:
            return 0.0, 0.0
        spreads = np.array([t.ask - t.bid for t in buf])
        p50 = float(np.percentile(spreads, 50))
        p95 = float(np.percentile(spreads, 95))
        return p50, p95

    def get_spread_expansion(self, symbol: str) -> float:
        """Relative widening of the latest spread vs its median."""
        buf = self._buffer(symbol)
        if not buf:
            return 0.0
        current = buf[-1].ask - buf[-1].bid
        p50, _ = self.get_spread_percentiles(symbol)
        if p50 == 0:
            return 0.0
        return (current / p50) - 1.0

    def get_trade_size_distribution(self, symbol: str) -> Tuple[float, float]:
        """Return (Pearson skewness of trade sizes, whale ratio).

        Whale ratio = volume in the top 5 % of trade sizes / total volume.
        """
        buf = self._buffer(symbol)
        if not buf:
            return 0.0, 0.0
        vols = np.array(
            [t.trade_volume if t.trade_volume > 0 else t.volume for t in buf]
        )
        if len(vols) == 0:
            return 0.0, 0.0
        mean_v = np.mean(vols)
        std_v = np.std(vols)
        if std_v == 0:
            return 0.0, 0.0
        skew = float(np.mean((vols - mean_v) ** 3) / (std_v**3))
        p95 = float(np.percentile(vols, 95))
        total = np.sum(vols)
        whale = float(np.sum(vols[vols >= p95]) / total) if total > 0 else 0.0
        return skew, whale

    def get_vwap_deviation(self, symbol: str, current_price: float) -> float:
        """(current_price - VWAP) / VWAP."""
        buf = self._buffer(symbol)
        if not buf:
            return 0.0
        num = 0.0
        den = 0.0
        for t in buf:
            vol = t.trade_volume if t.trade_volume > 0 else t.volume
            if vol > 0:
                num += t.mid * vol
                den += vol
        if den == 0:
            return 0.0
        vwap = num / den
        return (current_price - vwap) / vwap if vwap != 0 else 0.0

    def get_twap_deviation(self, symbol: str, current_price: float) -> float:
        """(current_price - TWAP) / TWAP."""
        buf = self._buffer(symbol)
        if not buf:
            return 0.0
        mids = np.array([t.mid for t in buf])
        twap = float(np.mean(mids))
        return (current_price - twap) / twap if twap != 0 else 0.0

    def get_dom_depth_imbalance(self, symbol: str, levels: int = 5) -> float:
        """Bid/ask depth ratio for the latest DOM snapshot."""
        buf = self._buffer(symbol)
        if not buf:
            return 0.0
        latest = buf[-1]
        bids = latest.dom_bids
        asks = latest.dom_asks
        if not bids or not asks:
            return 0.0
        bid_vol = sum(b[1] for b in bids[:levels])
        ask_vol = sum(a[1] for a in asks[:levels])
        total = bid_vol + ask_vol
        return (bid_vol - ask_vol) / total if total > 0 else 0.0

    def get_snapshot(self, symbol: str, current_price: float) -> MicrostructureSnapshot:
        """Return a fully-populated MicrostructureSnapshot."""
        p50, p95 = self.get_spread_percentiles(symbol)
        skew, whale = self.get_trade_size_distribution(symbol)
        return MicrostructureSnapshot(
            ofi=self.get_ofi(symbol),
            cvd=self.get_cvd(symbol),
            cvd_slope=self.get_cvd_slope(symbol),
            spread_p50=p50,
            spread_p95=p95,
            spread_expansion=self.get_spread_expansion(symbol),
            trade_size_skew=skew,
            whale_ratio=whale,
            vwap_deviation=self.get_vwap_deviation(symbol, current_price),
            twap_deviation=self.get_twap_deviation(symbol, current_price),
            dom_depth_imbalance=self.get_dom_depth_imbalance(symbol),
        )

    def get_feature_vector(self, symbol: str, current_price: float) -> np.ndarray:
        """Return an 11-dimensional normalized numpy vector."""
        snap = self.get_snapshot(symbol, current_price)
        total_vol = self._total_volume(symbol)
        price = current_price if current_price > 0 else 1.0

        ofi = float(np.clip(snap.ofi, -1.0, 1.0))
        cvd = (
            float(np.clip(snap.cvd / (total_vol + 1e-8), -1.0, 1.0))
            if total_vol > 0
            else 0.0
        )
        slope = float(np.clip(snap.cvd_slope * 10.0, -1.0, 1.0))

        # spreads expressed in basis points relative to price
        sp50_bp = snap.spread_p50 / price * 1e4
        sp95_bp = snap.spread_p95 / price * 1e4
        sp50 = float(np.tanh(sp50_bp))
        sp95 = float(np.tanh(sp95_bp))

        expansion = float(np.clip(snap.spread_expansion, -1.0, 1.0))
        skew = float(np.clip(snap.trade_size_skew / 3.0, -1.0, 1.0))
        whale = float(np.clip(snap.whale_ratio, 0.0, 1.0))
        vwap_dev = float(np.clip(snap.vwap_deviation * 100.0, -1.0, 1.0))
        twap_dev = float(np.clip(snap.twap_deviation * 100.0, -1.0, 1.0))
        dom = float(np.clip(snap.dom_depth_imbalance, -1.0, 1.0))

        return np.array(
            [
                ofi,
                cvd,
                slope,
                sp50,
                sp95,
                expansion,
                skew,
                whale,
                vwap_dev,
                twap_dev,
                dom,
            ],
            dtype=np.float32,
        )
