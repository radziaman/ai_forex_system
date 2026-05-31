"""
Level II DOM, order flow (CVD), and market depth analytics.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from .data_manager import DepthLevelData, MarketDepthData


class DOMEngine:
    """Manages market depth, order flow, CVD, and DOM analytics."""

    def __init__(self, dm):
        self.dm = dm  # Reference to DataManager for shared state

    # ------------------------------------------------------------------
    # Order flow (CVD)
    # ------------------------------------------------------------------

    def update_of(self, symbol: str, bid: float, ask: float, volume: float):
        """Update cumulative volume delta (CVD) for a symbol."""
        cvd_hist, bid_hist, ask_hist = self.dm._cvd[symbol]
        mid = (bid + ask) / 2.0
        if bid_hist:
            prev_mid = (bid_hist[-1] + ask_hist[-1]) / 2.0
            delta = volume if mid >= prev_mid else -volume
            new_cvd = (cvd_hist[-1] if cvd_hist else 0.0) + delta
        else:
            new_cvd = 0.0
        cvd_hist.append(new_cvd)
        bid_hist.append(bid)
        ask_hist.append(ask)
        if len(cvd_hist) > 1000:
            self.dm._cvd[symbol] = (cvd_hist[-500:], bid_hist[-500:], ask_hist[-500:])
        cvd_slope = (cvd_hist[-1] - cvd_hist[-20]) / 20 if len(cvd_hist) >= 20 else 0.0
        spread = ask - bid
        self.dm.order_flow[symbol] = {
            "cvd": new_cvd,
            "cvd_slope": cvd_slope,
            "imbalance": (bid - ask) / spread if spread > 0 else 0.0,
        }

    # ------------------------------------------------------------------
    # Level II DOM
    # ------------------------------------------------------------------

    def update_market_depth(self, depth: Any):
        """Accept both CtraderDepth and MarketDepthData (tuple-based) formats."""
        try:
            from api.ctrader_client import MarketDepth as CtraderDepth

            if isinstance(depth, CtraderDepth):
                sym = depth.symbol
                md: Optional[MarketDepthData] = self.dm.market_depth[sym]
                assert md is not None, f"Market depth not found for {sym}"
                md.symbol = sym
                md.bid = depth.bid
                md.ask = depth.ask
                md.spread = depth.spread
                md.volume = depth.volume
                md.timestamp = depth.timestamp
                md.bids = [
                    DepthLevelData(price=b.price, size=b.size) for b in depth.bids
                ]
                md.asks = [
                    DepthLevelData(price=a.price, size=a.size) for a in depth.asks
                ]
                dom = self.dm.dom_analyzer.get(sym)
                if dom is not None:
                    bids_t = [(b.price, b.size) for b in depth.bids]
                    asks_t = [(a.price, a.size) for a in depth.asks]
                    dom.ingest_dom(bids_t, asks_t)
                return
        except ImportError:
            pass

        if isinstance(depth, MarketDepthData):
            sym = depth.symbol
            self.dm.market_depth[sym] = depth
            dom = self.dm.dom_analyzer.get(sym)
            if dom is not None:
                bids_t = [(b.price, b.size) for b in depth.bids]
                asks_t = [(a.price, a.size) for a in depth.asks]
                dom.ingest_dom(bids_t, asks_t)
            return

        if (
            hasattr(depth, "symbol")
            and hasattr(depth, "bids")
            and hasattr(depth, "asks")
        ):
            sym = depth.symbol
            md = self.dm.market_depth.get(sym)
            if md is None:
                return
            md.bid = getattr(depth, "bid", md.bid)
            md.ask = getattr(depth, "ask", md.ask)
            md.spread = md.ask - md.bid
            md.timestamp = getattr(depth, "timestamp", time.time())
            if depth.bids and isinstance(depth.bids[0], (tuple, list)):
                md.bids = [DepthLevelData(price=p, size=s) for p, s in depth.bids]
            if depth.asks and isinstance(depth.asks[0], (tuple, list)):
                md.asks = [DepthLevelData(price=p, size=s) for p, s in depth.asks]
            dom = self.dm.dom_analyzer.get(sym)
            if dom is not None:
                bids_t = (
                    depth.bids
                    if depth.bids and isinstance(depth.bids[0], (tuple, list))
                    else [(b.price, b.size) for b in depth.bids]
                )
                asks_t = (
                    depth.asks
                    if depth.asks and isinstance(depth.asks[0], (tuple, list))
                    else [(a.price, a.size) for a in depth.asks]
                )
                dom.ingest_dom(bids_t, asks_t)
            return

        logger.debug(f"update_market_depth: unknown type {type(depth).__name__}")

    def get_market_depth(self, symbol: str) -> Optional[MarketDepthData]:
        """Return current market depth for a symbol."""
        return self.dm.market_depth.get(symbol)

    def get_dom_imbalance(self, symbol: str, levels: int = 5) -> float:
        """Calculate bid/ask volume imbalance from market depth."""
        md = self.dm.market_depth.get(symbol)
        if not md or not md.bids or not md.asks:
            return 0.0
        bid_vol = sum(b.size for b in md.bids[:levels])
        ask_vol = sum(a.size for a in md.asks[:levels])
        total = bid_vol + ask_vol
        return (bid_vol - ask_vol) / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # Order flow analytics
    # ------------------------------------------------------------------

    def get_order_flow_metrics(self, symbol: str) -> Dict:
        """Aggregate order flow, microstructure, DOM, alternative data, and options metrics."""
        cvd_hist, _, _ = self.dm._cvd.get(symbol, ([], [], []))
        of = self.dm.order_flow.get(symbol, {})
        ms_snapshot = self.dm.microstructure.get_snapshot(
            symbol, self.dm._last_realtime_price.get(symbol, 0.0)
        )
        dom = self.dm.dom_analyzer.get(symbol)
        dom_metrics = {}
        if dom is not None:
            dom_metrics = {
                "dom_depth_imbalance": dom.get_depth_imbalance(),
                "liquidity_score": dom.get_liquidity_score(),
                "book_pressure": dom.get_book_pressure(),
                "spoofing_detected": dom.detect_spoofing(),
                "iceberg_detected": dom.detect_iceberg(),
                "support": dom.get_support_resistance()[0],
                "resistance": dom.get_support_resistance()[1],
            }
        alt_scores = self.dm.alternative_data.compute_fx_impact_scores()
        opt_metrics = {}
        if self.dm.options_data.is_available():
            opt_metrics = {
                "risk_reversal_25d": self.dm.options_data.get_25d_risk_reversal(symbol),
                "butterfly_spread": self.dm.options_data.get_butterfly_spread(symbol),
                "skew_index": self.dm.options_data.get_skew_index(symbol),
                "term_structure": self.dm.options_data.get_term_structure(symbol),
            }
        return {
            "cvd": of.get("cvd", 0.0),
            "cvd_slope": of.get("cvd_slope", 0.0),
            "imbalance": of.get("imbalance", 0.0),
            "dom_imbalance": self.get_dom_imbalance(symbol),
            "cvd_hist": cvd_hist[-100:] if cvd_hist else [],
            "microstructure": ms_snapshot,
            "dom": dom_metrics,
            "alternative": alt_scores,
            "options": opt_metrics,
        }

    def calculate_gamma_exposure(self, symbol: str) -> Dict:
        """Calculate gamma exposure (placeholder)."""
        return {
            "gamma_exposure": 0.0,
            "gamma_flip_point": 0.0,
            "dealer_position": "neutral",
        }
