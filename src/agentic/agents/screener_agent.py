"""
Instrument Screener Agent — autonomously scans markets for tradeable edges.

The screener agent continuously monitors all instruments in its universe,
tests momentum strategies across multiple timeframes, and publishes only
those instruments that pass the statistical edge threshold to the agent bus.

Other agents (e.g., signal_agent, risk_agent) subscribe to INSTRUMENTS_UPDATED
to dynamically adjust which symbols they process.
"""

from __future__ import annotations
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from loguru import logger

_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _src not in sys.path:
    sys.path.insert(0, _src)

import yfinance as yf  # noqa: E402
from agentic.core.base_agent import BaseAgent  # noqa: E402
from agentic.core.agent_message import (  # noqa: E402
    MessageType,
    MessagePriority,
    AgentIntention,
)
from agentic.core.agent_consciousness import ConsciousnessLevel  # noqa: E402

# ─── Instrument Universe ──────────────────────────────────────────────────────
# The single source of truth for what the system can trade.
# The screener tests ALL of these and publishes only those with edge.

INSTRUMENT_UNIVERSE = {
    # Metals
    "GC=F": {"name": "Gold Futures", "asset_class": "metal", "pip": 0.1},
    "SI=F": {"name": "Silver Futures", "asset_class": "metal", "pip": 0.001},
    "HG=F": {"name": "Copper Futures", "asset_class": "metal", "pip": 0.0005},
    "PA=F": {"name": "Palladium Futures", "asset_class": "metal", "pip": 1.0},
    "PL=F": {"name": "Platinum Futures", "asset_class": "metal", "pip": 0.1},
    # Energy
    "CL=F": {"name": "Crude Oil Futures", "asset_class": "energy", "pip": 0.01},
    "HO=F": {"name": "Heating Oil Futures", "asset_class": "energy", "pip": 0.0001},
    "RB=F": {"name": "Gasoline Futures", "asset_class": "energy", "pip": 0.0001},
    "NG=F": {"name": "Natural Gas Futures", "asset_class": "energy", "pip": 0.001},
    # Bonds
    "ZB=F": {"name": "30yr T-Bond Futures", "asset_class": "bond", "pip": 0.01},
    "ZN=F": {"name": "10yr T-Note Futures", "asset_class": "bond", "pip": 0.005},
    "ZF=F": {"name": "5yr T-Note Futures", "asset_class": "bond", "pip": 0.0025},
    "ZT=F": {"name": "2yr T-Note Futures", "asset_class": "bond", "pip": 0.00125},
    # Equity Indices
    "ES=F": {"name": "S&P 500 Futures", "asset_class": "equity", "pip": 0.1},
    "NQ=F": {"name": "Nasdaq Futures", "asset_class": "equity", "pip": 0.25},
    "YM=F": {"name": "Dow Futures", "asset_class": "equity", "pip": 1.0},
    # FX
    "DX-Y.NYB": {"name": "US Dollar Index", "asset_class": "fx", "pip": 0.01},
    "EURUSD=X": {"name": "EUR/USD", "asset_class": "fx", "pip": 0.0001},
    "GBPUSD=X": {"name": "GBP/USD", "asset_class": "fx", "pip": 0.0001},
    # ETFs
    "GLD": {"name": "Gold ETF", "asset_class": "metal", "pip": 0.01},
    "SLV": {"name": "Silver ETF", "asset_class": "metal", "pip": 0.01},
    "USO": {"name": "Oil ETF", "asset_class": "energy", "pip": 0.01},
    "TLT": {"name": "20yr+ Treasury ETF", "asset_class": "bond", "pip": 0.01},
}

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", "historical"
)
os.makedirs(DATA_DIR, exist_ok=True)

# Edge thresholds — instruments must pass these to be tradeable
MIN_SHARPE = 0.3
MIN_PROFIT_FACTOR = 1.05
MIN_TRADES = 20


@dataclass
class ScreenResult:
    """Complete screening result for one instrument."""

    ticker: str
    name: str
    asset_class: str
    mom5_sharpe: float = 0.0
    mom5_pf: float = 0.0
    mom5_win_rate: float = 0.0
    mom5_trades: int = 0
    mom5_pnl: float = 0.0
    mom10_sharpe: float = 0.0
    mom10_pf: float = 0.0
    mom20_sharpe: float = 0.0
    mom20_pf: float = 0.0
    annual_return: float = 0.0
    annual_volatility: float = 0.0
    lag1_autocorr: float = 0.0
    n_bars: int = 0
    current_price: float = 0.0
    composite_score: float = 0.0
    edge_detected: bool = False
    recommendation: str = "HOLD"
    optimal_lookback: int = 5
    optimal_sl_atr: float = 3.0
    optimal_tp_atr: float = 6.0

    def to_dict(self) -> dict:
        return asdict(self)


class InstrumentScreenerAgent(BaseAgent):
    """
    Autonomously scans instruments, tests momentum strategies,
    and publishes tradeable instruments to the agent bus.

    Lifecycle:
      1. Perceive: Check if scan is due (daily) or if a SCREENING_REQUEST was received
      2. Reason: Run momentum tests on each instrument
      3. Act: Publish INSTRUMENTS_UPDATED with tradeable list
      4. Reflect: Store results in world state for other agents
    """

    def __init__(self, scan_interval_hours: int = 24):
        super().__init__(
            name="screener_agent",
            role="Autonomous Instrument Screener",
            purpose="Continuously scan markets for tradeable edges and publish findings",
            domain="screening",
            capabilities={
                "instrument_screening",
                "momentum_analysis",
                "edge_detection",
                "market_scanning",
                "portfolio_selection",
                "data_download",
            },
            tick_interval=60.0,  # Check every 60 seconds if scan is due
            consciousness_level=ConsciousnessLevel.REFLECTIVE,
        )
        self.scan_interval_hours = scan_interval_hours
        self._last_scan_time: float = 0.0
        self._tradeable_instruments: List[Dict] = []
        self._all_results: List[ScreenResult] = []
        self._scan_in_progress = False

        # Subscribe to on-demand screening requests
        self.subscribe(MessageType.SCREENING_REQUEST)

    async def perceive(self) -> Dict[str, Any]:
        """Check if a scan is needed."""
        now = time.time()
        hours_since_last = (now - self._last_scan_time) / 3600

        # Check for pending screening request in inbox
        has_request = False
        while not self._inbox.empty():
            try:
                msg = self._inbox.get_nowait()
                if msg.msg_type == MessageType.SCREENING_REQUEST:
                    has_request = True
                    self.log_state(
                        f"On-demand screening requested by {msg.source_agent}"
                    )
            except Exception:
                break

        should_scan = (
            self._last_scan_time == 0.0  # First run
            or hours_since_last >= self.scan_interval_hours  # Interval elapsed
            or has_request  # On-demand request
        )

        if not should_scan or self._scan_in_progress:
            return {"skip": True}

        return {
            "scan": True,
            "reason": (
                "initial"
                if self._last_scan_time == 0.0
                else (
                    "scheduled"
                    if hours_since_last >= self.scan_interval_hours
                    else "on_demand"
                )
            ),
        }

    async def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Run the screening algorithms across all instruments."""
        self._scan_in_progress = True
        self.consciousness.current_intention = (
            f"screening {len(INSTRUMENT_UNIVERSE)} instruments for tradeable edges"
        )

        results = []
        for ticker, info in INSTRUMENT_UNIVERSE.items():
            try:
                score = self._screen_instrument(ticker, info)
                if score:
                    results.append(score)
                    self.log_state(
                        f"{'✅' if score.edge_detected else '  '} {ticker}: "
                        f"Sharpe={score.mom5_sharpe:.3f} PF={score.mom5_pf:.2f} "
                        f"Score={score.composite_score:.3f} -> {score.recommendation}"
                    )
            except Exception as e:
                self.log_state(f"Failed to screen {ticker}: {e}", "warning")

        # Sort by composite score descending
        results.sort(key=lambda r: r.composite_score, reverse=True)
        self._all_results = results

        # Extract tradeable instruments
        tradeable = [r.to_dict() for r in results if r.edge_detected]

        return {
            "tradeable": tradeable,
            "total_screened": len(results),
            "tradeable_count": len(tradeable),
            "all_results": [r.to_dict() for r in results],
            "timestamp": time.time(),
        }

    async def act(self, decision: Dict[str, Any]):
        """Publish screening results to the agent bus and world state."""
        tradeable = decision.get("tradeable", [])

        # Update world state for other agents
        self.set_world(
            "screening.tradeable_instruments",
            tradeable,
            ttl=self.scan_interval_hours * 3600,
        )
        self.set_world("screening.last_scan", time.time())
        self.set_world("screening.total_screened", decision.get("total_screened", 0))

        # Update registry with current tradeable set
        for inst in tradeable:
            ticker = inst.get("ticker", "")
            try:
                self.registry.update_metadata(
                    self.name,
                    {f"instrument_{ticker}": inst.get("recommendation", "HOLD")},
                )
            except AttributeError:
                logger.debug(
                    f"Registry has no update_metadata; skipping metadata update for {ticker}"
                )

        # Publish to agent bus
        await self.send(
            MessageType.INSTRUMENTS_UPDATED,
            payload={
                "tradeable": tradeable,
                "total_screened": decision.get("total_screened", 0),
                "tradeable_count": decision.get("tradeable_count", 0),
                "timestamp": decision.get("timestamp", time.time()),
            },
            priority=MessagePriority.NORMAL,
            intention=AgentIntention(
                primary_goal="update all agents with current tradeable instruments",
                reasoning=f"Found {decision.get('tradeable_count', 0)} tradeable instruments "
                f"out of {decision.get('total_screened', 0)} screened",
                expected_outcome="signal_agent and risk_agent adjust their symbol sets",
                confidence=0.9 if tradeable else 0.5,
            ),
        )

        # Log summary
        self.log_state(
            f"Scan complete: {decision.get('tradeable_count', 0)} tradeable "
            f"out of {decision.get('total_screened', 0)} instruments"
        )
        for inst in tradeable:
            self.log_state(
                f"  ✅ TRADEABLE: {inst['ticker']} ({inst['name']}) — "
                f"Sharpe={inst['mom5_sharpe']:.2f} PF={inst['mom5_pf']:.2f} "
                f"Strategy: {inst['optimal_lookback']}d momentum"
            )

        self._tradeable_instruments = tradeable

    async def reflect(self, outcome: Dict[str, Any]):
        """Clean up after scan."""
        self._last_scan_time = time.time()
        self._scan_in_progress = False
        self.consciousness.current_intention = (
            "monitoring — next scan due in {self.scan_interval_hours}h"
        )

        # Save results to disk for persistence
        try:
            results_path = os.path.join(DATA_DIR, "..", "screen_results.json")
            with open(results_path, "w") as f:
                json.dump(
                    {
                        "timestamp": time.time(),
                        "tradeable": self._tradeable_instruments,
                        "all_results": [r.to_dict() for r in self._all_results],
                    },
                    f,
                    indent=2,
                    default=str,
                )
        except Exception as e:
            self.log_state(f"Failed to save results: {e}", "warning")

    # ─── Screening Logic ─────────────────────────────────────────────────

    def _load_prices(self, ticker: str) -> Optional[np.ndarray]:
        """Load or download daily price data."""
        safe = ticker.replace("=", "_").replace("-", "_")
        fpath = os.path.join(DATA_DIR, f"{safe}_daily.csv")

        if os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath, index_col=0, parse_dates=True)
                for col in ["Close", "close", "Adj Close"]:
                    if col in df.columns:
                        val = df[col].values
                        if val.dtype == "O":
                            val = pd.to_numeric(val, errors="coerce")
                        return np.nan_to_num(val, nan=0.0).astype(float)
                for col in df.columns:
                    if df[col].dtype in (np.float64, np.float32, np.int64):
                        return df[col].values.astype(float)
            except Exception:
                pass

        try:
            data = yf.download(
                ticker, period="10y", interval="1d", progress=False, auto_adjust=True
            )
            if data is None or len(data) < 100:
                return None
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [c[0] for c in data.columns]
            data.to_csv(fpath)
            for col in ["Close", "close", "Adj Close"]:
                if col in data.columns:
                    return data[col].values.astype(float)
            return data.iloc[:, 0].values.astype(float)
        except Exception:
            return None

    def _test_strategy(
        self, prices: np.ndarray, lookback: int, spread_cost: float = 0.0
    ) -> Dict:
        """Test momentum strategy and return metrics."""
        p = prices.flatten()
        if len(p) < 100:
            return {"sharpe": 0, "pf": 0, "wr": 0, "trades": 0, "pnl": 0}

        signals = np.zeros(len(p), dtype=int)
        for i in range(lookback, len(p)):
            signals[i] = 1 if p[i] > p[i - lookback] else -1

        tp = []
        pos = 0
        ep = 0
        for i in range(lookback, len(p)):
            sig = signals[i]
            if pos == 0 and sig != 0:
                pos = sig
                ep = p[i]
                continue
            if pos != 0 and (sig != 0 and sig != pos or i == len(p) - 1):
                pnl = (p[i] - ep) * pos - spread_cost * np.sign(pos)
                tp.append(pnl)
                pos = 0

        tpa = np.array(tp)
        if len(tpa) < 3:
            return {"sharpe": 0, "pf": 0, "wr": 0, "trades": 0, "pnl": 0}

        w = tpa[tpa > 0]
        losses = tpa[tpa < 0]
        sharpe = np.mean(tpa) / np.std(tpa) * np.sqrt(252) if np.std(tpa) > 1e-10 else 0
        pf = np.sum(w) / abs(np.sum(losses)) if np.sum(losses) != 0 else float("inf")

        return {
            "sharpe": float(sharpe),
            "pf": float(pf),
            "wr": float(np.mean(tpa > 0)),
            "trades": int(len(tpa)),
            "pnl": float(np.sum(tpa)),
        }

    def _screen_instrument(self, ticker: str, info: Dict) -> Optional[ScreenResult]:
        """Screen a single instrument for edge."""
        prices = self._load_prices(ticker)
        if prices is None or len(prices) < 100:
            return None

        p = prices.flatten()
        n = len(p)

        # Market characteristics
        returns = np.diff(p) / (p[:-1] + 1e-10)
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        if len(returns) < 10:
            return None

        ann_ret = float(np.mean(returns) * 252 * 100)
        ann_vol = float(np.std(returns) * np.sqrt(252) * 100)
        lag1 = (
            float(np.corrcoef(returns[1:], returns[:-1])[0, 1])
            if len(returns) > 2
            else 0.0
        )
        if np.isnan(lag1) or np.isinf(lag1):
            lag1 = 0.0

        spread_cost = info.get("pip", 0.0001) * 0.5

        # Test multiple lookbacks
        mom5 = self._test_strategy(p, 5, spread_cost)
        mom10 = self._test_strategy(p, 10, spread_cost)
        mom20 = self._test_strategy(p, 20, spread_cost)

        # Composite score
        composite = mom5["sharpe"] * 0.5 + mom10["sharpe"] * 0.3 + mom20["sharpe"] * 0.2

        # Best lookback
        lb_scores = {5: mom5["sharpe"], 10: mom10["sharpe"], 20: mom20["sharpe"]}
        best_lb = max(lb_scores, key=lb_scores.get)

        # Edge detection
        has_enough_trades = mom5["trades"] >= MIN_TRADES
        edge = (
            composite > MIN_SHARPE
            and mom5["pf"] >= MIN_PROFIT_FACTOR
            and has_enough_trades
        )

        if composite > 0.5 and mom5["pf"] > 1.15:
            rec = "STRONG_BUY"
        elif edge:
            rec = "BUY"
        elif composite > 0:
            rec = "WATCH"
        else:
            rec = "HOLD"

        return ScreenResult(
            ticker=ticker,
            name=info["name"],
            asset_class=info["asset_class"],
            mom5_sharpe=mom5["sharpe"],
            mom5_pf=mom5["pf"],
            mom5_win_rate=mom5["wr"],
            mom5_trades=mom5["trades"],
            mom5_pnl=mom5["pnl"],
            mom10_sharpe=mom10["sharpe"],
            mom10_pf=mom10["pf"],
            mom20_sharpe=mom20["sharpe"],
            mom20_pf=mom20["pf"],
            annual_return=ann_ret,
            annual_volatility=ann_vol,
            lag1_autocorr=lag1,
            n_bars=n,
            current_price=float(p[-1]),
            composite_score=composite,
            edge_detected=edge,
            recommendation=rec,
            optimal_lookback=best_lb,
        )

    # ─── Public API for other agents ──────────────────────────────────────

    def get_tradeable_instruments(self) -> List[Dict]:
        """Return currently tradeable instruments.

        Called by other agents (signal_agent, risk_agent) to get the
        current set of instruments with detected edge.
        """
        return self._tradeable_instruments

    def get_all_scores(self) -> List[Dict]:
        """Return full screening results for all instruments."""
        return [r.to_dict() for r in self._all_results]
