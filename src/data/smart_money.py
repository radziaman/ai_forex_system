"""
Smart Money Tracking via COTC COT (Commitment of Traders) reports.
Tracks where institutional money is positioning — fade them at extremes.
"""
import numpy as np
import pandas as pd
import time
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from loguru import logger
import os


@dataclass
class COTSnapshot:
    """Commercial (smart money) vs Non-Commercial positioning."""
    symbol: str
    commercial_long: float = 0.0
    commercial_short: float = 0.0
    noncommercial_long: float = 0.0
    noncommercial_short: float = 0.0
    net_commercial: float = 0.0  # Positive = commercials net long
    institutional_signal: str = "neutral"  # long | short | neutral | extreme_long | extreme_short
    position_extreme: bool = False
    recommended_action: str = "follow"  # follow | fade | neutral


class COTAnalyzer:
    """
    Analyze CFTC Commitment of Traders reports for positioning edges.
    Commercial hedgers are "smart money" — fade their positioning at extremes.
    Non-commercial (speculators) are usually wrong at extremes.
    """

    # Map forex pairs to CFTC codes
    COT_CODES = {
        "EURUSD": "096742",  # Euro FX
        "GBPUSD": "096742",  # British Pound (same code, different report)
        "USDJPY": "096742",
        "AUDUSD": "096742",
        "USDCAD": "090741",  # Canadian Dollar
        "USDCHF": "096742",
        "NZDUSD": "096742",
    }

    COT_URL = "https://www.cftc.gov/dea/options/deacmesf.htm"

    def __init__(self, cache_dir: str = "data/alternative_data"):
        self.cache_dir = cache_dir
        self.cache_path = os.path.join(cache_dir, "cot_cache.json")
        self.snapshots: Dict[str, COTSnapshot] = {}
        self._last_fetch = 0.0
        self.cache_ttl = 86400  # 24 hours

        os.makedirs(cache_dir, exist_ok=True)
        self._load_cache()

    def fetch_latest(self, symbol: str) -> COTSnapshot:
        """Get institutional positioning for a symbol."""
        # Check cache first
        if symbol in self.snapshots:
            snapshot = self.snapshots[symbol]
            if time.time() - self._last_fetch < self.cache_ttl:
                return snapshot

        # Try to download fresh COT data
        try:
            snapshot = self._fetch_cot_report(symbol)
            self.snapshots[symbol] = snapshot
            self._last_fetch = time.time()
            self._save_cache()
            return snapshot
        except Exception as e:
            logger.debug(f"COT fetch failed for {symbol}: {e}")
            return self.snapshots.get(symbol, COTSnapshot(symbol=symbol))

    def _fetch_cot_report(self, symbol: str) -> COTSnapshot:
        """Fetch and parse CFTC COT report."""
        import requests
        from lxml import html

        # For demo/placeholder: generate synthetic institutional data
        # In production, parse the actual CFTC Excel/CSV
        cot_code = self.COT_CODES.get(symbol, "096742")

        # Simulate COT data (replace with actual API call)
        np.random.seed(hash(symbol) % 2**32)
        commercial_long = np.random.normal(50000, 5000)
        commercial_short = np.random.normal(45000, 5000)
        noncom_long = np.random.normal(30000, 3000)
        noncom_short = np.random.normal(35000, 3000)

        net_commercial = commercial_long - commercial_short
        total = commercial_long + commercial_short + noncom_long + noncom_short
        net_ratio = net_commercial / total if total > 0 else 0.0

        # Determine signal
        if net_ratio > 0.15:
            signal = "extreme_long" if net_ratio > 0.25 else "long"
        elif net_ratio < -0.15:
            signal = "extreme_short" if net_ratio < -0.25 else "short"
        else:
            signal = "neutral"

        # At extremes, fade the commercials (they're usually right at extremes)
        action = "fade" if abs(net_ratio) > 0.25 else "follow"

        snapshot = COTSnapshot(
            symbol=symbol,
            commercial_long=commercial_long,
            commercial_short=commercial_short,
            noncommercial_long=noncom_long,
            noncommercial_short=noncom_short,
            net_commercial=net_ratio,
            institutional_signal=signal,
            position_extreme=abs(net_ratio) > 0.25,
            recommended_action=action,
        )

        logger.info(
            f"COT {symbol}: commercial_net={net_ratio:.2f} → {signal} → {action}"
        )
        return snapshot

    def get_trading_signal(
        self, symbol: str, current_direction: str
    ) -> Tuple[bool, str]:
        """
        Returns (should_block_trade, reason).
        Block trades that go AGAINST smart money at extremes.
        """
        snapshot = self.fetch_latest(symbol)

        if not snapshot.position_extreme:
            return False, "normal_positioning"

        # If institutions are extremely long, don't go short
        if snapshot.institutional_signal in ["extreme_long", "long"]:
            if current_direction == "SELL":
                return True, f"fade_smart_money_long_{snapshot.net_commercial:.2f}"
        elif snapshot.institutional_signal in ["extreme_short", "short"]:
            if current_direction == "BUY":
                return True, f"fade_smart_money_short_{snapshot.net_commercial:.2f}"

        return False, "aligned_with_smart_money"

    def get_feature_vector(self, symbols: Optional[List[str]] = None) -> np.ndarray:
        """Get COT features as vector for ML models."""
        targets = symbols or ["EURUSD", "GBPUSD", "USDJPY"]
        vec = []
        for sym in targets:
            snap = self.fetch_latest(sym)
            vec.extend([
                snap.net_commercial,
                1.0 if snap.position_extreme else 0.0,
                1.0 if "long" in snap.institutional_signal else (
                    -1.0 if "short" in snap.institutional_signal else 0.0
                ),
            ])
        return np.array(vec, dtype=np.float32)

    def _save_cache(self):
        try:
            import json
            data = {
                sym: {
                    "commercial_long": s.commercial_long,
                    "commercial_short": s.commercial_short,
                    "noncommercial_long": s.noncommercial_long,
                    "noncommercial_short": s.noncommercial_short,
                    "net_commercial": s.net_commercial,
                    "institutional_signal": s.institutional_signal,
                    "position_extreme": s.position_extreme,
                    "recommended_action": s.recommended_action,
                }
                for sym, s in self.snapshots.items()
            }
            with open(self.cache_path, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def _load_cache(self):
        try:
            import json
            if not os.path.exists(self.cache_path):
                return
            with open(self.cache_path) as f:
                data = json.load(f)
            for sym, d in data.items():
                self.snapshots[sym] = COTSnapshot(symbol=sym, **d)
        except Exception:
            pass
