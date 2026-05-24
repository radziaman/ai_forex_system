"""
Alternative Data Integration — commodity prices, central bank sentiment,
economic proxies, and cross-asset signals.

Extends the existing sentiment pipeline (src/ai/sentiment.py) with new
orthogonal data sources that provide low-correlation alpha.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import time


class CentralBankTone(Enum):
    HAWKISH = "hawkish"
    NEUTRAL = "neutral"
    DOVISH = "dovish"


@dataclass
class CentralBankSignal:
    bank_name: str
    tone: CentralBankTone
    confidence: float  # 0.0 to 1.0
    mention_count: int = 0
    timestamp: float = 0.0


@dataclass
class CommoditySignal:
    oil_signal: float = 0.0  # z-score
    gold_signal: float = 0.0  # z-score
    copper_signal: float = 0.0  # z-score
    composite: float = 0.0


@dataclass
class AlternativeDataSnapshot:
    central_bank: Dict[str, CentralBankSignal] = field(default_factory=dict)
    commodity: CommoditySignal = field(default_factory=CommoditySignal)
    economic_proxy: Dict[str, float] = field(default_factory=dict)
    composite_signal: Dict[str, float] = field(default_factory=dict)
    timestamp: float = 0.0


# Currency sensitivity to commodities (approximate)
COMMODITY_SENSITIVITY: Dict[str, Dict[str, float]] = {
    # (pair, commodity) -> sensitivity weight
    "USDCAD": {"oil": 0.7, "gold": 0.2, "copper": 0.1},
    "AUDUSD": {"oil": 0.2, "gold": 0.3, "copper": 0.5},
    "NZDUSD": {"oil": 0.1, "gold": 0.1, "copper": 0.1},
    "USDJPY": {"oil": 0.3, "gold": 0.1, "copper": 0.2},
    "EURUSD": {"oil": 0.2, "gold": 0.2, "copper": 0.2},
    "GBPUSD": {"oil": 0.2, "gold": 0.1, "copper": 0.1},
    "USDCHF": {"oil": 0.1, "gold": 0.5, "copper": 0.1},
    "XAUUSD": {"oil": 0.1, "gold": 0.9, "copper": 0.1},
}

# Currency pair sensitivity to central bank tone
CB_SENSITIVITY: Dict[str, str] = {
    "EURUSD": "ECB",
    "GBPUSD": "BOE",
    "USDJPY": "BOJ",
    "AUDUSD": "RBA",
    "NZDUSD": "RBNZ",
    "USDCAD": "BOC",
    "USDCHF": "SNB",
}


class AlternativeDataAggregator:
    """Aggregates alternative data sources into trading signals.

    Sources:
      1. Commodity prices (oil, gold, copper) -> FX pair impact
      2. Central bank speech/meeting tone -> FX pair impact
      3. Economic proxy signals (PMI, employment, CPI)
    """

    def __init__(
        self,
        lookback_commodity: int = 20,
        lookback_cb: int = 10,
    ):
        self.lookback_commodity = lookback_commodity
        self.lookback_cb = lookback_cb
        self._commodity_buffer: Dict[str, list] = {"oil": [], "gold": [], "copper": []}
        self._cb_history: Dict[str, List[CentralBankSignal]] = {}

    def process_commodity_prices(
        self,
        oil_price: Optional[float] = None,
        gold_price: Optional[float] = None,
        copper_price: Optional[float] = None,
    ) -> CommoditySignal:
        """Process commodity price changes into z-score signals."""
        data = {"oil": oil_price, "gold": gold_price, "copper": copper_price}
        z_scores = {}

        for commodity, price in data.items():
            if price is not None and price > 0:
                self._commodity_buffer[commodity].append(price)
                # Keep buffer
                if len(self._commodity_buffer[commodity]) > self.lookback_commodity:
                    self._commodity_buffer[commodity] = self._commodity_buffer[
                        commodity
                    ][-self.lookback_commodity :]

                vals = self._commodity_buffer[commodity]
                if len(vals) >= 10:
                    returns = np.diff(vals) / np.array(vals[:-1])
                    z = (returns[-1] - np.mean(returns)) / max(np.std(returns), 1e-10)
                    z_scores[commodity] = float(z)
                else:
                    z_scores[commodity] = 0.0
            else:
                z_scores[commodity] = 0.0

        composite = float(np.mean(list(z_scores.values())))
        return CommoditySignal(
            oil_signal=z_scores.get("oil", 0.0),
            gold_signal=z_scores.get("gold", 0.0),
            copper_signal=z_scores.get("copper", 0.0),
            composite=composite,
        )

    def process_central_bank_signal(
        self, bank_name: str, tone: CentralBankTone, confidence: float
    ) -> CentralBankSignal:
        """Process a central bank tone signal."""
        signal = CentralBankSignal(
            bank_name=bank_name,
            tone=tone,
            confidence=confidence,
            timestamp=time.time(),
        )
        if bank_name not in self._cb_history:
            self._cb_history[bank_name] = []
        self._cb_history[bank_name].append(signal)
        if len(self._cb_history[bank_name]) > self.lookback_cb:
            self._cb_history[bank_name] = self._cb_history[bank_name][
                -self.lookback_cb :
            ]
        return signal

    def get_recent_cb_tone(self, bank_name: str, window: int = 3) -> float:
        """Get recent central bank tone z-score (-1 = dovish, +1 = hawkish)."""
        history = self._cb_history.get(bank_name, [])
        recent = history[-window:] if len(history) >= window else history
        if not recent:
            return 0.0
        scores = []
        for s in recent:
            if s.tone == CentralBankTone.HAWKISH:
                scores.append(s.confidence)
            elif s.tone == CentralBankTone.DOVISH:
                scores.append(-s.confidence)
            else:
                scores.append(0.0)
        return float(np.mean(scores)) if scores else 0.0

    def compute_composite(
        self,
        commodity_signal: CommoditySignal,
        cb_signals: Dict[str, CentralBankSignal],
        economic_proxy_signals: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Compute composite alternative data signal per FX pair.

        Each pair gets a score based on:
          - Commodity sensitivity x commodity z-score
          - Central bank tone sensitivity
          - Economic proxy signals
        """
        composite = {}
        all_pairs = set(COMMODITY_SENSITIVITY.keys()) | set(CB_SENSITIVITY.keys())

        for pair in all_pairs:
            score = 0.0
            weight_sum = 0.0

            # Commodity contribution
            if pair in COMMODITY_SENSITIVITY:
                sensitivities = COMMODITY_SENSITIVITY[pair]
                for commodity, weight in sensitivities.items():
                    comm_z = getattr(commodity_signal, f"{commodity}_signal", 0.0)
                    score += weight * comm_z
                    weight_sum += weight

            # Central bank contribution
            if pair in CB_SENSITIVITY:
                bank = CB_SENSITIVITY[pair]
                if bank in cb_signals:
                    cb_signal = cb_signals[bank]
                    tone_score = (
                        1.0
                        if cb_signal.tone == CentralBankTone.HAWKISH
                        else (-1.0 if cb_signal.tone == CentralBankTone.DOVISH else 0.0)
                    )
                    score += 0.3 * tone_score * cb_signal.confidence
                    weight_sum += 0.3

            # Economic proxy
            if economic_proxy_signals and pair in economic_proxy_signals:
                score += 0.2 * economic_proxy_signals[pair]
                weight_sum += 0.2

            composite[pair] = score / max(weight_sum, 0.01) if weight_sum > 0 else 0.0

        return composite

    def get_snapshot(
        self,
        commodity_prices: Optional[Dict[str, float]] = None,
        cb_signals: Optional[List[Dict]] = None,
        economic_data: Optional[Dict[str, float]] = None,
    ) -> AlternativeDataSnapshot:
        """Get complete alternative data snapshot."""
        # Process commodities
        comm_signal = self.process_commodity_prices(
            oil_price=commodity_prices.get("oil") if commodity_prices else None,
            gold_price=commodity_prices.get("gold") if commodity_prices else None,
            copper_price=commodity_prices.get("copper") if commodity_prices else None,
        )

        # Process central bank signals
        cb_map: Dict[str, CentralBankSignal] = {}
        if cb_signals:
            for entry in cb_signals:
                bank = entry.get("bank", "")
                tone_str = entry.get("tone", "neutral")
                confidence = entry.get("confidence", 0.5)
                try:
                    tone = CentralBankTone(tone_str)
                except ValueError:
                    tone = CentralBankTone.NEUTRAL
                cb_map[bank] = self.process_central_bank_signal(bank, tone, confidence)

        # Compute composite
        composite = self.compute_composite(comm_signal, cb_map, economic_data)

        return AlternativeDataSnapshot(
            central_bank=cb_map,
            commodity=comm_signal,
            economic_proxy=economic_data or {},
            composite_signal=composite,
            timestamp=time.time(),
        )
