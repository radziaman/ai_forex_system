"""
Regime-Dependent Correlation Matrix — replaces static FX_CORR dict.
Maintains rolling correlation estimates per market regime.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class RegimeCorrelationStore:
    _store: Dict[str, Dict[Tuple[str, str], list]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    window: int = 50

    def update(self, sym_a: str, sym_b: str, correlation: float, regime: str):
        pair = tuple(sorted([sym_a.upper(), sym_b.upper()]))
        if regime not in self._store:
            self._store[regime] = {}
        if pair not in self._store[regime]:
            self._store[regime][pair] = []
        self._store[regime][pair].append(correlation)
        if len(self._store[regime][pair]) > self.window:
            self._store[regime][pair] = self._store[regime][pair][-self.window :]

    def get(self, sym_a: str, sym_b: str, regime: str) -> float:
        pair = tuple(sorted([sym_a.upper(), sym_b.upper()]))
        vals = self._store.get(regime, {}).get(pair, [])
        if not vals:
            for r, pairs in self._store.items():
                if pair in pairs and pairs[pair]:
                    return pairs[pair][-1]
            return 0.0
        return vals[-1]

    def to_dataframe(self, regime: str) -> pd.DataFrame:
        pairs_data = self._store.get(regime, {})
        symbols: List[str] = sorted({sym for pair in pairs_data for sym in pair})
        n = len(symbols)
        corr_matrix = np.eye(n)
        for (a, b), vals in pairs_data.items():
            if a in symbols and b in symbols:
                i, j = symbols.index(a), symbols.index(b)
                corr_matrix[i, j] = vals[-1]
                corr_matrix[j, i] = vals[-1]
        return pd.DataFrame(corr_matrix, index=symbols, columns=symbols)


class RegimeCorrelationMatrix:
    REGIMES = ["trending", "ranging", "volatile", "crisis"]

    def __init__(self, window: int = 100, decay_half_life: float = 0.5):
        self.store = RegimeCorrelationStore(window=window)
        self.window = window
        self.decay_half_life = decay_half_life

    def update_from_returns(
        self, returns: pd.DataFrame, regime: str
    ) -> RegimeCorrelationStore:
        if len(returns) < 10:
            return self.store
        for col in returns.columns:
            for other in returns.columns:
                if col >= other:
                    continue
                series_a = returns[col].values
                series_b = returns[other].values
                weights = np.exp(
                    self.decay_half_life * np.linspace(-1, 0, len(series_a))
                )
                weights /= weights.sum()
                mean_a = np.average(series_a, weights=weights)
                mean_b = np.average(series_b, weights=weights)
                cov = np.average(
                    (series_a - mean_a) * (series_b - mean_b), weights=weights
                )
                std_a = np.sqrt(np.average((series_a - mean_a) ** 2, weights=weights))
                std_b = np.sqrt(np.average((series_b - mean_b) ** 2, weights=weights))
                if std_a > 0 and std_b > 0:
                    corr = float(cov / (std_a * std_b))
                    self.store.update(col, other, corr, regime)
        return self.store

    def get(self, sym_a: str, sym_b: str, regime: Optional[str] = None) -> float:
        if regime:
            return self.store.get(sym_a, sym_b, regime)
        for r in self.REGIMES:
            corr = self.store.get(sym_a, sym_b, r)
            if abs(corr) > 1e-6:
                return corr
        return 0.0
