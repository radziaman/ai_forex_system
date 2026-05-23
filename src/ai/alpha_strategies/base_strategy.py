"""Base class for alpha strategies in the RTS AI Forex Trading System."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import pandas as pd


class BaseAlphaStrategy(ABC):
    """Abstract base class for alpha signal generators.

    Subclasses must implement:
      - generate_signal(features_df) -> signal dict
      - get_name() -> strategy name
      - get_required_features() -> list of required column names
    """

    def __init__(self, symbol: str, params: Dict[str, Any]):
        self.symbol = symbol
        self.params = params or {}

    @abstractmethod
    def generate_signal(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate a trading signal from the provided feature DataFrame.

        Returns a dict with keys:
          - direction: 'BUY', 'SELL', or 'HOLD'
          - confidence: float 0..1
          - meta: optional dict with strategy-specific metadata
        """
        raise NotImplementedError

    @abstractmethod
    def get_name(self) -> str:
        """Return the human-readable strategy name."""
        raise NotImplementedError

    @abstractmethod
    def get_required_features(self) -> List[str]:
        """Return the list of DataFrame columns this strategy requires."""
        raise NotImplementedError
