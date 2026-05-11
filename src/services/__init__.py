"""Domain types shared across all services."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
import time


class Regime(str, Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CRISIS = "crisis"


class SignalDirection(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    """Pure output of SignalEngine — no risk info, no execution details."""
    symbol: str
    direction: SignalDirection
    confidence: float
    regime: Regime
    price: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeDecision:
    """Risk-approved decision ready for execution."""
    signal: Signal
    volume: float
    sl_price: float
    tp_price: float
    max_slippage: float = 0.0
    execution_algo: str = "market"


@dataclass
class ExecutionResult:
    success: bool
    position_id: Optional[str] = None
    filled_price: Optional[float] = None
    filled_volume: Optional[float] = None
    error: Optional[str] = None


@dataclass
class FeatureUpdate:
    """Published when new features are ready for a symbol."""
    symbol: str
    timeframe: str
    features: Any  # np.ndarray
    ohlcv: Any  # pd.DataFrame
    price: float
    timestamp: float = field(default_factory=time.time)
