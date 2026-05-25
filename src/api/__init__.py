"""API module — broker execution providers and data providers."""

# Prefer the official Spotware-based wrapper; fall back to legacy client
try:
    from api.ctrader_wrapper import CtraderClient
except ImportError:
    from api.ctrader_client import CtraderClient

from api.ctrader_client import (
    MarketDepth,
    TradeOrder,
    TradeResult,
    AccountInfo as CtraderAccountInfo,
    SYMBOL_MAP,
    REVERSE_SYMBOL_MAP,
)
from api.base import (
    ExecutionProvider,
    DataProvider,
    AccountInfo,
    OrderRequest,
    OrderResult,
    PriceTick,
    Position,
    OHLCV,
)

__all__ = [
    "CtraderClient",
    "MarketDepth",
    "TradeOrder",
    "TradeResult",
    "CtraderAccountInfo",
    "SYMBOL_MAP",
    "REVERSE_SYMBOL_MAP",
    "ExecutionProvider",
    "DataProvider",
    "AccountInfo",
    "OrderRequest",
    "OrderResult",
    "PriceTick",
    "Position",
    "OHLCV",
]
