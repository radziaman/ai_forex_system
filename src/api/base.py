"""
Abstract base classes for execution providers and data providers.
All broker integrations implement these interfaces.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Callable
import time


@dataclass
class AccountInfo:
    account_id: str = ""
    balance: float = 0.0
    equity: float = 0.0
    margin: float = 0.0
    free_margin: float = 0.0
    margin_level: float = 0.0
    currency: str = "USD"
    leverage: str = "1:30"
    broker: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class OrderRequest:
    symbol: str = ""
    side: str = "BUY"  # BUY or SELL
    order_type: str = "MARKET"  # MARKET, LIMIT, STOP
    volume: float = 0.0  # in units
    price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    position_id: int = 0  # for closing/modifying


@dataclass
class OrderResult:
    order_id: str = ""
    position_id: str = ""
    status: str = ""  # FILLED, REJECTED, PENDING
    filled_price: float = 0.0
    filled_volume: float = 0.0
    error: str = ""


@dataclass
class PriceTick:
    symbol: str = ""
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0
    volume: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class Position:
    position_id: str = ""
    symbol: str = ""
    direction: str = ""
    volume: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0


@dataclass
class OHLCV:
    timestamp: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0


class ExecutionProvider(ABC):
    """Abstract interface for broker execution."""

    @abstractmethod
    async def start(self) -> bool:
        ...

    @abstractmethod
    async def stop(self):
        ...

    @abstractmethod
    async def get_account_info(self) -> Optional[AccountInfo]:
        ...

    @abstractmethod
    async def place_order(self, order: OrderRequest) -> Optional[OrderResult]:
        ...

    @abstractmethod
    async def close_position(self, position_id: str) -> Optional[OrderResult]:
        ...

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        ...

    # Optional callbacks
    on_price: Optional[Callable] = None
    on_order_update: Optional[Callable] = None


class DataProvider(ABC):
    """Abstract interface for market data (historical + real-time)."""

    @abstractmethod
    async def fetch_ohlcv(
        self, symbol: str, timeframe: str,
        start: str, end: str,
    ) -> List[OHLCV]:
        ...

    @abstractmethod
    async def fetch_ticks(
        self, symbol: str, date: str,
    ) -> List[PriceTick]:
        ...

    @abstractmethod
    async def stream_prices(self, symbols: List[str], callback: Callable):
        ...
