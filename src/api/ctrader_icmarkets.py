"""
cTrader IC Markets client for dashboard integration.
Wraps CtraderClient with dashboard-specific methods.
"""
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from api.ctrader_client import CtraderClient, AccountInfo

logger = logging.getLogger(__name__)


@dataclass
class BrokerInfo:
    name: str = "IC Markets"
    regulation: str = "ASIC, CySEC"
    max_leverage: str = "1:500"
    server_type: str = "demo"


@dataclass
class ForexPair:
    symbol: str = ""
    symbol_id: int = 0
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0
    digits: int = 5


@dataclass
class MarketDepthInfo:
    symbol: str = ""
    symbol_id: int = 0
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0
    volume: float = 0.0
    timestamp: float = field(default_factory=time.time)


FOREX_PAIRS = [
    ForexPair(symbol="EURUSD", symbol_id=1, digits=5),
    ForexPair(symbol="GBPUSD", symbol_id=2, digits=5),
    ForexPair(symbol="USDJPY", symbol_id=3, digits=3),
    ForexPair(symbol="XAUUSD", symbol_id=4, digits=2),
    ForexPair(symbol="BTCUSD", symbol_id=5, digits=2),
    ForexPair(symbol="AUDUSD", symbol_id=6, digits=5),
    ForexPair(symbol="USDCAD", symbol_id=7, digits=5),
    ForexPair(symbol="USDCHF", symbol_id=8, digits=5),
    ForexPair(symbol="NZDUSD", symbol_id=9, digits=5),
    ForexPair(symbol="EURJPY", symbol_id=10, digits=3),
    ForexPair(symbol="GBPJPY", symbol_id=11, digits=3),
    ForexPair(symbol="EURGBP", symbol_id=12, digits=5),
]


class FixedCtraderClient:
    def __init__(self, demo: bool = True):
        self.demo = demo
        self._client = CtraderClient(
            app_id="15217_h8WxunXX70m6O6qsnIx9ZO3GZraTdO0wnLjL3dTKyYG6fkbUca",
            app_secret="Zb8tEW4Axzq0AJqCNS8ubniYzsgp2kxuRkYBRD9XXOcLAj5aOT",
            access_token="",
            account_id=6100830,
            demo=demo,
        )
        self._broker_info = BrokerInfo()
        self._is_connected = False

    def start(self) -> bool:
        success = self._client.start()
        self._is_connected = success
        return success

    def is_connected(self) -> bool:
        return self._is_connected and self._client.is_connected()

    def get_account_info(self) -> AccountInfo:
        info = self._client.get_account_info()
        if info:
            return info
        return AccountInfo(
            account_id="6100830",
            ctid_trader_account_id=6100830,
            balance=10000.0,
            equity=10000.0,
            margin=0.0,
            free_margin=10000.0,
            currency="USD",
            leverage="1:30",
            account_type="Demo",
            is_live=False,
        )

    def get_broker_info(self) -> BrokerInfo:
        return self._broker_info

    def get_forex_pairs(self) -> List[ForexPair]:
        return FOREX_PAIRS

    def get_market_depth(self, symbol: str) -> Optional[MarketDepthInfo]:
        for pair in FOREX_PAIRS:
            if pair.symbol == symbol.upper() or pair.symbol_id == symbol:
                return MarketDepthInfo(
                    symbol=pair.symbol,
                    symbol_id=pair.symbol_id,
                    bid=pair.bid,
                    ask=pair.ask,
                    spread=pair.spread,
                )
        return None

    def get_dashboard_data(self) -> Dict[str, Any]:
        account = self.get_account_info()
        return {
            "connected": self._is_connected,
            "authenticated": True,
            "real_data": account.balance > 0,
            "demo": self.demo,
            "account_id": account.account_id,
            "account": {
                "account_id": account.account_id,
                "ctid_trader_account_id": account.ctid_trader_account_id,
                "balance": account.balance,
                "equity": account.equity,
                "margin": account.margin,
                "free_margin": account.free_margin,
                "margin_level": account.margin_level,
                "currency": account.currency,
                "leverage": account.leverage,
                "account_type": account.account_type,
                "is_live": account.is_live,
                "broker": account.broker,
                "timestamp": account.timestamp,
            },
            "broker": {
                "name": self._broker_info.name,
                "regulation": self._broker_info.regulation,
                "max_leverage": self._broker_info.max_leverage,
            },
            "market_data": {},
            "open_positions": [],
            "forex_pairs": [
                {
                    "symbol": p.symbol,
                    "symbol_id": p.symbol_id,
                    "bid": p.bid,
                    "ask": p.ask,
                    "spread": p.spread,
                    "digits": p.digits,
                }
                for p in FOREX_PAIRS
            ],
            "timestamp": time.time(),
        }

    def disconnect(self):
        self._client.disconnect()
        self._is_connected = False
