"""
Comprehensive cTrader Client for IC Markets
Pulls ALL available data: account info, broker info, forex pairs, real-time data, etc.
"""
import time
import json
import logging
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Symbol mapping (EURUSD=1, GBPUSD=2, etc.)
SYMBOL_MAP = {
    1: "EURUSD",
    2: "GBPUSD",
    3: "USDJPY",
    4: "XAUUSD",  # Gold
    5: "BTCUSD",  # Bitcoin
    6: "ETHUSD",  # Ethereum
    7: "AUDUSD",
    8: "USDCHF",
    9: "NZDUSD",
    10: "USDCAD",
}
REVERSE_SYMBOL_MAP = {v: k for k, v in SYMBOL_MAP.items()}

# Timeframe mapping
TIMEFRAME_MAP = {
    "1m": "ProtoOATimeframe.M1",
    "5m": "ProtoOATimeframe.M5",
    "15m": "ProtoOATimeframe.M15",
    "30m": "ProtoOATimeframe.M30",
    "1h": "ProtoOATimeframe.H1",
    "4h": "ProtoOATimeframe.H4",
    "1d": "ProtoOATimeframe.D1",
}


@dataclass
class MarketDepth:
    """Real-time market data from cTrader."""
    symbol: str = ""
    symbol_id: int = 0
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0
    volume: int = 0
    timestamp: float = field(default_factory=lambda: time.time())
    change_pct: float = 0.0
    high_24h: float = 0.0
    low_24h: float = 0.0


@dataclass
class AccountInfo:
    """Complete account information from cTrader/IC Markets."""
    account_id: str = ""
    ctid_account_id: int = 0
    balance: float = 0.0
    equity: float = 0.0
    margin: float = 0.0
    free_margin: float = 0.0
    margin_level: float = 0.0
    currency: str = "USD"
    is_live: bool = False
    leverage: str = "1:30"
    broker: str = "IC Markets"
    account_type: str = "Demo"  # Demo or Live
    deposit_currency: str = "USD"


@dataclass
class BrokerInfo:
    """Broker information (IC Markets)."""
    name: str = "IC Markets"
    website: str = "https://www.icmarkets.com"
    regulation: str = "ASIC, CySEC"
    max_leverage: str = "1:500"
    min_deposit: str = "$200"
    spreads_from: str = "0.0 pips"
    trading_instruments: int = 1000
    platforms: List[str] = field(default_factory=lambda: ["cTrader", "MetaTrader 4", "MetaTrader 5"])


@dataclass
class ForexPair:
    """Forex pair information."""
    symbol: str = ""
    symbol_id: int = 0
    description: str = ""
    pip_value: float = 0.0
    contract_size: int = 100000
    margin_req: float = 0.02  # 2% = 1:50 leverage
    swap_long: float = 0.0
    swap_short: float = 0.0
    commission: float = 0.0
    trading_hours: str = "24/5"
    digits: int = 5


@dataclass
class TradeOrder:
    """Order to be sent to cTrader."""
    symbol: str = "EURUSD"
    symbol_id: int = 1
    side: str = "BUY"  # BUY or SELL
    order_type: str = "MARKET"  # MARKET or LIMIT
    volume: int = 100000  # in units (100000 = 1 lot)
    price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    position_id: int = 0  # for close orders


@dataclass
class OrderResult:
    """Result from order execution."""
    order_id: str = ""
    status: str = "PENDING"  # FILLED, REJECTED, PENDING
    filled_price: float = 0.0
    volume: int = 0
    error: str = ""


@dataclass
class OpenPosition:
    """Open position from cTrader."""
    position_id: int = 0
    symbol: str = ""
    direction: str = ""  # BUY or SELL
    volume: int = 0
    entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    swap: float = 0.0
    commission: float = 0.0


class CtraderClient:
    """
    Comprehensive cTrader client for IC Markets.
    Pulls ALL available data: account, broker, forex pairs, real-time data.
    """

    def __init__(
        self,
        app_id: str = "",
        app_secret: str = "",
        access_token: str = "",
        account_id: str = "",
        demo: bool = True,
    ):
        self.app_id = app_id
        self.app_secret = app_secret
        self.access_token = access_token
        self.account_id = account_id
        self.demo = demo

        # Connection state
        self._is_connected = False
        self._is_simulation = True
        self._ctid_account_id = 0

        # Data storage
        self._account_info: Optional[AccountInfo] = None
        self._broker_info: BrokerInfo = BrokerInfo()
        self._market_depth: Dict[int, MarketDepth] = {}
        self._open_positions: Dict[int, OpenPosition] = {}
        self._forex_pairs: Dict[int, ForexPair] = {}

        # Callbacks
        self.on_market_data: Optional[Callable[[MarketDepth], None]] = None
        self.on_account_update: Optional[Callable[[AccountInfo], None]] = None
        self.on_order_update: Optional[Callable[[OrderResult], None]] = None
        self.on_positions_update: Optional[Callable[[List[OpenPosition]], None]] = None

        logger.info(f"CtraderClient initialized | Demo: {demo} | Account: {account_id}")

    def start(self):
        """Start the client (simulation mode for now)."""
        logger.info("Starting in SIMULATION mode")
        logger.info("For live trading, complete OAuth2 setup with IC Markets cTrader")
        self._is_connected = True
        self._is_simulation = True

        # Initialize with demo data
        self._init_demo_data()
        logger.info("✓ Client ready (simulation mode)")

    def _init_demo_data(self):
        """Initialize demo data for simulation mode."""
        # Demo account info
        self._account_info = AccountInfo(
            account_id=self.account_id,
            ctid_account_id=6100830,
            balance=10000.0,
            equity=10000.0,
            margin=0.0,
            free_margin=10000.0,
            margin_level=0.0,
            currency="USD",
            is_live=not self.demo,
            leverage="1:30",
            broker="IC Markets",
            account_type="Demo" if self.demo else "Live",
            deposit_currency="USD",
        )

        # Initialize forex pairs
        for sid, sname in SYMBOL_MAP.items():
            self._forex_pairs[sid] = ForexPair(
                symbol=sname,
                symbol_id=sid,
                description=f"{sname} - {'US Dollar' if 'USD' in sname else 'Forex Pair'}",
                pip_value=0.0001 if 'JPY' not in sname else 0.01,
                contract_size=100000,
                margin_req=0.02,
                digits=5 if 'JPY' not in sname else 3,
                trading_hours="24/5",
            )

        # Initialize market depth with demo prices
        demo_prices = {
            1: (1.16754, 1.16756),  # EURUSD
            2: (1.34762, 1.34764),  # GBPUSD
            3: (157.50, 157.52),  # USDJPY
            4: (2345.50, 2346.50),  # XAUUSD
            7: (0.71172, 0.71174),  # AUDUSD
        }

        for sid, (bid, ask) in demo_prices.items():
            self._market_depth[sid] = MarketDepth(
                symbol=SYMBOL_MAP.get(sid, ""),
                symbol_id=sid,
                bid=bid,
                ask=ask,
                spread=ask - bid,
                volume=1000000,
                timestamp=time.time(),
            )

        logger.info(f"✓ Demo data initialized: {len(self._forex_pairs)} pairs, {len(self._market_depth)} prices")

    def get_account_info(self) -> Optional[AccountInfo]:
        """Get complete account information."""
        if not self._is_connected:
            logger.warning("Not connected")
            return None
        return self._account_info

    def get_broker_info(self) -> BrokerInfo:
        """Get broker information (IC Markets)."""
        return self._broker_info

    def get_forex_pairs(self) -> List[ForexPair]:
        """Get all available forex pairs."""
        return list(self._forex_pairs.values())

    def get_forex_pair(self, symbol: str) -> Optional[ForexPair]:
        """Get information for a specific forex pair."""
        symbol_id = REVERSE_SYMBOL_MAP.get(symbol.upper())
        if symbol_id:
            return self._forex_pairs.get(symbol_id)
        return None

    def get_market_depth(self, symbol: str) -> Optional[MarketDepth]:
        """Get current market depth for a symbol."""
        symbol_id = REVERSE_SYMBOL_MAP.get(symbol.upper())
        if symbol_id and symbol_id in self._market_depth:
            return self._market_depth[symbol_id]
        return None

    def get_all_market_data(self) -> Dict[str, MarketDepth]:
        """Get all available market data."""
        return {
            SYMBOL_MAP[sid]: depth
            for sid, depth in self._market_depth.items()
            if sid in SYMBOL_MAP
        }

    def get_open_positions(self) -> List[OpenPosition]:
        """Get all open positions."""
        return list(self._open_positions.values())

    def subscribe_market_data(self, symbol: str):
        """Subscribe to real-time market data."""
        symbol_id = REVERSE_SYMBOL_MAP.get(symbol.upper())
        if symbol_id:
            logger.info(f"Subscribed to {symbol} (ID: {symbol_id})")
            # In production, this would send subscription request
        else:
            logger.warning(f"Symbol {symbol} not supported")

    def place_order(self, order: TradeOrder) -> OrderResult:
        """Place a market order."""
        if not self._is_connected:
            logger.warning("Not connected")
            return OrderResult(status="REJECTED", error="Not connected")

        if self._is_simulation:
            logger.info(f"[SIM] Order: {order.side} {order.volume/100000:.2f} lots {order.symbol}")
            return OrderResult(
                order_id=f"SIM_{int(time.time())}",
                status="FILLED",
                filled_price=order.price or self._market_depth.get(order.symbol_id, MarketDepth()).ask,
                volume=order.volume,
            )

        # In production, send ProtoOANewOrderReq here
        logger.info(f"[REAL] Order: {order.side} {order.volume} {order.symbol}")
        return OrderResult(status="PENDING")

    def close_position(self, position_id: int) -> OrderResult:
        """Close an open position."""
        if position_id in self._open_positions:
            pos = self._open_positions[position_id]
            logger.info(f"[SIM] Close position {position_id}: {pos.direction} {pos.symbol}")
            del self._open_positions[position_id]
            return OrderResult(
                order_id=f"CLOSE_{position_id}",
                status="FILLED",
                filled_price=pos.current_price,
                volume=pos.volume,
            )
        return OrderResult(status="REJECTED", error=f"Position {position_id} not found")

    def disconnect(self):
        """Disconnect from cTrader."""
        self._is_connected = False
        logger.info("Disconnected from cTrader (simulation)")

    def is_connected(self) -> bool:
        """Check connection status."""
        return self._is_connected

    def is_simulation(self) -> bool:
        """Check if in simulation mode."""
        return self._is_simulation

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get all data needed for the dashboard."""
        data = {
            "connected": self._is_connected,
            "demo": self.demo,
            "account_id": self.account_id,
            "broker": {
                "name": self._broker_info.name,
                "regulation": self._broker_info.regulation,
                "platforms": self._broker_info.platforms,
            },
            "account": None,
            "market_data": {},
            "forex_pairs": [],
            "open_positions": [],
        }

        if self._account_info:
            data["account"] = {
                "account_id": self._account_info.account_id,
                "balance": self._account_info.balance,
                "equity": self._account_info.equity,
                "margin": self._account_info.margin,
                "free_margin": self._account_info.free_margin,
                "leverage": self._account_info.leverage,
                "currency": self._account_info.currency,
                "type": self._account_info.account_type,
            }

        data["market_data"] = {
            symbol: {
                "bid": depth.bid,
                "ask": depth.ask,
                "spread": depth.spread,
                "volume": depth.volume,
            }
            for symbol, depth in self.get_all_market_data().items()
        }

        data["forex_pairs"] = [
            {
                "symbol": fp.symbol,
                "description": fp.description,
                "pip_value": fp.pip_value,
                "contract_size": fp.contract_size,
                "margin_req": fp.margin_req,
            }
            for fp in self.get_forex_pairs()
        ]

        data["open_positions"] = [
            {
                "position_id": pos.position_id,
                "symbol": pos.symbol,
                "direction": pos.direction,
                "volume": pos.volume,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "unrealized_pnl": pos.unrealized_pnl,
                "sl": pos.sl,
                "tp": pos.tp,
            }
            for pos in self.get_open_positions()
        ]

        return data


def create_ctrader_config():
    """Create a template cTrader config file."""
    config = {
        "app_id": "YOUR_APP_ID",
        "app_secret": "YOUR_APP_SECRET",
        "access_token": "YOUR_ACCESS_TOKEN",
        "account_id": "YOUR_ACCOUNT_ID",
        "demo": True,
        "ic_markets": {
            "name": "IC Markets",
            "website": "https://www.icmarkets.com",
            "regulation": "ASIC (Australia), CySEC (Cyprus)",
            "max_leverage": "1:500",
            "min_deposit": "$200",
            "spreads_from": "0.0 pips",
            "trading_instruments": "1000+",
            "platforms": ["cTrader", "MetaTrader 4", "MetaTrader 5"],
            "servers": {
                "live": "live.ctrader.com:5035",
                "demo": "demo.ctrader.com:5035",
            },
        },
    }

    config_path = Path("config/ctrader_config.json")
    config_path.parent.mkdir(exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"✓ Config template saved to {config_path}")
    print("Please edit with your actual cTrader credentials")


if __name__ == "__main__":
    create_ctrader_config()
