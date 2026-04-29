"""
cTrader Open API Python Client
Simplified version for IC Markets integration.
Uses ctrader-open-api SDK (Twisted-based, Protobuf).
"""

import time
import threading
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass, field
from loguru import logger

from twisted.internet import reactor
from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import *
from ctrader_open_api.messages.OpenApiMessages_pb2 import *
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import *


# Symbol mapping (EURUSD=1, GBPUSD=2, etc.)
SYMBOL_MAP = {
    1: "EURUSD",
    2: "GBPUSD",
    3: "USDJPY",
    4: "XAUUSD",
    5: "BTCUSD",
    6: "ETHUSD",
    7: "AUDUSD",
    8: "USDCHF",
    9: "NZDUSD",
    10: "USDCAD",
}
REVERSE_SYMBOL_MAP = {v: k for k, v in SYMBOL_MAP.items()}


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
class AccountInfo:
    """Account information from cTrader."""
    account_id: str = ""
    ctid_account_id: int = 0
    balance: float = 0.0
    equity: float = 0.0
    margin: float = 0.0
    free_margin: float = 0.0
    currency: str = "USD"
    is_live: bool = False


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


class CtraderClient:
    """
    cTrader Open API Python client (Twisted-based, async via deferreds).
    """

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        access_token: str,
        account_id: str,
        demo: bool = True,
    ):
        self.app_id = app_id
        self.app_secret = app_secret
        self.access_token = access_token
        self.account_id = account_id
        self.demo = demo

        # Connection state
        self._client: Optional[Client] = None
        self._is_connected = False
        self._is_simulation = True
        self._ctid_account_id = 0
        self._account_info: Optional[AccountInfo] = None
        self._open_positions: Dict[int, OpenPosition] = {}
        self._last_market_depth: Dict[int, MarketDepth] = {}
        self._last_order_filled = False

        # Callbacks (set by external components)
        self.on_market_data: Optional[Callable[[MarketDepth], None]] = None
        self.on_account_update: Optional[Callable[[AccountInfo], None]] = None
        self.on_order_update: Optional[Callable[[OrderResult], None]] = None
        self.on_positions_update: Optional[Callable[[List[OpenPosition]], None]] = None

        logger.info(f"CtraderClient initialized | Demo: {demo} | Account: {account_id}")

    def start(self):
        """Start the client and connect to cTrader."""
        host = (
            EndPoints.PROTOBUF_DEMO_HOST if self.demo else EndPoints.PROTOBUF_LIVE_HOST
        )
        port = EndPoints.PROTOBUF_PORT
        protocol = TcpProtocol

        logger.info(f"Connecting to {host}:{port} via {protocol.__name__}")

        self._client = Client(host, port, protocol)

        # Register message handlers
        self._client.add_handler(ProtoOAApplicationAuthRes, self._on_app_auth)
        self._client.add_handler(ProtoOAAccountAuthRes, self._on_account_auth)
        self._client.add_handler(
            ProtoOAGetAccountListByAccessTokenRes, self._on_account_list
        )
        self._client.add_handler(ProtoOASpotEvent, self._on_spot_event)
        self._client.add_handler(ProtoOAExecutionEvent, self._on_execution)
        self._client.add_handler(ProtoOAOrderErrorEvent, self._on_order_error)
        self._client.add_handler(ProtoOAErrorRes, self._on_error)

        # Start reactor in separate thread
        if not reactor.running:
            threading.Thread(
                target=reactor.run, kwargs={"installSignalHandlers": False}, daemon=True
            ).start()

        # Connect and authenticate
        self._connect_and_auth()

    def _connect_and_auth(self):
        """Connect and authenticate with cTrader."""
        time.sleep(1)  # Brief delay for reactor to start

        logger.info("Authenticating application...")
        req = ProtoOAApplicationAuthReq()
        req.client_id = self.app_id
        req.client_secret = self.app_secret

        def _on_app_auth_sent(result):
            logger.info("App auth request sent")

        d = self._client.send(req)
        d.addCallback(_on_app_auth_sent)
        d.addErrback(lambda f: logger.error(f"App auth failed: {f}"))

    def _on_app_auth(self, msg: ProtoOAApplicationAuthRes):
        """Handle application authentication response."""
        logger.success("Application authenticated successfully")
        self._is_connected = True
        self._get_account_list()

    def _get_account_list(self):
        """Get account list using access token."""
        if not self.access_token:
            logger.warning("No access token - running in simulation mode")
            self._is_simulation = True
            return

        logger.info("Getting account list...")
        req = ProtoOAGetAccountListByAccessTokenReq()
        req.access_token = self.access_token
        self._client.send(req)

    def _on_account_list(self, msg: ProtoOAGetAccountListByAccessTokenRes):
        """Handle account list response."""
        logger.info(f"Received {len(msg.ctid_trader_account)} accounts")

        matched_ctid = None
        matched_login = ""

        for acc in msg.ctid_trader_account:
            login = str(acc.trader_login)
            ctid = int(acc.ctid_trader_account_id)
            is_live = acc.is_live
            logger.info(f"  Account: {login} (CTID: {ctid}, Live: {is_live})")

            if login == self.account_id or str(ctid) == self.account_id:
                matched_ctid = ctid
                matched_login = login

        if matched_ctid:
            self._ctid_account_id = matched_ctid
            logger.info(f"Using account CTID: {matched_ctid} (Login: {matched_login})")
            self._auth_account()
        else:
            fallback = next((a for a in msg.ctid_trader_account if not a.is_live), None)
            if fallback:
                self._ctid_account_id = int(fallback.ctid_trader_account_id)
                logger.info(f"Using fallback DEMO CTID: {self._ctid_account_id}")
                self._auth_account()
            else:
                logger.warning("No matching account found - simulation mode")
                self._is_simulation = True

    def _auth_account(self):
        """Authenticate trading account."""
        logger.info(f"Authenticating account CTID: {self._ctid_account_id}...")
        req = ProtoOAAccountAuthReq()
        req.ctid_trader_account_id = self._ctid_account_id
        req.access_token = self.access_token
        self._client.send(req)

    def _on_account_auth(self, msg: ProtoOAAccountAuthRes):
        """Handle account authentication response."""
        logger.success(f"Account authenticated | CTID: {msg.ctid_trader_account_id}")
        self._is_simulation = False
        self._is_connected = True
        self.subscribe_market_data("EURUSD")

    def _on_spot_event(self, msg: ProtoOASpotEvent):
        """Handle real-time spot updates."""
        symbol_id = msg.symbol_id
        bid = msg.bid / 100000.0
        ask = msg.ask / 100000.0
        symbol = SYMBOL_MAP.get(symbol_id, f"UNKNOWN_{symbol_id}")

        depth = MarketDepth(
            symbol=symbol,
            symbol_id=symbol_id,
            bid=bid,
            ask=ask,
            spread=abs(ask - bid),
            timestamp=time.time(),
        )
        self._last_market_depth[symbol_id] = depth

        if self.on_market_data:
            self.on_market_data(depth)

    def subscribe_market_data(self, symbol: str):
        """Subscribe to real-time spot data."""
        symbol_id = REVERSE_SYMBOL_MAP.get(symbol.upper(), 1)
        logger.info(f"Subscribing to {symbol} (ID: {symbol_id})")

        req = ProtoOASubscribeSpotsReq()
        req.ctid_trader_account_id = self._ctid_account_id
        req.symbol_id.append(symbol_id)
        self._client.send(req)

    def place_order(self, order: TradeOrder) -> OrderResult:
        """
        Place a market order via cTrader.
        Returns OrderResult with status.
        """
        if self._is_simulation or not self._is_connected:
            logger.info(
                f"[SIM] Order: {order.side} {order.volume/100000:.2f} lots {order.symbol}"
            )
            return OrderResult(
                order_id=f"SIM_{int(time.time())}",
                status="FILLED",
                filled_price=order.price
                or (self._last_market_depth.get(order.symbol_id, MarketDepth())).ask,
                volume=order.volume,
            )

        req = ProtoOANewOrderReq()
        req.ctid_trader_account_id = self._ctid_account_id
        req.order_type = ProtoOAOrderType.Market
        req.trade_side = (
            ProtoOATradeSide.Buy
            if order.side.upper() == "BUY"
            else ProtoOATradeSide.Sell
        )
        req.volume = order.volume
        req.symbol_id = order.symbol_id

        if order.position_id > 0:
            req.position_id = order.position_id
            logger.info(
                f"[REAL] Close position {order.position_id}: {order.side} {order.volume/100000:.2f} lots"
            )
        else:
            logger.info(
                f"[REAL] Open order: {order.side} {order.volume/100000:.2f} lots {order.symbol}"
            )

        self._last_order_filled = False
        self._client.send(req)

        # Wait briefly for execution event
        for _ in range(10):
            time.sleep(0.5)
            if self._last_order_filled:
                break

        result = OrderResult(
            order_id=f"OPENAPI_{int(time.time())}",
            status="FILLED" if self._last_order_filled else "PENDING",
            filled_price=order.price,
            volume=order.volume,
        )

        if self.on_order_update:
            self.on_order_update(result)

        return result

    def _on_execution(self, msg: ProtoOAExecutionEvent):
        """Handle order execution."""
        logger.success(f"Order executed: ExecutionID={msg.execution_id}")
        self._last_order_filled = True

        result = OrderResult(
            order_id=str(msg.execution_id),
            status="FILLED",
            filled_price=msg.price / 100000.0 if msg.price else 0.0,
            volume=msg.volume,
        )
        if self.on_order_update:
            self.on_order_update(result)

    def _on_order_error(self, msg: ProtoOAOrderErrorEvent):
        """Handle order error."""
        logger.error(f"Order error: {msg}")
        result = OrderResult(status="REJECTED", error=str(msg))
        if self.on_order_update:
            self.on_order_update(result)

    def _on_error(self, msg: ProtoOAErrorRes):
        """Handle API error."""
        logger.error(f"API Error: {msg.error_code} - {msg.description}")

    def get_account_info(self) -> Optional[AccountInfo]:
        """Get current account info."""
        return self._account_info

    def get_open_positions(self) -> List[OpenPosition]:
        """Get all open positions."""
        return list(self._open_positions.values())

    def disconnect(self):
        """Disconnect from cTrader."""
        if self._client:
            self._client.stopService()
        if reactor.running:
            reactor.stop()
        self._is_connected = False
        logger.info("Disconnected from cTrader")
