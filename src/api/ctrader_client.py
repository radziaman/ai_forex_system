"""
cTrader Open API Client for RTS - AI FX Trading System
Uses: Standard ssl (LibreSSL 2.8.3) + Protobuf
"""
import ssl
import socket
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Callable

try:
    from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import ProtoMessage
    from ctrader_open_api.messages.OpenApiMessages_pb2 import (
        ProtoOAApplicationAuthReq,
        ProtoOAAccountAuthReq,
        ProtoOATraderReq,
        ProtoOATraderRes,
        ProtoOANewOrderReq,
        ProtoOAExecutionEvent,
    )
    _HAS_PROTOBUF = True
except ImportError:
    _HAS_PROTOBUF = False

logger = logging.getLogger(__name__)

# Symbol mapping: symbol -> cTrader symbol_id
SYMBOL_MAP = {
    "EURUSD": 1,
    "GBPUSD": 2,
    "USDJPY": 3,
    "XAUUSD": 4,
    "BTCUSD": 5,
    "AUDUSD": 6,
    "USDCAD": 7,
    "USDCHF": 8,
    "NZDUSD": 9,
    "EURJPY": 10,
    "GBPJPY": 11,
    "EURGBP": 12,
}

REVERSE_SYMBOL_MAP = {v: k for k, v in SYMBOL_MAP.items()}


@dataclass
class MarketDepth:
    symbol: str = ""
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0
    volume: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class TradeOrder:
    symbol: str = ""
    symbol_id: int = 0
    side: str = "BUY"
    order_type: str = "MARKET"
    volume: int = 0
    price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    position_id: int = 0
    stop_loss: float = 0.0
    take_profit: float = 0.0


@dataclass
class TradeResult:
    order_id: int = 0
    position_id: int = 0
    status: str = ""
    filled_price: float = 0.0
    error: str = ""


@dataclass
class AccountInfo:
    account_id: str = ""
    ctid_trader_account_id: int = 0
    balance: float = 0.0
    equity: float = 0.0
    margin: float = 0.0
    free_margin: float = 0.0
    margin_level: float = 0.0
    currency: str = "USD"
    leverage: str = "1:30"
    account_type: str = "Demo"
    is_live: bool = False
    broker: str = "IC Markets"
    timestamp: float = field(default_factory=time.time)


class CtraderClient:
    def __init__(
        self,
        app_id: str = "",
        app_secret: str = "",
        access_token: str = "",
        account_id: int = 0,
        demo: bool = True,
        use_websocket: bool = False,
    ):
        self.app_id = app_id
        self.app_secret = app_secret
        self.access_token = access_token
        self.account_id = account_id
        self.demo = demo
        self.use_websocket = use_websocket
        self.host = "demo.ctraderapi.com" if demo else "live.ctraderapi.com"
        self.port = 5035  # Protobuf port

        self.ssl_socket = None
        self.tcp_socket = None
        self._is_connected = False
        self._authenticated = False
        self._account_info: Optional[AccountInfo] = None
        self._last_market_depth: Dict[int, MarketDepth] = {}

        self.on_market_data: Optional[Callable] = None
        self.on_account_update: Optional[Callable] = None
        self.on_order_update: Optional[Callable] = None
        self.on_positions_update: Optional[Callable] = None

    def start(self) -> bool:
        try:
            if not _HAS_PROTOBUF:
                logger.warning("ctrader_open_api not installed, using simulation mode")
                return self._start_simulation()
            if not self._connect():
                logger.error("Failed to connect to cTrader")
                return self._start_simulation()
            if not self._auth_application():
                logger.error("Application auth failed")
                self.disconnect()
                return self._start_simulation()
            if not self._auth_account():
                logger.error("Account auth failed")
                self.disconnect()
                return self._start_simulation()
            self._fetch_account_info()
            self._is_connected = True
            logger.success("cTrader client connected and authenticated")
            return True
        except Exception as e:
            logger.error(f"Failed to start cTrader client: {e}")
            return self._start_simulation()

    def _start_simulation(self) -> bool:
        logger.info("Starting cTrader client in simulation mode")
        self._is_connected = True
        self._account_info = AccountInfo(
            account_id=str(self.account_id),
            ctid_trader_account_id=self.account_id,
            balance=100000.0,
            equity=100000.0,
            margin=0.0,
            free_margin=100000.0,
            currency="USD",
            leverage="1:30",
            account_type="Demo (Simulation)",
            is_live=False,
        )
        return True

    def _connect(self) -> bool:
        try:
            context = ssl.create_default_context()
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            self.tcp_socket = socket.create_connection(
                (self.host, self.port), timeout=10
            )
            self.ssl_socket = context.wrap_socket(
                self.tcp_socket, server_hostname=self.host
            )
            logger.info(f"Connected via {self.ssl_socket.version()}")
            return True
        except Exception as e:
            logger.error(f"SSL connection failed: {e}")
            return False

    def _send_msg(self, payload_type: int, payload_obj) -> bool:
        try:
            proto_msg = ProtoMessage()
            proto_msg.payloadType = payload_type
            proto_msg.payload = payload_obj.SerializeToString()
            data = proto_msg.SerializeToString()
            self.ssl_socket.sendall(len(data).to_bytes(4, "big") + data)
            return True
        except Exception as e:
            logger.error(f"Send failed: {e}")
            return False

    def _recv_msg(self, timeout: int = 15):
        try:
            self.ssl_socket.settimeout(timeout)
            length_bytes = b""
            while len(length_bytes) < 4:
                chunk = self.ssl_socket.recv(4 - len(length_bytes))
                if not chunk:
                    return None
                length_bytes += chunk
            msg_length = int.from_bytes(length_bytes, "big")
            data = b""
            while len(data) < msg_length:
                chunk = self.ssl_socket.recv(msg_length - len(data))
                if not chunk:
                    return None
                data += chunk
            response = ProtoMessage()
            response.ParseFromString(data)
            return response
        except Exception as e:
            logger.error(f"Receive failed: {e}")
            return None

    def _auth_application(self) -> bool:
        auth_req = ProtoOAApplicationAuthReq()
        auth_req.clientId = self.app_id
        auth_req.clientSecret = self.app_secret
        if not self._send_msg(2100, auth_req):
            return False
        response = self._recv_msg()
        return bool(response and response.payloadType == 2101)

    def _auth_account(self) -> bool:
        acc_auth = ProtoOAAccountAuthReq()
        acc_auth.ctidTraderAccountId = self.account_id
        acc_auth.accessToken = self.access_token
        if not self._send_msg(2102, acc_auth):
            return False
        response = self._recv_msg()
        if response and response.payloadType == 2103:
            self._authenticated = True
            return True
        return False

    def _fetch_account_info(self):
        if not self._authenticated:
            return
        trader_req = ProtoOATraderReq()
        trader_req.ctidTraderAccountId = self.account_id
        if not self._send_msg(201, trader_req):
            return
        response = self._recv_msg()
        if response and response.payloadType == 202:
            trader_res = ProtoOATraderRes()
            trader_res.ParseFromString(response.payload)
            self._account_info = AccountInfo(
                account_id=str(self.account_id),
                ctid_trader_account_id=self.account_id,
                balance=trader_res.balance / 100,
                equity=trader_res.equity / 100,
                margin=trader_res.margin / 100,
                free_margin=trader_res.freeMargin / 100,
                margin_level=trader_res.marginLevel,
                currency="USD",
                leverage=f"1:{int(trader_res.leverageInCents / 100)}"
                if trader_res.leverageInCents
                else "1:30",
                account_type="Demo" if self.demo else "Live",
                is_live=not self.demo,
            )

    def get_account_info(self) -> Optional[AccountInfo]:
        return self._account_info

    async def place_order(self, order: TradeOrder) -> Optional[TradeResult]:
        if not self._authenticated or not self._is_connected:
            return self._simulate_order(order)
        try:
            order_req = ProtoOANewOrderReq()
            order_req.ctidTraderAccountId = self.account_id
            order_req.symbolId = order.symbol_id
            order_req.orderType = 101 if order.order_type == "MARKET" else 102
            order_req.tradeSide = 1 if order.side == "BUY" else 2
            order_req.volume = order.volume
            order_req.stopLoss = int(order.sl * 100)
            order_req.takeProfit = int(order.tp * 100)
            if self._send_msg(203, order_req):
                response = self._recv_msg()
                if response and response.payloadType == 204:
                    exec_event = ProtoOAExecutionEvent()
                    exec_event.ParseFromString(response.payload)
                    return TradeResult(
                        order_id=exec_event.orderId,
                        position_id=exec_event.positionId,
                        status="FILLED",
                        filled_price=exec_event.price / 100,
                    )
            return TradeResult(status="REJECTED", error="Order failed")
        except Exception as e:
            logger.error(f"Place order error: {e}")
            return TradeResult(status="REJECTED", error=str(e))

    def _simulate_order(self, order: TradeOrder) -> TradeResult:
        return TradeResult(
            order_id=int(time.time()),
            position_id=int(time.time()) % 100000,
            status="FILLED",
            filled_price=order.price or 1.1200,
        )

    def disconnect(self):
        self._is_connected = False
        if self.ssl_socket:
            try:
                self.ssl_socket.close()
            except Exception:
                pass
            self.ssl_socket = None
        if self.tcp_socket:
            try:
                self.tcp_socket.close()
            except Exception:
                pass
            self.tcp_socket = None
        logger.info("cTrader client disconnected")

    def is_connected(self) -> bool:
        return self._is_connected
