"""
cTrader Open API Client — Async-safe SSL+Protobuf with retry, Level II DOM, and order lifecycle tracking.
"""
import ssl
import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Callable, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor

try:
    from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import ProtoMessage
    from ctrader_open_api.messages.OpenApiMessages_pb2 import (
        ProtoOAApplicationAuthReq,
        ProtoOAAccountAuthReq,
        ProtoOATraderReq,
        ProtoOATraderRes,
        ProtoOANewOrderReq,
        ProtoOAExecutionEvent,
        ProtoOASubscribeDepthQuotesReq,
        ProtoOAUnsubscribeDepthQuotesReq,
        ProtoOADepthEvent,
    )
    _HAS_PROTOBUF = True
    print("cTrader Open API protobuf loaded successfully")
except ImportError as e:
    print(f"cTrader protobuf import failed: {e}")
    _HAS_PROTOBUF = False

logger = logging.getLogger(__name__)

# Symbol mapping: symbol -> cTrader symbol_id (Open API uses numeric IDs too)
# ⚠️  VERIFIED from cTrader Symbol Info (Right-click → Symbol Info → FIX Symbol ID)
# Symbol IDs VERIFIED by user from cTrader Symbol Info
SYMBOL_MAP = {
    # Forex Majors (VERIFIED)
    "EURUSD": 1,       # VERIFIED
    "GBPUSD": 2,       # VERIFIED
    "USDJPY": 4,       # VERIFIED (was 3)
    "XAUUSD": 41,      # VERIFIED
    "BTCUSD": 114,     # VERIFIED (was 5)
    "AUDUSD": 5,       # VERIFIED (was 6)
    "USDCAD": 8,       # VERIFIED (was 7)
    "USDCHF": 6,       # VERIFIED (was 8)
    "NZDUSD": 12,      # VERIFIED (was 9)
    # Forex Minors (VERIFIED)
    "EURJPY": 3,       # VERIFIED (was 10)
    "GBPJPY": 7,       # VERIFIED (was 11)
    "EURGBP": 9,       # VERIFIED (was 12)
    # Commodities (Metals & Energy) - VERIFIED
    "XAGUSD": 42,      # VERIFIED (was 13)
    "XTIUSD": 99,      # VERIFIED (was 14)
    "XBRUSD": 100,     # VERIFIED (was 15)
    "XNGUSD": 121,     # VERIFIED (was 16)
    # Indices - VERIFIED
    "US500": 115,      # VERIFIED (was 17)
    "US30": 125,       # VERIFIED (was 18)
    "USTEC": 108,      # VERIFIED (was 19)
    "UK100": 116,      # VERIFIED (was 20)
    "DE40": 139,       # VERIFIED (was 21)
    # Crypto - VERIFIED
    "ETHUSD": 105,     # VERIFIED (was 22)
    "LTCUSD": 112,     # VERIFIED (was 23)
    "XRPUSD": 215,     # VERIFIED (was 24)
}

REVERSE_SYMBOL_MAP = {v: k for k, v in SYMBOL_MAP.items()}

# Message type constants
PROTO_OA_DEPTH_EVENT = 2155
PROTO_OA_SUBSCRIBE_DEPTH_QUOTES_REQ = 2156
PROTO_OA_SUBSCRIBE_DEPTH_QUOTES_RES = 2157
PROTO_OA_UNSUBSCRIBE_DEPTH_QUOTES_REQ = 2158
PROTO_OA_UNSUBSCRIBE_DEPTH_QUOTES_RES = 2159


@dataclass
class DepthLevel:
    """Single price level in the order book."""
    price: float = 0.0
    size: float = 0.0  # in units


@dataclass
class MarketDepth:
    """Full Level II market depth with bid/ask stacks."""
    symbol: str = ""
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0
    volume: float = 0.0
    timestamp: float = field(default_factory=time.time)
    bids: List[DepthLevel] = field(default_factory=list)  # Sorted best first
    asks: List[DepthLevel] = field(default_factory=list)  # Sorted best first
    symbol_id: int = 0


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
    """Async-safe cTrader client with thread-pool IO, Level II DOM, and retry logic."""
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
        self.port = 5035

        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._thread_pool = ThreadPoolExecutor(max_workers=2)
        self._is_connected = False
        self._authenticated = False
        self._account_info: Optional[AccountInfo] = None
        self._last_market_depth: Dict[int, MarketDepth] = {}
        self._subscribed_depth: Dict[int, bool] = {}  # symbol_id -> subscribed

        self.on_market_data: Optional[Callable] = None
        self.on_account_update: Optional[Callable] = None
        self.on_order_update: Optional[Callable] = None
        self.on_positions_update: Optional[Callable] = None
        self.on_depth_update: Optional[Callable] = None  # Callback for DOM updates

    async def start(self) -> bool:
        try:
            if not _HAS_PROTOBUF:
                logger.warning("ctrader_open_api not installed, using simulation")
                return self._start_simulation()
            if not await self._connect():
                logger.warning("cTrader connection failed, using simulation")
                return self._start_simulation()
            if not await self._auth_application():
                logger.warning("App auth failed, using simulation")
                await self.disconnect()
                return self._start_simulation()
            if not await self._auth_account():
                logger.warning("Account auth failed, using simulation")
                await self.disconnect()
                return self._start_simulation()
            await self._fetch_account_info()
            self._is_connected = True
            # Start background listener for DOM and other events
            await self.start_background_listener()
            logger.info("cTrader client connected, authenticated, and listening")
            return True
        except Exception as e:
            logger.warning(f"cTrader start failed: {e}, using simulation")
            return self._start_simulation()

    def _start_simulation(self) -> bool:
        self._is_connected = True
        self._account_info = AccountInfo(
            account_id=str(self.account_id),
            ctid_trader_account_id=self.account_id,
            balance=100_000.0,
            equity=100_000.0,
            margin=0.0,
            free_margin=100_000.0,
            currency="USD",
            leverage="1:30",
            account_type="Demo (Simulation)",
            is_live=False,
        )
        return True

    async def _connect(self) -> bool:
        try:
            context = ssl.create_default_context()
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port, ssl=context),
                timeout=10,
            )
            logger.info("Connected via async SSL")
            return True
        except Exception as e:
            logger.warning(f"SSL connection failed: {e}")
            return False

    async def _send_msg(self, payload_type: int, payload_obj) -> bool:
        try:
            proto_msg = ProtoMessage()
            proto_msg.payloadType = payload_type
            proto_msg.payload = payload_obj.SerializeToString()
            data = proto_msg.SerializeToString()
            self._writer.write(len(data).to_bytes(4, "big") + data)
            await self._writer.drain()
            return True
        except Exception as e:
            logger.warning(f"Send failed: {e}")
            return False

    async def _recv_msg(self, timeout: int = 15):
        try:
            length_bytes = await asyncio.wait_for(
                self._reader.readexactly(4), timeout=timeout
            )
            msg_length = int.from_bytes(length_bytes, "big")
            data = await asyncio.wait_for(
                self._reader.readexactly(msg_length), timeout=timeout
            )
            response = ProtoMessage()
            response.ParseFromString(data)
            return response
        except asyncio.TimeoutError:
            logger.warning("Receive timeout")
            return None
        except Exception as e:
            logger.warning(f"Receive failed: {e}")
            return None

    async def _auth_application(self) -> bool:
        auth_req = ProtoOAApplicationAuthReq()
        auth_req.clientId = self.app_id
        auth_req.clientSecret = self.app_secret
        if not await self._send_msg(2100, auth_req):
            return False
        response = await self._recv_msg()
        return bool(response and response.payloadType == 2101)

    async def _auth_account(self) -> bool:
        # Step 1: Get account list by access token
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
            ProtoOAGetAccountListByAccessTokenReq,
            ProtoOAGetAccountListByAccessTokenRes,
        )
        req = ProtoOAGetAccountListByAccessTokenReq()
        req.accessToken = self.access_token

        if not await self._send_msg(2149, req):
            return False

        response = await self._recv_msg()
        logger.info(f"Account list response: payloadType={response.payloadType if response else 'None'}")
        if response and response.payloadType == 2150:
            res = ProtoOAGetAccountListByAccessTokenRes()
            res.ParseFromString(response.payload)

            # Find the account by traderLogin (or use first)
            matching = [a for a in res.ctidTraderAccount if a.traderLogin == self.account_id]
            if matching:
                self.account_id = matching[0].ctidTraderAccountId
                logger.info(f"Matched account: traderLogin={matching[0].traderLogin} -> ctidTraderAccountId={self.account_id}")
            elif res.ctidTraderAccount and len(res.ctidTraderAccount) > 0:
                self.account_id = res.ctidTraderAccount[0].ctidTraderAccountId
                logger.info(f"Using first available account: ctidTraderAccountId={self.account_id}")
        elif response:
            logger.warning(f"Unexpected response type: {response.payloadType}")

        # Step 2: Authenticate with account ID and access token
        acc_auth = ProtoOAAccountAuthReq()
        acc_auth.ctidTraderAccountId = self.account_id
        acc_auth.accessToken = self.access_token

        if not await self._send_msg(2102, acc_auth):
            return False

        response = await self._recv_msg()
        if response and response.payloadType == 2103:
            self._authenticated = True
            return True
        elif response and response.payloadType == 2142:
            logger.warning(f"Account auth failed: Invalid/expired access token (2142)")
            return False
        elif response:
            logger.warning(f"Account auth unexpected response: payloadType={response.payloadType}")
        return False

    async def _fetch_account_info(self):
        if not self._authenticated:
            return
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOATraderRes, ProtoOAErrorRes
        from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOATrader
        trader_req = ProtoOATraderReq()
        trader_req.ctidTraderAccountId = self.account_id
        if not await self._send_msg(2121, trader_req):
            return
        response = await self._recv_msg()
        if response and response.payloadType == 2122:
            trader_res = ProtoOATraderRes()
            trader_res.ParseFromString(response.payload)
            trader = trader_res.trader
            self._account_info = AccountInfo(
                account_id=str(trader.traderLogin),
                ctid_trader_account_id=trader.ctidTraderAccountId,
                balance=trader.balance / (10 ** trader.moneyDigits),
                equity=trader.balance / (10 ** trader.moneyDigits),
                margin=0.0,
                free_margin=0.0,
                margin_level=0.0,
                currency="USD",
                leverage=f"1:{int(trader.leverageInCents / 100)}"
                if trader.leverageInCents else "1:30",
                account_type="Demo" if self.demo else "Live",
                is_live=not self.demo,
            )
            if self.on_account_update:
                self.on_account_update(self._account_info)
        elif response and response.payloadType == 2142:
            err = ProtoOAErrorRes()
            err.ParseFromString(response.payload)
            logger.warning(f"Trader info fetch failed: {err.errorCode} - {err.description}")

    def get_account_info(self) -> Optional[AccountInfo]:
        return self._account_info

    async def place_order(self, order: TradeOrder) -> Optional[TradeResult]:
        if not self._authenticated or not self._is_connected:
            return self._simulate_order(order)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                order_req = ProtoOANewOrderReq()
                order_req.ctidTraderAccountId = self.account_id
                order_req.symbolId = order.symbol_id
                order_req.orderType = 101 if order.order_type == "MARKET" else 102
                order_req.tradeSide = 1 if order.side == "BUY" else 2
                order_req.volume = order.volume
                order_req.stopLoss = int(order.sl * 100)
                order_req.takeProfit = int(order.tp * 100)
                if await self._send_msg(203, order_req):
                    response = await self._recv_msg()
                    if response and response.payloadType == 204:
                        exec_event = ProtoOAExecutionEvent()
                        exec_event.ParseFromString(response.payload)
                        result = TradeResult(
                            order_id=exec_event.orderId,
                            position_id=exec_event.positionId,
                            status="FILLED",
                            filled_price=exec_event.price / 100,
                        )
                        if self.on_order_update:
                            self.on_order_update(result)
                        return result
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.warning(f"Order attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        return TradeResult(status="REJECTED", error="Order failed after retries")

    def _simulate_order(self, order: TradeOrder) -> TradeResult:
        result = TradeResult(
            order_id=int(time.time()),
            position_id=int(time.time()) % 100000,
            status="FILLED",
            filled_price=order.price or 1.1200,
        )
        if self.on_order_update:
            self.on_order_update(result)
        return result

    async def disconnect(self):
        self._is_connected = False
        # Unsubscribe from all depth subscriptions
        for sym_id in list(self._subscribed_depth.keys()):
            await self.unsubscribe_depth(sym_id)
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
            self._writer = None
            self._reader = None
        logger.info("cTrader client disconnected")

    def is_connected(self) -> bool:
        return self._is_connected

    async def start_background_listener(self):
        """Start background listener for async events (Depth, Spot, etc.)."""
        asyncio.create_task(self._background_listener())

    async def _background_listener(self):
        """Continuously read messages and dispatch to handlers."""
        while self._is_connected and self._reader:
            try:
                length_bytes = await asyncio.wait_for(
                    self._reader.readexactly(4), timeout=30
                )
                msg_length = int.from_bytes(length_bytes, "big")
                data = await asyncio.wait_for(
                    self._reader.readexactly(msg_length), timeout=30
                )
                response = ProtoMessage()
                response.ParseFromString(data)
                await self._handle_message(response)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if self._is_connected:
                    logger.debug(f"Background listener error: {e}")
                break

    async def _handle_message(self, msg):
        """Dispatch incoming messages to appropriate handlers."""
        if msg.payloadType == PROTO_OA_DEPTH_EVENT:
            await self._handle_depth_event(msg)
        elif msg.payloadType == 2103:  # ProtoOAAccountAuthRes
            logger.debug("Received account auth response")
        elif msg.payloadType == 2101:  # ProtoOAApplicationAuthRes
            logger.debug("Received app auth response")
        else:
            logger.debug(f"Unhandled message type: {msg.payloadType}")

    async def _handle_depth_event(self, msg):
        """Process Level II DOM update from ProtoOADepthEvent."""
        try:
            depth_event = ProtoOADepthEvent()
            depth_event.ParseFromString(msg.payload)
            symbol_id = depth_event.symbolId
            symbol = REVERSE_SYMBOL_MAP.get(symbol_id, str(symbol_id))

            # Initialize or update market depth
            if symbol_id not in self._last_market_depth:
                self._last_market_depth[symbol_id] = MarketDepth(
                    symbol=symbol, symbol_id=symbol_id
                )
            md = self._last_market_depth[symbol_id]
            md.timestamp = time.time()

            # Process new/updated quotes
            bids_dict = {b.price: b for b in md.bids}
            asks_dict = {a.price: a for a in md.asks}

            for quote in depth_event.newQuotes:
                price = quote.bid / 100000.0 if quote.HasField("bid") else quote.ask / 100000.0
                size = quote.size / 100.0  # Convert cents to units
                level = DepthLevel(price=price, size=size)
                if quote.HasField("bid"):
                    bids_dict[price] = level
                elif quote.HasField("ask"):
                    asks_dict[price] = level

            # Process deleted quotes
            deleted_prices = set()
            for quote_id in depth_event.deletedQuotes:
                deleted_prices.add(quote_id)
            bids_dict = {p: l for p, l in bids_dict.items() if p not in deleted_prices}
            asks_dict = {p: l for p, l in asks_dict.items() if p not in deleted_prices}

            # Sort: bids descending (best bid first), asks ascending (best ask first)
            md.bids = sorted(bids_dict.values(), key=lambda x: x.price, reverse=True)[:20]
            md.asks = sorted(asks_dict.values(), key=lambda x: x.price)[:20]

            # Update best bid/ask
            md.bid = md.bids[0].price if md.bids else 0.0
            md.ask = md.asks[0].price if md.asks else 0.0
            md.spread = (md.ask - md.bid) if (md.ask and md.bid) else 0.0
            md.volume = sum(b.size for b in md.bids) + sum(a.size for a in md.asks)

            # Notify callback
            if self.on_depth_update:
                self.on_depth_update(md)

            logger.debug(f"DOM update {symbol}: {len(md.bids)} bids, {len(md.asks)} asks")
        except Exception as e:
            logger.warning(f"Failed to parse depth event: {e}")

    async def subscribe_depth(self, symbol_id: int) -> bool:
        """Subscribe to Level II DOM for a symbol."""
        if not _HAS_PROTOBUF or not self._authenticated:
            return False
        if symbol_id in self._subscribed_depth:
            return True
        try:
            req = ProtoOASubscribeDepthQuotesReq()
            req.ctidTraderAccountId = self.account_id
            req.symbolId.append(symbol_id)
            if await self._send_msg(PROTO_OA_SUBSCRIBE_DEPTH_QUOTES_REQ, req):
                self._subscribed_depth[symbol_id] = True
                logger.info(f"Subscribed to Level II DOM for {REVERSE_SYMBOL_MAP.get(symbol_id, symbol_id)}")
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to subscribe to depth: {e}")
            return False

    async def unsubscribe_depth(self, symbol_id: int) -> bool:
        """Unsubscribe from Level II DOM for a symbol."""
        if not _HAS_PROTOBUF or symbol_id not in self._subscribed_depth:
            return False
        try:
            req = ProtoOAUnsubscribeDepthQuotesReq()
            req.ctidTraderAccountId = self.account_id
            req.symbolId.append(symbol_id)
            if await self._send_msg(PROTO_OA_UNSUBSCRIBE_DEPTH_QUOTES_REQ, req):
                del self._subscribed_depth[symbol_id]
                logger.info(f"Unsubscribed from Level II DOM for {REVERSE_SYMBOL_MAP.get(symbol_id, symbol_id)}")
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to unsubscribe from depth: {e}")
            return False

    def get_market_depth(self, symbol: str) -> Optional[MarketDepth]:
        """Get latest Level II market depth for a symbol."""
        symbol_id = SYMBOL_MAP.get(symbol.upper())
        if symbol_id and symbol_id in self._last_market_depth:
            return self._last_market_depth[symbol_id]
        return None
