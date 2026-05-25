"""
cTrader Open API Client — Async-safe SSL+Protobuf with retry, Level II DOM, and order lifecycle tracking.  # noqa: E501
"""

import ssl
import socket
import asyncio
import time
import functools
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Callable, Any, List
from concurrent.futures import ThreadPoolExecutor

try:
    from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import (
        ProtoMessage,
        ProtoHeartbeatEvent,
    )
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
        ProtoOAGetTrendbarsReq,
        ProtoOAGetTrendbarsRes,
        # Additional protobuf message types for richer account + market data
        ProtoOAAssetListReq,
        ProtoOAAssetListRes,
        ProtoOAMarginChangedEvent,
        ProtoOATraderUpdatedEvent,
        ProtoOAOrderErrorEvent,
        ProtoOASpotEvent,
        ProtoOASubscribeSpotsReq,
        ProtoOAUnsubscribeSpotsReq,
        ProtoOAReconcileReq,
        ProtoOAReconcileRes,
        ProtoOAGetPositionUnrealizedPnLReq,
        ProtoOAGetPositionUnrealizedPnLRes,
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
    "EURUSD": 1,
    "GBPUSD": 2,
    "USDJPY": 4,
    "XAUUSD": 41,
    "BTCUSD": 114,
    "AUDUSD": 5,
    "USDCAD": 8,
    "USDCHF": 6,
    "NZDUSD": 12,
    "XTIUSD": 99,
    "US500": 115,
}

REVERSE_SYMBOL_MAP = {v: k for k, v in SYMBOL_MAP.items()}

# Message type constants — use the ProtoOAPayloadType enum from the installed
# protobuf package.  DO NOT hardcode values here; they differ between
# ctrader-open-api package versions.
try:
    from ctrader_open_api.messages.OpenApiModelMessages_pb2 import (
        ProtoOAPayloadType as _PT,
    )

    PROTO_OA_DEPTH_EVENT = _PT.Value("PROTO_OA_DEPTH_EVENT")
    PROTO_OA_SUBSCRIBE_DEPTH_QUOTES_REQ = _PT.Value(
        "PROTO_OA_SUBSCRIBE_DEPTH_QUOTES_REQ"
    )
    PROTO_OA_SUBSCRIBE_DEPTH_QUOTES_RES = _PT.Value(
        "PROTO_OA_SUBSCRIBE_DEPTH_QUOTES_RES"
    )
    PROTO_OA_UNSUBSCRIBE_DEPTH_QUOTES_REQ = _PT.Value(
        "PROTO_OA_UNSUBSCRIBE_DEPTH_QUOTES_REQ"
    )
    PROTO_OA_UNSUBSCRIBE_DEPTH_QUOTES_RES = _PT.Value(
        "PROTO_OA_UNSUBSCRIBE_DEPTH_QUOTES_RES"
    )
    PROTO_OA_MARGIN_CHANGED_EVENT = _PT.Value("PROTO_OA_MARGIN_CHANGED_EVENT")
    PROTO_OA_SPOT_EVENT = _PT.Value("PROTO_OA_SPOT_EVENT")
    PROTO_OA_SUBSCRIBE_SPOTS_REQ = _PT.Value("PROTO_OA_SUBSCRIBE_SPOTS_REQ")
    PROTO_OA_SUBSCRIBE_SPOTS_RES = _PT.Value("PROTO_OA_SUBSCRIBE_SPOTS_RES")
    PROTO_OA_UNSUBSCRIBE_SPOTS_REQ = _PT.Value("PROTO_OA_UNSUBSCRIBE_SPOTS_REQ")
    PROTO_OA_UNSUBSCRIBE_SPOTS_RES = _PT.Value("PROTO_OA_UNSUBSCRIBE_SPOTS_RES")
    PROTO_OA_TRADER_UPDATED_EVENT = _PT.Value("PROTO_OA_TRADER_UPDATE_EVENT")
    PROTO_OA_ORDER_ERROR_EVENT = _PT.Value("PROTO_OA_ORDER_ERROR_EVENT")
    PROTO_OA_ASSET_LIST_REQ = _PT.Value("PROTO_OA_ASSET_LIST_REQ")
    PROTO_OA_ASSET_LIST_RES = _PT.Value("PROTO_OA_ASSET_LIST_RES")
    PROTO_OA_RECONCILE_REQ = _PT.Value("PROTO_OA_RECONCILE_REQ")
    PROTO_OA_RECONCILE_RES = _PT.Value("PROTO_OA_RECONCILE_RES")
    PROTO_OA_GET_POSITION_UNREALIZED_PNL_REQ = _PT.Value(
        "PROTO_OA_GET_POSITION_UNREALIZED_PNL_REQ"
    )
    PROTO_OA_GET_POSITION_UNREALIZED_PNL_RES = _PT.Value(
        "PROTO_OA_GET_POSITION_UNREALIZED_PNL_RES"
    )
    PROTO_OA_NEW_ORDER_REQ = _PT.Value("PROTO_OA_NEW_ORDER_REQ")
    PROTO_OA_EXECUTION_EVENT = _PT.Value("PROTO_OA_EXECUTION_EVENT")
    PROTO_OA_ACCOUNT_AUTH_RES = _PT.Value("PROTO_OA_ACCOUNT_AUTH_RES")
    PROTO_OA_APPLICATION_AUTH_RES = _PT.Value("PROTO_OA_APPLICATION_AUTH_RES")
except (ImportError, AttributeError) as _e:
    logger.warning(f"Failed to load ProtoOAPayloadType enum: {_e}")
    # Fallback defaults (may be wrong for some server versions)
    PROTO_OA_DEPTH_EVENT = 2155
    PROTO_OA_SUBSCRIBE_DEPTH_QUOTES_REQ = 2156
    PROTO_OA_SUBSCRIBE_DEPTH_QUOTES_RES = 2157
    PROTO_OA_UNSUBSCRIBE_DEPTH_QUOTES_REQ = 2158
    PROTO_OA_UNSUBSCRIBE_DEPTH_QUOTES_RES = 2159
    PROTO_OA_MARGIN_CHANGED_EVENT = 2141
    PROTO_OA_SPOT_EVENT = 2131
    PROTO_OA_SUBSCRIBE_SPOTS_REQ = 2127
    PROTO_OA_SUBSCRIBE_SPOTS_RES = 2128
    PROTO_OA_UNSUBSCRIBE_SPOTS_REQ = 2129
    PROTO_OA_UNSUBSCRIBE_SPOTS_RES = 2130
    PROTO_OA_TRADER_UPDATED_EVENT = 2123
    PROTO_OA_ORDER_ERROR_EVENT = 2132
    PROTO_OA_ASSET_LIST_REQ = 2112
    PROTO_OA_ASSET_LIST_RES = 2113
    PROTO_OA_RECONCILE_REQ = 2124
    PROTO_OA_RECONCILE_RES = 2125
    PROTO_OA_GET_POSITION_UNREALIZED_PNL_REQ = 2187
    PROTO_OA_GET_POSITION_UNREALIZED_PNL_RES = 2188
    PROTO_OA_NEW_ORDER_REQ = 2106
    PROTO_OA_EXECUTION_EVENT = 2126
    PROTO_OA_ACCOUNT_AUTH_RES = 2103
    PROTO_OA_APPLICATION_AUTH_RES = 2101


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
        self._trader_login = account_id  # Preserve original login for re-auth matching
        self.demo = demo
        self.use_websocket = use_websocket
        self.host = "demo.ctraderapi.com" if demo else "live.ctraderapi.com"
        self.port = 5035

        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._thread_pool = ThreadPoolExecutor(max_workers=2)
        self._is_connected = False
        self._authenticated = False
        self._simulation = False  # True when running in simulation fallback mode
        self._account_info: Optional[AccountInfo] = None
        self._last_market_depth: Dict[int, MarketDepth] = {}
        self._subscribed_depth: Dict[int, bool] = {}  # symbol_id -> subscribed

        # Pending request-response futures keyed by payloadType (int).
        # All requests are sequential (each await-s its response before next),
        # so no key collision occurs between requests of different types.
        # subscribe_depth uses a compound key (int, int) via tuple to allow
        # multiple sequential depth subscriptions without collision.
        self._pending_responses: Dict[Any, asyncio.Future] = {}
        self._next_id = 0
        self._start_lock: Optional[asyncio.Lock] = None  # Created lazily in start()

        # Heartbeat health tracking
        self._heartbeat_ok = 0
        self._heartbeat_fail = 0
        self._last_heartbeat_ts = 0.0

        self.on_market_data: Optional[Callable] = None
        self.on_account_update: Optional[Callable] = None
        self.on_order_update: Optional[Callable] = None
        self.on_positions_update: Optional[Callable] = None
        self.on_depth_update: Optional[Callable] = None  # Callback for DOM updates
        self.on_disconnect: Optional[Callable] = None  # Called when connection drops
        self._listener_task: Optional[asyncio.Task] = None
        self._listener_phase: int = 0  # Atomic phase counter for listener lifecycle
        self._heartbeat_task: Optional[asyncio.Task] = None

        # Asset ID → currency name mapping (populated by _fetch_asset_list)
        self._asset_id_to_currency: Dict[int, str] = {}
        # Spot price tracking (authoritative price source)
        self._subscribed_spots: Dict[int, bool] = {}
        self._last_spot_prices: Dict[int, Dict[str, float]] = {}
        # Position cache (populated by reconcile)
        self._cached_positions: List[Dict] = []

    async def start(self) -> bool:
        """Connect and authenticate to cTrader.

        Safe to call in ANY state:
        - Already connected → returns True immediately
        - Partially connected → full disconnect then reconnect
        - Disconnected → normal connect flow

        Returns True if fully connected and listening.
        """
        # Serialize concurrent start() calls to prevent races.
        # The background _connect_and_subscribe() task from _on_start()
        # can race with AGENT_DIRECTIVE-triggered _reconnect() calls,
        # both trying to auth against the same StreamReader.
        if self._start_lock is None:
            self._start_lock = asyncio.Lock()
        async with self._start_lock:
            return await self._start_impl()

    async def _start_impl(self) -> bool:
        """Internal start implementation (serialized by _start_lock)."""
        # Idempotent: if already connected and listening, skip.
        if self._is_connected and self._listener_task and not self._listener_task.done():
            return True

        try:
            if not _HAS_PROTOBUF:
                logger.warning("ctrader_open_api not installed, no trading possible")
                return False

            # Always fully disconnect before connecting — ensures no stale
            # reader, writer, or listener tasks remain to race against the
            # new connection.  Without this, old listeners can leave zombie
            # _waiters on StreamReaders that crash new listeners with:
            #   "readexactly() called while another coroutine is
            #    already waiting for incoming data"
            await self._full_disconnect()

            if not await self._connect():
                logger.warning("cTrader SSL connection failed")
                return False
            if not await self._auth_application():
                logger.warning("cTrader application auth failed")
                await self._full_disconnect()
                return False
            if not await self._auth_account():
                logger.warning("cTrader account auth failed — attempting token refresh")
                await self._full_disconnect()
                # Token might be expired — refresh and retry once
                token_refreshed = await self._refresh_access_token()
                if token_refreshed:
                    if not await self._connect():
                        return False
                    if not await self._auth_application():
                        await self._full_disconnect()
                        return False
                    if not await self._auth_account():
                        logger.warning("Account auth still failed after token refresh")
                        await self._full_disconnect()
                        return False
                else:
                    logger.warning("Token refresh failed — will retry on next cycle")
                    return False

            # Start background listener BEFORE fetching account info.
            # This is critical: after account auth, the server may push
            # unsolicited events (margin changed, spot prices, etc.).
            # If we use _recv_msg synchronously, it may consume these
            # events instead of the expected response.  By starting the
            # listener first, all messages go through _handle_message
            # which routes events to handlers and resolves futures.
            self._is_connected = True
            # Start background listener for DOM and other events.
            # Phase lock: incrementing the counter causes any old listener
            # (still running after cancellation) to exit on its next loop
            # iteration, BEFORE it tries to read from the StreamReader.
            # This eliminates the race where two listeners read concurrently.
            self._listener_phase += 1
            current_phase = self._listener_phase
            # At this point _full_disconnect() already cleaned up any old
            # listener, but we keep the guard for defense-in-depth.
            if self._listener_task and not self._listener_task.done():
                self._listener_task.cancel()
                try:
                    await self._listener_task
                except (asyncio.CancelledError, Exception):
                    pass
                self._listener_task = None
                # CRITICAL: yield so StreamReader's _waiter fully cleans up
                await asyncio.sleep(0)
            self._listener_task = asyncio.create_task(
                self._background_listener(phase=current_phase)
            )
            # Start heartbeat sender (required by cTrader — every 10s)
            if self._heartbeat_task and not self._heartbeat_task.done():
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except (asyncio.CancelledError, Exception):
                    pass
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            # Now fetch account info via future-based pattern (listener resolves it)
            await self._fetch_account_info()

            # Asset list / currency mapping: best-effort, may not be supported
            await self._fetch_asset_list()

            logger.info("cTrader client connected, authenticated, and listening")
            return True
        except Exception as e:
            logger.warning(f"cTrader start failed: {e}")
            return False

    async def _full_disconnect(self):
        """Complete teardown: stop listeners, close socket, reset state.
        
        Unlike disconnect(), this is guaranteed to fully clean up regardless
        of current connection state.  Safe to call multiple times.
        """
        self._is_connected = False
        self._listener_phase += 1
        self._authenticated = False
        self._subscribed_depth.clear()
        self._subscribed_spots.clear()
        self._account_info = None
        self._pending_responses.clear()
        
        # Cancel listener and heartbeat tasks
        for task_name in ("_listener_task", "_heartbeat_task"):
            task = getattr(self, task_name, None)
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
                setattr(self, task_name, None)
        
        # Yield to let StreamReader clean up stale _waiter before close
        await asyncio.sleep(0)
        
        # Close the socket
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
        self._writer = None
        self._reader = None
        
        # Reset heartbeat health
        self._heartbeat_ok = 0
        self._heartbeat_fail = 0
        self._last_heartbeat_ts = 0.0
        
        logger.debug("cTrader client fully disconnected")

    def _start_simulation(self) -> bool:
        self._simulation = True
        self._is_connected = False  # NOT connected — no background listener runs
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
        """Connect to cTrader server via SSL in a thread pool.

        Running the SSL handshake in a thread pool completely isolates it
        from asyncio event-loop congestion. This avoids ConnectionResetError
        caused by delayed SSL handshakes when 21 agents share the loop.
        """
        try:
            loop = asyncio.get_running_loop()

            def _do_connect():
                context = ssl.create_default_context()
                context.minimum_version = ssl.TLSVersion.TLSv1_2
                sock = socket.create_connection((self.host, self.port), timeout=60)
                ssock = context.wrap_socket(sock, server_hostname=self.host)
                ssock.setblocking(False)
                return ssock

            ssock = await loop.run_in_executor(None, _do_connect)
            self._reader, self._writer = await asyncio.open_connection(
                host=None, port=None, ssl=None, sock=ssock
            )
            logger.info("Connected via thread-pool SSL")
            return True
        except Exception as e:
            logger.warning(f"SSL connection failed: {type(e).__name__}: {e}")
            return False

    async def _send_msg(
        self,
        payload_type: int,
        payload_obj,
        client_msg_id: Optional[str] = None,
    ):
        """Serialize and send a protobuf message over the SSL stream.

        Args:
            payload_type: The ProtoOAPayloadType value for the message.
            payload_obj: The protobuf message instance to serialize.
            client_msg_id: Optional unique request ID for response matching.
                When provided, the ProtoMessage.clientMsgId field is set,
                and the function returns the ID string on success.
                When omitted, returns True on success (backward compatible).

        Returns:
            client_msg_id string if provided (for future-based callers),
            True if no client_msg_id (legacy callers),
            None on failure.
        """
        if self._writer is None:
            return None
        try:
            proto_msg = ProtoMessage()
            proto_msg.payloadType = payload_type
            proto_msg.payload = payload_obj.SerializeToString()
            if client_msg_id is not None:
                proto_msg.clientMsgId = client_msg_id
            data = proto_msg.SerializeToString()
            self._writer.write(len(data).to_bytes(4, "big") + data)
            await self._writer.drain()
            return client_msg_id if client_msg_id is not None else True
        except Exception as e:
            logger.debug(f"Send failed: {e}")
            return None

    async def _recv_msg(self, timeout: int = 15):
        if self._reader is None:
            return None
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
        sent = await self._send_msg(2100, auth_req)
        if not sent:
            logger.warning("App auth: send failed")
            return False
        response = await self._recv_msg()
        if response is None:
            logger.warning("App auth: no response")
            return False
        if response.payloadType != 2101:
            logger.warning(f"App auth: unexpected response type {response.payloadType}")
            return False
        return True

    async def _refresh_access_token(self) -> bool:
        """Refresh the access token using the refresh token from environment."""
        try:
            import requests as http_requests

            # Load refresh token from env — search common locations
            import os, re

            env_path = None
            for candidate in [
                os.path.join(os.getcwd(), ".env"),
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"
                ),
                os.path.expanduser("~/.env"),
            ]:
                candidate = os.path.normpath(candidate)
                if os.path.exists(candidate):
                    env_path = candidate
                    break

            if env_path is None:
                logger.warning("Cannot refresh token: .env not found")
                return False

            with open(env_path) as f:
                env = f.read()

            match = re.search(r"CTRADER_REFRESH_TOKEN=(\S+)", env)
            if not match:
                logger.warning("Cannot refresh token: no refresh token in .env")
                return False

            refresh_token = match.group(1)
            loop = asyncio.get_running_loop()
            resp = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    functools.partial(
                        http_requests.post,
                        "https://openapi.ctrader.com/apps/token",
                        data={
                            "grant_type": "refresh_token",
                            "refresh_token": refresh_token,
                            "client_id": self.app_id,
                            "client_secret": self.app_secret,
                        },
                        timeout=10,
                    ),
                ),
                timeout=15,
            )

            if resp.status_code == 429:
                logger.warning(
                    f"Token refresh rate-limited (429). "
                    f"Will retry on next reconnect attempt."
                )
                return False
            if resp.status_code != 200:
                logger.warning(f"Token refresh failed: {resp.status_code}")
                return False

            data = resp.json()
            new_token = data.get("accessToken") or data.get("access_token", "")
            new_refresh = data.get("refreshToken") or data.get("refresh_token", "")

            if not new_token or len(new_token) < 10:
                logger.warning("Token refresh returned empty token")
                return False

            # Update in-memory token
            self.access_token = new_token

            # Persist to .env
            env = re.sub(
                r"CTRADER_ACCESS_TOKEN=.*",
                f"CTRADER_ACCESS_TOKEN={new_token}",
                env,
            )
            if new_refresh:
                env = re.sub(
                    r"CTRADER_REFRESH_TOKEN=.*",
                    f"CTRADER_REFRESH_TOKEN={new_refresh}",
                    env,
                )
            with open(env_path, "w") as f:
                f.write(env)

            logger.info("cTrader access token refreshed successfully")
            return True
        except Exception as e:
            logger.warning(f"Token refresh error: {e}")
            return False

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
        logger.info(
            f"Account list response: payloadType={response.payloadType if response else 'None'}"  # noqa: E501
        )
        if response and response.payloadType == 2150:
            res = ProtoOAGetAccountListByAccessTokenRes()
            res.ParseFromString(response.payload)

            # Find the account by traderLogin (the original login number, not
            # the ctidTraderAccountId which was set during a previous auth).
            matching = [
                a for a in res.ctidTraderAccount if a.traderLogin == self._trader_login
            ]
            if matching:
                self.account_id = matching[0].ctidTraderAccountId
                logger.info(
                    f"Matched account: traderLogin={matching[0].traderLogin} -> ctidTraderAccountId={self.account_id}"  # noqa: E501
                )
            elif res.ctidTraderAccount and len(res.ctidTraderAccount) > 0:
                self.account_id = res.ctidTraderAccount[0].ctidTraderAccountId
                logger.info(
                    f"Using first available account: ctidTraderAccountId={self.account_id}"  # noqa: E501
                )
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
            # Try refreshing the token and retry once
            logger.info("Attempting token refresh...")
            if await self._refresh_access_token():
                logger.info("Token refreshed, retrying account auth...")
                acc_auth.accessToken = self.access_token
                if not await self._send_msg(2102, acc_auth):
                    return False
                response2 = await self._recv_msg()
                if response2 and response2.payloadType == 2103:
                    self._authenticated = True
                    return True
                logger.warning("Account auth still failed after token refresh")
            return False
        elif response:
            logger.warning(
                f"Account auth unexpected response: payloadType={response.payloadType}"
            )
        return False

    async def _fetch_account_info(self):
        if not self._authenticated:
            return
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
            ProtoOATraderRes,
            ProtoOAErrorRes,
        )

        trader_req = ProtoOATraderReq()
        trader_req.ctidTraderAccountId = self.account_id

        # Use future-based pattern (listener is already running).
        # _recv_msg would race with the listener for data — the listener
        # might consume unsolicited events (margin, spot) that the server
        # pushes after auth, leaving _recv_msg with the wrong response.
        fut = asyncio.get_event_loop().create_future()
        self._pending_responses[2122] = fut

        if not await self._send_msg(2121, trader_req):
            self._pending_responses.pop(2122, None)
            return

        try:
            response = await asyncio.wait_for(fut, timeout=10.0)
        except asyncio.TimeoutError:
            self._pending_responses.pop(2122, None)
            logger.warning("Trader info request timed out")
            return

        if response and response.payloadType == 2122:
            trader_res = ProtoOATraderRes()
            trader_res.ParseFromString(response.payload)
            trader = trader_res.trader
            # NOTE: ProtoOATrader (cTrader Open API v0.9.2) does NOT expose
            # equity, margin, freeMargin, or marginLevel fields.
            # Only balance, leverageInCents, and depositAssetId are available.
            # - equity is compensated by ExecutionEngine (balance + unrealised PnL)
            # - margin comes from ProtoOAMarginChangedEvent push events
            # - currency comes from _asset_id_to_currency (populated by _fetch_asset_list)  # noqa: E501
            money_div = 10**trader.moneyDigits
            dep_id = trader.depositAssetId if hasattr(trader, "depositAssetId") else 0
            currency = self._asset_id_to_currency.get(dep_id, "USD")
            self._account_info = AccountInfo(
                account_id=str(trader.traderLogin),
                ctid_trader_account_id=trader.ctidTraderAccountId,
                balance=trader.balance / money_div,
                equity=trader.balance / money_div,  # placeholder — see engine.py
                margin=0.0,  # populated by _handle_margin_changed_event
                free_margin=0.0,  # recalculated by _handle_margin_changed_event
                margin_level=0.0,  # recalculated by _handle_margin_changed_event
                currency=currency,
                leverage=(
                    f"1:{int(trader.leverageInCents / 100)}"
                    if trader.leverageInCents
                    else "1:30"
                ),
                account_type="Demo" if self.demo else "Live",
                is_live=not self.demo,
            )
            if self.on_account_update:
                self.on_account_update(self._account_info)
        elif response and response.payloadType == 2142:
            err = ProtoOAErrorRes()
            err.ParseFromString(response.payload)
            logger.warning(
                f"Trader info fetch failed: {err.errorCode} - {err.description}"
            )

    async def _fetch_asset_list(self):
        """Fetch asset list from broker to map depositAssetId → currency name.

        Populates self._asset_id_to_currency so _fetch_account_info can
        resolve the deposit currency instead of hardcoding "USD".
        """
        if not self._authenticated:
            return
        req = ProtoOAAssetListReq()
        req.ctidTraderAccountId = self.account_id

        fut = asyncio.get_event_loop().create_future()
        self._pending_responses[PROTO_OA_ASSET_LIST_RES] = fut

        if not await self._send_msg(PROTO_OA_ASSET_LIST_REQ, req):
            self._pending_responses.pop(PROTO_OA_ASSET_LIST_RES, None)
            return

        try:
            response = await asyncio.wait_for(fut, timeout=5.0)
        except asyncio.TimeoutError:
            self._pending_responses.pop(PROTO_OA_ASSET_LIST_RES, None)
            logger.debug("Asset list request timed out (non-critical)")
            return

        if response and response.payloadType == PROTO_OA_ASSET_LIST_RES:
            res = ProtoOAAssetListRes()
            res.ParseFromString(response.payload)
            for asset in res.asset:
                self._asset_id_to_currency[asset.assetId] = asset.name
            logger.info(
                f"Loaded {len(self._asset_id_to_currency)} assets: "
                f"{dict(list(self._asset_id_to_currency.items())[:5])}..."
            )

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
                if await self._send_msg(PROTO_OA_NEW_ORDER_REQ, order_req):
                    response = await self._recv_msg()
                    if response and response.payloadType == PROTO_OA_EXECUTION_EVENT:
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
                    await asyncio.sleep(2**attempt)
            except Exception as e:
                logger.warning(f"Order attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)
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

    async def modify_position(
        self, position_id: int, sl: Optional[float] = None, tp: Optional[float] = None
    ) -> bool:
        """Modify stop loss and/or take profit on an existing position.

        Args:
            position_id: cTrader position ID
            sl: New stop loss price (absolute, e.g. 1.1150). None = leave unchanged.
            tp: New take profit price (absolute). None = leave unchanged.

        Returns:
            True if modification request was sent successfully.
        """
        if not self._authenticated or not self._is_connected:
            return False
        try:
            from ctrader_open_api.messages.OpenApiMessages_pb2 import (
                ProtoOAAmendPositionSLTPReq,
            )

            req = ProtoOAAmendPositionSLTPReq()
            req.ctidTraderAccountId = self.account_id
            req.positionId = position_id
            if sl is not None:
                req.stopLoss = int(sl * 100)
            if tp is not None:
                req.takeProfit = int(tp * 100)
            if await self._send_msg(2110, req):
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to modify position {position_id}: {e}")
            return False

    async def disconnect(self):
        """Disconnect from cTrader and clean up.
        
        Delegates to _full_disconnect() for consistent teardown.
        Safe to call multiple times.
        """
        await self._full_disconnect()
        logger.info("cTrader client disconnected")

    def is_connected(self) -> bool:
        return self._is_connected and not self._simulation

    def is_simulation(self) -> bool:
        """True when running in simulation fallback (no real broker connection)."""
        return self._simulation

    async def _fetch_trendbar_batch(
        self,
        symbol_id: int,
        period: int,
        from_ms: int,
        to_ms: int,
        count: int = 10000,
    ) -> Optional[List]:
        """Single trendbar request with future-based response.

        IMPORTANT: The future is registered in _pending_responses BEFORE
        _send_msg to eliminate a race condition with _background_listener.
        If the response arrives before we await the future, _handle_message
        will find and resolve it.  Cleanup is performed on send failure or
        timeout to prevent future leaks.

        count: Max bars per request. cTrader supports up to 10000,
        but may return fewer based on server limits.
        """
        req = ProtoOAGetTrendbarsReq()
        req.ctidTraderAccountId = self.account_id
        req.symbolId = symbol_id
        req.fromTimestamp = from_ms
        req.toTimestamp = to_ms
        req.period = period
        req.count = count

        # Register future BEFORE send to win the race with background listener
        fut = asyncio.get_event_loop().create_future()
        self._pending_responses[2138] = fut

        if not await self._send_msg(2137, req):
            self._pending_responses.pop(2138, None)
            return None

        try:
            response = await asyncio.wait_for(fut, timeout=60.0)
        except asyncio.TimeoutError:
            self._pending_responses.pop(2138, None)
            logger.warning("cTrader trendbar request timed out")
            return None

        if response is None or response.payloadType != 2138:
            return None
        res = ProtoOAGetTrendbarsRes()
        res.ParseFromString(response.payload)
        return list(res.trendbar)

    async def fetch_historical_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        days_back: int = 365,
        max_bars: int = 10000,
    ) -> Optional[List[Dict]]:
        """Fetch historical OHLCV via cTrader protobuf, auto-batching for full range."""
        if not _HAS_PROTOBUF or not self._authenticated or not self._is_connected:
            return None

        sym = symbol.upper()
        symbol_id = SYMBOL_MAP.get(sym)
        if symbol_id is None:
            return None

        tf_map = {
            "1m": 1,
            "5m": 5,
            "15m": 7,
            "30m": 8,
            "1h": 9,
            "4h": 10,
            "1d": 12,
            "1w": 13,
        }
        period = tf_map.get(timeframe)
        if period is None:
            return None

        from datetime import datetime, timezone, timedelta

        now = datetime.now(timezone.utc)

        batch_window_hours = {9: 4000, 10: 16000, 12: 60000}.get(period, 4000)
        window_secs = batch_window_hours * 3600
        all_bars: Dict[int, Any] = {}
        batch_to = int(now.timestamp() * 1000)
        batch_from = int((now.timestamp() - window_secs) * 1000)
        earliest = int((now - timedelta(days=days_back)).timestamp() * 1000)

        while batch_to > earliest:
            try:
                bars = await self._fetch_trendbar_batch(
                    symbol_id,
                    period,
                    max(batch_from, earliest),
                    batch_to,
                )
                if not bars:
                    break
                for tb in bars:
                    all_bars[tb.utcTimestampInMinutes] = tb
                oldest_min = bars[0].utcTimestampInMinutes
                batch_to = (oldest_min - 1) * 60 * 1000
                batch_from = batch_to - int(window_secs * 1000)
            except asyncio.TimeoutError:
                logger.debug(
                    f"cTrader batch timeout for {sym}, using {len(all_bars)} bars so far"  # noqa: E501
                )
                break
            except Exception as e:
                logger.debug(f"cTrader batch error for {sym}: {e}")
                break

        if not all_bars:
            return None

        # Sort and convert
        # Per cTrader docs: all trendbar prices are in 1/100,000 format
        sorted_minutes = sorted(all_bars.keys())
        divisor = 100000.0

        result = []
        for ts_min in sorted_minutes:
            tb = all_bars[ts_min]
            low_f = tb.low / divisor
            result.append(
                {
                    "timestamp": ts_min * 60,
                    "open": round(low_f + tb.deltaOpen / divisor, 5),
                    "high": round(low_f + tb.deltaHigh / divisor, 5),
                    "low": round(low_f, 5),
                    "close": round(low_f + tb.deltaClose / divisor, 5),
                    "volume": tb.volume,
                }
            )
        logger.info(
            f"cTrader proto: {len(result)} {timeframe} bars for {sym} ({days_back}d)"
        )
        return result

    async def start_background_listener(self):
        """Start background listener for async events (Depth, Spot, etc.)."""
        self._listener_phase += 1
        current_phase = self._listener_phase
        # Yield to let any old listener's stale _waiter clean up
        await asyncio.sleep(0)
        asyncio.create_task(self._background_listener(phase=current_phase))

    async def _background_listener(self, phase: int):
        """Continuously read messages and dispatch to handlers.

        On connection loss, sets _is_connected = False and fires the
        on_disconnect callback (with safety guard to prevent double-fire).
        The existing reconnect infrastructure (ConnectionAgent →
        ExecutionAgent._reconnect) handles full reconnection + resubscription.

        Phase-based lifecycle (CRITICAL):
          A ``phase`` integer is received at creation time.  Before each
          iteration the method checks whether ``self._listener_phase`` still
          matches — if a newer listener has been started (phase was
          incremented), this task exits cleanly WITHOUT touching the
          StreamReader.  This eliminates the race where a cancelled old
          listener's ``readexactly()`` resurfaces after a new listener
          already started reading from the same StreamReader.

          The ``was_cancelled`` flag tracks ONLY ``CancelledError`` (explicit
          task cancellation), not stale-phase exit — both suppress the
          ``on_disconnect`` callback during reconnect scenarios.
        """
        was_cancelled = False

        while self._is_connected and self._reader:
            # Phase check: if a newer listener started, exit immediately
            # without touching the StreamReader.
            if phase != self._listener_phase:
                break

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
                # Normal: no message within 30s keepalive window
                continue
            except ssl.SSLWantReadError:
                # Non-blocking SSL needs retry — yield and try again
                await asyncio.sleep(0.01)
                continue
            except ssl.SSLWantWriteError:
                await asyncio.sleep(0.01)
                continue
            except asyncio.CancelledError:
                # Task was cancelled during graceful shutdown (reconnect).
                # DO NOT fire on_disconnect — the new listener owns the
                # connection now.  See docstring above.
                was_cancelled = True
                break
            except RuntimeError as e:
                if "already waiting for incoming data" in str(e):
                    # Race during reconnect: a previous listener's
                    # readexactly() left a stale _waiter on the reader.
                    # This happens when asyncio's StreamReader doesn't
                    # clear its internal waiter synchronously after a
                    # CancelledError during readexactly().  Yield and
                    # retry once — the stale waiter should resolve.
                    logger.debug(
                        "Listener race (stale _waiter) — retrying once"
                    )
                    await asyncio.sleep(0)
                    continue
                logger.warning(
                    f"Background listener disconnected: {type(e).__name__}: {e}",
                    exc_info=True,
                )
                break
            except Exception as e:
                # Connection lost, stream error, protocol error, etc.
                logger.warning(
                    f"Background listener disconnected: {type(e).__name__}: {e}",
                    exc_info=True,
                )
                break

        # Connection is dead — notify the system (once).
        # Skip notification if we were cancelled (reconnect case).
        if not was_cancelled:
            was_connected = self._is_connected
            self._is_connected = False
            if was_connected and self.on_disconnect:
                try:
                    await self.on_disconnect()
                except Exception as e:
                    logger.warning(f"Disconnect callback error: {e}")

    async def _heartbeat_loop(self):
        """Send ProtoHeartbeatEvent to the server every 8 seconds.

        Required by cTrader Open API — without heartbeats the server
        closes the connection after a period of inactivity.
        See: https://help.ctrader.com/open-api/faq/

        Interval is 8s (not 10s) to provide safety margin against
        network jitter and server-side timeouts (~100-120s).

        Tracks success/failure counts so diagnostics can detect
        stale heartbeats before the server disconnects us.
        """
        count = 0
        while self._is_connected:
            try:
                await asyncio.sleep(8)
                if self._is_connected and self._writer:
                    heartbeat = ProtoHeartbeatEvent()
                    ok = await self._send_msg(51, heartbeat)
                    count += 1
                    if ok:
                        self._heartbeat_ok += 1
                        self._last_heartbeat_ts = time.time()
                    else:
                        self._heartbeat_fail += 1
                    if count % 10 == 0:  # Log every ~80s
                        logger.debug(
                            f"Heartbeat #{count} sent (ok={ok}, "
                            f"ok={self._heartbeat_ok} fail={self._heartbeat_fail})"
                        )
                    # Proactive disconnect if 3 consecutive heartbeats fail
                    if self._heartbeat_fail >= 3 and self._heartbeat_ok == 0:
                        logger.warning(
                            "3 consecutive heartbeats failed — "
                            "connection is dead, forcing disconnect"
                        )
                        self._is_connected = False
                        break
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._heartbeat_fail += 1
                logger.debug(f"Heartbeat error: {e}")

    async def _handle_message(self, msg):
        """Dispatch incoming messages to appropriate handlers.

        Routes to:
          - Request-response futures (by payloadType key)
          - Market data (depth events, spot events)
          - Account/margin updates (margin changed, trader updated)
          - Order lifecycle events (execution, errors)
          - Heartbeat events (keepalive from server — acknowledged silently)
        """
        # 1. Check for pending request-response futures by payloadType.
        #    Simple int keys for all requests (they're sequential so no
        #    collision).  subscribe_depth uses (int, symbol_id) tuples.
        pt = msg.payloadType
        fut = self._pending_responses.pop(pt, None)
        if fut is None:
            # Try compound key (payloadType, symbol_id) for depth subs
            for key in list(self._pending_responses.keys()):
                if isinstance(key, tuple) and key[0] == pt:
                    fut = self._pending_responses.pop(key)
                    break
        if fut is not None and not fut.done():
            fut.set_result(msg)
            return

        # 2. Route by payload type
        if msg.payloadType == PROTO_OA_DEPTH_EVENT:
            await self._handle_depth_event(msg)
        elif msg.payloadType == PROTO_OA_SPOT_EVENT:
            await self._handle_spot_event(msg)
        elif msg.payloadType == PROTO_OA_MARGIN_CHANGED_EVENT:
            await self._handle_margin_changed_event(msg)
        elif msg.payloadType == PROTO_OA_TRADER_UPDATED_EVENT:
            await self._handle_trader_updated_event(msg)
        elif msg.payloadType == PROTO_OA_ORDER_ERROR_EVENT:
            await self._handle_order_error_event(msg)
        elif msg.payloadType == PROTO_OA_ACCOUNT_AUTH_RES:
            logger.debug("Received account auth response")
        elif msg.payloadType == PROTO_OA_APPLICATION_AUTH_RES:
            logger.debug("Received app auth response")
        elif msg.payloadType == 51:
            # ProtoHeartbeatEvent from server — silently acknowledge.
            # cTrader sends heartbeats to verify the connection is alive.
            # No action needed on our side; our _heartbeat_loop handles
            # the client-to-server direction.
            pass
        else:
            # Log unhandled message type — don't try to parse as ProtoErrorRes
            # unless the payload type actually signals an error (2155+ range).
            # Some unhandled messages are legitimate responses we don't care
            # about (e.g. ProtoOASubscribeDepthQuotesRes routed by clientMsgId).
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Unhandled message: payloadType={msg.payloadType}, "
                    f"payload={len(msg.payload)}B"
                )

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
                raw = quote.bid if quote.HasField("bid") else quote.ask
                divisor = self._price_divisor(symbol, raw)
                price = raw / divisor if divisor else raw

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
            md.bids = sorted(bids_dict.values(), key=lambda x: x.price, reverse=True)[
                :20
            ]
            md.asks = sorted(asks_dict.values(), key=lambda x: x.price)[:20]

            # Update best bid/ask
            md.bid = md.bids[0].price if md.bids else 0.0
            md.ask = md.asks[0].price if md.asks else 0.0
            # Safety: ensure bid < ask; swap if cTrader provides reversed
            if md.bid > md.ask and md.ask > 0:
                md.bid, md.ask = md.ask, md.bid
            md.spread = (md.ask - md.bid) if (md.ask and md.bid) else 0.0
            md.volume = sum(b.size for b in md.bids) + sum(a.size for a in md.asks)

            # Spot events are the authoritative price source
            # ------------------------------------------------
            # cTrader depth events and spot events both use a divisor of
            # 100000 (see _price_divisor).  However, depth events do NOT
            # carry moneyDigits, so the divisor is always a flat guess.
            # Spot events DO carry moneyDigits → exact scaling.
            #
            # Strategy:
            #   - If we have a cached spot price for this symbol, skip the
            #     depth event entirely.  The spot event handler will fire
            #     on_market_data with the correct bid/ask prices.
            #   - If no spot price is available (e.g. during initial connect),
            #     use the raw depth prices ÷ 100000 as a fallback.  No
            #     correction factor is applied — the divisor is already correct.
            spot = self._last_spot_prices.get(symbol_id)
            if spot and spot["bid"] > 0:
                logger.debug(
                    f"Skipping depth event for {symbol}, "
                    f"using spot price {spot['bid']:.5f}"
                )
                return

            # Notify callbacks
            if self.on_depth_update:
                result = self.on_depth_update(md)
                if hasattr(result, "__await__"):
                    await result
            if self.on_market_data and md.bid > 0:
                await self.on_market_data(md)

            logger.debug(
                f"DOM update {symbol}: {len(md.bids)} bids, {len(md.asks)} asks"
            )
        except Exception as e:
            logger.warning(f"Failed to parse depth event: {e}")

    async def subscribe_depth(self, symbol_id: int) -> bool:
        """Subscribe to Level II DOM for a symbol.

        Waits for the server's response before returning success.
        Uses unique clientMsgId per request so multiple concurrent
        subscriptions don't collide in _pending_responses.
        """
        if not _HAS_PROTOBUF or not self._authenticated:
            return False
        if symbol_id in self._subscribed_depth:
            return True
        try:
            req = ProtoOASubscribeDepthQuotesReq()
            req.ctidTraderAccountId = self.account_id
            req.symbolId.append(symbol_id)

            # Register future with compound key (payloadType, symbol_id)
            # to prevent collision between multiple sequential depth subs.
            depth_key = (PROTO_OA_SUBSCRIBE_DEPTH_QUOTES_RES, symbol_id)
            fut = asyncio.get_event_loop().create_future()
            self._pending_responses[depth_key] = fut

            if not await self._send_msg(PROTO_OA_SUBSCRIBE_DEPTH_QUOTES_REQ, req):
                self._pending_responses.pop(depth_key, None)
                return False

            response = await asyncio.wait_for(fut, timeout=10.0)
            self._pending_responses.pop(depth_key, None)
            if response and response.payloadType == PROTO_OA_SUBSCRIBE_DEPTH_QUOTES_RES:
                self._subscribed_depth[symbol_id] = True
                logger.info(
                    f"Subscribed to Level II DOM for "
                    f"{REVERSE_SYMBOL_MAP.get(symbol_id, symbol_id)}"
                )
                return True
            return False
        except asyncio.TimeoutError:
            logger.debug(f"Depth subscribe timeout for symbol {symbol_id}")
            self._pending_responses.pop(depth_key, None)
            return False
        except Exception as e:
            logger.warning(f"Failed to subscribe to depth: {e}")
            self._pending_responses.pop(depth_key, None)
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
                logger.info(
                    f"Unsubscribed from Level II DOM for {REVERSE_SYMBOL_MAP.get(symbol_id, symbol_id)}"  # noqa: E501
                )
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to unsubscribe from depth: {e}")
            return False

    # ------------------------------------------------------------------
    # Spot price subscription (simpler alternative to Level II DOM)
    # ------------------------------------------------------------------

    async def subscribe_spots(self, symbol_id: int) -> bool:
        """Subscribe to spot price updates for a symbol (lighter than depth).

        NOTE: cTrader does NOT send ProtoOASubscribeSpotsRes for spot
        subscriptions.  Instead, the server immediately starts pushing
        ProtoOASpotEvent messages.  We fire-and-forget the request and
        mark as subscribed; errors arrive as ProtoOAErrorRes which are
        picked up by _handle_message.
        """
        if not _HAS_PROTOBUF or not self._authenticated:
            return False
        if symbol_id in self._subscribed_spots:
            return True
        try:
            req = ProtoOASubscribeSpotsReq()
            req.ctidTraderAccountId = self.account_id
            req.symbolId.append(symbol_id)
            if await self._send_msg(PROTO_OA_SUBSCRIBE_SPOTS_REQ, req):
                self._subscribed_spots[symbol_id] = True
                sym_name = REVERSE_SYMBOL_MAP.get(symbol_id, str(symbol_id))
                logger.info(f"Subscribed to spot prices for {sym_name}")
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to subscribe to spot: {e}")
            return False

    async def unsubscribe_spots(self, symbol_id: int) -> bool:
        """Unsubscribe from spot price updates."""
        if not _HAS_PROTOBUF or symbol_id not in self._subscribed_spots:
            return False
        try:
            req = ProtoOAUnsubscribeSpotsReq()
            req.ctidTraderAccountId = self.account_id
            req.symbolId.append(symbol_id)
            if await self._send_msg(PROTO_OA_UNSUBSCRIBE_SPOTS_REQ, req):
                del self._subscribed_spots[symbol_id]
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to unsubscribe spot: {e}")
            return False

    # ------------------------------------------------------------------
    # Event handlers (push messages from broker)
    # ------------------------------------------------------------------

    async def _handle_margin_changed_event(self, msg):
        """Process ProtoOAMarginChangedEvent → updates _account_info.margin."""
        try:
            event = ProtoOAMarginChangedEvent()
            event.ParseFromString(msg.payload)
            if self._account_info is None:
                return
            money_div = 10**event.moneyDigits
            used_margin = event.usedMargin / money_div
            self._account_info.margin = used_margin
            # Recalculate free_margin and margin_level
            eq = self._account_info.equity or self._account_info.balance
            self._account_info.free_margin = max(eq - used_margin, 0.0)
            self._account_info.margin_level = (
                (eq / used_margin * 100) if used_margin > 0 else 0.0
            )
            self._account_info.timestamp = time.time()
            logger.debug(
                f"Margin changed: {used_margin:.2f} "
                f"(free={self._account_info.free_margin:.2f}, "
                f"level={self._account_info.margin_level:.2f}%)"
            )
            if self.on_account_update:
                await self._on_account_update_safe()
        except Exception as e:
            logger.warning(f"Failed to parse margin changed event: {e}")

    async def _handle_trader_updated_event(self, msg):
        """Process ProtoOATraderUpdatedEvent → updated balance/equity/etc."""
        try:
            event = ProtoOATraderUpdatedEvent()
            event.ParseFromString(msg.payload)
            if self._account_info is None:
                return
            trader = event.trader  # ProtoOATrader nested message
            if trader is None:
                return
            money_div = 10**trader.moneyDigits
            old_balance = self._account_info.balance
            self._account_info.balance = trader.balance / money_div
            self._account_info.timestamp = time.time()
            if old_balance != self._account_info.balance:
                logger.info(
                    f"Balance updated: {old_balance:.2f} → "
                    f"{self._account_info.balance:.2f}"
                )
            if self.on_account_update:
                await self._on_account_update_safe()
        except Exception as e:
            logger.warning(f"Failed to parse trader updated event: {e}")

    async def _handle_order_error_event(self, msg):
        """Process ProtoOAOrderErrorEvent → log order rejection details."""
        try:
            from ctrader_open_api.messages.OpenApiMessages_pb2 import (
                ProtoOAOrderErrorEvent,
            )

            event = ProtoOAOrderErrorEvent()
            event.ParseFromString(msg.payload)
            logger.warning(
                f"Order error: orderId={event.orderId} "
                f"errorCode={event.errorCode} "
                f"description={getattr(event, 'description', '')}"
            )
        except Exception as e:
            logger.debug(f"Failed to parse order error event: {e}")

    async def _handle_spot_event(self, msg):
        """Process ProtoOASpotEvent → store latest spot bid/ask per symbol.

        Per cTrader docs, spot event prices are in '1/100,000' format,
        same as depth events.  Divisor is always 100000 for ALL symbols.
        """
        try:
            event = ProtoOASpotEvent()
            event.ParseFromString(msg.payload)
            symbol_id = event.symbolId
            bid = event.bid / 100000.0 if event.bid else 0.0
            ask = event.ask / 100000.0 if event.ask else 0.0
            self._last_spot_prices[symbol_id] = {
                "bid": bid,
                "ask": ask,
                "timestamp": time.time(),
            }
            # Forward spot prices as market data.  Spot events have moneyDigits
            # which gives EXACT price scaling (unlike depth events where
            # _price_divisor may guess wrong).  We forward unconditionally so
            # downstream consumers get accurate prices regardless of depth
            # subscription status.
            if self.on_market_data and bid > 0:
                symbol = REVERSE_SYMBOL_MAP.get(symbol_id, str(symbol_id))

                spot_depth = MarketDepth(
                    symbol=symbol,
                    bid=bid,
                    ask=ask,
                    spread=(ask - bid) if ask > bid else 0.0,
                    volume=0,
                    timestamp=time.time(),
                    symbol_id=symbol_id,
                )
                await self.on_market_data(spot_depth)
        except Exception as e:
            logger.debug(f"Failed to parse spot event: {e}")

    async def _on_account_update_safe(self):
        """Fire on_account_update callback with error guarding."""
        try:
            if self.on_account_update:
                result = self.on_account_update(self._account_info)
                if hasattr(result, "__await__"):
                    await result
        except Exception as e:
            logger.warning(f"Account update callback error: {e}")

    # ------------------------------------------------------------------
    # Position reconciliation
    # ------------------------------------------------------------------

    async def reconcile(self) -> Optional[List[Dict]]:
        """Send ProtoOAReconcileReq to get current positions from broker.

        Returns a list of position dicts or None on failure.
        """
        if not self._authenticated or not self._is_connected:
            return None
        req = ProtoOAReconcileReq()
        req.ctidTraderAccountId = self.account_id

        fut = asyncio.get_event_loop().create_future()
        self._pending_responses[PROTO_OA_RECONCILE_RES] = fut

        if not await self._send_msg(PROTO_OA_RECONCILE_REQ, req):
            self._pending_responses.pop(PROTO_OA_RECONCILE_RES, None)
            return None

        try:
            response = await asyncio.wait_for(fut, timeout=30.0)
        except asyncio.TimeoutError:
            self._pending_responses.pop(PROTO_OA_RECONCILE_RES, None)
            logger.warning("Reconcile request timed out")
            return None

        if response and response.payloadType == PROTO_OA_RECONCILE_RES:
            res = ProtoOAReconcileRes()
            res.ParseFromString(response.payload)
            positions = []
            for pos in res.position:
                positions.append(
                    {
                        "position_id": pos.positionId,
                        "symbol_id": pos.symbolId,
                        "volume": pos.volume,
                        "price": pos.price,
                        "direction": "BUY" if pos.tradeSide == 1 else "SELL",
                        "swap": pos.swap,
                    }
                )
            self._cached_positions = positions
            if self.on_positions_update:
                try:
                    self.on_positions_update(positions)
                except Exception:
                    pass
            logger.debug(f"Reconcile: {len(positions)} positions")
            return positions
        return None

    # ------------------------------------------------------------------
    # Unrealized PnL query
    # ------------------------------------------------------------------

    async def fetch_unrealized_pnl(self, position_id: int) -> Optional[float]:
        """Query broker for unrealized PnL of a specific position."""
        if not self._authenticated or not self._is_connected:
            return None
        req = ProtoOAGetPositionUnrealizedPnLReq()
        req.ctidTraderAccountId = self.account_id
        req.positionId = position_id

        fut = asyncio.get_event_loop().create_future()
        self._pending_responses[PROTO_OA_GET_POSITION_UNREALIZED_PNL_RES] = fut

        if not await self._send_msg(PROTO_OA_GET_POSITION_UNREALIZED_PNL_REQ, req):
            self._pending_responses.pop(PROTO_OA_GET_POSITION_UNREALIZED_PNL_RES, None)
            return None

        try:
            response = await asyncio.wait_for(fut, timeout=15.0)
        except asyncio.TimeoutError:
            self._pending_responses.pop(PROTO_OA_GET_POSITION_UNREALIZED_PNL_RES, None)
            logger.debug("Unrealized PnL request timed out")
            return None

        if (
            response
            and response.payloadType == PROTO_OA_GET_POSITION_UNREALIZED_PNL_RES
        ):
            res = ProtoOAGetPositionUnrealizedPnLRes()
            res.ParseFromString(response.payload)
            pnl = res.positionUnrealizedPnL
            money_div = 10 ** res.moneyDigits if hasattr(res, "moneyDigits") else 1
            return pnl / money_div
        return None

    @staticmethod
    def _price_divisor(symbol: str, raw: float) -> float:
        """Return the divisor for cTrader price scaling.

        Per cTrader Open API docs, ALL prices (depth, spot, trendbar) use
        '1/100,000' format — divisor is ALWAYS 100000 for every symbol type.
        See https://help.ctrader.com/open-api/symbol-data/

        Note: ``symbol`` and ``raw`` are unused (kept for API compatibility
        with callers that pass them).  Unlike spot events which carry
        moneyDigits, depth events always need the flat 100000 divisor.
        """
        return 100000.0

    def get_market_depth(self, symbol: str) -> Optional[MarketDepth]:
        """Get latest Level II market depth for a symbol."""
        symbol_id = SYMBOL_MAP.get(symbol.upper())
        if symbol_id and symbol_id in self._last_market_depth:
            return self._last_market_depth[symbol_id]
        return None
