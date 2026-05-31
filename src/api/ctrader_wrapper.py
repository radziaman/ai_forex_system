"""
cTrader Open API Wrapper — uses the official Spotware ctrader_open_api library.

Replaces the hand-rolled protobuf client with the official Twisted-based library,
integrated with our asyncio event loop via AsyncioSelectorReactor.

The official library handles:
  - SSL connection management with auto-reconnect
  - Protobuf message serialization/deserialization
  - Message routing with deferred-based response matching
  - Heartbeat management

We handle:
  - Application + account authentication
  - Token refresh with rate-limit awareness
  - Market data subscription (depth, spots)
  - Order execution and position management
  - Price feed callback dispatch
"""

from __future__ import annotations

# ── Install Twisted's asyncio reactor BEFORE any other Twisted import ──
# This makes Twisted use the same asyncio event loop as our agent system.
import platform

if platform.system() != "Windows":
    import asyncio

    try:
        from twisted.internet import asyncioreactor

        asyncioreactor.install()
    except (AssertionError, Exception):
        pass  # Already installed or not needed

import time
from typing import Optional, Callable, Dict, Any, List

from twisted.internet import defer, reactor
from loguru import logger

from ctrader_open_api import Client, TcpProtocol, EndPoints
from ctrader_open_api.protobuf import Protobuf
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import (
    ProtoMessage,
    ProtoHeartbeatEvent,
)
from ctrader_open_api.messages.OpenApiMessages_pb2 import (
    ProtoOAAccountAuthReq,
    ProtoOAAccountAuthRes,
    ProtoOAApplicationAuthReq,
    ProtoOAClosePositionReq,
    ProtoOADepthEvent,
    ProtoOAExecutionEvent,
    ProtoOAGetAccountListByAccessTokenReq,
    ProtoOAGetAccountListByAccessTokenRes,
    ProtoOAMarginChangedEvent,
    ProtoOANewOrderReq,
    ProtoOAReconcileReq,
    ProtoOAReconcileRes,
    ProtoOASpotEvent,
    ProtoOASubscribeDepthQuotesReq,
    ProtoOASubscribeSpotsReq,
    ProtoOATraderUpdatedEvent,
)


# ── Constants ──────────────────────────────────────────────────────────

MSG_TIMEOUT = 30  # seconds to wait for response messages


# ── Helper: convert Twisted Deferred to asyncio Future ────────────────


async def await_deferred(d: defer.Deferred, timeout: float = MSG_TIMEOUT) -> Any:
    """Await a Twisted Deferred from asyncio code."""
    future: asyncio.Future = asyncio.Future()

    def callback(value):
        if not future.done():
            future.set_result(value)

    def errback(failure):
        if not future.done():
            future.set_exception(failure.value)

    d.addCallbacks(callback, errback)
    try:
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        d.cancel()
        raise


# ── CtraderClient ─────────────────────────────────────────────────────


class CtraderClient:
    """cTrader Open API client using the official Spotware library.

    Wraps ``ctrader_open_api.Client`` (Twisted-based) and exposes an
    asyncio-compatible interface for the rest of the RTS agent system.
    """

    def __init__(
        self,
        app_id: str = "",
        app_secret: str = "",
        access_token: str = "",
        account_id: int = 0,
        demo: bool = True,
    ):
        self.app_id = app_id
        self.app_secret = app_secret
        self.access_token = access_token
        self.account_id = account_id
        self.demo = demo

        host = EndPoints.PROTOBUF_DEMO_HOST if demo else EndPoints.PROTOBUF_LIVE_HOST
        self._client = Client(host, EndPoints.PROTOBUF_PORT, TcpProtocol)
        self._client.setConnectedCallback(self._on_connected)
        self._client.setDisconnectedCallback(self._on_disconnected)
        self._client.setMessageReceivedCallback(self._on_message)

        self._is_connected: bool = False
        self._simulation: bool = True  # Start in simulation until proven otherwise
        self._ctid_trader_account_id: int = 0

        # Callbacks
        self.on_price: Optional[Callable] = None
        self.on_market_data: Optional[Callable] = None
        self.on_disconnect: Optional[Callable] = None
        self.on_order_update: Optional[Callable] = None

        # Subscription state
        self._depth_subscriptions: Dict[int, bool] = {}
        self._spot_subscriptions: Dict[int, bool] = {}

        # Last heartbeat timestamp (for health monitor)
        self._last_heartbeat_ts: float = 0.0

    # ── Public API ─────────────────────────────────────────────────

    async def start(self) -> bool:
        """Connect to cTrader, authenticate, and start listening.

        Returns True if a live connection is established, False otherwise.
        """
        # Refresh access token first (non-blocking HTTP via thread pool)
        await self._refresh_access_token()

        # Start the Twisted client service (connects via SSL)
        self._client.startService()

        # Wait for connection callback (timeout after 60s)
        t0 = time.time()
        while not self._is_connected and time.time() - t0 < 60:
            await asyncio.sleep(0.1)

        if not self._is_connected:
            logger.warning("cTrader SSL connection timed out")
            self._client.stopService()
            return False

        # Step 1: Application authentication
        try:
            app_req = ProtoOAApplicationAuthReq()
            app_req.clientId = self.app_id
            app_req.clientSecret = self.app_secret
            resp = await self._send_and_wait(2100, app_req, expected_type=2101)
            if resp is None or resp.payloadType != 2101:
                logger.warning("cTrader application auth failed")
                self._client.stopService()
                return False
            logger.info("Application auth OK")
        except Exception as e:
            logger.warning(f"Application auth error: {e}")
            self._client.stopService()
            return False

        # Step 2: Account authentication
        try:
            acct_req = ProtoOAAccountAuthReq()
            acct_req.ctidTraderAccountId = self.account_id
            acct_req.accessToken = self.access_token
            resp = await self._send_and_wait(2102, acct_req, expected_type=2103)
            if resp is None:
                logger.warning("cTrader account auth failed (no response)")
                self._client.stopService()
                return False
            if resp.payloadType == 2103:
                # Success — extract ctidTraderAccountId from response
                auth_res = ProtoOAAccountAuthRes()
                auth_res.ParseFromString(resp.protoOAPayload)
                self._ctid_trader_account_id = auth_res.ctidTraderAccountId
                self._simulation = False
                logger.info("Account auth OK — LIVE connection")
            else:
                logger.warning(f"Account auth failed: payloadType={resp.payloadType}")
                self._client.stopService()
                return False
        except Exception as e:
            logger.warning(f"Account auth error: {e}")
            self._client.stopService()
            return False

        logger.info("cTrader client connected and authenticated")
        return True

    async def stop(self):
        """Disconnect from cTrader."""
        self._client.stopService()
        self._is_connected = False

    def is_connected(self) -> bool:
        return self._is_connected

    def is_simulation(self) -> bool:
        return self._simulation

    @property
    def raw(self):
        """Return self for backward compatibility (subscription methods are on this class)."""
        return self

    @property
    def account_info(self):
        return getattr(self, "_account_info", None)

    # ── Subscriptions ──────────────────────────────────────────────

    async def subscribe_depth(self, symbol_id: int) -> bool:
        """Subscribe to Level II Depth of Market for a symbol."""
        try:
            req = ProtoOASubscribeDepthQuotesReq()
            req.ctidTraderAccountId = self._ctid_trader_account_id
            req.symbolId = symbol_id
            resp = await self._send_and_wait(2104, req, expected_type=2105)
            ok = resp is not None
            if ok:
                self._depth_subscriptions[symbol_id] = True
            return ok
        except Exception:
            return False

    async def subscribe_spots(self, symbol_id: int) -> bool:
        """Subscribe to spot price updates for a symbol."""
        try:
            req = ProtoOASubscribeSpotsReq()
            req.ctidTraderAccountId = self._ctid_trader_account_id
            req.symbolId = symbol_id
            # Spot subscriptions are fire-and-forget (no response expected)
            await self._send(2106, req)
            self._spot_subscriptions[symbol_id] = True
            return True
        except Exception:
            return False

    # ── Order Management ───────────────────────────────────────────

    async def open_position(
        self,
        symbol_id: int,
        order_type: str,
        volume: float,
        stop_loss_price: float = 0.0,
        take_profit_price: float = 0.0,
        comment: str = "",
    ) -> Optional[Dict]:
        """Open a new position. Returns execution event dict or None."""
        try:
            order_req = ProtoOANewOrderReq()
            order_req.ctidTraderAccountId = self._ctid_trader_account_id
            order_req.symbolId = symbol_id
            order_req.orderType = 50 if order_type == "BUY" else 51  # MARKET
            order_req.tradeSide = 100 if order_type == "BUY" else 101
            order_req.volume = int(volume * 100)  # cents
            if stop_loss_price > 0:
                order_req.stopLossPrice = int(stop_loss_price * 1e5)
            if take_profit_price > 0:
                order_req.takeProfitPrice = int(take_profit_price * 1e5)
            order_req.comment = comment

            resp = await self._send_and_wait(2108, order_req, expected_type=2109)
            if resp is None:
                return None

            exec_event = ProtoOAExecutionEvent()
            exec_event.ParseFromString(resp.protoOAPayload)
            return Protobuf.extract(exec_event)
        except Exception as e:
            logger.debug(f"Open position error: {e}")
            return None

    async def close_position(self, position_id: int, volume: float) -> bool:
        """Close a position (fully or partially)."""
        try:
            req = ProtoOAClosePositionReq()
            req.ctidTraderAccountId = self._ctid_trader_account_id
            req.positionId = position_id
            req.volume = int(volume * 100)
            resp = await self._send_and_wait(2118, req, expected_type=2119)
            return resp is not None
        except Exception:
            return False

    async def get_open_positions(self) -> List[Dict]:
        """Fetch all open positions via reconciliation."""
        try:
            req = ProtoOAReconcileReq()
            req.ctidTraderAccountId = self._ctid_trader_account_id
            resp = await self._send_and_wait(2110, req, expected_type=2111)
            if resp is None:
                return []
            rec = ProtoOAReconcileRes()
            rec.ParseFromString(resp.protoOAPayload)
            return [Protobuf.extract(p) for p in rec.position]
        except Exception:
            return []

    async def fetch_account_info(self) -> Optional[Dict]:
        """Fetch account details from cTrader."""
        try:
            req = ProtoOAGetAccountListByAccessTokenReq()
            req.accessToken = self.access_token
            resp = await self._send_and_wait(2106, req, expected_type=2107)
            if resp is None:
                return None
            acct_res = ProtoOAGetAccountListByAccessTokenRes()
            acct_res.ParseFromString(resp.protoOAPayload)
            accounts = [Protobuf.extract(a) for a in acct_res.account]
            if accounts:
                self._account_info = accounts[0]
            return accounts
        except Exception:
            return None
            acct_res = ProtoOAGetAccountListByAccessTokenRes()
            acct_res.ParseFromString(resp.protoOAPayload)
            accounts = [Protobuf.extract(a) for a in acct_res.account]
            if accounts:
                self._account_info = accounts[0]
            return accounts
        except Exception:
            return None

    # ── Internal: message send/receive ─────────────────────────────

    async def _send(self, payload_type: int, payload_obj) -> bool:
        """Send a protobuf message without waiting for response."""
        try:
            msg = ProtoMessage()
            msg.payloadType = payload_type
            msg.protoOAPayload = payload_obj.SerializeToString()
            self._client.send(msg)
            return True
        except Exception:
            return False

    async def _send_and_wait(
        self, payload_type: int, payload_obj, expected_type: Optional[int] = None
    ) -> Optional[ProtoMessage]:
        """Send a protobuf message and wait for the response."""
        msg = ProtoMessage()
        msg.payloadType = payload_type
        msg.protoOAPayload = payload_obj.SerializeToString()

        deferred = self._client.send(msg)

        try:
            response = await await_deferred(deferred, timeout=MSG_TIMEOUT)
            if expected_type and response.payloadType != expected_type:
                logger.warning(
                    f"Expected response type {expected_type}, got {response.payloadType}"
                )
                return None
            return response
        except Exception as e:
            logger.debug(f"Send/wait failed (type={payload_type}): {e}")
            return None

    # ── Callbacks from official Client ─────────────────────────────

    def _on_connected(self, client):
        """Called by the official library when SSL connection is established."""
        self._is_connected = True
        self._last_heartbeat_ts = time.time()
        logger.info("Connected via official cTrader library")

    def _on_disconnected(self, client, reason):
        """Called by the official library when connection drops."""
        self._is_connected = False
        logger.warning(f"cTrader disconnected: {reason}")
        if self.on_disconnect:
            try:
                import asyncio

                asyncio.ensure_future(self.on_disconnect())
            except Exception:
                pass

    def _on_message(self, client, message: ProtoMessage):
        """Called by the official library for every received message."""
        self._last_heartbeat_ts = time.time()

        pt = message.payloadType

        # Spot price updates (most frequent)
        if pt == 2109:  # ProtoOASpotEvent
            self._handle_spot_event(message)

        # Depth of Market updates
        elif pt == 2111:  # ProtoOADepthEvent
            self._handle_depth_event(message)

        # Execution events (order fills, rejections)
        elif pt == 2113:  # ProtoOAExecutionEvent
            self._handle_execution_event(message)

        # Margin changed
        elif pt == 2120:  # ProtoOAMarginChangedEvent
            pass  # Tracked by engine

        # Trader updated (positions modified)
        elif pt == 2122:  # ProtoOATraderUpdatedEvent
            pass

        # Heartbeat (keep-alive)
        elif pt == 2001:  # ProtoHeartbeatEvent
            pass

        # Everything else — ignore
        else:
            pass

    def _handle_spot_event(self, message: ProtoMessage):
        """Handle spot price update from cTrader."""
        try:
            event = ProtoOASpotEvent()
            event.ParseFromString(message.protoOAPayload)
            extracted = Protobuf.extract(event)

            if self.on_price:
                from api.base import PriceTick

                tick = PriceTick(
                    symbol=extracted.get("symbolName", ""),
                    bid=extracted.get("bid", 0),
                    ask=extracted.get("ask", 0),
                    spread=extracted.get("spread", 0),
                    timestamp=time.time(),
                    volume=0,
                )
                try:
                    import asyncio

                    asyncio.ensure_future(self.on_price(tick))
                except Exception:
                    pass
        except Exception:
            pass

    def _handle_depth_event(self, message: ProtoMessage):
        """Handle Level II depth update from cTrader."""
        try:
            event = ProtoOADepthEvent()
            event.ParseFromString(message.protoOAPayload)

            if self.on_market_data:
                try:
                    import asyncio

                    asyncio.ensure_future(self.on_market_data(event))
                except Exception:
                    pass
        except Exception:
            pass

    def _handle_execution_event(self, message: ProtoMessage):
        """Handle order execution event (fill, partial fill, rejection)."""
        try:
            event = ProtoOAExecutionEvent()
            event.ParseFromString(message.protoOAPayload)

            if self.on_order_update:
                try:
                    import asyncio

                    asyncio.ensure_future(self.on_order_update(event))
                except Exception:
                    pass
        except Exception:
            pass

    # ── Token Refresh ──────────────────────────────────────────────

    async def _refresh_access_token(self) -> bool:
        """Refresh access token via cTrader's OAuth endpoint.

        Runs the HTTP request in a thread pool to avoid blocking the
        asyncio event loop. Handles 429 rate limiting gracefully.
        """
        import os
        import re
        import functools

        env_path = None
        for candidate in [
            os.path.join(os.getcwd(), ".env"),
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"
            ),
        ]:
            candidate = os.path.normpath(candidate)
            if os.path.exists(candidate):
                env_path = candidate
                break

        if env_path is None:
            logger.debug("No .env found for token refresh")
            return True  # Current token might still work

        with open(env_path) as f:
            env = f.read()

        match = re.search(r"CTRADER_REFRESH_TOKEN=(\S+)", env)
        if not match:
            return True  # No refresh token, use current access token

        refresh_token = match.group(1)

        try:
            import requests as http_requests

            loop = asyncio.get_running_loop()
            resp = await loop.run_in_executor(
                None,
                functools.partial(
                    http_requests.post,
                    EndPoints.TOKEN_URI,
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": refresh_token,
                        "client_id": self.app_id,
                        "client_secret": self.app_secret,
                    },
                    timeout=30,
                ),
            )

            if resp.status_code == 429:
                logger.warning(
                    "Token refresh rate-limited (429), current token may still work"
                )
                return True
            if resp.status_code != 200:
                logger.warning(f"Token refresh failed: {resp.status_code}")
                return True  # Current token might still be valid

            data = resp.json()
            new_token = data.get("accessToken") or data.get("access_token", "")
            new_refresh = data.get("refreshToken") or data.get("refresh_token", "")

            if new_token and len(new_token) >= 10:
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
                logger.info("Access token refreshed successfully")
        except Exception as e:
            logger.debug(f"Token refresh error: {e}")

        return True
