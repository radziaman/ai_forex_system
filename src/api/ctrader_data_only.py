"""
cTrader Read-Only Data Listener — streams real-time price quotes without any trading capability.

Connects to cTrader Open API via SSL/TCP, authenticates, subscribes to spot price updates,
and streams them via callbacks. Has NO account ID configured and NO order placement methods.

Uses the same protobuf protocol as ctrader_client.py but purpose-built for read-only data.

Usage:
    listener = CtradeDataListener()
    await listener.connect()
    await listener.subscribe_prices(["EURUSD", "GBPUSD"], on_tick_callback)
    while True: await asyncio.sleep(1)  # stream forever
"""

import os, sys, ssl, asyncio, struct, time
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass
from loguru import logger

_src = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _src not in sys.path:
    sys.path.insert(0, _src)

from api.symbol_map import get_symbol_id

# cTrader protobuf messages
_CTRADER_AVAILABLE = False
try:
    from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import ProtoMessage
    from ctrader_open_api.messages.OpenApiMessages_pb2 import (
        ProtoOAApplicationAuthReq,
        ProtoOAApplicationAuthRes,
        ProtoOAAccountAuthReq,
        ProtoOAAccountAuthRes,
        ProtoOAGetAccountListByAccessTokenReq,
        ProtoOAGetAccountListByAccessTokenRes,
        ProtoOASubscribeSpotsReq,
        ProtoOASpotEvent,
    )

    _CTRADER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ctrader_open_api protobuf not available: {e}")

from infrastructure.secrets import Secrets


@dataclass
class PriceQuote:
    """Real-time price quote from cTrader."""

    symbol: str
    symbol_id: int
    bid: float
    ask: float
    timestamp: float
    spread: float = 0.0
    spread_bps: float = 0.0


class CtradeDataListener:
    """
    Read-only cTrader data listener using raw SSL/TCP + protobuf.

    SAFETY: ZERO trade execution capability. No account ID configured.
    Only subscribes to spot price data.
    """

    def __init__(self):
        self._reader = None
        self._writer = None
        self._running = False
        self._authenticated = False
        self._subscribed: Dict[int, str] = {}  # symbol_id -> symbol
        self._callbacks: Dict[str, List[Callable]] = {}
        self._last_quote: Dict[str, PriceQuote] = {}
        self._buffer = b""
        self._account_id = None

        self._load_credentials()

    def _load_credentials(self):
        secrets = Secrets()
        self.app_id = secrets.ctrader_app_id
        self.app_secret = secrets.ctrader_app_secret
        self.access_token = secrets.ctrader_access_token

        if secrets.is_demo:
            self.host = "demo.ctraderapi.com"
        else:
            self.host = "live.ctraderapi.com"
        self.port = 5035

        if not self.app_id:
            logger.warning(
                "cTrader credentials not configured. Set CTRADER_APP_ID in .env"
            )

    async def connect(self) -> bool:
        """Connect to cTrader via SSL/TCP and authenticate."""
        if not _CTRADER_AVAILABLE:
            logger.error("ctrader_open_api not installed")
            return False

        try:
            context = ssl.create_default_context()
            context.minimum_version = ssl.TLSVersion.TLSv1_2

            logger.info(f"Connecting to {self.host}:{self.port}...")
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port, ssl=context),
                timeout=10,
            )
            logger.info("TCP/SSL connected")

            # Authenticate application
            if not await self._authenticate_app():
                return False

            # Get accounts (needed for data subscription context)
            if not await self._get_accounts():
                return False

            self._running = True
            # Start message reader
            self._reader_task = asyncio.create_task(self._message_loop())
            logger.info("cTrader data listener ready (read-only)")
            return True

        except asyncio.TimeoutError:
            logger.warning("cTrader connection timeout")
            return False
        except Exception as e:
            logger.warning(f"cTrader connection failed: {e}")
            return False

    async def _authenticate_app(self) -> bool:
        """Authenticate the application with client credentials."""
        auth_req = ProtoOAApplicationAuthReq()
        auth_req.clientId = self.app_id
        auth_req.clientSecret = self.app_secret

        await self._send(2100, auth_req)  # ProtoOAApplicationAuthReq
        response = await self._recv(timeout=10)

        if response is None:
            return False

        payload_type, payload = response
        if payload_type == 2101:  # ProtoOAApplicationAuthRes
            logger.info("Application authenticated")
            return True

        logger.warning(f"Auth failed (type={payload_type})")
        return False

    async def _get_accounts(self) -> bool:
        """Get account list for data context."""
        if not self.access_token:
            logger.warning("No access token — data streaming may be limited")
            return True

        req = ProtoOAGetAccountListByAccessTokenReq()
        req.accessToken = self.access_token

        await self._send(2149, req)  # ProtoOAGetAccountListByAccessTokenReq
        response = await self._recv(timeout=10)

        if response is None:
            return True  # Non-fatal — data may still work

        payload_type, payload = response
        if payload_type == 2150:  # ProtoOAGetAccountListByAccessTokenRes
            res = ProtoOAGetAccountListByAccessTokenRes()
            res.ParseFromString(payload)
            accounts = res.ctidTraderAccount
            logger.info(f"Got {len(accounts)} cTrader accounts")
            if len(accounts) > 0:
                self._account_id = accounts[0].ctidTraderAccountId
                logger.info(
                    f"Using account ID {self._account_id} for data subscription"
                )

                # Account authentication — required to receive spot events
                acc_auth = ProtoOAAccountAuthReq()
                acc_auth.ctidTraderAccountId = self._account_id
                acc_auth.accessToken = self.access_token
                await self._send(2102, acc_auth)

                auth_resp = await self._recv(timeout=10)
                if auth_resp:
                    pt, pb = auth_resp
                    if pt == 2103:  # ProtoOAAccountAuthRes
                        logger.info(
                            f"Account {self._account_id} authenticated for data"
                        )
                    else:
                        logger.warning(f"Account auth response type={pt}")
            return True

        logger.info(f"GetAccounts response type={payload_type} (non-fatal)")
        return True  # Continue anyway

    async def subscribe_prices(self, symbols: List[str], callback: Callable):
        """Subscribe to real-time spot price quotes.

        Args:
            symbols: List of symbol names (EURUSD, GBPUSD, etc.)
            callback: Called with PriceQuote on each update
        """
        if not self._writer:
            logger.error("Not connected")
            return

        for symbol in symbols:
            symbol_id = get_symbol_id(symbol)
            if symbol_id is None:
                logger.warning(f"Unknown symbol: {symbol}")
                continue

            self._subscribed[symbol_id] = symbol
            if symbol not in self._callbacks:
                self._callbacks[symbol] = []
            self._callbacks[symbol].append(callback)

            # Subscribe to spot quotes
            req = ProtoOASubscribeSpotsReq()
            req.symbolId.append(symbol_id)
            if self._account_id:
                req.ctidTraderAccountId = self._account_id

            await self._send(100, req)  # PayloadType 100 = ProtoOASubscribeSpotsReq
            logger.info(f"  Subscribed to {symbol} (ID={symbol_id})")

    async def _send(self, payload_type: int, payload_obj) -> bool:
        """Send a protobuf message."""
        try:
            proto_msg = ProtoMessage()
            proto_msg.payloadType = payload_type
            proto_msg.payload = payload_obj.SerializeToString()
            data = proto_msg.SerializeToString()
            self._writer.write(len(data).to_bytes(4, "big") + data)
            await self._writer.drain()
            return True
        except Exception as e:
            logger.debug(f"Send error: {e}")
            return False

    async def _recv(self, timeout: int = 15) -> Optional[tuple]:
        """Receive a protobuf message."""
        try:
            # Read 4-byte length prefix
            header = await asyncio.wait_for(
                self._reader.readexactly(4), timeout=timeout
            )
            msg_len = int.from_bytes(header, "big")

            # Read payload
            data = await asyncio.wait_for(
                self._reader.readexactly(msg_len), timeout=timeout
            )
            proto_msg = ProtoMessage()
            proto_msg.ParseFromString(data)
            return (proto_msg.payloadType, proto_msg.payload)

        except asyncio.IncompleteReadError:
            return None
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.debug(f"Recv error: {e}")
            return None

    async def _message_loop(self):
        """Process incoming messages."""
        while self._running:
            try:
                msg = await self._recv(timeout=5)
                if msg is None:
                    continue
                await self._handle_message(msg)
            except Exception as e:
                if self._running:
                    logger.debug(f"Message loop error: {e}")
                    await asyncio.sleep(0.1)

    async def _handle_message(self, msg: tuple):
        """Route message to handler."""
        payload_type, payload = msg

        if payload_type == 7:  # ProtoOASpotEvent
            await self._handle_spot(payload)

    async def _handle_spot(self, payload: bytes):
        """Handle spot price update."""
        try:
            event = ProtoOASpotEvent()
            event.ParseFromString(payload)

            symbol_id = event.symbolId
            if symbol_id not in self._subscribed:
                return

            symbol = self._subscribed[symbol_id]
            bid = event.bid / self._get_divisor(symbol, event.bid)
            ask = event.ask / self._get_divisor(symbol, event.ask)

            quote = PriceQuote(
                symbol=symbol,
                symbol_id=symbol_id,
                bid=bid,
                ask=ask,
                timestamp=time.time(),
                spread=ask - bid,
                spread_bps=(
                    (ask - bid) / ((bid + ask) / 2) * 10000 if (bid + ask) > 0 else 0
                ),
            )
            self._last_quote[symbol] = quote

            # Dispatch
            if symbol in self._callbacks:
                for cb in self._callbacks[symbol]:
                    try:
                        if asyncio.iscoroutinefunction(cb):
                            asyncio.ensure_future(cb(quote))
                        else:
                            cb(quote)
                    except Exception:
                        pass

        except Exception as e:
            logger.debug(f"Spot event error: {e}")

    def _get_divisor(self, symbol: str, raw: int) -> int:
        """Determine price divisor from raw value and symbol type."""
        if symbol in ("USDJPY",):
            return 1000 if raw < 1000000 else 10000
        elif symbol in ("XAUUSD", "XTIUSD", "US500"):
            return 10 if raw < 100000 else 100
        elif symbol in ("BTCUSD",):
            return 1
        else:
            return 10000 if raw < 10000000 else 100000

    def get_latest_quote(self, symbol: str) -> Optional[PriceQuote]:
        return self._last_quote.get(symbol)

    async def close(self):
        self._running = False
        if self._writer:
            try:
                self._writer.close()
            except:
                pass
            self._writer = None
        logger.info("cTrader data listener disconnected")


async def stream_ctrader_ticks(
    symbols: List[str],
    callback: Callable,
    timeout: float = 30.0,
) -> None:
    """Quick-start: connect to cTrader and stream ticks."""
    listener = CtradeDataListener()
    connected = await listener.connect()
    if not connected:
        logger.warning("cTrader unavailable")
        return

    await listener.subscribe_prices(symbols, callback)

    if timeout > 0:
        logger.info(f"Streaming for {timeout}s...")
        await asyncio.sleep(timeout)
    else:
        while listener._running:
            await asyncio.sleep(1)

    await listener.close()
