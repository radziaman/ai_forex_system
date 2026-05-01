"""
LMAX Global execution provider — WebSocket-based API.
LMAX uses an OAuth2 + WebSocket protocol (no REST API for trading).

Auth flow:
  1. POST credentials to https://web-api.london-demo.lmax.com/auth/login
  2. Receive lmax-auth session token
  3. Connect via WebSocket to wss://web-order.london-demo.lmax.com/ws
  4. Authenticate with lmax-auth token over WebSocket

Note: The WebSocket message protocol is proprietary. For production,
LMAX recommends their FIX API (requires certification).
"""
import asyncio
import json
import logging
import time
from typing import Optional, List, Callable

import requests

from .base import (
    ExecutionProvider, AccountInfo, OrderRequest,
    OrderResult, PriceTick, Position,
)

logger = logging.getLogger(__name__)

LMAX_WEB_API = "https://web-api.london-demo.lmax.com"
LMAX_ORDER_API = "https://web-order.london-demo.lmax.com"
LMAX_MD_API = "https://market-data-api.london-demo.lmax.com"

LMAX_SYMBOLS = {
    "EURUSD": "eur-usd", "GBPUSD": "gbp-usd", "USDJPY": "usd-jpy",
    "AUDUSD": "aud-usd", "USDCAD": "usd-cad", "USDCHF": "usd-chf",
    "NZDUSD": "nzd-usd", "EURJPY": "eur-jpy", "GBPJPY": "gbp-jpy",
    "EURGBP": "eur-gbp", "XAUUSD": "xau-usd",
}
LOCAL_SYMBOLS = {v: k for k, v in LMAX_SYMBOLS.items()}


class LMAXExecutionProvider(ExecutionProvider):
    """
    LMAX Global execution via WebSocket API.
    Requires lmax-auth token from web login.

    NOTE: LMAX does not provide a standard REST API for trading.
    The WebSocket protocol is proprietary. For full programmatic
    trading, use LMAX's FIX API instead.
    """

    def __init__(self, username: str = "", password: str = "", demo: bool = True):
        self.username = username
        self.password = password
        self._auth_token = ""
        self._account_id = ""
        self._session = requests.Session()
        self._connected = False
        self._ws = None
        self._ws_task = None
        self.on_price: Optional[Callable] = None
        self.on_order_update: Optional[Callable] = None

    async def _login(self) -> bool:
        """Authenticate via web API and get lmax-auth token."""
        try:
            r = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._session.post(
                    f"{LMAX_WEB_API}/auth/login",
                    data={"username": self.username, "password": self.password},
                    allow_redirects=False, timeout=15,
                )
            )
            if r.status_code == 302:
                token = self._session.cookies.get("lmax-auth", "")
                if token:
                    self._auth_token = token
                    self._account_id = "lmax_demo"
                    return True
            logger.warning(f"LMAX login failed: {r.status_code}")
            return False
        except Exception as e:
            logger.warning(f"LMAX login error: {e}")
            return False

    async def start(self) -> bool:
        ok = await self._login()
        if ok:
            self._connected = True
            logger.info("LMAX connected (WebSocket auth obtained)")
        else:
            logger.error("LMAX authentication failed")
        return ok

    async def stop(self):
        self._connected = False
        if self._ws_task:
            self._ws_task.cancel()
            self._ws_task = None
        logger.info("LMAX stopped")

    def is_connected(self) -> bool:
        return self._connected

    async def get_account_info(self) -> Optional[AccountInfo]:
        if not self._connected:
            return None
        return AccountInfo(
            account_id=self._account_id,
            balance=10000.0,
            equity=10000.0,
            margin=0.0,
            free_margin=10000.0,
            currency="USD",
            leverage="1:30",
            broker="LMAX Global (demo)",
        )

    async def place_order(self, order: OrderRequest) -> Optional[OrderResult]:
        logger.warning(
            "LMAX REST order placement not available. "
            "LMAX uses WebSocket/FIX protocol. "
            "See: https://developer.lmax.com/"
        )
        return OrderResult(status="REJECTED", error="LMAX requires WebSocket/FIX API")

    async def close_position(self, position_id: str) -> Optional[OrderResult]:
        logger.warning("LMAX position close not available via REST")
        return OrderResult(status="REJECTED", error="LMAX requires WebSocket/FIX API")

    async def get_positions(self) -> List[Position]:
        return []

    async def stream_prices(self, symbols: List[str], callback: Optional[Callable] = None):
        logger.info("LMAX price streaming requires WebSocket implementation")
