"""
LMAX Global — credential verification only.

LMAX does NOT provide a public REST or WebSocket API for
algorithmic trading. Their only programmatic access is via
the FIX protocol (requires institutional account and certification).

What works:
  ✅ POST /auth/login -> lmax-auth token (credential verification)
  ❌ No REST API for account info, orders, or positions
  ❌ No WebSocket API (endpoints do not exist at any path)

For automated trading with LMAX:
  - Use their FIX API: https://developer.lmax.com/
  - Requires FIX certification and institutional account
"""
import asyncio
import logging
from typing import Optional, List

import requests

from .base import (
    ExecutionProvider, AccountInfo, OrderRequest, OrderResult, Position,
)

logger = logging.getLogger(__name__)

LMAX_LOGIN = "https://web-api.london-demo.lmax.com/auth/login"


class LMAXExecutionProvider(ExecutionProvider):
    """
    LMAX Global — credential verification only.

    LMAX does not have a public programmatic trading API.
    Automated trading requires FIX protocol (institutional).

    This provider validates that your LMAX credentials are correct.
    For actual trading, use cTrader or Dukascopy instead.
    """

    def __init__(self, username: str = "", password: str = "", demo: bool = True):
        self.username = username
        self.password = password
        self._token = ""
        self._connected = False
        self.on_price = None
        self.on_order_update = None

    async def start(self) -> bool:
        try:
            r = await asyncio.get_event_loop().run_in_executor(
                None, lambda: requests.post(
                    LMAX_LOGIN,
                    data={"username": self.username, "password": self.password},
                    allow_redirects=False, timeout=15,
                )
            )
            token = r.cookies.get("lmax-auth", "")
            if r.status_code == 302 and token:
                self._token = token
                self._connected = True
                logger.info("LMAX credentials verified (no trading API available)")
                return True
            logger.warning(f"LMAX login failed: {r.status_code}")
            return False
        except Exception as e:
            logger.warning(f"LMAX connection error: {e}")
            return False

    async def stop(self):
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    async def get_account_info(self) -> Optional[AccountInfo]:
        # LMAX has no REST API for account info
        return AccountInfo(
            account_id="lmax_demo",
            balance=0.0, equity=0.0, margin=0.0, free_margin=0.0,
            currency="USD", broker="LMAX Global",
        )

    async def place_order(self, order: OrderRequest) -> Optional[OrderResult]:
        logger.warning("LMAX has no trading API. Use FIX protocol for automated trading.")
        return OrderResult(status="REJECTED", error="LMAX requires FIX API")

    async def close_position(self, position_id: str) -> Optional[OrderResult]:
        return OrderResult(status="REJECTED", error="LMAX requires FIX API")

    async def get_positions(self) -> List[Position]:
        return []
