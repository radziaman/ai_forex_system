"""
LMAX Global REST API execution and data provider.
API docs: https://developer.lmax.com/
"""
import asyncio
import base64
import logging
import time  # noqa
from typing import Optional, List, Callable
from datetime import datetime  # noqa

import requests

from .base import (
    ExecutionProvider, AccountInfo, OrderRequest,
    OrderResult, PriceTick, Position,
)

logger = logging.getLogger(__name__)

LMAX_LIVE = "https://api.lmax.com"
LMAX_DEMO = "https://api.lmax.com/demo"
LMAX_STREAM = "https://stream.lmax.com"

LMAX_SYMBOLS = {
    "EURUSD": "eur-usd", "GBPUSD": "gbp-usd", "USDJPY": "usd-jpy",
    "AUDUSD": "aud-usd", "USDCAD": "usd-cad", "USDCHF": "usd-chf",
    "NZDUSD": "nzd-usd", "EURJPY": "eur-jpy", "GBPJPY": "gbp-jpy",
    "EURGBP": "eur-gbp", "XAUUSD": "xau-usd",
}
LOCAL_SYMBOLS = {v: k for k, v in LMAX_SYMBOLS.items()}


class LMAXExecutionProvider(ExecutionProvider):
    """
    LMAX Global REST API execution provider.
    Direct market access (DMA), true ECN/STP.

    Setup:
        Get API credentials from LMAX account management
        Demo: https://demo.lmax.com
    """

    def __init__(self, username: str = "", password: str = "", demo: bool = True):
        self.username = username
        self.password = password
        self.base = LMAX_DEMO if demo else LMAX_LIVE
        self._auth_token = ""
        self._account_id = ""
        self._session = requests.Session()
        self._connected = False
        self.on_price: Optional[Callable] = None
        self.on_order_update: Optional[Callable] = None

    async def _authenticate(self) -> bool:
        """Authenticate and get bearer token."""
        auth_str = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
        try:
            resp = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._session.post(
                    f"{self.base}/api/authenticate",
                    params={"authorization": auth_str},
                )
            )
            if resp.status_code == 200:
                data = resp.json()
                self._auth_token = data.get("token", "")
                self._account_id = str(data.get("accountId", ""))
                self._session.headers.update({"Authorization": f"Bearer {self._auth_token}"})
                return True
            return False
        except Exception:
            return False

    async def start(self) -> bool:
        success = await self._authenticate()
        if success:
            self._connected = True
            logger.info("LMAX connected")
        else:
            logger.error("LMAX authentication failed")
        return success

    async def stop(self):
        self._connected = False
        logger.info("LMAX stopped")

    def is_connected(self) -> bool:
        return self._connected

    async def get_account_info(self) -> Optional[AccountInfo]:
        try:
            resp = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._session.get(
                    f"{self.base}/api/accounts/{self._account_id}"
                )
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            return AccountInfo(
                account_id=self._account_id,
                balance=float(data.get("balance", 0)),
                equity=float(data.get("equity", 0)),
                margin=float(data.get("marginRequirement", 0)),
                free_margin=max(0, float(data.get("cashAvailableToTrade", 0))),
                currency=data.get("currency", "USD"),
                leverage="1:30",
                broker="LMAX Global",
            )
        except Exception as e:
            logger.warning(f"LMAX account error: {e}")
            return None

    async def place_order(self, order: OrderRequest) -> Optional[OrderResult]:
        lmax_sym = LMAX_SYMBOLS.get(order.symbol.upper(), order.symbol)
        body = {
            "accountId": self._account_id,
            "instrument": lmax_sym,
            "side": order.side.upper(),
            "quantity": str(order.volume),
            "orderType": "MARKET" if order.order_type == "MARKET" else "LIMIT",
        }
        if order.price and order.order_type != "MARKET":
            body["price"] = str(order.price)
        if order.stop_loss:
            body["stopLoss"] = str(order.stop_loss)
        if order.take_profit:
            body["takeProfit"] = str(order.take_profit)
        try:
            resp = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._session.post(
                    f"{self.base}/api/orders", json=body
                )
            )
            if resp.status_code in (200, 201):
                data = resp.json()
                result = OrderResult(
                    order_id=str(data.get("orderId", "")),
                    position_id=str(data.get("positionId", "")),
                    status="FILLED",
                    filled_price=float(data.get("price", 0)),
                )
                if self.on_order_update:
                    self.on_order_update(result)
                return result
            else:
                return OrderResult(status="REJECTED", error=resp.text[:200])
        except Exception as e:
            return OrderResult(status="REJECTED", error=str(e))

    async def close_position(self, position_id: str) -> Optional[OrderResult]:
        try:
            resp = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._session.delete(
                    f"{self.base}/api/positions/{position_id}"
                )
            )
            if resp.status_code in (200, 201, 204):
                return OrderResult(order_id="", position_id=position_id, status="FILLED")
            return OrderResult(status="REJECTED", error=resp.text[:200])
        except Exception as e:
            return OrderResult(status="REJECTED", error=str(e))

    async def get_positions(self) -> List[Position]:
        try:
            resp = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._session.get(
                    f"{self.base}/api/accounts/{self._account_id}/positions"
                )
            )
            if resp.status_code != 200:
                return []
            result = []
            for p in resp.json().get("positions", []):
                sym = LOCAL_SYMBOLS.get(p.get("instrument", ""), p.get("instrument", ""))
                result.append(Position(
                    position_id=str(p.get("positionId", "")),
                    symbol=sym,
                    direction=p.get("side", ""),
                    volume=float(abs(p.get("quantity", 0))),
                    entry_price=float(p.get("openPrice", 0)),
                    unrealized_pnl=float(p.get("unrealizedPnl", 0)),
                ))
            return result
        except Exception:
            return []
