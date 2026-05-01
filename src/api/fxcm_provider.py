"""
FXCM REST API execution and data provider.
API docs: https://fxcm.github.io/rest-api/
"""
import asyncio
import logging
import time
from typing import Optional, List, Callable

import requests

from .base import (
    ExecutionProvider, DataProvider, AccountInfo, OrderRequest,
    OrderResult, PriceTick, Position, OHLCV,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://api.fxcm.com"
FXCM_SYMBOLS = {
    "EURUSD": "EUR/USD", "GBPUSD": "GBP/USD", "USDJPY": "USD/JPY",
    "AUDUSD": "AUD/USD", "USDCAD": "USD/CAD", "USDCHF": "USD/CHF",
    "NZDUSD": "NZD/USD", "EURJPY": "EUR/JPY", "GBPJPY": "GBP/JPY",
    "EURGBP": "EUR/GBP",
}
LOCAL_SYMBOLS = {v: k for k, v in FXCM_SYMBOLS.items()}
PIP_MAP = {"JPY": 0.01}
DEFAULT_PIP = 0.0001


class FXCMExecutionProvider(ExecutionProvider, DataProvider):
    """
    FXCM REST API execution and data provider.

    Setup:
        Get API key from FXCM App Store Portal: https://fxcm.com/apps
        Use demo account for testing: https://demo.fxcm.com
    """

    def __init__(self, access_token: str = "", account_id: str = "", demo: bool = True):
        self.access_token = access_token
        self.account_id = account_id
        self.base = "https://api-demo.fxcm.com" if demo else BASE_URL
        self._headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        }
        self._session = requests.Session()
        self._session.headers.update(self._headers)
        self._connected = False
        self.on_price: Optional[Callable] = None
        self.on_order_update: Optional[Callable] = None

    async def start(self) -> bool:
        try:
            resp = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._session.get(f"{self.base}/trading/v1/accounts")
            )
            if resp.status_code == 200:
                self._connected = True
                logger.info("FXCM connected")
                return True
            else:
                logger.error(f"FXCM connection failed: {resp.status_code} {resp.text[:200]}")
                return False
        except Exception as e:
            logger.error(f"FXCM connection error: {e}")
            return False

    async def stop(self):
        self._connected = False
        logger.info("FXCM stopped")

    def is_connected(self) -> bool:
        return self._connected

    async def get_account_info(self) -> Optional[AccountInfo]:
        try:
            resp = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._session.get(f"{self.base}/trading/v1/accounts/{self.account_id}")
            )
            if resp.status_code != 200:
                return None
            data = resp.json().get("accounts", [{}])[0]
            return AccountInfo(
                account_id=str(data.get("accountId", self.account_id)),
                balance=float(data.get("balance", 0)),
                equity=float(data.get("equity", 0)),
                margin=float(data.get("margin", 0)),
                free_margin=float(data.get("usableMargin", 0)),
                leverage=f"1:{data.get('leverage', 50)}",
                broker="FXCM",
            )
        except Exception as e:
            logger.warning(f"FXCM account error: {e}")
            return None

    async def place_order(self, order: OrderRequest) -> Optional[OrderResult]:
        fxcm_sym = FXCM_SYMBOLS.get(order.symbol.upper(), order.symbol)
        body = {
            "accountId": self.account_id,
            "instrument": fxcm_sym,
            "amount": int(order.volume),
            "isBuy": order.side.upper() == "BUY",
            "orderType": "AtMarket",
            "timeInForce": "IOC",
        }
        if order.stop_loss:
            body["stopLoss"] = order.stop_loss
        if order.take_profit:
            body["takeProfit"] = order.take_profit
        try:
            resp = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._session.post(
                    f"{self.base}/trading/v1/orders", json=body
                )
            )
            if resp.status_code in (200, 201):
                data = resp.json()
                result = OrderResult(
                    order_id=str(data.get("orderId", "")),
                    position_id=str(data.get("tradeId", "")),
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
                    f"{self.base}/trading/v1/positions/{position_id}"
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
                    f"{self.base}/trading/v1/positions?accountId={self.account_id}"
                )
            )
            if resp.status_code != 200:
                return []
            result = []
            for p in resp.json().get("positions", []):
                sym = LOCAL_SYMBOLS.get(p.get("instrument", ""), p.get("instrument", ""))
                result.append(Position(
                    position_id=str(p.get("tradeId", "")),
                    symbol=sym,
                    direction="BUY" if p.get("isBuy") else "SELL",
                    volume=float(abs(p.get("amount", 0))),
                    entry_price=float(p.get("openPrice", 0)),
                    unrealized_pnl=float(p.get("unrealizedPL", 0)),
                ))
            return result
        except Exception:
            return []

    # ------------------------------------------------------------
    # Historical data
    # ------------------------------------------------------------
    async def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1h",
        start: str = "", end: str = "",
    ) -> List[OHLCV]:
        fxcm_sym = FXCM_SYMBOLS.get(symbol.upper(), symbol)
        period_map = {"1m": "m1", "5m": "m5", "15m": "m15", "30m": "m30",
                      "1h": "H1", "4h": "H4", "1d": "D1"}
        period = period_map.get(timeframe, "H1")
        params = {"instrument": fxcm_sym, "period": period, "number": 500}
        try:
            resp = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._session.get(
                    f"{self.base}/trading/v1/candles", params=params
                )
            )
            if resp.status_code != 200:
                return []
            result = []
            for c in resp.json().get("candles", []):
                result.append(OHLCV(
                    timestamp=c.get("time", 0) / 1000,
                    open=float(c.get("open", 0)),
                    high=float(c.get("high", 0)),
                    low=float(c.get("low", 0)),
                    close=float(c.get("close", 0)),
                    volume=float(c.get("volume", 0)),
                ))
            return result
        except Exception as e:
            logger.warning(f"FXCM historical error: {e}")
            return []

    async def stream_prices(self, symbols: List[str], callback: Optional[Callable] = None):
        cb = callback or self.on_price
        if not cb:
            return
        instruments = ",".join(FXCM_SYMBOLS.get(s.upper(), s) for s in symbols)
        url = f"{self.base}/trading/v1/subscribe?instruments={instruments}"

            def _subscribe():
            try:
                requests.post(url, headers=self._headers, stream=False)
                for sym in symbols:
                    tick = PriceTick(
                        symbol=sym, bid=0.0, ask=0.0, spread=0.0,
                        timestamp=time.time(),
                    )
                    cb(tick)
            except Exception as e:
                logger.warning(f"FXCM stream error: {e}")

        asyncio.create_task(
            asyncio.get_event_loop().run_in_executor(None, _subscribe)
        )
