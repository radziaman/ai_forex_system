"""
OANDA v20 REST + WebSocket execution and data provider.
API docs: https://developer.oanda.com/rest-live-v20/introduction/
"""
import asyncio
import json
import time
import logging
from datetime import datetime
from typing import Optional, List, Dict, Callable

import requests

from .base import (
    ExecutionProvider, DataProvider, AccountInfo, OrderRequest,
    OrderResult, PriceTick, Position, OHLCV,
)

logger = logging.getLogger(__name__)

OANDA_PRACTICE = "https://api-fxpractice.oanda.com"
OANDA_LIVE = "https://api-fxtrade.oanda.com"
OANDA_STREAM_PRACTICE = "https://stream-fxpractice.oanda.com"
OANDA_STREAM_LIVE = "https://stream-fxtrade.oanda.com"

GRANULARITY_MAP = {
    "1m": "M1", "5m": "M5", "15m": "M15", "30m": "M30",
    "1h": "H1", "4h": "H4", "8h": "H8", "12h": "H12",
    "1d": "D", "1w": "W", "1mo": "M",
}

INVERSE_MAP = {v: k for k, v in GRANULARITY_MAP.items()}

SYMBOL_TO_OANDA = {
    "EURUSD": "EUR_USD", "GBPUSD": "GBP_USD", "USDJPY": "USD_JPY",
    "AUDUSD": "AUD_USD", "USDCAD": "USD_CAD", "USDCHF": "USD_CHF",
    "NZDUSD": "NZD_USD", "EURJPY": "EUR_JPY", "GBPJPY": "GBP_JPY",
    "EURGBP": "EUR_GBP", "XAUUSD": "XAU_USD", "BTCUSD": "BTC_USD",
}
OANDA_TO_SYMBOL = {v: k for k, v in SYMBOL_TO_OANDA.items()}


def _oanda_symbol(symbol: str) -> str:
    return SYMBOL_TO_OANDA.get(symbol.upper(), symbol.upper().replace("=X", "").replace("/", "_"))


def _local_symbol(oanda: str) -> str:
    return OANDA_TO_SYMBOL.get(oanda, oanda.replace("_", ""))


def _parse_datetime(dt_str: str) -> float:
    return datetime.fromisoformat(dt_str.replace("Z", "+00:00")).timestamp()


class OandaExecutionProvider(ExecutionProvider, DataProvider):
    """
    OANDA v20 execution via REST API + WebSocket streaming.

    Usage:
        provider = OandaExecutionProvider(api_key="...", account_id="...", practice=True)
        await provider.start()
        info = await provider.get_account_info()
        result = await provider.place_order(OrderRequest(...))
    """

    def __init__(self, api_key: str = "", account_id: str = "", practice: bool = True):
        self.api_key = api_key
        self.account_id = account_id
        self.base = OANDA_PRACTICE if practice else OANDA_LIVE
        self.stream_base = OANDA_STREAM_PRACTICE if practice else OANDA_STREAM_LIVE
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self._session = requests.Session()
        self._session.headers.update(self._headers)
        self._connected = False
        self._ws_task: Optional[asyncio.Task] = None
        self._price_cache: Dict[str, PriceTick] = {}

        self.on_price: Optional[Callable] = None
        self.on_order_update: Optional[Callable] = None

    # ------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------
    async def start(self) -> bool:
        try:
            resp = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._session.get(f"{self.base}/v3/accounts/{self.account_id}/summary")
            )
            if resp.status_code == 200:
                self._connected = True
                data = resp.json()
                logger.info(f"OANDA connected: {data.get('account', {}).get('currency', 'USD')} account")
                return True
            else:
                logger.error(f"OANDA connection failed: {resp.status_code} {resp.text[:200]}")
                return False
        except Exception as e:
            logger.error(f"OANDA connection error: {e}")
            return False

    async def stop(self):
        self._connected = False
        if self._ws_task:
            self._ws_task.cancel()
            self._ws_task = None
        logger.info("OANDA client stopped")

    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------
    # Account
    # ------------------------------------------------------------
    async def get_account_info(self) -> Optional[AccountInfo]:
        try:
            resp = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._session.get(f"{self.base}/v3/accounts/{self.account_id}/summary")
            )
            if resp.status_code != 200:
                return None
            data = resp.json().get("account", {})
            return AccountInfo(
                account_id=str(data.get("id", self.account_id)),
                balance=float(data.get("balance", 0)),
                equity=float(data.get("NAV", 0)),
                margin=float(data.get("marginUsed", 0)),
                free_margin=float(data.get("marginAvailable", 0)),
                currency=data.get("currency", "USD"),
                leverage=("1:" + str(int(data.get("marginRate", 0.03) * 100))) if data.get("marginRate") else "1:30",
                broker="OANDA",
            )
        except Exception as e:
            logger.warning(f"OANDA get_account_info error: {e}")
            return None

    # ------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------
    async def get_pricing(self, symbols: List[str]) -> Dict[str, PriceTick]:
        instruments = ",".join(_oanda_symbol(s) for s in symbols)
        try:
            resp = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._session.get(
                    f"{self.base}/v3/accounts/{self.account_id}/pricing",
                    params={"instruments": instruments},
                )
            )
            if resp.status_code != 200:
                return {}
            result = {}
            for p in resp.json().get("prices", []):
                sym = _local_symbol(p.get("instrument", ""))
                bt = float(p.get("bids", [{}])[0].get("price", 0)) if p.get("bids") else 0
                at = float(p.get("asks", [{}])[0].get("price", 0)) if p.get("asks") else 0
                tick = PriceTick(
                    symbol=sym, bid=bt, ask=at,
                    spread=round(at - bt, 5) if at and bt else 0.0,
                    timestamp=time.time(),
                )
                result[sym] = tick
                self._price_cache[sym] = tick
                if self.on_price:
                    self.on_price(tick)
            return result
        except Exception as e:
            logger.warning(f"OANDA pricing error: {e}")
            return {}

    # ------------------------------------------------------------
    # Historical data
    # ------------------------------------------------------------
    async def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1h",
        start: str = "", end: str = "",
    ) -> List[OHLCV]:
        oanda_sym = _oanda_symbol(symbol)
        gran = GRANULARITY_MAP.get(timeframe, "H1")
        params = {"instrument": oanda_sym, "granularity": gran, "price": "MBA"}
        if start:
            dt = datetime.fromisoformat(start) if "T" in start else datetime.strptime(start, "%Y-%m-%d")
            params["from"] = dt.isoformat() + "Z"
        if end:
            dt = datetime.fromisoformat(end) if "T" in end else datetime.strptime(end, "%Y-%m-%d")
            params["to"] = dt.isoformat() + "Z"
        try:
            resp = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._session.get(
                    f"{self.base}/v3/accounts/{self.account_id}/candles",
                    params=params,
                )
            )
            if resp.status_code != 200:
                logger.warning(f"OANDA candles error: {resp.status_code}")
                return []
            result = []
            for c in resp.json().get("candles", []):
                if c.get("complete", False):
                    mid = c.get("mid", {})
                    result.append(OHLCV(
                        timestamp=_parse_datetime(c["time"]),
                        open=float(mid.get("o", 0)),
                        high=float(mid.get("h", 0)),
                        low=float(mid.get("l", 0)),
                        close=float(mid.get("c", 0)),
                        volume=float(c.get("volume", 0)),
                    ))
            return result
        except Exception as e:
            logger.warning(f"OANDA historical error: {e}")
            return []

    # ------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------
    async def place_order(self, order: OrderRequest) -> Optional[OrderResult]:
        oanda_sym = _oanda_symbol(order.symbol)
        units = int(order.volume)
        if order.side == "SELL":
            units = -abs(units)
        body = {
            "order": {
                "type": "MARKET",
                "instrument": oanda_sym,
                "units": str(units),
            }
        }
        if order.stop_loss:
            body["order"]["stopLossOnFill"] = {"price": str(round(order.stop_loss, 5))}
        if order.take_profit:
            body["order"]["takeProfitOnFill"] = {"price": str(round(order.take_profit, 5))}
        try:
            resp = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._session.post(
                    f"{self.base}/v3/accounts/{self.account_id}/orders",
                    json=body,
                )
            )
            if resp.status_code in (200, 201):
                data = resp.json()
                fill = data.get("orderFillTransaction", data.get("orderCreateTransaction", {}))
                ord_id = fill.get("id", "")
                price = float(data.get("orderFillTransaction", {}).get("price", 0))
                result = OrderResult(
                    order_id=ord_id,
                    position_id=data.get("orderFillTransaction", {}).get("tradeOpened", {}).get("tradeID", ord_id),
                    status="FILLED",
                    filled_price=price,
                    filled_volume=float(data.get("orderFillTransaction", {}).get("units", 0)),
                )
                if self.on_order_update:
                    self.on_order_update(result)
                return result
            else:
                err = resp.json().get("errorCode", "Unknown")
                logger.warning(f"OANDA order rejected: {err}")
                return OrderResult(status="REJECTED", error=err)
        except Exception as e:
            logger.warning(f"OANDA order error: {e}")
            return OrderResult(status="REJECTED", error=str(e))

    async def close_position(self, position_id: str) -> Optional[OrderResult]:
        try:
            resp = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._session.put(
                    f"{self.base}/v3/accounts/{self.account_id}/trades/{position_id}/close",
                    json={},
                )
            )
            if resp.status_code in (200, 201):
                data = resp.json()
                return OrderResult(
                    order_id=data.get("orderFillTransaction", {}).get("id", ""),
                    position_id=position_id,
                    status="FILLED",
                    filled_price=float(data.get("orderFillTransaction", {}).get("price", 0)),
                )
            return OrderResult(status="REJECTED", error=resp.text[:200])
        except Exception as e:
            return OrderResult(status="REJECTED", error=str(e))

    async def get_positions(self) -> List[Position]:
        try:
            resp = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._session.get(
                    f"{self.base}/v3/accounts/{self.account_id}/openPositions"
                )
            )
            if resp.status_code != 200:
                return []
            result = []
            for p in resp.json().get("positions", []):
                sym = _local_symbol(p.get("instrument", ""))
                long_info = p.get("long", {})
                short_info = p.get("short", {})
                if long_info.get("units", "0") != "0":
                    result.append(Position(
                        symbol=sym, direction="BUY",
                        volume=float(long_info["units"]),
                        entry_price=float(long_info.get("averagePrice", 0)),
                        unrealized_pnl=float(long_info.get("unrealizedPL", 0)),
                    ))
                if short_info.get("units", "0") != "0":
                    result.append(Position(
                        symbol=sym, direction="SELL",
                        volume=abs(float(short_info["units"])),
                        entry_price=float(short_info.get("averagePrice", 0)),
                        unrealized_pnl=float(short_info.get("unrealizedPL", 0)),
                    ))
            return result
        except Exception as e:
            logger.warning(f"OANDA positions error: {e}")
            return []

    # ------------------------------------------------------------
    # WebSocket streaming
    # ------------------------------------------------------------
    async def stream_prices(self, symbols: List[str], callback: Optional[Callable] = None):
        instruments = ",".join(_oanda_symbol(s) for s in symbols)
        url = f"{self.stream_base}/v3/accounts/{self.account_id}/pricing/stream"
        headers = {**self._headers, "Accept": "text/event-stream"}

        def _stream():
            try:
                resp = requests.get(url, headers=headers, params={"instruments": instruments}, stream=True)
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line.decode("utf-8"))
                        if "type" in data and data["type"] == "PRICE":
                            sym = _local_symbol(data.get("instrument", ""))
                            bt = float(data.get("bids", [{}])[0].get("price", 0))
                            at = float(data.get("asks", [{}])[0].get("price", 0))
                            tick = PriceTick(
                                symbol=sym, bid=bt, ask=at,
                                spread=round(at - bt, 5),
                                timestamp=_parse_datetime(data.get("time", "")) if data.get("time") else time.time(),
                            )
                            cb = callback or self.on_price
                            if cb:
                                cb(tick)
                    except json.JSONDecodeError:
                        continue
            except Exception as e:
                logger.warning(f"OANDA stream ended: {e}")

        self._ws_task = asyncio.create_task(
            asyncio.get_event_loop().run_in_executor(None, _stream)
        )
