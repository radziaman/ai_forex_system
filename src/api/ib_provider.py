"""
Interactive Brokers execution and data provider via ib_insync.
Connects to locally-running TWS or IB Gateway (required).
"""
import asyncio
import logging
from typing import Optional, List, Callable
from datetime import datetime

from .base import (
    ExecutionProvider, DataProvider, AccountInfo, OrderRequest,
    OrderResult, PriceTick, Position, OHLCV,
)

logger = logging.getLogger(__name__)

# Symbol mapping: our format -> IBKR format (IDEALPRO)
IB_SYMBOLS = {
    "EURUSD": ("EUR", "USD"), "GBPUSD": ("GBP", "USD"), "USDJPY": ("USD", "JPY"),
    "AUDUSD": ("AUD", "USD"), "USDCAD": ("USD", "CAD"), "USDCHF": ("USD", "CHF"),
    "NZDUSD": ("NZD", "USD"), "EURJPY": ("EUR", "JPY"), "GBPJPY": ("GBP", "JPY"),
    "EURGBP": ("EUR", "GBP"),
}

BAR_SIZE_MAP = {
    "1m": "1 min", "5m": "5 mins", "15m": "15 mins",
    "30m": "30 mins", "1h": "1 hour", "4h": "4 hours",
    "1d": "1 day", "1w": "1 week",
}


def _ib_contract(symbol: str):
    """Create an IBKR forex contract."""
    from ib_insync.contract import Forex
    parts = IB_SYMBOLS.get(symbol.upper())
    if not parts:
        raise ValueError(f"Symbol {symbol} not supported by IBKR")
    return Forex(parts[0], parts[1], exchange="IDEALPRO")


def _bar_size(timeframe: str) -> str:
    return BAR_SIZE_MAP.get(timeframe, "1 hour")


class IBExecutionProvider(ExecutionProvider, DataProvider):
    """
    Interactive Brokers execution and data via ib_insync.
    Requires TWS or IB Gateway running locally.

    Setup:
        pip install ib_insync
        # Start TWS/Gateway with API enabled on port 7497 (live) or 7496 (paper)
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        self.host = host
        self.port = port
        self.client_id = client_id
        self._ib = None
        self._connected = False
        self._loop = None
        self.on_price: Optional[Callable] = None
        self.on_order_update: Optional[Callable] = None

    async def start(self) -> bool:
        try:
            from ib_insync import IB
            self._ib = IB()
            loop = asyncio.get_event_loop()
            # ib_insync runs in a separate thread
            def _connect():
                self._ib.connect(self.host, self.port, clientId=self.client_id)
                return self._ib.isConnected()
            result = await loop.run_in_executor(None, _connect)
            self._connected = result
            if result:
                logger.info(f"IBKR connected (host={self.host}:{self.port})")
            else:
                logger.error("IBKR connection failed")
            return result
        except Exception as e:
            logger.error(f"IBKR connection error: {e}")
            return False

    async def stop(self):
        if self._ib and self._connected:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._ib.disconnect)
            except Exception:
                pass
        self._connected = False
        logger.info("IBKR stopped")

    def is_connected(self) -> bool:
        return self._connected and (self._ib is not None and self._ib.isConnected())

    # ------------------------------------------------------------
    # Account
    # ------------------------------------------------------------
    async def get_account_info(self) -> Optional[AccountInfo]:
        try:
            loop = asyncio.get_event_loop()
            summary = await loop.run_in_executor(None, self._ib.accountSummary)
            acct = await loop.run_in_executor(None, lambda: self._ib.managedAccounts()[0] if self._ib.managedAccounts() else "")
            info = {}
            for item in summary:
                info[item.tag] = item.value
            return AccountInfo(
                account_id=acct,
                balance=float(info.get("TotalCashValue", 0)),
                equity=float(info.get("NetLiquidation", 0)),
                margin=float(info.get("FullInitMarginReq", 0)),
                free_margin=float(info.get("AvailableFunds", 0)),
                currency=info.get("Currency", "USD"),
                leverage="1:50",
                broker="Interactive Brokers",
            )
        except Exception as e:
            logger.warning(f"IBKR account info error: {e}")
            return None

    # ------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------
    async def place_order(self, order: OrderRequest) -> Optional[OrderResult]:
        try:
            from ib_insync.order import MarketOrder, LimitOrder, StopOrder
            contract = _ib_contract(order.symbol)
            if order.order_type == "MARKET":
                ib_order = MarketOrder(order.side, int(order.volume))
            elif order.order_type == "LIMIT" and order.price:
                ib_order = LimitOrder(order.side, int(order.volume), order.price)
            elif order.order_type == "STOP" and order.price:
                ib_order = StopOrder(order.side, int(order.volume), order.price)
            else:
                ib_order = MarketOrder(order.side, int(order.volume))

            if order.stop_loss or order.take_profit:
                from ib_insync.order import AttachableOrder
                attach = AttachableOrder()
                if order.stop_loss:
                    attach.stopLoss = order.stop_loss
                if order.take_profit:
                    attach.takeProfit = order.take_profit

            loop = asyncio.get_event_loop()
            trade = await loop.run_in_executor(
                None, lambda: self._ib.placeOrder(contract, ib_order)
            )
            if trade and hasattr(trade, 'orderStatus'):
                status = str(trade.orderStatus.status)
                filled = float(trade.orderStatus.filled) if hasattr(trade.orderStatus, 'filled') else 0
                price = float(trade.orderStatus.avgFillPrice) if hasattr(trade.orderStatus, 'avgFillPrice') else 0
                result = OrderResult(
                    order_id=str(trade.order.orderId),
                    position_id=str(trade.order.orderId),
                    status="FILLED" if "Filled" in status else "PENDING",
                    filled_price=price,
                    filled_volume=filled,
                )
                if self.on_order_update:
                    self.on_order_update(result)
                return result
            return OrderResult(status="REJECTED", error="No trade returned")
        except Exception as e:
            logger.warning(f"IBKR order error: {e}")
            return OrderResult(status="REJECTED", error=str(e))

    async def close_position(self, position_id: str) -> Optional[OrderResult]:
        try:
            pid = int(position_id) if position_id.isdigit() else 0
            loop = asyncio.get_event_loop()
            positions = await loop.run_in_executor(None, self._ib.positions)
            for pos in positions:
                if pos.account == position_id or pos.position > 0:
                    side = "SELL" if pos.position > 0 else "BUY"
                    contract = pos.contract
                    from ib_insync.order import MarketOrder
                    mkt = MarketOrder(side, abs(pos.position))
                    trade = await loop.run_in_executor(None, lambda: self._ib.placeOrder(contract, mkt))
                    if trade:
                        return OrderResult(order_id=str(trade.order.orderId), position_id=position_id, status="FILLED")
            return OrderResult(status="REJECTED", error="Position not found")
        except Exception as e:
            return OrderResult(status="REJECTED", error=str(e))

    async def get_positions(self) -> List[Position]:
        try:
            loop = asyncio.get_event_loop()
            positions = await loop.run_in_executor(None, self._ib.positions)
            result = []
            for pos in positions:
                sym = f"{pos.contract.symbol}{pos.contract.currency}"
                result.append(Position(
                    position_id=str(pos.account),
                    symbol=sym, direction="BUY" if pos.position > 0 else "SELL",
                    volume=abs(pos.position),
                    entry_price=float(pos.avgCost),
                    unrealized_pnl=float(pos.unrealizedPNL),
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
        try:
            contract = _ib_contract(symbol)
            duration = "1 M" if not start else None
            if start and end:
                dt_start = datetime.fromisoformat(start) if "T" in start else datetime.strptime(start, "%Y-%m-%d")
                dt_end = datetime.fromisoformat(end) if "T" in end else datetime.strptime(end, "%Y-%m-%d")
                duration_seconds = (dt_end - dt_start).total_seconds()
                days = int(duration_seconds / 86400) + 1
                duration = f"{min(days, 365)} D"

            loop = asyncio.get_event_loop()
            bars = await loop.run_in_executor(
                None, lambda: self._ib.reqHistoricalData(
                    contract, endDateTime="", durationStr=duration or "1 M",
                    barSizeSetting=_bar_size(timeframe),
                    whatToShow="MIDPOINT", useRTH=True,
                )
            )
            result = []
            for bar in bars:
                result.append(OHLCV(
                    timestamp=bar.date.timestamp(),
                    open=bar.open, high=bar.high, low=bar.low,
                    close=bar.close, volume=bar.volume,
                ))
            return result
        except Exception as e:
            logger.warning(f"IBKR historical data error: {e}")
            return []

    # ------------------------------------------------------------
    # Real-time streaming
    # ------------------------------------------------------------
    async def stream_prices(self, symbols: List[str], callback: Optional[Callable] = None):
        cb = callback or self.on_price
        if not cb:
            return
        loop = asyncio.get_event_loop()
        for sym in symbols:
            try:
                contract = _ib_contract(sym)
                def _on_price(ticker):
                    ts = PriceTick(
                        symbol=sym, bid=ticker.bid, ask=ticker.ask,
                        spread=round(ticker.ask - ticker.bid, 5),
                        volume=ticker.size or 0,
                    )
                    cb(ts)
                self._ib.reqMktData(contract, "", False, False)
                self._ib.pendingTickersEvent += _on_price
            except Exception as e:
                logger.warning(f"IBKR stream {sym}: {e}")
