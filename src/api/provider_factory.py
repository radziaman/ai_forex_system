"""
Provider factory — auto-selects the right execution and data providers based on TRADING_PROVIDER.
"""
import logging
from typing import Tuple, Optional

from infrastructure.secrets import Secrets
from api.base import ExecutionProvider, DataProvider, AccountInfo, OrderRequest, OrderResult

logger = logging.getLogger(__name__)


class CtraderExecutionAdapter(ExecutionProvider):
    """Wraps CtraderClient to conform to the ExecutionProvider ABC."""

    def __init__(self, secrets: Secrets):
        from api.ctrader_client import CtraderClient
        self._client = CtraderClient(
            app_id=secrets.ctrader_app_id,
            app_secret=secrets.ctrader_app_secret,
            access_token=secrets.ctrader_access_token,
            account_id=secrets.ctrader_account_id,
            demo=secrets.is_demo,
        )
        self._client.on_market_data = self._on_market_data
        self.on_price = None
        self.on_order_update = None

    def _on_market_data(self, depth):
        if self.on_price:
            from api.base import PriceTick
            self.on_price(PriceTick(
                symbol=depth.symbol, bid=depth.bid, ask=depth.ask,
                spread=depth.spread, volume=depth.volume,
            ))

    @property
    def raw(self):
        return self._client

    async def start(self) -> bool:
        return await self._client.start()

    async def stop(self):
        await self._client.disconnect()

    async def get_account_info(self) -> Optional[AccountInfo]:
        info = self._client.get_account_info()
        if info is None:
            return None
        return AccountInfo(
            account_id=str(info.ctid_trader_account_id),
            balance=info.balance, equity=info.equity,
            margin=info.margin, free_margin=info.free_margin,
            currency="USD", broker="IC Markets (cTrader)",
        )

    async def place_order(self, order: OrderRequest) -> Optional[OrderResult]:
        from api.ctrader_client import TradeOrder
        trade_order = TradeOrder(
            symbol=order.symbol,
            symbol_id={"EURUSD": 1, "GBPUSD": 2, "USDJPY": 3, "XAUUSD": 4, "BTCUSD": 5}.get(order.symbol.upper(), 1),
            side=order.side, order_type=order.order_type,
            volume=int(order.volume),
            price=order.price, sl=order.stop_loss, tp=order.take_profit,
        )
        result = await self._client.place_order(trade_order)
        if result is None:
            return None
        or_ = OrderResult(
            order_id=str(result.order_id),
            position_id=str(result.position_id),
            status=result.status,
            filled_price=result.filled_price,
            filled_volume=order.volume,
            error=result.error,
        )
        if self.on_order_update:
            self.on_order_update(or_)
        return or_

    async def close_position(self, position_id: str) -> Optional[OrderResult]:
        pid = int(position_id) if position_id.isdigit() else 0
        if pid in self._client._open_positions:
            # Simulate close
            return OrderResult(order_id="0", position_id=position_id, status="FILLED")
        return OrderResult(status="REJECTED", error="Position not found")

    async def get_positions(self) -> list:
        return []

    def is_connected(self) -> bool:
        return self._client.is_connected()


def create_execution_provider(secrets: Secrets) -> Tuple[ExecutionProvider, Optional[DataProvider]]:
    provider = secrets.provider

    if provider == "dukascopy":
        logger.info("Provider: Dukascopy data + cTrader execution")
        from data.dukascopy_provider import DukascopyDataProvider
        return CtraderExecutionAdapter(secrets), DukascopyDataProvider()

    else:
        logger.info("Provider: cTrader Open API (default)")
        return CtraderExecutionAdapter(secrets), None
