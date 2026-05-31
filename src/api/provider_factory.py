"""
Provider factory — auto-selects the right execution and data providers based on TRADING_PROVIDER.  # noqa: E501
"""

from typing import Tuple, Optional
from loguru import logger

from infrastructure.secrets import Secrets
from api.base import (
    ExecutionProvider,
    DataProvider,
    AccountInfo,
    OrderRequest,
    OrderResult,
)


class CtraderExecutionAdapter(ExecutionProvider):
    """Wraps CtraderClient to conform to the ExecutionProvider ABC."""

    def __init__(self, secrets: Secrets):
        from api.ctrader_client import CtraderClient

        self._client = CtraderClient(
            app_id=secrets.ctrader_app_id,
            app_secret=secrets.ctrader_app_secret,
            access_token=secrets.ctrader_access_token,
            account_id=int(secrets.ctrader_account_id),
            demo=secrets.is_demo,
        )
        self._client.on_market_data = self._on_market_data
        self.on_price = None
        self.on_order_update = None

    async def _on_market_data(self, depth):
        if self.on_price:
            from api.base import PriceTick

            result = self.on_price(
                PriceTick(
                    symbol=depth.symbol,
                    bid=depth.bid,
                    ask=depth.ask,
                    spread=depth.spread,
                    volume=depth.volume,
                )
            )
            if hasattr(result, "__await__"):
                await result

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
            balance=float(info.balance),
            equity=float(info.equity),
            margin=float(info.margin),
            free_margin=float(info.free_margin),
            currency="USD",
            broker="IC Markets (cTrader)",
        )

    async def place_order(self, order: OrderRequest) -> Optional[OrderResult]:
        from api.symbol_map import get_symbol_id

        sid = get_symbol_id(order.symbol)
        result = await self._client.open_position(
            symbol_id=sid,
            order_type=order.side,
            volume=order.volume,
            stop_loss_price=order.stop_loss or 0.0,
            take_profit_price=order.take_profit or 0.0,
            comment="RTS Agentic",
        )
        if result is None:
            return None
        return OrderResult(
            order_id=str(result.get("orderId", 0)),
            position_id=str(result.get("positionId", 0)),
            status=(
                "FILLED" if result.get("executionType") == "ORDER_FILLED" else "PENDING"
            ),
            filled_price=float(result.get("price", 0)),
            filled_volume=order.volume,
            error="",
        )

    async def close_position(self, position_id: str) -> Optional[OrderResult]:
        try:
            pid = int(position_id) if position_id.isdigit() else 0
            if pid > 0:
                ok = await self._client.close_position(pid, volume=0)
                if ok:
                    return OrderResult(
                        order_id="0", position_id=position_id, status="FILLED"
                    )
        except Exception as e:
            logger.warning(f"close_position failed: {e}")
        return OrderResult(order_id="0", position_id=position_id, status="FILLED")

    async def get_positions(self) -> list:
        return await self._client.get_open_positions()

    def is_connected(self) -> bool:
        return self._client.is_connected()


def create_execution_provider(
    secrets: Secrets,
) -> Tuple[ExecutionProvider, Optional[DataProvider]]:
    provider = secrets.provider

    logger.info(f"Provider: cTrader Open API ({provider})")

    if provider == "dukascopy":
        from data.dukascopy_realtime import DukascopyProvider

        return CtraderExecutionAdapter(secrets), DukascopyProvider(
            cache=True, poll_interval=1.0
        )

    return CtraderExecutionAdapter(secrets), None
