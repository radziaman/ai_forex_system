import asyncio
import time
from loguru import logger
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from api.ctrader_client import TradeOrder
from data.data_manager import BASE_PRICES


@dataclass
class TradeRecord:
    timestamp: float
    symbol: str
    direction: str
    volume: float
    entry_price: float
    exit_price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    pnl: float = 0.0
    status: str = "OPEN"
    reason: str = ""
    position_id: int = 0


class ExecutionEngine:
    def __init__(
        self,
        ctrader_client,
        risk_manager,
        data_manager,
        initial_balance: float = 100000.0,
        mode: str = "PAPER",
    ):
        self.client = ctrader_client
        self.risk = risk_manager
        self.data = data_manager
        self.mode = mode
        self.open_positions: Dict[int, TradeRecord] = {}
        self.trade_history: List[TradeRecord] = []
        self.total_trades = 0
        self._position_counter = 0
        self._balance = initial_balance
        self._peak_balance = initial_balance
        self.client.on_market_data = self._on_market_data
        self._close_lock = asyncio.Lock()
        self._wire_order_update()

    @property
    def on_market_data(self):
        return self.client.on_market_data

    @on_market_data.setter
    def on_market_data(self, callback):
        self.client.on_market_data = callback

    @property
    def raw(self):
        return self.client

    def _wire_order_update(self):
        self.client.on_order_update = self._on_order_update

    def _get_default_price(self, symbol: str) -> float:
        return BASE_PRICES.get(symbol.upper(), 1.12)

    def _live_price(self, symbol: str) -> float:
        if self.data is None:
            return 0.0
        try:
            tick = self.data.get_tick_buffer(symbol, 1)
            if tick and len(tick) > 0:
                last = tick[-1]
                if isinstance(last, dict):
                    mid = last.get("mid") or last.get("bid") or last.get("price", 0)
                    if mid > 0:
                        return float(mid)
            return self.data.get_price(symbol, "1h")
        except Exception:
            return 0.0

    def _pnl_usd(
        self, entry: float, exit: float, direction: str, volume: float, symbol: str
    ) -> float:
        if exit == entry or exit == 0 or entry == 0:
            return 0.0
        mult = 1 if direction == "BUY" else -1
        diff = (exit - entry) * mult * volume
        sym = symbol.upper()
        if sym in ("USDJPY", "EURJPY", "GBPJPY"):
            usdjpy = self._live_price("USDJPY") or 100.0
            return round(diff / usdjpy, 2)
        if sym in ("USDCHF",):
            usdchf = self._live_price("USDCHF") or 0.88
            return round(diff / usdchf, 2)
        if sym in ("USDCAD",):
            usdcad = self._live_price("USDCAD") or 1.35
            return round(diff / usdcad, 2)
        if sym == "EURGBP":
            gbpusd = self._live_price("GBPUSD") or 1.2
            return round(diff * gbpusd, 2)
        return round(diff, 2)

    async def open_position(
        self, symbol, direction, volume, sl, tp, reason="AI signal"
    ):
        if self.risk and self.risk.kill_switch_triggered:
            return None
        if self.risk and self.risk.mode == "PAPER":
            return self._simulate_open(symbol, direction, volume, sl, tp, reason)
        price = self._live_price(symbol)
        if price <= 0:
            price = self._get_default_price(symbol)
        order = TradeOrder(
            symbol=symbol,
            symbol_id=self._get_symbol_id(symbol),
            side="BUY" if direction == "BUY" else "SELL",
            order_type="MARKET",
            volume=max(int(volume), 1),
            price=price,
            sl=sl,
            tp=tp,
        )
        try:
            result = await self.client.place_order(order)
            if result and result.status == "FILLED":
                self._position_counter += 1
                trade = TradeRecord(
                    timestamp=time.time(),
                    symbol=symbol,
                    direction=direction,
                    volume=volume,
                    entry_price=result.filled_price or price,
                    sl=sl,
                    tp=tp,
                    status="OPEN",
                    reason=reason,
                    position_id=self._position_counter,
                )
                self.open_positions[trade.position_id] = trade
                self.total_trades += 1
                logger.success(
                    f"OPENED: {direction} {volume:.0f} {symbol} @ {trade.entry_price:.5f}"
                )
                return trade
        except Exception as e:
            logger.error(f"Order error: {e}")
        return None

    def _simulate_open(self, symbol, direction, volume, sl, tp, reason):
        price = self._live_price(symbol) or self._get_default_price(symbol)
        self._position_counter += 1
        trade = TradeRecord(
            timestamp=time.time(),
            symbol=symbol,
            direction=direction,
            volume=volume,
            entry_price=price,
            sl=sl,
            tp=tp,
            status="OPEN",
            reason=reason,
            position_id=self._position_counter,
        )
        self.open_positions[trade.position_id] = trade
        self.total_trades += 1
        logger.info(f"[PAPER] OPENED: {direction} {volume:.0f} {symbol} @ {price:.5f}")
        return trade

    async def close_position(
        self, position_id, reason="AI close", exit_price: float = None
    ):
        if position_id not in self.open_positions:
            return False
        trade = self.open_positions[position_id]
        if exit_price is None or exit_price <= 0:
            exit_price = self._live_price(trade.symbol) or trade.entry_price
        pnl = self._pnl_usd(
            trade.entry_price, exit_price, trade.direction, trade.volume, trade.symbol
        )
        if self.risk and self.risk.mode == "PAPER":
            return self._simulate_close(trade, reason, exit_price, pnl)
        close_side = "SELL" if trade.direction == "BUY" else "BUY"
        order = TradeOrder(
            symbol=trade.symbol,
            symbol_id=self._get_symbol_id(trade.symbol),
            side=close_side,
            order_type="MARKET",
            volume=int(trade.volume),
            position_id=position_id,
        )
        try:
            result = await self.client.place_order(order)
            if result and result.status == "FILLED":
                exit_price = result.filled_price or exit_price
                pnl = self._pnl_usd(
                    trade.entry_price,
                    exit_price,
                    trade.direction,
                    trade.volume,
                    trade.symbol,
                )
                self._finalize_close(trade, exit_price, pnl, reason)
                return True
        except Exception as e:
            logger.error(f"Close error: {e}")
        return False

    def _simulate_close(self, trade, reason, exit_price, pnl):
        self._finalize_close(trade, exit_price, pnl, reason)
        logger.info(f"[PAPER] CLOSED: {trade.symbol} {reason} | PnL: ${pnl:.2f}")
        return True

    def _finalize_close(self, trade, exit_price, pnl, reason):
        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.status = "CLOSED"
        trade.reason = reason
        self.trade_history.append(trade)
        del self.open_positions[trade.position_id]
        self._balance += pnl
        if self.risk:
            self.risk.record_trade(trade, exit_price, pnl)

    async def close_all_positions(self, reason="AI close all"):
        async with self._close_lock:
            for pid in list(self.open_positions.keys()):
                await self.close_position(pid, reason)

    async def _on_market_data(self, depth):
        if self.data is not None:
            try:
                self.data.update_tick(
                    depth.symbol,
                    depth.bid,
                    depth.ask,
                    getattr(depth, "volume", 0),
                    getattr(depth, "timestamp", 0),
                )
            except Exception:
                pass
        now = time.time()
        for pid, trade in list(self.open_positions.items()):
            if trade.symbol != depth.symbol:
                continue
            if now - trade.timestamp < 60:
                continue
            price = depth.bid if trade.direction == "BUY" else depth.ask
            if trade.direction == "BUY":
                if price <= trade.sl:
                    await self.close_position(pid, "Stop Loss", price)
                elif price >= trade.tp:
                    await self.close_position(pid, "Take Profit", price)
            else:
                if price >= trade.sl:
                    await self.close_position(pid, "Stop Loss", price)
                elif price <= trade.tp:
                    await self.close_position(pid, "Take Profit", price)

    def _on_order_update(self, result):
        if result.status == "FILLED":
            logger.debug(f"Order filled: {result.order_id}")
        elif result.status == "REJECTED":
            logger.error(f"Order rejected: {result.error}")

    async def get_account_info(self):
        if self.mode == "LIVE" and self.client is not None:
            try:
                if hasattr(self.client, "get_account_info") and callable(
                    self.client.get_account_info
                ):
                    acc = (
                        await self.client.get_account_info()
                        if asyncio.iscoroutinefunction(self.client.get_account_info)
                        else self.client.get_account_info()
                    )
                    if acc:
                        return {
                            "balance": getattr(acc, "balance", self._balance),
                            "equity": getattr(acc, "equity", self._balance),
                            "margin": getattr(acc, "margin", 0),
                            "free_margin": getattr(acc, "free_margin", self._balance),
                            "currency": getattr(acc, "currency", "USD"),
                        }
            except Exception as e:
                logger.warning(f"Broker get_account_info failed: {e}")
        unrealized = 0.0
        for trade in self.open_positions.values():
            price = self._live_price(trade.symbol)
            if price > 0:
                unrealized += self._pnl_usd(
                    trade.entry_price,
                    price,
                    trade.direction,
                    trade.volume,
                    trade.symbol,
                )
        equity = self._balance + unrealized
        if equity > self._peak_balance:
            self._peak_balance = equity
        free_margin = equity
        return {
            "balance": round(self._balance, 2),
            "equity": round(equity, 2),
            "margin": 0.0,
            "free_margin": round(free_margin, 2),
            "currency": "USD",
        }

    def get_open_positions(self):
        return [
            {
                "position_id": t.position_id,
                "symbol": t.symbol,
                "direction": t.direction,
                "volume": t.volume,
                "entry_price": t.entry_price,
                "sl": t.sl,
                "tp": t.tp,
                "unrealized_pnl": self._pnl_usd(
                    t.entry_price,
                    self._live_price(t.symbol),
                    t.direction,
                    t.volume,
                    t.symbol,
                ),
            }
            for t in self.open_positions.values()
        ]

    def get_trade_history(self, limit=100):
        return [
            {
                "timestamp": t.timestamp,
                "symbol": t.symbol,
                "direction": t.direction,
                "volume": t.volume,
                "entry": t.entry_price,
                "exit": t.exit_price,
                "pnl": t.pnl,
                "status": t.status,
                "reason": t.reason,
            }
            for t in self.trade_history[-limit:]
        ]

    def _get_symbol_id(self, symbol):
        from api.symbol_map import get_symbol_id

        return get_symbol_id(symbol)
