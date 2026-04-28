import time
import json
from loguru import logger
from typing import Optional, Dict, List
from dataclasses import dataclass, field


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
    def __init__(self, ctrader_client, risk_manager, data_manager):
        self.client = ctrader_client
        self.risk = risk_manager
        self.data = data_manager
        self.open_positions: Dict[int, TradeRecord] = {}
        self.trade_history: List[TradeRecord] = []
        self.total_trades = 0
        self._position_counter = 0
        self.client.on_market_data = self._on_market_data
        self.client.on_order_update = self._on_order_update
        logger.info("ExecutionEngine initialized")

    async def open_position(self, symbol, direction, volume, sl, tp, reason="AI signal"):
        if self.risk.kill_switch_triggered:
            logger.warning("Kill switch triggered")
            return None
        if self.risk.mode == "PAPER":
            return self._simulate_open(symbol, direction, volume, sl, tp, reason)
        snapshot = self.data.latest_snapshot
        if not snapshot:
            return None
        price = snapshot.bid if direction == "SELL" else snapshot.ask
        symbol_id = self._get_symbol_id(symbol)
        order = TradeOrder(
            symbol=symbol, symbol_id=symbol_id,
            side="BUY" if direction == "BUY" else "SELL",
            order_type="MARKET", volume=int(volume*100000),
            price=price, sl=sl, tp=tp
        )
        try:
            result = await self.client.place_order(order)
            if result and result.status == "FILLED":
                self._position_counter += 1
                trade = TradeRecord(
                    timestamp=time.time(), symbol=symbol, direction=direction,
                    volume=volume, entry_price=result.filled_price or price,
                    sl=sl, tp=tp, status="OPEN", reason=reason,
                    position_id=self._position_counter
                )
                self.open_positions[trade.position_id] = trade
                self.total_trades += 1
                logger.success(f"OPENED: {direction} {volume} {symbol} @ {trade.entry_price:.5f}")
                return trade
        except Exception as e:
            logger.error(f"Order error: {e}")
        return None

    def _simulate_open(self, symbol, direction, volume, sl, tp, reason):
        snapshot = self.data.latest_snapshot
        price = snapshot.bid if direction == "SELL" else snapshot.ask if snapshot else 1.1200
        self._position_counter += 1
        trade = TradeRecord(
            timestamp=time.time(), symbol=symbol, direction=direction,
            volume=volume, entry_price=price, sl=sl, tp=tp,
            status="OPEN", reason=reason, position_id=self._position_counter
        )
        self.open_positions[trade.position_id] = trade
        self.total_trades += 1
        logger.info(f"[PAPER] OPENED: {direction} {volume} {symbol} @ {price:.5f}")
        return trade

    async def close_position(self, position_id, reason="AI close"):
        if position_id not in self.open_positions:
            return False
        trade = self.open_positions[position_id]
        if self.risk.mode == "PAPER":
            return self._simulate_close(trade, reason)
        symbol_id = self._get_symbol_id(trade.symbol)
        close_side = "SELL" if trade.direction == "BUY" else "BUY"
        order = TradeOrder(symbol=trade.symbol, symbol_id=symbol_id,
                         side=close_side, order_type="MARKET",
                         volume=int(trade.volume*100000), position_id=position_id)
        try:
            result = await self.client.place_order(order)
            if result and result.status == "FILLED":
                snapshot = self.data.latest_snapshot
                exit_price = result.filled_price or (snapshot.bid if snapshot else trade.entry_price)
                pnl = self._calculate_pnl(trade, exit_price)
                trade.exit_price = exit_price
                trade.pnl = pnl
                trade.status = "CLOSED"
                self.trade_history.append(trade)
                del self.open_positions[position_id]
                self.risk.record_trade(trade, exit_price, pnl)
                logger.success(f"CLOSED: {trade.symbol} | PnL: ${pnl:.2f}")
                return True
        except Exception as e:
            logger.error(f"Close error: {e}")
        return False

    def _simulate_close(self, trade, reason):
        snapshot = self.data.latest_snapshot
        exit_price = snapshot.bid if trade.direction == "BUY" else snapshot.ask if snapshot else trade.entry_price
        pnl = self._calculate_pnl(trade, exit_price)
        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.status = "CLOSED"
        self.trade_history.append(trade)
        del self.open_positions[trade.position_id]
        self.risk.record_trade(trade, exit_price, pnl)
        logger.info(f"[PAPER] CLOSED: {trade.symbol} | PnL: ${pnl:.2f}")
        return True

    async def close_all_positions(self, reason="AI close all"):
        for pid in list(self.open_positions.keys()):
            await self.close_position(pid, reason)

    def _calculate_pnl(self, trade, exit_price):
        if trade.direction == "BUY":
            pips = (exit_price - trade.entry_price) * 10000
        else:
            pips = (trade.entry_price - exit_price) * 10000
        return round(pips * trade.volume * 10, 2)

    def _get_symbol_id(self, symbol):
        return {"EURUSD":1,"GBPUSD":2,"USDJPY":3,"XAUUSD":4,"BTCUSD":5}.get(symbol.upper(),1)

    def _on_market_data(self, depth):
        prices = {depth.symbol: depth.bid}
        self.risk.update_trailing_stops(prices)
        for pid, trade in list(self.open_positions.items()):
            if trade.symbol != depth.symbol:
                continue
            current_price = depth.bid if trade.direction == "BUY" else depth.ask
            if trade.direction == "BUY" and current_price <= trade.sl:
                self.close_position(pid, "Stop Loss")
            elif trade.direction == "SELL" and current_price >= trade.sl:
                self.close_position(pid, "Stop Loss")
            if trade.direction == "BUY" and current_price >= trade.tp:
                self.close_position(pid, "Take Profit")
            elif trade.direction == "SELL" and current_price <= trade.tp:
                self.close_position(pid, "Take Profit")

    def _on_order_update(self, result):
        if result.status == "FILLED":
            logger.debug(f"Order filled: {result.order_id}")
        elif result.status == "REJECTED":
            logger.error(f"Order rejected: {result.error}")

    def get_account_info(self):
        acc = self.client.get_account_info()
        if acc:
            return {'balance':acc.balance,'equity':acc.equity,'margin':acc.margin,
                    'free_margin':acc.free_margin,'currency':acc.currency}
        return {'balance':100000,'equity':100000,'margin':0,'free_margin':100000,'currency':'USD'}

    def get_open_positions(self):
        return [{'position_id':t.position_id,'symbol':t.symbol,'direction':t.direction,
                 'volume':t.volume,'entry_price':t.entry_price,'sl':t.sl,'tp':t.tp,
                 'unrealized_pnl':self._calculate_pnl(t, self.data.latest_snapshot.bid if self.data.latest_snapshot else t.entry_price)}
                for t in self.open_positions.values()]

    def get_trade_history(self, limit=100):
        return [{'timestamp':t.timestamp,'symbol':t.symbol,'direction':t.direction,
                 'volume':t.volume,'entry':t.entry_price,'exit':t.exit_price,
                 'pnl':t.pnl,'status':t.status,'reason':t.reason}
                for t in self.trade_history[-limit:]]
