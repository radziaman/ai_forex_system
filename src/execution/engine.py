import asyncio
import time
import json
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
    def __init__(self, ctrader_client, risk_manager, data_manager,
                 initial_balance: float = 100000.0):
        self.client = ctrader_client
        self.risk = risk_manager
        self.data = data_manager
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
        """Delegate to underlying client's on_market_data callback."""
        return self.client.on_market_data
    
    @on_market_data.setter
    def on_market_data(self, callback):
        self.client.on_market_data = callback

    @property
    def raw(self):
        """Access underlying client."""
        return self.client

    def _wire_order_update(self):
        self.client.on_order_update = self._on_order_update

    def _get_default_price(self, symbol: str) -> float:
        return BASE_PRICES.get(symbol.upper(), 1.12)

    async def open_position(
        self, symbol, direction, volume, sl, tp, reason="AI signal"
    ):
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
            symbol=symbol,
            symbol_id=symbol_id,
            side="BUY" if direction == "BUY" else "SELL",
            order_type="MARKET",
            volume=max(int(volume), 1),  # Kelly returns units, use directly
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
                    f"OPENED: {direction} {volume} {symbol} @ {trade.entry_price:.5f}"
                )
                return trade
        except Exception as e:
            logger.error(f"Order error: {e}")
        return None

    def _simulate_open(self, symbol, direction, volume, sl, tp, reason):
        price = self._get_default_price(symbol)
        if self.data is not None:
            try:
                price = self.data.get_price(symbol, "1h")
            except Exception:
                pass
        if price is None or price == 0:
            price = self._get_default_price(symbol)
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
        logger.info(f"[PAPER] OPENED: {direction} {volume} {symbol} @ {price:.5f}")
        return trade

    async def close_position(self, position_id, reason="AI close", exit_price: float = None):
        if position_id not in self.open_positions:
            return False
        trade = self.open_positions[position_id]
        if exit_price is None:
            if self.data is not None:
                try:
                    exit_price = self.data.get_price(trade.symbol, "1h")
                except Exception:
                    exit_price = trade.entry_price
            else:
                exit_price = trade.entry_price
        if self.risk.mode == "PAPER":
            return self._simulate_close(trade, reason, exit_price)
        symbol_id = self._get_symbol_id(trade.symbol)
        close_side = "SELL" if trade.direction == "BUY" else "BUY"
        order = TradeOrder(
            symbol=trade.symbol,
            symbol_id=symbol_id,
            side=close_side,
            order_type="MARKET",
            volume=int(trade.volume * 100000),
            position_id=position_id,
        )
        try:
            result = await self.client.place_order(order)
            if result and result.status == "FILLED":
                exit_price = result.filled_price or exit_price
                pnl = self._calculate_pnl(trade, exit_price)
                trade.exit_price = exit_price
                trade.pnl = pnl
                trade.status = "CLOSED"
                self.trade_history.append(trade)
                del self.open_positions[position_id]
                self._balance += pnl
                self.risk.record_trade(trade, exit_price, pnl)
                logger.success(f"CLOSED: {trade.symbol} | PnL: ${pnl:.2f}")
                return True
        except Exception as e:
            logger.error(f"Close error: {e}")
        return False

    def _simulate_close(self, trade, reason, exit_price: float = None):
        if exit_price is None or exit_price == trade.entry_price:
            if self.data is not None:
                try:
                    exit_price = self.data.get_price(trade.symbol, "1h")
                except Exception:
                    exit_price = trade.entry_price
            else:
                exit_price = trade.entry_price
        pnl = self._calculate_pnl(trade, exit_price)
        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.status = "CLOSED"
        self.trade_history.append(trade)
        del self.open_positions[trade.position_id]
        self._balance += pnl
        self.risk.record_trade(trade, exit_price, pnl)
        logger.info(f"[PAPER] CLOSED: {trade.symbol} | PnL: ${pnl:.2f}")
        return True

    async def close_all_positions(self, reason="AI close all"):
        async with self._close_lock:
            for pid in list(self.open_positions.keys()):
                await self.close_position(pid, reason)

    def _calculate_pnl(self, trade, exit_price):
        if exit_price is None or exit_price == 0.0 or exit_price == trade.entry_price:
            return 0.0
        # Pip value depends on pair type
        sym = trade.symbol.upper()
        if "JPY" in sym:
            pip_size = 0.01
            pip_value_per_lot = 1000 / exit_price if "USD" not in sym else 9.0  # USDJPY ~$9/lot
        elif sym in ("XAUUSD", "XAGUSD"):
            pip_size = 0.1  # XAUUSD pip is 0.1
            pip_value_per_lot = 10.0
        elif "BTC" in sym or "ETH" in sym or "XRP" in sym or "LTC" in sym:
            pip_size = 1.0  # Crypto quoted in whole dollars
            pip_value_per_lot = 1.0
        elif "USD" in sym:  # XXXUSD pairs
            pip_size = 0.0001
            pip_value_per_lot = 10.0
        else:  # Crosses and indices
            pip_size = 0.0001
            pip_value_per_lot = 10.0
        if trade.direction == "BUY":
            pips = (exit_price - trade.entry_price) / pip_size
        else:
            pips = (trade.entry_price - exit_price) / pip_size
        return round(pips * (trade.volume / 100_000.0) * pip_value_per_lot, 2)

    def _get_symbol_id(self, symbol):
        """Map symbol to cTrader symbol ID (VERIFIED WORKING from test_fix_working.py)."""
        symbol_map = {
            "EURUSD": 1, "GBPUSD": 2, "EURJPY": 3, "USDJPY": 4, "AUDUSD": 5,
            "USDCHF": 6, "GBPJPY": 7, "USDCAD": 8, "EURGBP": 9, "NZDUSD": 12,
            "XAUUSD": 41, "XAGUSD": 42, "XTIUSD": 99, "XBRUSD": 100, "XNGUSD": 121,
            "US500": 115, "US30": 125, "USTEC": 108, "UK100": 116, "DE40": 139,
            "BTCUSD": 114, "ETHUSD": 105, "LTCUSD": 112, "XRPUSD": 215,
        }
        return symbol_map.get(symbol.upper(), 1)

    async def _on_market_data(self, depth):
        prices = {depth.symbol: depth.bid}
        self.risk.update_trailing_stops(prices)
        for pid, trade in list(self.open_positions.items()):
            if trade.symbol != depth.symbol:
                continue
            current_price = depth.bid if trade.direction == "BUY" else depth.ask
            if trade.direction == "BUY" and current_price <= trade.sl:
                await self.close_position(pid, "Stop Loss")
            elif trade.direction == "SELL" and current_price >= trade.sl:
                await self.close_position(pid, "Stop Loss")
            if trade.direction == "BUY" and current_price >= trade.tp:
                await self.close_position(pid, "Take Profit")
            elif trade.direction == "SELL" and current_price <= trade.tp:
                await self.close_position(pid, "Take Profit")

    def _on_order_update(self, result):
        if result.status == "FILLED":
            logger.debug(f"Order filled: {result.order_id}")
        elif result.status == "REJECTED":
            logger.error(f"Order rejected: {result.error}")

    async def get_account_info(self):
        if self.risk and self.risk.mode == "LIVE":
            if hasattr(self.client, 'get_account_info') and callable(self.client.get_account_info):
                if asyncio.iscoroutinefunction(self.client.get_account_info):
                    acc = await self.client.get_account_info()
                else:
                    acc = self.client.get_account_info()
                if acc:
                    return {
                        "balance": getattr(acc, 'balance', self._balance),
                        "equity": getattr(acc, 'equity', self._balance),
                        "margin": getattr(acc, 'margin', 0),
                        "free_margin": getattr(acc, 'free_margin', self._balance),
                        "currency": getattr(acc, 'currency', 'USD'),
                    }
        margin = 0.0
        unrealized = 0.0
        for pid, trade in self.open_positions.items():
            price = self._current_price(trade.symbol)
            if price > 0:
                unrealized += self._calculate_pnl(trade, price)
        equity = self._balance + unrealized
        if equity > self._peak_balance:
            self._peak_balance = equity
        free_margin = equity - margin if margin < equity else 0
        return {
            "balance": round(self._balance, 2),
            "equity": round(equity, 2),
            "margin": round(margin, 2),
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
                "unrealized_pnl": self._calculate_pnl(t, self._current_price(t.symbol)),
            }
            for t in self.open_positions.values()
        ]

    def _current_price(self, symbol: str) -> float:
        if self.data is not None:
            try:
                return self.data.get_price(symbol, "1h")
            except Exception:
                pass
        return 0.0

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
