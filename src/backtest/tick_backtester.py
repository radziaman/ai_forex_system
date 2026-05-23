"""
Tick-by-Tick Backtester for microstructure-accurate strategy evaluation.

Processes ticks in chronological order with realistic costs, floating PnL,
and support for market, limit, and stop orders.
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Union, Any


class TickBacktester:
    """Event-driven tick backtester for <15m holding strategies."""

    def __init__(
        self,
        tick_data: Dict[str, np.ndarray],
        signal_fn: Callable,
        spread_model: Optional[Callable] = None,
        commission_per_lot: float = 7.0,
        lot_size: float = 100_000,
        pip_size: float = 0.0001,
    ):
        self.tick_data = tick_data
        self.signal_fn = signal_fn
        self.spread_model = spread_model
        self.commission_per_lot = commission_per_lot
        self.lot_size = lot_size
        self.pip_size = pip_size

        self._validate_tick_data()

        self.position = 0
        self.entry_price = 0.0
        self.entry_idx = 0
        self.trades: List[Dict[str, Any]] = []
        self.equity: List[float] = [1.0]
        self.floating_pnls: List[float] = []
        self.realized_pnl = 0.0

    def _validate_tick_data(self) -> None:
        if "timestamp" not in self.tick_data:
            raise ValueError(
                f"tick_data must contain 'timestamp', got {list(self.tick_data.keys())}"
            )
        has_bid_ask = "bid" in self.tick_data and "ask" in self.tick_data
        has_mid = "mid" in self.tick_data or "close" in self.tick_data
        if not has_bid_ask and not has_mid:
            raise ValueError("tick_data must contain ('bid', 'ask') or 'mid'/'close'")

    def _get_prices(self, idx: int):
        if "bid" in self.tick_data and "ask" in self.tick_data:
            return float(self.tick_data["bid"][idx]), float(self.tick_data["ask"][idx])

        mid_key = "mid" if "mid" in self.tick_data else "close"
        mid = float(self.tick_data[mid_key][idx])

        if self.spread_model is not None:
            spread = self.spread_model(mid, idx)
        else:
            spread = self.pip_size * 0.5

        return mid - spread / 2, mid + spread / 2

    def _calc_floating_pnl(self, bid: float, ask: float) -> float:
        if self.position == 0:
            return 0.0
        if self.position == 1:
            price_diff = bid - self.entry_price
        else:
            price_diff = self.entry_price - ask
        return price_diff / self.pip_size * 10.0

    def _transaction_cost(self, bid: float, ask: float) -> float:
        lots = 1.0
        pip_value = self.pip_size * self.lot_size
        spread_pips = (ask - bid) / self.pip_size
        spread_cost = spread_pips * pip_value * lots
        commission_cost = self.commission_per_lot * lots
        return spread_cost + commission_cost

    def _slippage(self, side: str, fill_price: float, expected_price: float) -> float:
        """Slippage in pips — next tick price vs expected fill."""
        slippage_price = abs(fill_price - expected_price)
        return slippage_price / self.pip_size

    def _execute_fill(
        self, side: str, fill_price: float, bid: float, ask: float, idx: int
    ) -> None:
        if side == "BUY":
            expected = self._get_prices(max(0, idx - 1))[1] if idx > 0 else fill_price
        else:
            expected = self._get_prices(max(0, idx - 1))[0] if idx > 0 else fill_price

        slippage_pips = self._slippage(side, fill_price, expected)

        if self.position == 0:
            self.position = 1 if side == "BUY" else -1
            self.entry_price = fill_price
            self.entry_idx = idx
        elif (self.position == 1 and side == "SELL") or (
            self.position == -1 and side == "BUY"
        ):
            if self.position == 1:
                price_diff = fill_price - self.entry_price
            else:
                price_diff = self.entry_price - fill_price

            raw_pnl = price_diff / self.pip_size * 10.0
            cost = self._transaction_cost(bid, ask)
            net_pnl = raw_pnl - cost

            self.trades.append(
                {
                    "entry_idx": self.entry_idx,
                    "exit_idx": idx,
                    "entry_price": self.entry_price,
                    "exit_price": fill_price,
                    "position": self.position,
                    "raw_pnl": raw_pnl,
                    "cost": cost,
                    "net_pnl": net_pnl,
                    "slippage_pips": slippage_pips,
                }
            )

            self.realized_pnl += net_pnl
            self.position = 0
            self.entry_price = 0.0
        # Same direction: ignore (already in position)

    @staticmethod
    def _parse_signal(signal_output: Union[int, Dict]) -> List[Dict[str, Any]]:
        if isinstance(signal_output, (int, np.integer)):
            if signal_output == 0:
                return []
            side = "BUY" if signal_output == 1 else "SELL"
            return [{"side": side, "order_type": "MARKET", "price": None}]

        if isinstance(signal_output, dict):
            signal = signal_output.get("signal", 0)
            if signal == 0:
                return []
            side = "BUY" if signal == 1 else "SELL"
            order_type = signal_output.get("order_type", "MARKET")
            price = signal_output.get("price")
            return [{"side": side, "order_type": order_type, "price": price}]

        return []

    def _check_pending_orders(
        self, pending: List[Dict[str, Any]], bid: float, ask: float
    ) -> List[Dict[str, Any]]:
        filled = []
        for order in pending:
            ot = order["order_type"]
            side = order["side"]
            price = order.get("price", 0.0)

            if ot == "LIMIT":
                if side == "BUY" and ask <= price:
                    filled.append(order)
                elif side == "SELL" and bid >= price:
                    filled.append(order)
            elif ot == "STOP":
                if side == "BUY" and ask >= price:
                    filled.append(order)
                elif side == "SELL" and bid <= price:
                    filled.append(order)
        return filled

    def run(self) -> None:
        n = len(self.tick_data["timestamp"])
        if n < 2:
            return

        pending_orders: List[Dict[str, Any]] = []

        for i in range(n - 1):
            next_bid, next_ask = self._get_prices(i + 1)

            # Generate orders based on tick i
            context = {
                "position": self.position,
                "entry_price": self.entry_price,
                "entry_idx": self.entry_idx,
                "floating_pnl": (self.floating_pnls[-1] if self.floating_pnls else 0.0),
                "equity": self.equity[-1],
            }
            signal_output = self.signal_fn(i, self.tick_data, context)
            orders = self._parse_signal(signal_output)

            # Add new limit/stop orders to pending
            for order in orders:
                if order["order_type"] != "MARKET":
                    pending_orders.append(order)

            # Check ALL pending orders (including newly placed) against tick i+1
            filled = self._check_pending_orders(pending_orders, next_bid, next_ask)
            filled_ids = {id(o) for o in filled}
            for order in filled:
                fill_price = next_ask if order["side"] == "BUY" else next_bid
                self._execute_fill(order["side"], fill_price, next_bid, next_ask, i + 1)
            pending_orders = [o for o in pending_orders if id(o) not in filled_ids]

            # Execute market orders at tick i+1
            for order in orders:
                if order["order_type"] == "MARKET":
                    fill_price = next_ask if order["side"] == "BUY" else next_bid
                    self._execute_fill(
                        order["side"], fill_price, next_bid, next_ask, i + 1
                    )

            # Update equity with post-fill floating PnL
            floating = self._calc_floating_pnl(next_bid, next_ask)
            self.floating_pnls.append(floating)
            total_pnl = self.realized_pnl + floating
            pnl_pct = total_pnl / 100.0
            self.equity.append(max(0.0, self.equity[0] * (1 + pnl_pct)))

        # Close any remaining position at last tick
        if self.position != 0:
            last_bid, last_ask = self._get_prices(n - 1)
            side = "SELL" if self.position == 1 else "BUY"
            fill_price = last_bid if side == "SELL" else last_ask
            self._execute_fill(side, fill_price, last_bid, last_ask, n - 1)
            total_pnl = self.realized_pnl
            pnl_pct = total_pnl / 100.0
            self.equity.append(max(0.0, self.equity[0] * (1 + pnl_pct)))

    def get_results(self) -> Dict[str, Any]:
        trade_pnls = np.array([t["net_pnl"] for t in self.trades])
        equity_curve = np.array(self.equity)
        n = len(self.tick_data["timestamp"])

        if len(trade_pnls) == 0:
            return {
                "total_return_pct": 0.0,
                "annual_return_pct": 0.0,
                "sharpe": 0.0,
                "max_drawdown_pct": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_trades": 0,
                "avg_hold_bars": 0.0,
                "avg_win_pct": 0.0,
                "avg_loss_pct": 0.0,
                "calmar_ratio": 0.0,
                "sortino": 0.0,
            }

        total_return = float(equity_curve[-1] - 1.0) * 100
        years = n / 252
        base = 1 + total_return / 100
        if base <= 0:
            annual_return = -1.0
        else:
            annual_return = base ** (1 / max(years, 0.01)) - 1
        annual_return_pct = float(annual_return) * 100

        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe = (
            float(np.mean(returns) / max(np.std(returns), 1e-10) * np.sqrt(252))
            if len(returns) > 1
            else 0.0
        )

        cum = np.maximum.accumulate(equity_curve)
        dd = (cum - equity_curve) / cum
        max_dd = float(np.max(dd)) * 100

        wins = trade_pnls[trade_pnls > 0]
        losses = trade_pnls[trade_pnls < 0]
        win_rate = len(wins) / max(len(trade_pnls), 1)
        profit_factor = float(np.sum(wins) / max(abs(np.sum(losses)), 1e-10))

        avg_hold = (
            float(np.mean([t["exit_idx"] - t["entry_idx"] for t in self.trades]))
            if self.trades
            else 0.0
        )
        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0

        calmar = annual_return_pct / max(max_dd, 0.01)
        downside = returns[returns < 0]
        sortino = (
            float(np.mean(returns) / max(np.std(downside), 1e-10) * np.sqrt(252))
            if len(downside) > 1
            else 0.0
        )

        return {
            "total_return_pct": round(total_return, 2),
            "annual_return_pct": round(annual_return_pct, 2),
            "sharpe": round(sharpe, 3),
            "max_drawdown_pct": round(max_dd, 2),
            "win_rate": round(win_rate, 3),
            "profit_factor": round(profit_factor, 3),
            "total_trades": len(trade_pnls),
            "avg_hold_bars": round(avg_hold, 1),
            "avg_win_pct": round(avg_win, 3),
            "avg_loss_pct": round(avg_loss, 3),
            "calmar_ratio": round(calmar, 3),
            "sortino": round(sortino, 3),
        }
