"""
Off-Site Simulation Runner — Live tick data, local position management, no broker posting.

Connects to Dukascopy for FREE real-time tick data, runs the full agentic
pipeline (screener → data → signal → risk → execution), logs every decision
and virtual position locally. ZERO risk — nothing is sent to any broker.

Usage:
    python -m src.scripts.run_simulation                  # Default simulation
    python -m src.scripts.run_simulation --mode paper      # With agent bus
    python -m src.scripts.run_simulation --ticks-only      # Just stream ticks
"""

import os
import sys
import asyncio
import time
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict

_src = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _src not in sys.path:
    sys.path.insert(0, _src)

from loguru import logger  # noqa: E402


@dataclass
class VirtualPosition:
    """A position managed locally — never sent to any broker."""

    symbol: str
    direction: str  # LONG or SHORT
    entry_price: float
    entry_time: float
    volume: float  # in lots
    sl_price: float = 0.0
    tp_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    status: str = "OPEN"  # OPEN, CLOSED, SL_HIT, TP_HIT
    exit_price: float = 0.0
    exit_time: float = 0.0
    exit_reason: str = ""


@dataclass
class SimulationLog:
    """Complete log of simulation activity."""

    timestamp: float = 0.0
    event_type: str = ""  # TICK, SIGNAL, OPEN, CLOSE, PNL, SCREEN
    symbol: str = ""
    detail: str = ""
    price: float = 0.0
    pnl: float = 0.0
    metadata: Dict = field(default_factory=dict)


class OffSiteSimulation:
    """
    Complete off-site simulation with live Dukascopy tick data.

    Architecture:
      1. DukascopyProvider streams real ticks → tick handler
      2. Tick handler aggregates to 1-second bars → features → signals
      3. Signal handler generates trade decisions
      4. Virtual position manager tracks positions locally
      5. Logger records EVERYTHING to disk
      6. NO data is sent to any broker

    The screener_agent runs separately on a schedule, scanning Yahoo Finance
    data to determine which instruments have edge.
    """

    def __init__(
        self,
        log_dir: str = "data/simulation",
        initial_balance: float = 100_000.0,
        tick_symbols: Optional[List[str]] = None,
        trade_symbols: Optional[List[str]] = None,
    ):
        self.log_dir = log_dir
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance

        # Default tick symbols (Dukascopy-supported FX pairs)
        self.tick_symbols = tick_symbols or ["EURUSD", "GBPUSD", "USDJPY"]
        # Default trade symbols (from screener — only trade what has edge)
        self.trade_symbols = trade_symbols or []

        os.makedirs(log_dir, exist_ok=True)

        # Position management
        self.positions: Dict[str, VirtualPosition] = {}
        self.trade_history: List[VirtualPosition] = []
        self.total_trades = 0
        self.winning_trades = 0

        # Tick data
        self._tick_buffer: Dict[str, List] = defaultdict(list)
        self._second_bars: Dict[str, Dict] = {}
        self._running = False
        self._provider = None

        # Signal state
        self._signal_state: Dict[str, Dict] = defaultdict(
            lambda: {"last_signal": 0, "last_price": 0.0, "streak": 0}
        )

        # Logging
        self._log: List[SimulationLog] = []
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info("Off-Site Simulation initialized")
        logger.info(f"  Balance: ${initial_balance:,.0f}")
        logger.info(f"  Tick symbols: {self.tick_symbols}")
        logger.info(f"  Trade symbols: {self.trade_symbols}")
        logger.info(f"  Log dir: {log_dir}")

    # ─── Tick Handler ────────────────────────────────────────────────────

    async def _on_tick(self, tick):
        """Process incoming tick from Dukascopy stream."""
        symbol = tick.symbol if hasattr(tick, "symbol") else "EURUSD"
        price = tick.mid if hasattr(tick, "mid") else (tick.bid + tick.ask) / 2
        ts = tick.timestamp if hasattr(tick, "timestamp") else time.time()
        bid = tick.bid if hasattr(tick, "bid") else price
        ask = tick.ask if hasattr(tick, "ask") else price

        # Log tick
        self._log_event(
            SimulationLog(
                timestamp=ts,
                event_type="TICK",
                symbol=symbol,
                price=price,
                detail=f"bid={bid:.5f} ask={ask:.5f} spread={(ask-bid)*10000:.1f}bp",
            )
        )

        # Update 1-second bar
        bar_key = int(ts)
        if symbol not in self._second_bars:
            self._second_bars[symbol] = {
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "ts": bar_key,
            }
        else:
            bar = self._second_bars[symbol]
            if bar["ts"] != bar_key:
                # Bar closed — process it
                await self._on_bar_close(symbol, bar)
                self._second_bars[symbol] = {
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "ts": bar_key,
                }
            else:
                bar["high"] = max(bar["high"], price)
                bar["low"] = min(bar["low"], price)
                bar["close"] = price

        # Update open positions
        if symbol in self.positions:
            pos = self.positions[symbol]
            pos.current_price = price
            pos.unrealized_pnl = (
                (price - pos.entry_price) * pos.volume * 100000
                if pos.direction == "LONG"
                else (pos.entry_price - price) * pos.volume * 100000
            )

            # Check SL/TP
            if pos.sl_price > 0:
                if (pos.direction == "LONG" and price <= pos.sl_price) or (
                    pos.direction == "SHORT" and price >= pos.sl_price
                ):
                    await self._close_position(symbol, price, "SL_HIT")
            if pos.tp_price > 0:
                if (pos.direction == "LONG" and price >= pos.tp_price) or (
                    pos.direction == "SHORT" and price <= pos.tp_price
                ):
                    await self._close_position(symbol, price, "TP_HIT")

    async def _on_bar_close(self, symbol: str, bar: Dict):
        """Process a closed 1-second bar — generate signals."""
        # Simple momentum signal: if close > open (green bar) → long else → short
        if symbol not in self.trade_symbols:
            return

        price = bar["close"]
        signal = (
            1 if bar["close"] > bar["open"] else -1 if bar["close"] < bar["open"] else 0
        )

        if signal != 0:
            state = self._signal_state[symbol]
            if signal == state["last_signal"]:
                state["streak"] += 1
            else:
                state["streak"] = 1
            state["last_signal"] = signal
            state["last_price"] = price

            # Trade on streak of 3+ in same direction
            if state["streak"] >= 3 and symbol not in self.positions:
                direction = "LONG" if signal == 1 else "SHORT"
                atr_est = price * 0.0005  # ~5 pip ATR estimate
                await self._open_position(
                    symbol=symbol,
                    direction=direction,
                    entry_price=price,
                    volume=0.01,
                    sl_price=(
                        price - 2 * atr_est
                        if direction == "LONG"
                        else price + 2 * atr_est
                    ),
                    tp_price=(
                        price + 4 * atr_est
                        if direction == "LONG"
                        else price - 4 * atr_est
                    ),
                )

    # ─── Position Management ─────────────────────────────────────────────

    async def _open_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        volume: float = 0.01,
        sl_price: float = 0.0,
        tp_price: float = 0.0,
    ):
        """Open a virtual position — no broker interaction."""
        pos = VirtualPosition(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            entry_time=time.time(),
            volume=volume,
            sl_price=sl_price,
            tp_price=tp_price,
            current_price=entry_price,
        )
        self.positions[symbol] = pos
        self.total_trades += 1

        self._log_event(
            SimulationLog(
                timestamp=time.time(),
                event_type="OPEN",
                symbol=symbol,
                price=entry_price,
                detail=f"{direction} {volume} lots @ {entry_price:.5f}"
                f"{' SL=' + str(round(sl_price,5)) if sl_price else ''}"
                f"{' TP=' + str(round(tp_price,5)) if tp_price else ''}",
            )
        )
        logger.info(f"[OPEN] {symbol} {direction} {volume} lots @ {entry_price:.5f}")

    async def _close_position(
        self, symbol: str, exit_price: float, reason: str = "SIGNAL"
    ):
        """Close a virtual position — calculate PnL locally."""
        if symbol not in self.positions:
            return

        pos = self.positions.pop(symbol)
        pos.exit_price = exit_price
        pos.exit_time = time.time()
        pos.exit_reason = reason
        pos.status = "CLOSED"

        if pos.direction == "LONG":
            pos.realized_pnl = (exit_price - pos.entry_price) * pos.volume * 100000
        else:
            pos.realized_pnl = (pos.entry_price - exit_price) * pos.volume * 100000

        self.balance += pos.realized_pnl
        self.equity = self.balance
        if pos.realized_pnl > 0:
            self.winning_trades += 1

        self.trade_history.append(pos)

        self._log_event(
            SimulationLog(
                timestamp=time.time(),
                event_type="CLOSE",
                symbol=symbol,
                price=exit_price,
                pnl=pos.realized_pnl,
                detail=f"{reason} PnL=${pos.realized_pnl:.2f}",
            )
        )
        logger.info(
            f"[CLOSE] {symbol} {reason} @ {exit_price:.5f} | PnL=${pos.realized_pnl:.2f}"
        )

    # ─── Screener Integration ─────────────────────────────────────────────

    async def run_screener(self):
        """Run the autonomous screener to update tradeable symbols."""
        try:
            from agentic.agents.screener_agent import (
                InstrumentScreenerAgent,
                INSTRUMENT_UNIVERSE,
            )

            screener = InstrumentScreenerAgent(scan_interval_hours=24)

            logger.info("Running autonomous screener...")
            tradeable = []

            # Screen HO=F (Heating Oil) — the current best candidate
            if "HO=F" in INSTRUMENT_UNIVERSE:
                score = screener._screen_instrument("HO=F", INSTRUMENT_UNIVERSE["HO=F"])
                if score and score.edge_detected:
                    tradeable.append(score.ticker)
                    logger.info(
                        f"  ✅ HO=F tradeable: Sharpe={score.mom5_sharpe:.2f} PF={score.mom5_pf:.2f}"
                    )

            # Screen EURUSD for tick-level simulation
            if "EURUSD=X" in INSTRUMENT_UNIVERSE:
                score = screener._screen_instrument(
                    "EURUSD=X", INSTRUMENT_UNIVERSE["EURUSD=X"]
                )
                if score:
                    logger.info(
                        f"  EURUSD: Sharpe={score.mom5_sharpe:.2f} -> {score.recommendation}"
                    )

            self.trade_symbols = tradeable
            self._log_event(
                SimulationLog(
                    timestamp=time.time(),
                    event_type="SCREEN",
                    symbol=", ".join(tradeable) if tradeable else "none",
                    detail=f"Tradeable instruments: {tradeable}",
                )
            )

        except Exception as e:
            logger.warning(f"Screener failed: {e}")

    # ─── Reporting ────────────────────────────────────────────────────────

    def _log_event(self, event: SimulationLog):
        self._log.append(event)
        # Trim log to last 10000 events
        if len(self._log) > 10000:
            self._log = self._log[-5000:]

    def print_status(self):
        """Print current simulation status."""
        print()
        print("=" * 70)
        print("  OFF-SITE SIMULATION STATUS")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        print(f"  Balance:     ${self.balance:>10,.2f}")
        print(f"  Equity:      ${self.equity:>10,.2f}")
        print(f"  Total PnL:   ${self.balance - self.initial_balance:>+10,.2f}")
        print(f"  Open Positions: {len(self.positions)}")
        for sym, pos in self.positions.items():
            print(
                f"    {sym}: {pos.direction} {pos.volume} lots @ {pos.entry_price:.5f} "
                f"UPnL=${pos.unrealized_pnl:.2f}"
            )
        print(f"  Closed Trades: {len(self.trade_history)}")
        if self.trade_history:
            win_rate = self.winning_trades / len(self.trade_history) * 100
            total_pnl = sum(t.realized_pnl for t in self.trade_history)
            print(f"    Win Rate: {win_rate:.1f}%")
            print(f"    Total PnL: ${total_pnl:.2f}")
        print(f"  Tick Rate:   {len(self._log):,} events")
        print(f"  Trade Symbols: {self.trade_symbols}")
        print(f"  Tick Symbols: {self.tick_symbols}")
        print("=" * 70)

    def save_log(self):
        """Save simulation log to disk."""
        log_path = os.path.join(self.log_dir, f"simulation_{self._session_id}.json")
        with open(log_path, "w") as f:
            json.dump(
                {
                    "session_id": self._session_id,
                    "timestamp": time.time(),
                    "initial_balance": self.initial_balance,
                    "final_balance": self.balance,
                    "total_trades": self.total_trades,
                    "winning_trades": self.winning_trades,
                    "trade_history": [asdict(t) for t in self.trade_history],
                    "events": [asdict(e) for e in self._log[-1000:]],
                },
                f,
                indent=2,
                default=str,
            )
        logger.info(f"Simulation log saved: {log_path}")

    # ─── Main Loop ────────────────────────────────────────────────────────

    async def run(self, duration_minutes: int = 5):
        """Run the simulation for a specified duration."""
        self._running = True

        # Step 1: Run screener
        await self.run_screener()

        # Step 2: Start multi-source tick stream
        logger.info("Starting multi-source tick provider...")
        try:
            from data.multi_source_tick_provider import MultiSourceTickProvider

            # Build combined symbol list: tick symbols + trade symbols
            all_symbols = list(set(self.tick_symbols + self.trade_symbols))
            logger.info(f"Streaming symbols: {all_symbols}")

            # Pre-download futures data so it's ready
            provider = MultiSourceTickProvider(poll_interval=1.0)
            futures_syms = [s for s in all_symbols if provider.is_futures_symbol(s)]
            if futures_syms:
                await provider.prepare_futures_data(futures_syms, days=7)

            self._provider = provider
            await provider.stream_prices(all_symbols, self._on_tick)
        except Exception as e:
            logger.error(f"Failed to start tick stream: {e}")
            import traceback

            traceback.print_exc()
            return

        # Step 3: Run for duration
        logger.info(f"Running simulation for {duration_minutes} minutes...")
        start = time.time()
        status_interval = 30  # print status every 30 seconds
        last_status = 0

        try:
            while self._running and (time.time() - start) < duration_minutes * 60:
                await asyncio.sleep(1)

                # Periodic status
                elapsed = time.time() - start
                if elapsed - last_status >= status_interval:
                    last_status = int(elapsed)
                    self.print_status()

        except asyncio.CancelledError:
            logger.info("Simulation cancelled")
        finally:
            # Cleanup
            if self._provider:
                await self._provider.close()
            self._running = False
            self.print_status()
            self.save_log()

        # Print summary
        print()
        print("=" * 70)
        print("  SIMULATION COMPLETE")
        print("=" * 70)
        total_pnl = self.balance - self.initial_balance
        if len(self.trade_history) > 0:
            win_rate = self.winning_trades / len(self.trade_history) * 100
            sharpe = 0
            if len(self.trade_history) > 1:
                pnls = [t.realized_pnl for t in self.trade_history]
                if np.std(pnls) > 0:
                    sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252)
            print(f"  Duration:      {duration_minutes} minutes")
            print(f"  Total Trades:  {len(self.trade_history)}")
            print(f"  Win Rate:      {win_rate:.1f}%")
            print(f"  Total PnL:     ${total_pnl:,.2f}")
            print(f"  Sharpe:        {sharpe:.2f}")
        print(f"  Log:           data/simulation/simulation_{self._session_id}.json")
        print("=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Off-Site Simulation Runner")
    parser.add_argument("--duration", type=int, default=5, help="Duration in minutes")
    parser.add_argument("--balance", type=float, default=100000, help="Initial balance")
    parser.add_argument(
        "--tick-symbols",
        nargs="+",
        default=["EURUSD", "GBPUSD"],
        help="Symbols for tick streaming",
    )
    parser.add_argument(
        "--no-screener",
        action="store_true",
        help="Skip screener scan (uses default trade symbols)",
    )
    args = parser.parse_args()

    sim = OffSiteSimulation(
        initial_balance=args.balance,
        tick_symbols=args.tick_symbols,
    )
    if args.no_screener:
        sim.trade_symbols = args.tick_symbols

    asyncio.run(sim.run(duration_minutes=args.duration))


if __name__ == "__main__":
    main()
