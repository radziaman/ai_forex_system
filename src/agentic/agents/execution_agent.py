"""
Execution Agent — G1: tick wiring, G3: position publishing, G5: delivery ack.
"""

from __future__ import annotations
import time
import asyncio
from typing import Dict, List, Optional, Any, Set
from loguru import logger

from agentic.core.base_agent import BaseAgent
from agentic.core.agent_message import (
    MessageType,
    MessagePriority,
    AgentIntention,
    AgentMessage,
)
from agentic.core.agent_consciousness import ConsciousnessLevel


class ExecutionAgent(BaseAgent):
    def __init__(self, config, secrets, initial_balance: float = 100_000.0):
        super().__init__(
            name="execution_agent",
            role="Order Execution Engine",
            purpose="Execute trades through broker, wire live ticks, publish position state",
            domain="execution",
            capabilities={
                "order_execution",
                "broker_connection",
                "position_tracking",
                "sl_tp_monitoring",
                "paper_trading",
                "order_lifecycle",
                "pnl_calculation",
                "account_info",
                "tick_wiring",  # G1
            },
            tick_interval=1.0,
            consciousness_level=ConsciousnessLevel.REFLECTIVE,
        )
        self.config = config
        self.secrets = secrets
        self.initial_balance = initial_balance
        self.engine = None
        self.ctrader = None
        self._executed = 0
        self._failed = 0
        self._reconnecting = False
        self._position_signals: Dict[int, Dict] = {}  # position_id -> signal payload

        self.subscribe(MessageType.RISK_APPROVED)
        self.subscribe(MessageType.AGENT_DIRECTIVE)
        self.subscribe(MessageType.DIAGNOSTIC_REQUEST)
        self.subscribe(MessageType.INSTRUMENTS_UPDATED)
        self.subscribe(MessageType.POSITION_MODIFIED)

        # Active symbols: starts with all core SYMBOLS, dynamically updated by screener
        from data.data_manager import SYMBOLS

        self._active_symbols = list(SYMBOLS)
        self._core_symbols = list(SYMBOLS)  # Preserved for subscription fallback

    async def _on_start(self):
        self.consciousness.current_intention = "initializing execution provider"
        from api.provider_factory import create_execution_provider
        from execution.engine import ExecutionEngine

        self.ctrader, data_provider = create_execution_provider(self.secrets)
        engine_mode = "LIVE" if not self.secrets.is_demo else "PAPER"
        self.engine = ExecutionEngine(
            self.ctrader,
            None,
            None,
            initial_balance=self.initial_balance,
            mode=engine_mode,
        )

        # G1: Wire live price feed from broker to data pipeline
        async def _on_tick(depth):
            """Forward broker ticks to data_agent and store live spread in world state.

            IMPORTANT: depth.bid / .ask arrive ALREADY divided by the correct
            _price_divisor from _handle_depth_event in ctrader_client.py.
            Do NOT re-divide — prices are in standard decimal format for ALL
            symbols (EURUSD=1.1234, XAUUSD=1987.50, BTCUSD=45000.0, etc.).
            """
            try:
                symbol = getattr(depth, "symbol", "")
                bid = getattr(depth, "bid", 0)
                ask = getattr(depth, "ask", 0)
                ts = time.time()

                if bid > 0 and ask > 0:
                    from execution.cost_model import CostModel

                    pip_size = CostModel.pip_to_price(symbol)
                    spread_pips = (ask - bid) / pip_size if pip_size > 0 else 0

                    # Quality check: reject ticks with abnormal spreads
                    # Use percentage of price (works across all asset types)
                    mid_price = (bid + ask) / 2
                    spread_pct = (
                        abs(spread_pips * pip_size / mid_price * 100)
                        if mid_price > 0
                        else 0
                    )
                    # Max acceptable spread as % of mid price.
                    # 0.5% covers normal + volatile periods for all asset classes.
                    # During extreme volatility, spreads on XAUUSD can reach 0.15%,
                    # BTCUSD 0.10%, while normal EURUSD is ~0.009%.
                    MAX_GOOD_SPREAD_PCT = 0.5
                    if spread_pct > MAX_GOOD_SPREAD_PCT or spread_pips < 0:
                        logger.warning(
                            f"TICK REJECTED {symbol}: spread={spread_pips:.1f}pips "
                            f"({spread_pct:.3f}% of price, max={MAX_GOOD_SPREAD_PCT}%)"
                        )
                        # Don't publish bad ticks to the bus
                        return

                    self.set_world(f"data.spread.{symbol}", round(spread_pips, 2))
                    self.set_world(f"data.bid.{symbol}", round(float(bid), 5))
                    self.set_world(f"data.ask.{symbol}", round(float(ask), 5))
                    self.set_world(
                        f"data.price.{symbol}", round(float((bid + ask) / 2), 5)
                    )
                    logger.debug(
                        f"TICK {symbol}: bid={bid:.5f} ask={ask:.5f} spread={spread_pips:.2f}"
                    )

                await self.send(
                    MessageType.TICK_RECEIVED,
                    payload={
                        "symbol": symbol,
                        "bid": bid,
                        "ask": ask,
                        "volume": getattr(depth, "volume", 0),
                        "timestamp": ts,
                    },
                    target_capability="tick_ingestion",
                    priority=MessagePriority.HIGH,
                )
            except Exception:
                pass

        if self.ctrader:
            self.ctrader.on_price = (
                _on_tick if hasattr(self.ctrader, "on_price") else None
            )
            if hasattr(self.ctrader, "on_market_data"):
                self.ctrader.on_market_data = _on_tick

        raw = getattr(self.ctrader, "raw", None)
        if raw:
            raw.on_disconnect = lambda: asyncio.ensure_future(self._on_disconnect())

        connected = await self.ctrader.start()
        if (
            connected
            and hasattr(self.ctrader, "is_connected")
            and self.ctrader.is_connected()
        ):
            self.log_state(f"cTrader connected ({engine_mode})")
            self.set_world("execution.connected", True)
            # Expose raw CtraderClient so data_agent can use it for
            # historical data refreshes (load_from_ctrader)
            raw = getattr(self.ctrader, "raw", None)
            if raw:
                self.set_world("execution.ctrader_client", raw, ttl=3600)
            from data.data_manager import SYMBOLS
            from api.symbol_map import get_symbol_id

            if raw:
                depth_ok = 0
                spot_ok = 0
                # Subscribe to ALL core symbols — depth first, then spot.
                #
                # NOTE: Depth (Level II DOM) may not be available for all
                # symbols or account types (IC Markets Raw Spread accounts
                # often restrict depth).  If depth subscription fails or
                # times out, we still have spot prices as fallback.
                #
                # Rate limiting: 1s between depth subs (cTrader allows
                # max 50 req/s, but depth responses are heavyweight).
                # Spot subs are fire-and-forget (no response expected),
                # so they can be sent more aggressively.
                for sym in self._core_symbols:
                    try:
                        sid = get_symbol_id(sym)
                        # Check connection health — abort if disconnected
                        # (the subscribe callback may have fired during a
                        # previous subscription's await)
                        if not self.get_world("execution.connected", True):
                            self.log_state(
                                f"Connection lost during subscription loop, "
                                f"aborting ({sym} pending)", "warning"
                            )
                            break
                        # Level II depth subscription (order book)
                        if hasattr(raw, "subscribe_depth"):
                            if await raw.subscribe_depth(sid):
                                depth_ok += 1
                            await asyncio.sleep(1.0)  # Rate limit: 1 depth req/s
                        # Spot price subscription (fire-and-forget)
                        if hasattr(raw, "subscribe_spots") and await raw.subscribe_spots(sid):
                            spot_ok += 1
                            await asyncio.sleep(0.2)
                    except Exception as e:
                        logger.debug(f"Subscribe failed for {sym}: {e}")
                self.log_state(
                    f"Subscriptions: depth={depth_ok}/{len(self._core_symbols)}, "
                    f"spot={spot_ok}/{len(self._core_symbols)}"
                )
        else:
            self.log_state(f"Running in {engine_mode} mode (simulation)", "warning")
            self.set_world("execution.connected", False)

        self.set_world("execution.mode", engine_mode)
        self.set_world("execution.status", "ready")

    async def _on_disconnect(self):
        """Called when cTrader connection drops — idempotent (safe to call multiple times)."""
        if not self.get_world("execution.connected", False):
            return  # Already disconnected
        self.log_state("cTrader disconnected", "warning")
        self.set_world("execution.connected", False)
        self.set_world("execution.ctrader_client", None)

    async def _reconnect(self):
        """Reconnect to cTrader and re-subscribe depth/spot subscriptions.

        Governed by a minimum 5s cooldown between attempts to prevent
        rapid reconnect cycling.  Each attempt performs a full SSL connect +
        app auth + account auth + subscription resubscribe.

        On repeated failure, falls back to exponential backoff managed
        by ConnectionAgent (which reads execution.connected from world state).
        """
        if self._reconnecting:
            self.log_state("Reconnect already in progress, skipping", "debug")
            return

        # Minimum cooldown: don't attempt reconnect more than once per 5s
        now = time.time()
        if hasattr(self, "_last_reconnect_ts") and now - self._last_reconnect_ts < 5:
            self.log_state("Reconnect cooldown active, skipping", "debug")
            return
        self._last_reconnect_ts = now

        self._reconnecting = True
        self.log_state("Beginning cTrader reconnection...", "info")
        try:
            # Gracefully stop existing connection (best-effort)
            # Only call stop() if writer is still alive
            if self.ctrader:
                raw = getattr(self.ctrader, "raw", None)
                if raw and getattr(raw, "_writer", None) is not None:
                    try:
                        await self.ctrader.stop()
                        self.log_state("Stopped existing connection", "debug")
                    except Exception as e:
                        self.log_state(f"Stop existing connection: {e}", "debug")
                else:
                    self.log_state("No active connection to stop", "debug")
            await asyncio.sleep(1)

            # Start fresh connection — this does SSL connect + app auth +
            # account auth + fetch info + start background listener + heartbeat
            ok = await self.ctrader.start()
            raw = getattr(self.ctrader, "raw", None)

            if not ok or raw is None:
                self.log_state("Reconnect failed — cannot start broker client", "warning")
                self.set_world("execution.connected", False)
                return

            # Check simulation mode
            sim = raw.is_simulation() if hasattr(raw, "is_simulation") else True
            if sim:
                self.log_state(
                    "Reconnect failed — broker unreachable (simulation fallback)",
                    "warning",
                )
                self.set_world("execution.connected", False)
                return

            # Real connection succeeded — wire disconnect handler
            raw.on_disconnect = lambda: asyncio.ensure_future(self._on_disconnect())
            # Expose raw client for data_agent historical refreshes
            self.set_world("execution.ctrader_client", raw, ttl=3600)

            # Re-subscribe to depth and spot for all core symbols
            from api.symbol_map import get_symbol_id

            depth_ok = 0
            spot_ok = 0
            for sym in self._core_symbols:
                try:
                    sid = get_symbol_id(sym)
                    # Abort subscribe loop if we disconnected during a previous
                    # subscription's await (e.g. server rejected depth request)
                    if not self.get_world("execution.connected", True):
                        self.log_state(
                            f"Connection lost during re-subscribe, "
                            f"aborting ({sym} pending)", "warning"
                        )
                        break
                    if hasattr(raw, "subscribe_depth"):
                        if await raw.subscribe_depth(sid):
                            depth_ok += 1
                        await asyncio.sleep(1.0)
                    if hasattr(raw, "subscribe_spots") and await raw.subscribe_spots(
                        sid
                    ):
                        spot_ok += 1
                        await asyncio.sleep(0.2)
                except Exception as sub_err:
                    self.log_state(f"Subscribe failed for {sym}: {sub_err}", "debug")

            # Only mark connected if we completed the subscribe loop
            # (if we broke early due to disconnect, don't flip to True)
            if self.get_world("execution.connected", False):
                self.log_state(
                    f"Reconnected: depth={depth_ok}/{len(self._core_symbols)}, "
                    f"spot={spot_ok}/{len(self._core_symbols)}"
                )
            self._executed = 0
            self._failed = 0
            self.set_world("execution.connected", True)

        except asyncio.CancelledError:
            self.log_state("Reconnect cancelled", "debug")
        except Exception as e:
            self.log_state(f"Reconnect error: {type(e).__name__}: {e}", "warning")
            self.set_world("execution.connected", False)
        finally:
            self._reconnecting = False

    async def perceive(self) -> Dict[str, Any]:
        if self.engine is None:
            return {"skip": True}
        open_positions = self.engine.get_open_positions()
        account = await self._get_account_info()
        n_positions = len(open_positions)

        # G3: Publish positions to world state
        self.set_world("account.balance", account.get("balance", self.initial_balance))
        self.set_world("account.equity", account.get("equity", self.initial_balance))
        self.set_world("account.margin", account.get("margin", 0))
        self.set_world("account.open_positions", n_positions)
        self.set_world("execution.open_positions_raw", open_positions)

        sl_hits, tp_hits = [], []
        for pos in open_positions:
            price = self._get_live_price(pos["symbol"])
            if price <= 0:
                continue
            if pos["direction"] == "BUY":
                if price <= pos["sl"]:
                    sl_hits.append(pos)
                elif price >= pos["tp"]:
                    tp_hits.append(pos)
            else:
                if price >= pos["sl"]:
                    sl_hits.append(pos)
                elif price <= pos["tp"]:
                    tp_hits.append(pos)

        return {
            "open_positions": n_positions,
            "account": account,
            "sl_hits": sl_hits,
            "tp_hits": tp_hits,
        }

    async def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        close_positions = []
        for pos in perception.get("sl_hits", []):
            close_positions.append({"pid": pos["position_id"], "reason": "Stop Loss"})
        for pos in perception.get("tp_hits", []):
            close_positions.append({"pid": pos["position_id"], "reason": "Take Profit"})
        return {"close_positions": close_positions}

    async def act(self, decision: Dict[str, Any]):
        for cp in decision.get("close_positions", []):
            pid, reason = cp["pid"], cp["reason"]
            assert self.engine is not None
            success = await self.engine.close_position(int(pid), reason)
            if success:
                # Get PnL from the closed trade
                pnl = 0.0
                trade = self.engine.get_trade_by_id(pid)
                if trade:
                    pnl = trade.pnl
                # Look up cached signal data
                signal_data = self._position_signals.pop(pid, {})
                await self.send(
                    MessageType.POSITION_CLOSED,
                    payload={
                        "position_id": pid,
                        "reason": reason,
                        "pnl": pnl,
                        "symbol": signal_data.get("symbol", ""),
                        "signal": signal_data,
                        "timestamp": time.time(),
                    },
                    priority=MessagePriority.NORMAL,
                )

    async def reflect(self, outcome: Dict[str, Any]):
        if self.consciousness.cycle_count % 50 == 0:
            self.memory.know("execution.executed", self._executed, ttl=3600)
            self.memory.know("execution.failed", self._failed, ttl=3600)

    async def on_message(self, message: AgentMessage):
        if message.msg_type == MessageType.INSTRUMENTS_UPDATED:
            payload = message.payload if isinstance(message.payload, dict) else {}
            tradeable = payload.get("tradeable", [])
            if tradeable:
                new_symbols = [
                    t.get("ticker", "") for t in tradeable if t.get("ticker")
                ]
                if new_symbols:
                    old_count = len(self._active_symbols)
                    from data.data_manager import SYMBOLS as CORE_SYMBOLS

                    # Core symbols (EURUSD, GBPUSD, ...) are always tradeable.
                    # Screener findings are MERGED in — they use Yahoo tickers
                    # (e.g. HO=F, RB=F) which may not have a cTrader symbol_id,
                    # so subscription/trading will gracefully fall through.
                    merged = list(dict.fromkeys(list(CORE_SYMBOLS) + new_symbols))
                    self._active_symbols = merged
                    self.log_state(
                        f"Active symbols: {len(merged)} "
                        f"({len(CORE_SYMBOLS)} core + {len(new_symbols)} screener)"
                    )
            return

        if message.msg_type == MessageType.RISK_APPROVED:
            await self._execute(message)
        elif message.msg_type == MessageType.AGENT_DIRECTIVE:
            payload = message.payload if isinstance(message.payload, dict) else {}
            action = payload.get("action", "")
            if action == "close_all":
                assert self.engine is not None
                await self.engine.close_all_positions(
                    payload.get("reason", "directive")
                )
            elif action == "close_position":
                pid = payload.get("position_id")
                if pid:
                    assert self.engine is not None
                    await self.engine.close_position(
                        int(pid), payload.get("reason", "directive")
                    )
            elif action == "reconnect":
                self.log_state("Reconnecting to cTrader...")
                await self._reconnect()
        elif message.msg_type == MessageType.POSITION_MODIFIED:
            payload = message.payload if isinstance(message.payload, dict) else {}
            pid = payload.get("position_id", 0)
            sl = payload.get("sl")
            tp = payload.get("tp")
            close_ratio = payload.get("close_ratio", 0)
            symbol = payload.get("symbol", "")
            if pid:
                assert self.engine is not None
                # Handle partial close
                if close_ratio > 0:
                    pnl = self.engine.partial_close(int(pid), close_ratio)
                    if pnl is not None:
                        self.log_state(
                            f"Partial close {pid} ({symbol}): "
                            f"{close_ratio*100:.0f}%, PnL=${pnl:.2f}"
                        )
                        # Apply SL update for remaining position
                        if sl is not None:
                            await self.engine.modify_position(int(pid), sl=sl, tp=tp)
                        # Notify system about the partial close
                        await self.send(
                            MessageType.POSITION_CLOSED,
                            payload={
                                "position_id": pid,
                                "reason": "partial_take_profit",
                                "pnl": pnl,
                                "symbol": symbol,
                                "timestamp": time.time(),
                            },
                        )
                    else:
                        self.log_state(
                            f"Failed to partial close position {pid}", "warning"
                        )
                else:
                    success = await self.engine.modify_position(int(pid), sl=sl, tp=tp)
                    if success:
                        self.log_state(
                            f"Modified position {pid} ({symbol}): "
                            f"{'SL=' + str(sl) if sl is not None else ''}"
                            f"{' TP=' + str(tp) if tp is not None else ''}"
                        )
                    else:
                        self.log_state(f"Failed to modify position {pid}", "warning")
        elif message.msg_type == MessageType.DIAGNOSTIC_REQUEST:
            await self.send(
                MessageType.DIAGNOSTIC_RESULT,
                payload={
                    "agent": self.name,
                    "executed": self._executed,
                    "failed": self._failed,
                    "connected": self.get_world("execution.connected", False),
                    "open_positions": (
                        len(self.engine.get_open_positions()) if self.engine else 0
                    ),
                },
                target=message.source_agent,
            )

    async def _execute(self, message: AgentMessage):
        if self.engine is None:
            return
        payload = message.payload if isinstance(message.payload, dict) else {}
        signal = payload.get("signal", {})
        symbol = signal.get("symbol", "")
        direction = signal.get("direction", "")
        volume = payload.get("volume", 0)
        sl_price = payload.get("sl_price", 0)
        tp_price = payload.get("tp_price", 0)
        if direction == "HOLD" or volume <= 0:
            return

        trade = await self.engine.open_position(
            symbol=symbol,
            direction=direction,
            volume=volume,
            sl=sl_price,
            tp=tp_price,
            reason=f"agentic_signal_conf={signal.get('confidence', 0):.2f}",
        )
        if trade:
            self._executed += 1
            # Cache original signal for PnL tracking at close time
            self._position_signals[trade.position_id] = signal
            await self.send(
                MessageType.EXECUTION_RESULT,
                payload={
                    "success": True,
                    "symbol": symbol,
                    "direction": direction,
                    "volume": volume,
                    "filled_price": trade.entry_price,
                    "position_id": trade.position_id,
                    "signal": signal,  # Include original signal for expert tracking
                    "timestamp": time.time(),
                },
                priority=MessagePriority.HIGH,
                requires_ack=True,
            )
            await self.send(
                MessageType.POSITION_OPENED,
                payload={
                    "position_id": trade.position_id,
                    "symbol": symbol,
                    "direction": direction,
                    "volume": volume,
                    "entry": trade.entry_price,
                    "sl_price": trade.sl,
                    "tp_price": trade.tp,
                    "strategy": payload.get("strategy", "?"),
                    "session": payload.get("session", "?"),
                    "confidence": payload.get("confidence", 0.5),
                },
            )
        else:
            self._failed += 1
            await self.send(
                MessageType.EXECUTION_RESULT,
                payload={
                    "success": False,
                    "symbol": symbol,
                    "error": "order_failed",
                    "timestamp": time.time(),
                },
                priority=MessagePriority.HIGH,
            )

    async def _get_account_info(self) -> Dict:
        if self.engine is None:
            return {
                "balance": self.initial_balance,
                "equity": self.initial_balance,
                "margin": 0,
            }
        result = self.engine.get_account_info()
        return await result if asyncio.iscoroutine(result) else result

    def _get_live_price(self, symbol: str) -> float:
        if self.engine is None or self.engine.data is None:
            return 0.0
        try:
            tick = self.engine.data.get_tick_buffer(symbol, 1)
            if tick and len(tick) > 0:
                last = tick[-1]
                if isinstance(last, dict):
                    mid = last.get("mid") or last.get("bid") or last.get("price", 0)
                    return float(mid) if mid > 0 else 0.0
            return self.engine.data.get_price(symbol, "1h")
        except Exception:
            return 0.0
