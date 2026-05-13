"""
Execution Agent — G1: tick wiring, G3: position publishing, G5: delivery ack.
"""

from __future__ import annotations
import time
import asyncio
from typing import Dict, List, Optional, Any, Set
from loguru import logger

from agentic.core.base_agent import BaseAgent
from agentic.core.agent_message import MessageType, MessagePriority, AgentIntention
from agentic.core.agent_consciousness import ConsciousnessLevel


class ExecutionAgent(BaseAgent):
    def __init__(self, config, secrets, initial_balance: float = 100_000.0):
        super().__init__(
            name="execution_agent",
            role="Order Execution Engine",
            purpose="Execute trades through broker, wire live ticks, publish position state",
            domain="execution",
            capabilities={
                "order_execution", "broker_connection", "position_tracking",
                "sl_tp_monitoring", "paper_trading", "order_lifecycle",
                "pnl_calculation", "account_info", "tick_wiring",  # G1
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

        self.subscribe(MessageType.RISK_APPROVED)
        self.subscribe(MessageType.AGENT_DIRECTIVE)
        self.subscribe(MessageType.DIAGNOSTIC_REQUEST)

    async def _on_start(self):
        self.consciousness.current_intention = "initializing execution provider"
        from api.provider_factory import create_execution_provider
        from execution.engine import ExecutionEngine

        self.ctrader, data_provider = create_execution_provider(self.secrets)
        engine_mode = "LIVE" if not self.secrets.is_demo else "PAPER"
        self.engine = ExecutionEngine(
            self.ctrader, None, None,
            initial_balance=self.initial_balance, mode=engine_mode,
        )

        # G1: Wire live price feed from broker to data pipeline
        async def _on_tick(depth):
            """Forward broker ticks to data_agent and store live spread in world state."""
            try:
                symbol = getattr(depth, 'symbol', '')
                bid = getattr(depth, 'bid', 0)
                ask = getattr(depth, 'ask', 0)
                ts = time.time()

                # Compute live spread in pips and store in world state
                if bid > 0 and ask > 0:
                    from execution.cost_model import CostModel
                    pip_size = CostModel.pip_to_price(symbol)
                    spread_pips = (ask - bid) / pip_size if pip_size > 0 else 0
                    self.set_world(f"data.spread.{symbol}", round(spread_pips, 2))
                    self.set_world(f"data.bid.{symbol}", round(float(bid), 5))
                    self.set_world(f"data.ask.{symbol}", round(float(ask), 5))
                    self.set_world(f"data.price.{symbol}", round(float((bid + ask) / 2), 5))

                await self.send(
                    MessageType.TICK_RECEIVED,
                    payload={
                        "symbol": symbol,
                        "bid": bid,
                        "ask": ask,
                        "volume": getattr(depth, 'volume', 0),
                        "timestamp": ts,
                    },
                    target_capability="tick_ingestion",
                    priority=MessagePriority.HIGH,
                )
            except Exception:
                pass

        if self.ctrader:
            self.ctrader.on_price = _on_tick if hasattr(self.ctrader, 'on_price') else None
            if hasattr(self.ctrader, 'on_market_data'):
                self.ctrader.on_market_data = _on_tick

        connected = await self.ctrader.start()
        if connected and hasattr(self.ctrader, 'is_connected') and self.ctrader.is_connected():
            self.log_state(f"cTrader connected ({engine_mode})")
            self.set_world("execution.connected", True)
        else:
            self.log_state(f"Running in {engine_mode} mode (simulation)", "warning")
            self.set_world("execution.connected", False)

        self.set_world("execution.mode", engine_mode)
        self.set_world("execution.status", "ready")

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

        return {"open_positions": n_positions, "account": account,
                "sl_hits": sl_hits, "tp_hits": tp_hits}

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
            success = await self.engine.close_position(int(pid), reason)
            if success:
                await self.send(
                    MessageType.POSITION_CLOSED,
                    payload={"position_id": pid, "reason": reason, "timestamp": time.time()},
                    priority=MessagePriority.NORMAL,
                )

    async def reflect(self, outcome: Dict[str, Any]):
        if self.consciousness.cycle_count % 50 == 0:
            self.memory.know("execution.executed", self._executed, ttl=3600)
            self.memory.know("execution.failed", self._failed, ttl=3600)

    async def on_message(self, message: AgentMessage):
        if message.msg_type == MessageType.RISK_APPROVED:
            await self._execute(message)
        elif message.msg_type == MessageType.AGENT_DIRECTIVE:
            payload = message.payload if isinstance(message.payload, dict) else {}
            action = payload.get("action", "")
            if action == "close_all":
                await self.engine.close_all_positions(payload.get("reason", "directive"))
            elif action == "close_position":
                pid = payload.get("position_id")
                if pid:
                    await self.engine.close_position(int(pid), payload.get("reason", "directive"))
        elif message.msg_type == MessageType.DIAGNOSTIC_REQUEST:
            await self.send(MessageType.DIAGNOSTIC_RESULT, payload={
                "agent": self.name, "executed": self._executed, "failed": self._failed,
                "connected": self.get_world("execution.connected", False),
                "open_positions": len(self.engine.get_open_positions()) if self.engine else 0,
            }, target=message.source_agent)

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
            symbol=symbol, direction=direction, volume=volume,
            sl=sl_price, tp=tp_price,
            reason=f"agentic_signal_conf={signal.get('confidence', 0):.2f}",
        )
        if trade:
            self._executed += 1
            await self.send(MessageType.EXECUTION_RESULT, payload={
                "success": True, "symbol": symbol, "direction": direction,
                "volume": volume, "filled_price": trade.entry_price,
                "position_id": trade.position_id, "timestamp": time.time(),
            }, priority=MessagePriority.HIGH, requires_ack=True)
            await self.send(MessageType.POSITION_OPENED, payload={
                "position_id": trade.position_id, "symbol": symbol,
                "direction": direction, "volume": volume, "entry": trade.entry_price,
            })
        else:
            self._failed += 1
            await self.send(MessageType.EXECUTION_RESULT, payload={
                "success": False, "symbol": symbol, "error": "order_failed",
                "timestamp": time.time(),
            }, priority=MessagePriority.HIGH)

    async def _get_account_info(self) -> Dict:
        if self.engine is None:
            return {"balance": self.initial_balance, "equity": self.initial_balance, "margin": 0}
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
