"""
Unified Dashboard — FastAPI app wired to AgentBus for real-time streaming.
Replaces both app.py and smart_dashboard.py.

Serves:
  - /  (HTML dashboard from dashboard.html)
  - /api/*  (REST state endpoints)
  - /ws  (WebSocket for real-time streaming)
  - /health  (health check)
"""

from __future__ import annotations

import asyncio
import time
import os
from typing import Dict, List, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from loguru import logger

from agentic.core.agent_bus import get_agent_bus
from agentic.core.agent_message import MessageType

app = FastAPI(title="RTS: Agentic FX System Elite Dashboard", version="4.1")

# Connected WebSocket clients
connected_clients: List[WebSocket] = []

# Consolidated dashboard state (updated by bus listeners)
dashboard_state: Dict[str, Any] = {
    "connected": False,
    "balance": 100000.0,
    "equity": 100000.0,
    "margin": 0.0,
    "free_margin": 100000.0,
    "initial_balance": 100000.0,
    "total_trades": 0,
    "win_rate": 0.0,
    "mode": "PAPER",
    "regime": "waiting...",
    "open_positions": [],
    "trade_history": [],
    "market_data": {"prices": {}, "spread": 0.0},
    "ai_metrics": {},
    "risk_metrics": {},
    "equity_curve": [],
    "signal_count": 0,
    "agent_health": {},
    "alerts": [],
    "strategy_attribution": {},
    "last_update": time.time(),
}


@app.on_event("startup")
async def startup():
    """Start background task that bridges AgentBus to WebSocket clients."""
    asyncio.create_task(_bridge_agent_bus_to_dashboard())


@app.on_event("shutdown")
async def shutdown():
    """Cleanly disconnect WebSocket clients on shutdown."""
    dead = list(connected_clients)
    connected_clients.clear()
    for client in dead:
        try:
            await client.close()
        except Exception:
            pass


@app.get("/")
async def get_dashboard():
    """Serve the HTML dashboard."""
    html_path = os.path.join(os.path.dirname(__file__), "..", "..", "dashboard.html")
    if os.path.exists(html_path):
        with open(html_path) as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>RTS Dashboard</h1><p>dashboard.html not found</p>")


@app.get("/api/data")
async def get_data():
    """REST endpoint — returns full dashboard state."""
    return dashboard_state


@app.get("/api/positions")
async def get_positions():
    """Get current open positions."""
    return dashboard_state.get("open_positions", [])


@app.get("/api/equity_curve")
async def get_equity_curve():
    """Get equity curve data points."""
    return dashboard_state.get("equity_curve", [])


@app.get("/api/risk_metrics")
async def get_risk_metrics():
    """Get risk metrics."""
    return dashboard_state.get("risk_metrics", {})


@app.get("/api/agent_health")
async def get_agent_health():
    """Get agent health status."""
    return dashboard_state.get("agent_health", {})


@app.get("/api/alerts")
async def get_alerts():
    """Get recent alerts."""
    return dashboard_state.get("alerts", [])


@app.get("/api/trade_history")
async def get_trade_history():
    """Get trade history."""
    return dashboard_state.get("trade_history", [])


@app.get("/health")
async def health():
    """Health check endpoint."""
    elapsed = time.time() - dashboard_state.get("last_update", time.time())
    return {
        "status": "ok" if elapsed < 60 else "stale",
        "timestamp": time.time(),
        "version": "4.1.0-dashboard",
        "uptime_s": round(elapsed, 1),
        "clients": len(connected_clients),
        "trading": dashboard_state.get("mode", "UNKNOWN"),
        "balance": round(dashboard_state.get("balance", 0), 2),
        "equity": round(dashboard_state.get("equity", 0), 2),
        "open_positions": len(dashboard_state.get("open_positions", [])),
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket for real-time streaming updates.

    - Sends full dashboard state immediately on connect
    - Responds to 'ping' with {'type': 'pong'}
    - Broadcasts state updates every ~1s via bridge task
    """
    await ws.accept()
    connected_clients.append(ws)
    logger.info(f"Dashboard WS client connected ({len(connected_clients)} total)")

    # Send current state immediately
    try:
        await ws.send_json(dashboard_state)
    except Exception:
        pass

    try:
        while True:
            msg = await ws.receive_text()
            if msg == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        if ws in connected_clients:
            connected_clients.remove(ws)
    except Exception:
        if ws in connected_clients:
            connected_clients.remove(ws)


async def _bridge_agent_bus_to_dashboard():
    """Listen to AgentBus messages and update dashboard state.

    Runs in the background, subscribing to key message types and
    updating dashboard_state + broadcasting to WebSocket clients.
    """
    try:
        bus = get_agent_bus()

        # Define handlers

        async def on_tick(msg):
            """Update market data on tick."""
            payload = msg.payload if isinstance(msg.payload, dict) else {}
            symbol = payload.get("symbol", "")
            bid = payload.get("bid", 0)
            ask = payload.get("ask", 0)
            if symbol and bid > 0:
                dashboard_state.setdefault("market_data", {})
                dashboard_state["market_data"].setdefault("prices", {})
                dashboard_state["market_data"]["prices"][symbol] = (bid + ask) / 2
                dashboard_state["market_data"]["spread"] = round(ask - bid, 5)

        async def on_position_opened(msg):
            """Update open positions list."""
            payload = msg.payload if isinstance(msg.payload, dict) else {}
            pos = {
                "symbol": payload.get("symbol", ""),
                "direction": payload.get("direction", ""),
                "volume": payload.get("volume", 0),
                "entry_price": payload.get("entry", 0),
                "sl": payload.get("sl_price", 0),
                "tp": payload.get("tp_price", 0),
                "position_id": payload.get("position_id", 0),
                "timestamp": time.time(),
            }
            positions = dashboard_state.get("open_positions", [])
            positions.append(pos)
            dashboard_state["open_positions"] = positions
            dashboard_state["total_trades"] = dashboard_state.get("total_trades", 0) + 1

        async def on_position_closed(msg):
            """Remove from open positions, add to trade history."""
            payload = msg.payload if isinstance(msg.payload, dict) else {}
            pid = payload.get("position_id", 0)
            pnl = payload.get("pnl", 0)
            # Remove from open positions
            positions = dashboard_state.get("open_positions", [])
            dashboard_state["open_positions"] = [
                p for p in positions if p.get("position_id") != pid
            ]
            # Add to trade history
            history = dashboard_state.get("trade_history", [])
            history.append(
                {
                    "position_id": pid,
                    "pnl": pnl,
                    "reason": payload.get("reason", ""),
                    "timestamp": time.time(),
                }
            )
            dashboard_state["trade_history"] = history[-500:]  # Keep last 500

        async def on_signal(msg):
            """Update signal count."""
            dashboard_state["signal_count"] = dashboard_state.get("signal_count", 0) + 1

        async def on_heartbeat(msg):
            """Update agent health."""
            payload = msg.payload if isinstance(msg.payload, dict) else {}
            agent = msg.source_agent
            dashboard_state.setdefault("agent_health", {})
            dashboard_state["agent_health"][agent] = {
                "status": payload.get("status", "unknown"),
                "health": payload.get("health", 1.0),
                "last_seen": time.time(),
            }

        async def on_risk_alert(msg):
            """Add alert to dashboard."""
            payload = msg.payload if isinstance(msg.payload, dict) else {}
            alerts = dashboard_state.get("alerts", [])
            alerts.append(
                {
                    "type": payload.get("type", "alert"),
                    "reason": payload.get("reason", ""),
                    "timestamp": time.time(),
                }
            )
            dashboard_state["alerts"] = alerts[-100:]  # Keep last 100

        # Subscribe to bus messages
        bus.subscribe(MessageType.TICK_RECEIVED, on_tick)
        bus.subscribe(MessageType.POSITION_OPENED, on_position_opened)
        bus.subscribe(MessageType.POSITION_CLOSED, on_position_closed)
        bus.subscribe(MessageType.SIGNAL_GENERATED, on_signal)
        bus.subscribe(MessageType.AGENT_HEARTBEAT, on_heartbeat)
        bus.subscribe(MessageType.RISK_ALERT, on_risk_alert)

        # Periodic broadcast loop: every 1s, push state to WebSocket clients
        while True:
            await asyncio.sleep(1.0)
            dashboard_state["last_update"] = time.time()

            # Update equity curve
            equity = dashboard_state.get("equity", dashboard_state.get("balance", 0))
            curve = dashboard_state.get("equity_curve", [])
            curve.append(
                {
                    "timestamp": time.time(),
                    "equity": equity,
                    "balance": dashboard_state.get("balance", 0),
                }
            )
            dashboard_state["equity_curve"] = curve[-1000:]  # Keep last 1000 points

            # Broadcast to all connected clients
            dead: List[WebSocket] = []
            for client in connected_clients:
                try:
                    await client.send_json(dashboard_state)
                except Exception:
                    dead.append(client)
            for client in dead:
                if client in connected_clients:
                    connected_clients.remove(client)

    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Dashboard bridge error: {e}")


def update_state(**kwargs):
    """External API for the bot to push state updates directly."""
    dashboard_state.update(kwargs)
    dashboard_state["last_update"] = time.time()
