"""
Dashboard — FastAPI server that exposes bot state via REST + WebSocket.
The bot writes to latest_state; the dashboard reads it.
"""
import time
import json
import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from loguru import logger

app = FastAPI(title="RTS: AI Moneybot System Elite Dashboard", version="4.0")
connected_clients: List[WebSocket] = []

latest_state: Dict[str, Any] = {
    "connected": False,
    "balance": 100000.0,
    "equity": 100000.0,
    "margin": 0.0,
    "free_margin": 100000.0,
    "initial_balance": 100000.0,
    "total_trades": 0,
    "win_rate": 0.0,
    "mode": "PAPER",
    "regime": "starting...",
    "open_positions": [],
    "trade_history": [],
    "market_data": {"prices": {}, "spread": 0.0},
    "ai_metrics": {
        "regime": "starting...",
        "sentiment": 0.0,
        "econ_suppressed": False,
        "econ_next_event": "",
        "upcoming_events": 0,
        "active_decisions": 0,
        "var": 0.0,
        "cvar": 0.0,
    },
    "status": "initializing",
    "timestamp": time.time(),
}


def update_state(**kwargs):
    """Called by main bot to push state updates."""
    latest_state.update(kwargs)
    latest_state["timestamp"] = time.time()


@app.get("/")
async def get_dashboard():
    html_path = os.path.join(os.path.dirname(__file__), "..", "..", "dashboard.html")
    if os.path.exists(html_path):
        with open(html_path) as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Dashboard</h1><p>dashboard.html not found</p>")


@app.get("/api/data")
async def get_data():
    """REST endpoint — returns full bot state for polling clients."""
    return latest_state


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.append(ws)
    logger.info(f"Dashboard client connected ({len(connected_clients)} total)")
    try:
        # Send current state immediately on connect
        await ws.send_json(latest_state)
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        connected_clients.remove(ws)


async def broadcast_update(state: dict):
    """Push state to all connected WebSocket clients."""
    dead = []
    for c in connected_clients:
        try:
            await c.send_json(state)
        except Exception:
            dead.append(c)
    for c in dead:
        if c in connected_clients:
            connected_clients.remove(c)


@app.get("/health")
async def health():
    elapsed = time.time() - latest_state.get("timestamp", time.time())
    return {
        "status": "ok" if elapsed < 60 else "stale",
        "last_update_s": round(elapsed, 1),
        "clients": len(connected_clients),
    }
