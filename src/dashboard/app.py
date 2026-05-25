"""
Dashboard — FastAPI server that exposes bot state via REST + WebSocket.
The bot writes to latest_state; the dashboard reads it.
"""

import time
import os
from typing import List, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from loguru import logger

app = FastAPI(title="RTS: Agentic FX System Elite Dashboard", version="4.0")
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
    state = latest_state
    pos_count = len(state.get("open_positions", []))
    bal = state.get("balance", 0)
    eq = state.get("equity", 0)
    dd = (
        (state.get("initial_balance", bal) - bal)
        / max(state.get("initial_balance", bal), 1)
        * 100
    )

    # System-level health checks
    from pathlib import Path

    checks = {}

    # Key imports
    for mod_name in ["numpy", "pandas", "torch", "tensorflow"]:
        try:
            __import__(mod_name)
            checks[mod_name] = True
        except ImportError:
            checks[mod_name] = False

    # Config files
    checks["config.yaml"] = Path("config.yaml").exists()
    checks[".env"] = Path(".env").exists()

    # Models directory
    checks["models/"] = Path("models").is_dir()

    # Ensemble config
    checks["ensemble_config.json"] = Path("models/ensemble_config.json").exists()

    # Feature normalization
    checks["feature_norm.npz"] = Path("models/feature_norm.npz").exists()

    # Data directory
    checks["data/"] = Path("data/historical").is_dir()

    ok_count = sum(1 for v in checks.values() if v)
    fail_count = sum(1 for v in checks.values() if not v)

    if fail_count == 0:
        sys_status = "ok"
    elif ok_count > fail_count:
        sys_status = "degraded"
    else:
        sys_status = "failed"

    return {
        "status": sys_status,
        "timestamp": time.time(),
        "version": "4.0.0-agentic",
        "checks": checks,
        "summary": {"ok": ok_count, "fail": fail_count},
        "runtime": {
            "status": "ok" if elapsed < 60 else "stale",
            "last_update_s": round(elapsed, 1),
            "clients": len(connected_clients),
            "trading": state.get("mode", "UNKNOWN"),
            "balance": round(bal, 2),
            "equity": round(eq, 2),
            "drawdown_pct": round(max(dd, 0), 2),
            "open_positions": pos_count,
            "total_trades": state.get("total_trades", 0),
            "win_rate": state.get("win_rate", 0),
            "regime": state.get("regime", "unknown"),
            "signal_count": state.get("signal_count", 0),
        },
    }
