"""
Dashboard for AI Forex Trading System v3.0
Provides real-time monitoring via FastAPI and WebSockets.
Integrates with cTrader (IC Markets) for live data.
"""
import asyncio
import json
import time
import os
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from loguru import logger
from typing import List, Dict, Any

app = FastAPI(title="AI Forex Trading System Dashboard", version="3.0")

connected_clients: List[WebSocket] = []

latest_state = {
    "equity": 100000.0,
    "balance": 100000.0,
    "margin": 0.0,
    "free_margin": 100000.0,
    "open_positions": [],
    "trade_history": [],
    "market_data": {},
    "ai_metrics": {},
    "signals": [],
    "ctrader": {"connected": False, "account_id": "6100830", "demo": True},
    "timestamp": time.time(),
}

@app.get("/")
async def get_dashboard(request: Request):
    """Serve the main dashboard HTML page."""
    # Try multiple possible locations for dashboard.html
    possible_paths = [
        Path(__file__).parent.parent.parent / "dashboard.html",
        Path(__file__).parent.parent / "dashboard.html",
        Path("dashboard.html"),
    ]
    
    dashboard_path = None
    for path in possible_paths:
        if path.exists():
            dashboard_path = path
            break
    
    if dashboard_path:
        with open(dashboard_path, "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    else:
        return HTMLResponse(content="<h1>Dashboard not found</h1><p>Expected at: dashboard.html</p>")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    logger.info(f"Dashboard client connected | Total: {len(connected_clients)}")
    try:
        await websocket.send_json(latest_state)
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_clients.remove(websocket)
        logger.info(f"Client disconnected | Total: {len(connected_clients)}")

async def broadcast_update(state: dict):
    """Broadcast updates to all connected dashboard clients."""
    disconnected = []
    for client in connected_clients:
        try:
            await client.send_json(state)
        except:
            disconnected.append(client)
    for client in disconnected:
        if client in connected_clients:
            connected_clients.remove(client)

def update_state(**kwargs):
    """Update the latest state with new data."""
    global latest_state
    latest_state.update(kwargs)
    latest_state["timestamp"] = time.time()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time(), "version": "3.0"}

@app.get("/api/state")
async def get_state():
    """Return current state for API consumers."""
    return latest_state

@app.post("/api/update")
async def update_from_trader(data: Dict[str, Any]):
    """Update dashboard state from trader system."""
    update_state(**data)
    await broadcast_update(data)
    return {"status": "updated"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
