"""
Dashboard for AI Forex Trading System v3.0
Provides real-time monitoring via FastAPI and WebSockets.
Integrates with cTrader (IC Markets) for ALL available data.
"""
import asyncio
import json
import time
import os
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from loguru import logger
from typing import List, Dict, Any, Optional

app = FastAPI(title="AI Forex Trading System Dashboard", version="3.0")

connected_clients: List[WebSocket] = []

# cTrader client instance
ctrader_client: Optional[object] = None


def init_ctrader():
    """Initialize cTrader client with proper IC Markets connection."""
    global ctrader_client
    try:
        # Use the proper cTrader client with correct endpoints
        from api.ctrader_icmarkets import FixedCtraderClient
        ctrader_client = FixedCtraderClient(demo=True)
        
        # Try to connect (will fail on macOS due to network blocking)
        success = ctrader_client.start()
        
        if success and ctrader_client.is_connected():
            logger.info("✓ cTrader REAL connection established!")
            logger.info("  Account data will be pulled from IC Markets")
        else:
            logger.warning("✗ cTrader connection failed (network blocked on macOS)")
            logger.info("  System will use simulation mode for testing")
            logger.info("  To get real data, deploy to Linux server or use Docker")
            ctrader_client = None
    except Exception as e:
        logger.error(f"Failed to initialize cTrader client: {e}")
        ctrader_client = None
    except Exception as e:
        logger.error(f"Failed to initialize cTrader: {e}")
        ctrader_client = None


latest_state = {
    "connected": False,
    "equity": 10000.0,
    "balance": 10000.0,
    "margin": 0.0,
    "free_margin": 10000.0,
    "leverage": "1:30",
    "open_positions": [],
    "trade_history": [],
    "market_data": {},
    "ai_metrics": {},
    "signals": [],
    "ctrader": {"connected": False, "account_id": "6100830", "demo": True},
    "broker": {
        "name": "IC Markets",
        "regulation": "ASIC, CySEC",
        "max_leverage": "1:500",
    },
    "forex_pairs": [],
    "account_info": {},
    "timestamp": time.time(),
}


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    init_ctrader()
    logger.info("Dashboard started with cTrader integration")


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
        # Send initial state with all cTrader data
        state = get_full_state()
        await websocket.send_json(state)
        
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


def get_full_state() -> Dict[str, Any]:
    """Get full state including ALL cTrader data."""
    global latest_state
    
    if ctrader_client and ctrader_client.is_connected():
        # Get comprehensive data from cTrader client
        dashboard_data = ctrader_client.get_dashboard_data()
        
        # Update latest_state with all available data
        latest_state.update({
            "connected": True,
            "equity": dashboard_data.get("account", {}).get("equity", 10000.0),
            "balance": dashboard_data.get("account", {}).get("balance", 10000.0),
            "margin": dashboard_data.get("account", {}).get("margin", 0.0),
            "free_margin": dashboard_data.get("account", {}).get("free_margin", 10000.0),
            "leverage": dashboard_data.get("account", {}).get("leverage", "1:30"),
            "market_data": dashboard_data.get("market_data", {}),
            "open_positions": dashboard_data.get("open_positions", []),
            "forex_pairs": dashboard_data.get("forex_pairs", []),
            "broker": dashboard_data.get("broker", {}),
            "account_info": dashboard_data.get("account", {}),
            "ctrader": {
                "connected": True,
                "account_id": dashboard_data.get("account_id", ""),
                "demo": dashboard_data.get("demo", True),
            },
            "timestamp": time.time(),
        })
    else:
        latest_state["connected"] = False
    
    return latest_state


@app.get("/api/data")
async def get_data():
    """Get ALL data for the dashboard from cTrader."""
    try:
        state = get_full_state()
        
        # Get AI signals from latest file
        signals = []
        signal_file = "signals_live.json"
        if os.path.exists(signal_file):
            try:
                with open(signal_file) as f:
                    latest = json.load(f)
                    signals = latest.get("signals", [])
            except:
                pass
        
        # Get equity data (last 30 days)
        equity_data = []
        equity_file = "equity_curve.csv"
        if os.path.exists(equity_file):
            try:
                import pandas as pd
                df = pd.read_csv(equity_file)
                equity_data = df.tail(30)["equity"].tolist()
            except:
                equity_data = [10000] * 30
        else:
            equity_data = [10000] * 30
        
        # Combine all data
        state["signals"] = signals[:10]
        state["equity_data"] = equity_data
        
        return state
    except Exception as e:
        logger.error(f"Error getting data: {e}")
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ctrader/account")
async def get_ctrader_account():
    """Get complete account information from cTrader."""
    if ctrader_client and ctrader_client.is_connected():
        return ctrader_client.get_account_info().__dict__
    return {"error": "cTrader not connected"}


@app.get("/api/ctrader/broker")
async def get_ctrader_broker():
    """Get broker information (IC Markets)."""
    if ctrader_client:
        return ctrader_client.get_broker_info().__dict__
    return {"error": "cTrader not initialized"}


@app.get("/api/ctrader/pairs")
async def get_ctrader_pairs():
    """Get all available forex pairs."""
    if ctrader_client and ctrader_client.is_connected():
        return [p.__dict__ for p in ctrader_client.get_forex_pairs()]
    return {"error": "cTrader not connected"}


@app.get("/api/ctrader/market/{symbol}")
async def get_market_data(symbol: str):
    """Get market data for a specific symbol."""
    if ctrader_client and ctrader_client.is_connected():
        depth = ctrader_client.get_market_depth(symbol)
        if depth:
            return depth.__dict__
    return {"error": "Symbol not found or cTrader not connected"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "ctrader_connected": ctrader_client.is_connected() if ctrader_client else False,
        "timestamp": time.time(),
        "version": "3.0",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
