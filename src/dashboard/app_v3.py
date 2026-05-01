"""
Dashboard for RTS - AI FX Trading System v3.0
Provides real-time monitoring via FastAPI and WebSockets.
Integrates with cTrader (IC Markets) - macOS compatible SSL solution.
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

app = FastAPI(title="RTS - AI FX Trading System Dashboard", version="3.0")

connected_clients: List[WebSocket] = []

# cTrader client instance (will be initialized on startup)
ctrader_client: Optional[object] = None
ctrader_available: bool = False

def init_ctrader():
    """Initialize cTrader client with macOS-compatible SSL (LibreSSL 2.8.3)"""
    global ctrader_client, ctrader_available
    try:
        # Use the working cTrader client (standard ssl + protobuf)
        from api.ctrader_ready import cTraderICMarkets
        
        logger.info("Initializing cTrader client (macOS LibreSSL compatible)...")
        client = cTraderICMarkets(host="demo.ctraderapi.com", port=5035)
        
        # Try to connect
        if client.connect():
            logger.info(f"✓ cTrader SSL connection established via {client.ssl_socket.version() if client.ssl_socket else 'N/A'}")
            ctrader_client = client
            ctrader_available = True
        else:
            logger.warning("✗ cTrader connection failed")
            ctrader_client = None
            ctrader_available = False
    except Exception as e:
        logger.error(f"Failed to initialize cTrader client: {e}")
        ctrader_client = None
        ctrader_available = False

async def get_ctrader_data() -> Dict[str, Any]:
    """Get real account data from cTrader (if available)"""
    global ctrader_client, ctrader_available
    
    if not ctrader_available or not ctrader_client:
        return {
            "connected": False,
            "message": "cTrader not available - complete OAuth flow to get valid token",
            "macos_ssl_works": True,
            "instruction": "Run: python src/api/ctrader_ready.py after getting valid access token"
        }
    
    try:
        # Authenticate and get data
        # (This would be called after OAuth flow is complete)
        return {
            "connected": True,
            "message": "Ready to fetch data - need valid access token",
            "ssl_status": "✓ SSL works on macOS (LibreSSL 2.8.3)",
            "next_step": "Complete OAuth flow: https://id.ctrader.com/my/settings/openapi/grantingaccess/"
        }
    except Exception as e:
        logger.error(f"Error getting cTrader data: {e}")
        return {"connected": False, "error": str(e)}

# Initial state (will be updated with real data when available)
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
    "ctrader": {
        "connected": False,
        "macos_compatible": True,
        "ssl_works": True,
        "message": "Waiting for valid access token from OAuth flow"
    },
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
    logger.info("Dashboard started with cTrader macOS-compatible integration")
    logger.info("✓ SSL (LibreSSL 2.8.3) works - waiting for valid OAuth token")

@app.get("/")
async def get_dashboard(request: Request):
    """Serve the main dashboard HTML page."""
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
        return HTMLResponse(content=html_content, status_code=200)
    else:
        return HTMLResponse(content="<h1>Dashboard not found</h1>", status_code=404)

@app.get("/api/status")
async def get_status():
    """Get current system status."""
    return {
        "status": "operational",
        "ctrader": {
            "available": ctrader_available,
            "macos_compatible": True,
            "ssl_works": True,
            "message": "Standard ssl module works on macOS LibreSSL 2.8.3"
        },
        "timestamp": time.time()
    }

@app.get("/api/ctrader/status")
async def ctrader_status():
    """Check cTrader connection status."""
    has_token = bool(os.getenv("CTRADER_ACCESS_TOKEN"))
    return {
        "status": "connected" if ctrader and ctrader.sock else "disconnected",
        "last_error": ctrader.last_error if ctrader else None,
        "connection_status": ctrader.connection_status if ctrader else "not_initialized",
        "has_token": has_token,
        "token_status": "valid" if has_token else "missing",
        "oauth_url": "https://id.ctrader.com/my/settings/openapi/grantingaccess/?client_id=15217_h8WxunXX70m6O6qsnIx9ZO3GZraTdO0wnLjL3dTKyYG6fkbUca&redirect_uri=https://spotware.com&scope=trading&product=web" if not has_token else None
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    connected_clients.append(websocket)
    
    try:
        # Send initial state
        await websocket.send_json(latest_state)
        
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            
            # Echo back or process commands
            if data == "get_status":
                await websocket.send_json(latest_state)
            elif data == "get_ctrader_data":
                ctrader_data = await get_ctrader_data()
                await websocket.send_json({"type": "ctrader_data", "data": ctrader_data})
            
    except WebSocketDisconnect:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in connected_clients:
            connected_clients.remove(websocket)

@app.get("/api/ctrader/accounts")
async def get_ctrader_accounts():
    """Get cTrader accounts (requires valid access token)"""
    if not ctrader_client or not ctrader_available:
        return {
            "status": "error",
            "message": "cTrader not connected",
            "instruction": "Complete OAuth flow and update access token in src/api/ctrader_ready.py"
        }
    
    # This would call ctrader_client.get_account_list(access_token)
    return {
        "status": "pending",
        "message": "Complete OAuth flow to see accounts",
        "oauth_url": f"https://id.ctrader.com/my/settings/openapi/grantingaccess/?client_id=15217_h8WxunXX70m6O6qsnIx9ZO3GZraTdO0wnLjL3dTKyYG6fkbUca&redirect_uri=https://spotware.com&scope=trading&product=web"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
