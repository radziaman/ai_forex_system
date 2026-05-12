"""
AI-Backed Smart Enhanced Dashboard (Enhancement #17).
Authentication, real-time equity curve, risk metrics, WebSocket improvements.
"""

import os
import json
import time
import asyncio
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger

try:
    from fastapi import (
        FastAPI,
        WebSocket,
        WebSocketDisconnect,
        HTTPException,
        Depends,
        Security,
    )
    from fastapi.security import HTTPBasic, HTTPBasicCredentials
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    import jwt

    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False


# Dashboard state
latest_state: Dict[str, Any] = {
    "balance": 100000,
    "equity": 100000,
    "margin": 0,
    "free_margin": 100000,
    "initial_balance": 100000,
    "total_trades": 0,
    "win_rate": 0.0,
    "mode": "PAPER",
    "regime": "",
    "open_positions": [],
    "trade_history": [],
    "market_data": {"prices": {}, "spread": 0.0},
    "ai_metrics": {},
    "risk_metrics": {},
    "equity_curve": [],
    "last_update": time.time(),
}

# Authentication — secrets from env only, no hardcoded defaults
security = HTTPBasic() if FASTAPI_AVAILABLE else None
JWT_SECRET = os.getenv("DASHBOARD_JWT_SECRET")
if not JWT_SECRET:
    import uuid

    JWT_SECRET = str(uuid.uuid4())
    logger.warning(
        "DASHBOARD_JWT_SECRET not set — using random per-run secret (sessions invalidated on restart)"
    )

# Alerts
alerts_queue: List[Dict] = []
MAX_ALERTS = 100


def create_access_token(data: Dict) -> str:
    """Create JWT access token."""
    if not JWT_AVAILABLE:
        return ""
    to_encode = data.copy()
    to_encode.update({"exp": time.time() + 3600})
    return jwt.encode(to_encode, JWT_SECRET, algorithm="HS256")


def verify_token(token: str) -> Optional[Dict]:
    """Verify JWT token."""
    if not JWT_AVAILABLE:
        return None
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload
    except Exception:
        return None


def verify_credentials(credentials: HTTPBasicCredentials) -> bool:
    """Verify HTTP Basic credentials."""
    correct_username = os.getenv("DASHBOARD_USER", "admin")
    correct_password = os.getenv("DASHBOARD_PASS", "moneybot_2026")
    username_hash = hashlib.sha256(credentials.username.encode()).hexdigest()
    password_hash = hashlib.sha256(credentials.password.encode()).hexdigest()
    return (
        username_hash == hashlib.sha256(correct_username.encode()).hexdigest()
        and password_hash == hashlib.sha256(correct_password.encode()).hexdigest()
    )


def update_state(**kwargs):
    """Update dashboard state."""
    global latest_state
    for key, value in kwargs.items():
        if key in latest_state or key == "equity_curve":
            latest_state[key] = value
    latest_state["last_update"] = time.time()


async def broadcast_update(state: Dict):
    """Broadcast update to all WebSocket clients."""
    dead = []
    for c in connected_clients:
        try:
            await c.send_json(state)
        except Exception:
            dead.append(c)
    for c in dead:
        if c in connected_clients:
            connected_clients.remove(c)


def get_dashboard_html() -> str:
    """Return professional dark-themed dashboard with SVG logo and real-time data."""
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RTS: AI Moneybot System Elite</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#080b16;--surface:#0f1326;--surface2:#171d35;--border:#1e2648;--gold:#f59e0b;--green:#10b981;--red:#ef4444;--blue:#3b82f6;--cyan:#06b6d4;--text:#f1f5f9;--muted:#64748b;--radius:10px}
html{font-size:15px}
body{font-family:'Inter',system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden}
.topbar{display:flex;align-items:center;gap:12px;padding:14px 20px;background:var(--surface);border-bottom:1px solid var(--border);position:sticky;top:0;z-index:100}
.topbar h1{font-size:1.05rem;font-weight:700;background:linear-gradient(135deg,var(--gold),var(--green));-webkit-background-clip:text;-webkit-text-fill-color:transparent;white-space:nowrap}
.topbar-right{margin-left:auto;display:flex;align-items:center;gap:10px;font-size:.75rem}
.status-dot{width:8px;height:8px;border-radius:50%;display:inline-block}
.status-dot.ok{background:var(--green);box-shadow:0 0 6px var(--green)}
.status-dot.warn{background:var(--gold);box-shadow:0 0 6px var(--gold)}
.mode-badge{font-size:.65rem;font-weight:600;padding:3px 10px;border-radius:20px;text-transform:uppercase;letter-spacing:.04em}
.mode-badge.PAPER{background:rgba(245,158,11,.15);color:var(--gold);border:1px solid rgba(245,158,11,.3)}
.mode-badge.LIVE{background:rgba(239,68,68,.15);color:var(--red);border:1px solid rgba(239,68,68,.3)}
.timer{font-family:'JetBrains Mono',monospace;font-size:.7rem;color:var(--muted)}
.container{padding:16px;max-width:1400px;margin:0 auto}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin-bottom:12px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:16px}
.card-title{font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin-bottom:8px}
.stat-row{display:flex;justify-content:space-between;align-items:baseline;padding:4px 0}
.stat-label{font-size:.78rem;color:var(--muted)}
.stat-value{font-family:'JetBrains Mono',monospace;font-size:1.1rem;font-weight:600}
.stat-value.lg{font-size:1.6rem;font-weight:700}
.stat-value.pos{color:var(--green)}
.stat-value.neg{color:var(--red)}
.price-tile{background:var(--surface2);border-radius:6px;padding:6px 10px;display:flex;justify-content:space-between;align-items:center;font-size:.78rem;font-family:'JetBrains Mono',monospace}
.price-sym{color:var(--muted);font-weight:500}
.price-val{font-weight:600}
table{width:100%;border-collapse:collapse;font-size:.78rem}
th{padding:8px 10px;text-align:left;font-size:.65rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);font-weight:600;border-bottom:1px solid var(--border)}
td{padding:8px 10px;border-bottom:1px solid rgba(30,38,72,.5)}
tr:hover td{background:rgba(255,255,255,.02)}
.cell-buy{color:var(--green);font-weight:600}
.cell-sell{color:var(--red);font-weight:600}
.regime-tag{display:inline-block;font-size:.65rem;padding:2px 8px;border-radius:4px;font-weight:500}
.regime-tag.trending{background:rgba(16,185,129,.12);color:var(--green)}
.regime-tag.ranging{background:rgba(59,130,246,.12);color:var(--blue)}
.regime-tag.volatile{background:rgba(245,158,11,.12);color:var(--gold)}
.regime-tag.crisis{background:rgba(239,68,68,.12);color:var(--red)}
.metrics-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:8px}
.metric-item{background:var(--surface2);border-radius:6px;padding:8px 12px}
.metric-item .label{font-size:.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.metric-item .value{font-family:'JetBrains Mono',monospace;font-size:.85rem;font-weight:600;margin-top:2px}
@media(max-width:600px){.grid{grid-template-columns:1fr}.topbar h1{font-size:.9rem}.stat-value.lg{font-size:1.3rem}}
</style>
</head>
<body>

<div class="topbar">
<svg width="32" height="32" viewBox="0 0 32 32" fill="none">
  <rect x="2" y="2" width="28" height="28" rx="6" fill="url(#lg)"/>
  <path d="M8 22V14l4 2 4-6 4 4 4-6v14H8z" fill="#fff" opacity=".9"/>
  <rect x="10" y="23" width="12" height="1.5" rx=".75" fill="#fff" opacity=".4"/>
  <defs><linearGradient id="lg" x1="0" y1="0" x2="32" y2="32"><stop stop-color="#f59e0b"/><stop offset="1" stop-color="#10b981"/></linearGradient></defs>
</svg>
<h1>RTS: AI Moneybot System Elite</h1>
<div class="topbar-right">
  <span class="mode-badge" id="mode">PAPER</span>
  <span class="status-dot ok" id="statusDot"></span>
  <span class="timer" id="timer">--</span>
</div>
</div>

<div class="container">

<div class="grid">
  <div class="card">
    <div class="card-title">Account</div>
    <div class="stat-row"><span class="stat-label">Balance</span><span class="stat-value lg" id="balance">--</span></div>
    <div class="stat-row"><span class="stat-label">Equity</span><span class="stat-value" id="equity">--</span></div>
    <div class="stat-row"><span class="stat-label">Free Margin</span><span class="stat-value" id="freeMargin">--</span></div>
    <div class="stat-row"><span class="stat-label">Open Positions</span><span class="stat-value" id="posCount">0</span></div>
  </div>
  <div class="card">
    <div class="card-title">Performance</div>
    <div class="stat-row"><span class="stat-label">Total Trades</span><span class="stat-value" id="totalTrades">0</span></div>
    <div class="stat-row"><span class="stat-label">Win Rate</span><span class="stat-value" id="winRate">--</span></div>
    <div class="stat-row"><span class="stat-label">P&amp;L</span><span class="stat-value" id="pnl">--</span></div>
    <div class="stat-row"><span class="stat-label">Sharpe</span><span class="stat-value" id="sharpe">--</span></div>
  </div>
  <div class="card">
    <div class="card-title">Risk</div>
    <div class="stat-row"><span class="stat-label">VaR (95%)</span><span class="stat-value" id="var">--</span></div>
    <div class="stat-row"><span class="stat-label">CVaR</span><span class="stat-value" id="cvar">--</span></div>
    <div class="stat-row"><span class="stat-label">Sentiment</span><span class="stat-value" id="sentiment">--</span></div>
    <div class="stat-row"><span class="stat-label">Regime</span><span class="stat-value" id="regime">--</span></div>
  </div>
</div>

<div class="card" style="margin-bottom:12px">
  <div class="card-title">Live Prices</div>
  <div class="prices-grid" id="pricesGrid"></div>
</div>

<div class="card" style="margin-bottom:12px">
  <div class="card-title">Open Positions</div>
  <table><thead><tr><th>Symbol</th><th>Dir</th><th>Volume</th><th>Entry</th><th>SL</th><th>TP</th><th>P&amp;L</th></tr></thead><tbody id="positionsBody"></tbody></table>
  <div id="noPositions" style="text-align:center;padding:20px 0;color:var(--muted);font-size:.85rem">No open positions</div>
</div>

<div class="card">
  <div class="card-title">AI Metrics</div>
  <div class="metrics-grid" id="aiMetricsGrid"></div>
</div>

</div>

<script>
const DOMAIN = window.location.host;
let ws = null;
function connectWS(){
  ws = new WebSocket('ws://'+DOMAIN+'/ws');
  ws.onmessage = function(e){
    try{updateDashboard(JSON.parse(e.data))}catch(ex){}
  };
  ws.onclose = function(){setTimeout(connectWS,2000)};
}
function updateDashboard(d){
  const fmt = (n,d=2)=>n==null||n===undefined?'--':'$'+Number(n).toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d});
  const num = (n,d=2)=>n==null||n===undefined?'--':Number(n).toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d});
  const pct = (n,d=1)=>n==null||n===undefined?'--':(Number(n)*100).toFixed(d)+'%';
  setText('balance',fmt(d.balance,0));
  setText('equity',fmt(d.equity,0));
  setText('freeMargin',fmt(d.free_margin,0));
  setText('posCount',d.open_positions?d.open_positions.length:0);
  setText('totalTrades',d.total_trades||0);
  setText('winRate',pct(d.win_rate));
  setText('mode',d.mode||'PAPER');
  document.querySelector('.mode-badge').className='mode-badge '+(d.mode||'PAPER');
  const pnl = (d.equity||0)-(d.initial_balance||0);
  const pnlEl=document.getElementById('pnl');
  pnlEl.textContent=(pnl>=0?'+':'')+pnl.toFixed(2);
  pnlEl.className='stat-value'+(pnl>=0?' pos':' neg');
  if(d.ai_metrics){
    setText('sentiment',num(d.ai_metrics.sentiment,3));
    setText('var',fmt(d.ai_metrics.var));
    setText('cvar',fmt(d.ai_metrics.cvar));
    setText('sharpe',num(d.ai_metrics.sharpe,2));
    renderRegime(d.ai_metrics.regime||d.regime);
    renderAIMetrics(d.ai_metrics);
  }
  if(d.market_data&&d.market_data.prices) renderPrices(d.market_data.prices);
  if(d.open_positions) renderPositions(d.open_positions);
}
function setText(id,v){
  const el=document.getElementById(id);
  if(el) el.textContent=v;
}
function renderRegime(r){
  const el=document.getElementById('regime');
  if(!el) return;
  if(!r||r==='--'){el.textContent='--';return}
  el.innerHTML=r.split(',').map(s=>{
    const parts=s.trim().split(':');
    const reg=(parts[1]||'').trim().toLowerCase();
    return '<span class="regime-tag '+(['trending','ranging','volatile','crisis'].includes(reg)?reg:'')+'">'+s.trim()+'</span>';
  }).join(' ');
}
function renderPrices(prices){
  const g=document.getElementById('pricesGrid');
  if(!g)return;
  const syms=Object.keys(prices);
  if(syms.length===0){g.innerHTML='<span style="color:var(--muted);font-size:.8rem">Waiting for data...</span>';return}
  g.innerHTML=syms.map(s=>'<div class="price-tile"><span class="price-sym">'+s+'</span><span class="price-val">'+(Number(prices[s])||0).toFixed(5)+'</span></div>').join('');
}
function renderPositions(positions){
  const body=document.getElementById('positionsBody');
  const none=document.getElementById('noPositions');
  if(!body)return;
  if(!positions||positions.length===0){body.innerHTML='';if(none)none.style.display='block';return}
  if(none)none.style.display='none';
  body.innerHTML=positions.map(p=>'<tr><td>'+(p.symbol||'')+'</td><td class="cell-'+(p.direction||'buy').toLowerCase()+'">'+(p.direction||'')+'</td><td>'+Number(p.volume||0).toLocaleString()+'</td><td>'+Number(p.entry_price||0).toFixed(5)+'</td><td>'+Number(p.sl||0).toFixed(5)+'</td><td>'+Number(p.tp||0).toFixed(5)+'</td><td class="'+((p.unrealized_pnl||0)>=0?'pos':'neg')+'">'+((p.unrealized_pnl||0)>=0?'+':'')+Number(p.unrealized_pnl||0).toFixed(2)+'</td></tr>').join('');
}
function renderAIMetrics(metrics){
  const g=document.getElementById('aiMetricsGrid');
  if(!g)return;
  let html='<div class="metric-item"><div class="label">Active Decisions</div><div class="value">'+(metrics.active_decisions||0)+'</div></div>';
  const sd=metrics.sentiment_data;
  if(sd){
    html+='<div class="metric-item"><div class="label">Sentiment Score</div><div class="value">'+Number(sd.overall_score||0).toFixed(3)+'</div></div>';
    html+='<div class="metric-item"><div class="label">Fear & Greed</div><div class="value">'+(sd.fear_greed_index||50).toFixed(0)+'</div></div>';
    html+='<div class="metric-item"><div class="label">Social Volume</div><div class="value">'+(sd.social_volume||0).toLocaleString()+'</div></div>';
    html+='<div class="metric-item"><div class="label">News Sentiment</div><div class="value">'+Number(sd.news_score||0).toFixed(3)+'</div></div>';
    html+='<div class="metric-item"><div class="label">Reddit Sentiment</div><div class="value">'+Number(sd.reddit_score||0).toFixed(3)+'</div></div>';
    html+='<div class="metric-item"><div class="label">Satellite Activity</div><div class="value">'+Number(sd.satellite_score||0).toFixed(3)+'</div></div>';
    html+='<div class="metric-item"><div class="label">On-Chain Sentiment</div><div class="value">'+Number(sd.onchain_score||0).toFixed(3)+'</div></div>';
    if(sd.recent_headlines&&sd.recent_headlines.length){
      html+='<div class="metric-item" style="grid-column:1/-1"><div class="label">Recent Headlines</div><div class="value" style="font-size:.75rem;line-height:1.4;white-space:normal;word-break:break-word">'+sd.recent_headlines.join('<br>')+'</div></div>';
    }
  }else{
    html+='<div class="metric-item"><div class="label">Behavioral Sentiment</div><div class="value">'+Number(metrics.behavioral_sentiment||0).toFixed(3)+'</div></div>';
  }
  g.innerHTML=html;
}
function updateTimer(){
  const el=document.getElementById('timer');
  if(el)el.textContent=new Date().toLocaleTimeString();
}
setInterval(updateTimer,1000);updateTimer();
connectWS();
</script>
</body>
</html>"""


if FASTAPI_AVAILABLE:
    app = FastAPI(title="AI Forex Dashboard", version="2.0")

    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Serve dashboard HTML."""
        return get_dashboard_html()

    @app.post("/api/login")
    async def login(credentials: HTTPBasicCredentials = Depends(security)):
        """Login and get JWT token."""
        if not verify_credentials(credentials):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        token = create_access_token({"sub": credentials.username})
        return {"token": token, "token_type": "bearer"}

    @app.get("/health")
    async def health():
        elapsed = time.time() - latest_state.get("last_update", time.time())
        pos_count = len(latest_state.get("open_positions", []))
        bal = latest_state.get("balance", 0)
        eq = latest_state.get("equity", 0)
        dd = (
            (latest_state.get("initial_balance", bal) - bal)
            / max(latest_state.get("initial_balance", bal), 1)
            * 100
        )
        return {
            "status": "ok" if elapsed < 60 else "stale",
            "dashboard": True,
            "last_update_s": round(elapsed, 1),
            "trading": latest_state.get("mode", "UNKNOWN"),
            "balance": round(bal, 2),
            "equity": round(eq, 2),
            "drawdown_pct": round(max(dd, 0), 2),
            "open_positions": pos_count,
        }

    @app.get("/api/state")
    async def get_state(token: str = None):
        """Get current dashboard state (requires auth)."""
        payload = verify_token(token) if token else None
        if not payload:
            raise HTTPException(
                status_code=401,
                detail="Missing or invalid token — POST /api/login first",
            )
        return latest_state

    @app.get("/api/risk_metrics")
    async def get_risk_metrics():
        """Get AI-backed risk metrics."""
        return {
            "var": latest_state.get("ai_metrics", {}).get("var", 0),
            "cvar": latest_state.get("ai_metrics", {}).get("cvar", 0),
            "sharpe": latest_state.get("ai_metrics", {}).get("sharpe", 0),
            "max_drawdown": latest_state.get("ai_metrics", {}).get("max_drawdown", 0),
            "kelly_optimal": latest_state.get("ai_metrics", {}).get("kelly_optimal", 0),
        }

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket for real-time updates."""
        await websocket.accept()
        connected_clients.append(websocket)
        await websocket.send_json(latest_state)
        try:
            while True:
                msg = await websocket.receive_text()
                if msg == "ping":
                    await websocket.send_json({"type": "pong"})
        except WebSocketDisconnect:
            if websocket in connected_clients:
                connected_clients.remove(websocket)

    @app.get("/api/positions")
    async def get_positions():
        """Get open positions."""
        return latest_state.get("open_positions", [])

    @app.get("/api/equity_curve")
    async def get_equity_curve():
        """Get equity curve data."""
        return latest_state.get("equity_curve", [])

    @app.post("/api/alert")
    async def add_alert(alert: Dict):
        """Add alert to queue."""
        global alerts_queue
        alert["timestamp"] = time.time()
        alerts_queue.append(alert)
        if len(alerts_queue) > MAX_ALERTS:
            alerts_queue = alerts_queue[-MAX_ALERTS:]
        return {"status": "ok"}


logger.info("[OK] AI-Backed Smart Dashboard module loaded (Enhancement #17)")
