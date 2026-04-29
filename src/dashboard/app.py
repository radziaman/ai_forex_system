import asyncio
import json
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from loguru import logger
from typing import List, Dict, Any
import os

app = FastAPI(title="Institutional Forex AI Dashboard", version="3.0")

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
    "timestamp": time.time(),
}


@app.get("/")
async def get_dashboard(request: Request):
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Institutional Forex AI | Live Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@2.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
    <style>
        :root {
            --bg-dark: #0a0e17;
            --bg-card: #111827;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-blue: #3b82f6;
            --accent-gold: #f59e0b;
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --border-color: #1f2937;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
        }
        .navbar {
            background: linear-gradient(180deg, #111827 0%, #0a0e17 100%);
            border-bottom: 1px solid var(--border-color);
            padding: 1rem 0;
        }
        .navbar-brand {
            font-weight: 700;
            font-size: 1.5rem;
            background: linear-gradient(135deg, var(--accent-gold) 0%, var(--accent-green) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .status-indicator {
            width: 10px; height: 10px; border-radius: 50%;
            display: inline-block; margin-right: 8px;
            animation: pulse 2s infinite;
        }
        .status-online { background: var(--accent-green); box-shadow: 0 0 10px var(--accent-green); }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .stat-card {
            background: var(--bg-card); border: 1px solid var(--border-color);
            border-radius: 12px; padding: 1.5rem; transition: transform 0.3s;
        }
        .stat-card:hover { transform: translateY(-4px); }
        .stat-value { font-size: 1.75rem; font-weight: 700; }
        .stat-value.positive { color: var(--accent-green); }
        .stat-value.negative { color: var(--accent-red); }
        .card {
            background: var(--bg-card); border: 1px solid var(--border-color);
            border-radius: 12px; margin-bottom: 1.5rem;
        }
        .card-header {
            background: transparent; border-bottom: 1px solid var(--border-color);
            padding: 1rem 1.5rem; font-weight: 600;
        }
        .chart-container { position: relative; height: 300px; }
        table { width: 100%; border-collapse: collapse; }
        th {
            color: var(--text-secondary); font-weight: 500; font-size: 0.75rem;
            text-transform: uppercase; letter-spacing: 0.05em;
            padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid var(--border-color);
        }
        td { padding: 0.75rem 1rem; border-bottom: 1px solid var(--border-color); }
        tr:hover { background: rgba(255,255,255,0.02); }
        .badge-win { background: rgba(16,185,129,0.15); color: var(--accent-green); padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.75rem; }
        .badge-loss { background: rgba(239,68,68,0.15); color: var(--accent-red); padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.75rem; }
        .live-price { font-family: 'Courier New', monospace; font-size: 2rem; font-weight: 700; }
        .spread-badge { background: rgba(59,130,246,0.15); color: var(--accent-blue); padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.75rem; }
        footer {
            background: var(--bg-card); border-top: 1px solid var(--border-color);
            padding: 1.5rem 0; margin-top: 3rem;
        }
        .footer-text { color: var(--text-secondary); font-size: 0.875rem; }
    </style>
</head>
<body>
    <nav class="navbar sticky-top">
        <div class="container">
            <div class="d-flex align-items-center">
                <i class="fas fa-brain text-warning me-2" style="font-size: 1.5rem;"></i>
                <span class="navbar-brand">Institutional Forex AI</span>
                <span class="badge bg-secondary ms-3">v3.0</span>
            </div>
            <div class="d-flex align-items-center">
                <span class="status-indicator" id="connectionIndicator"></span>
                <span class="text-success me-4" id="connectionStatus">Connecting...</span>
                <span class="text-secondary" id="currentSymbol">EURUSD</span>
            </div>
        </div>
    </nav>
    <div class="container mt-4">
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="stat-card">
                    <div class="text-secondary mb-2">Account Balance</div>
                    <div class="stat-value" id="balance">$100,000.00</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card">
                    <div class="text-secondary mb-2">Equity</div>
                    <div class="stat-value" id="equity">$100,000.00</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card">
                    <div class="text-secondary mb-2">Total Profit</div>
                    <div class="stat-value positive" id="totalProfit">$0.00</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card">
                    <div class="text-secondary mb-2">Total Trades</div>
                    <div class="stat-value" id="totalTrades">0</div>
                </div>
            </div>
        </div>
        <div class="row mb-4">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header"><i class="fas fa-robot me-2"></i>AI Brain Metrics</div>
                    <div class="card-body">
                        <div class="row text-center">
                            <div class="col-md-2"><div class="stat-label">Win Rate</div><div class="stat-value" id="winRate">--%</div></div>
                            <div class="col-md-2"><div class="stat-label">Mode</div><div class="stat-value" id="tradingMode">PAPER</div></div>
                            <div class="col-md-2"><div class="stat-label">Regime</div><div class="stat-value" id="marketRegime">--</div></div>
                            <div class="col-md-2"><div class="stat-label">AI Action</div><div class="stat-value" id="aiAction">HOLD</div></div>
                            <div class="col-md-2"><div class="stat-label">Confidence</div><div class="stat-value" id="aiConfidence">--%</div></div>
                            <div class="col-md-2"><div class="stat-label">Open Positions</div><div class="stat-value" id="openPositionsCount">0</div></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div class="row mb-4">
            <div class="col-lg-8">
                <div class="card">
                    <div class="card-header"><i class="fas fa-chart-area me-2"></i>Equity Curve</div>
                    <div class="card-body"><div class="chart-container"><canvas id="equityChart"></canvas></div></div>
                </div>
            </div>
            <div class="col-lg-4">
                <div class="card">
                    <div class="card-header"><i class="fas fa-tachometer-alt me-2"></i>Live Market</div>
                    <div class="card-body text-center">
                        <div class="live-price mb-2" id="livePrice">1.11985</div>
                        <div class="d-flex justify-content-center gap-3 mb-3">
                            <span class="text-success"><i class="fas fa-arrow-up"></i> Bid: <span id="bidPrice">1.11983</span></span>
                            <span class="spread-badge" id="spread">0.2 pip</span>
                            <span class="text-danger"><i class="fas fa-arrow-down"></i> Ask: <span id="askPrice">1.11985</span></span>
                        </div>
                        <hr style="border-color: var(--border-color);">
                        <div class="text-start">
                            <div class="d-flex justify-content-between mb-2"><span class="text-secondary">Regime</span><span class="badge bg-info" id="regimeBadge">--</span></div>
                            <div class="d-flex justify-content-between mb-2"><span class="text-secondary">Features</span><span class="badge bg-primary" id="featuresCount">222+</span></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div class="row">
            <div class="col-lg-5">
                <div class="card">
                    <div class="card-header"><i class="fas fa-briefcase me-2"></i>Open Positions</div>
                    <div class="card-body p-0">
                        <table>
                            <thead><tr><th>Symbol</th><th>Dir</th><th>Volume</th><th>Entry</th><th>PnL</th></tr></thead>
                            <tbody id="openPositionsTable"><tr><td colspan="5" class="text-center text-secondary py-4">No open positions</td></tr></tbody>
                        </table>
                    </div>
                </div>
            </div>
            <div class="col-lg-7">
                <div class="card">
                    <div class="card-header"><i class="fas fa-history me-2"></i>Trade History</div>
                    <div class="card-body p-0">
                        <table>
                            <thead><tr><th>Time</th><th>Symbol</th><th>Dir</th><th>Entry</th><th>Exit</th><th>PnL</th></tr></thead>
                            <tbody id="tradeHistoryTable"><tr><td colspan="6" class="text-center text-secondary py-4">No trades yet</td></tr></tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <footer>
        <div class="container">
            <div class="d-flex justify-content-between align-items-center">
                <div class="footer-text"><i class="fas fa-robot me-2"></i>Institutional Forex AI v3.0 | Powered by Python, RL & cTrader</div>
                <div class="footer-text"><i class="fab fa-github me-2"></i><a href="https://github.com/radziaman/ai_forex_system" class="text-decoration-none text-secondary">Source</a></div>
            </div>
        </div>
    </footer>
    <script>
        let ws = null;
        const equityData = { labels: [], datasets: [{ label: 'Equity', data: [], borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.1)', fill: true, tension: 0.4, pointRadius: 0 }] };
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            ws.onopen = () => {
                document.getElementById('connectionIndicator').className = 'status-indicator status-online';
                document.getElementById('connectionStatus').textContent = 'Live Trading';
            };
            ws.onmessage = (e) => { updateDashboard(JSON.parse(e.data)); };
            ws.onclose = () => {
                document.getElementById('connectionIndicator').className = 'status-indicator';
                document.getElementById('connectionStatus').textContent = 'Disconnected';
                setTimeout(connectWebSocket, 3000);
            };
        }
        function updateDashboard(data) {
            if (data.balance !== undefined) document.getElementById('balance').textContent = '$' + data.balance.toLocaleString('en-US', {minimumFractionDigits: 2});
            if (data.equity !== undefined) {
                document.getElementById('equity').textContent = '$' + data.equity.toLocaleString('en-US', {minimumFractionDigits: 2});
                const profit = data.equity - (data.initial_balance || 100000);
                const el = document.getElementById('totalProfit');
                el.textContent = (profit >= 0 ? '+' : '') + '$' + profit.toFixed(2);
                el.className = 'stat-value ' + (profit >= 0 ? 'positive' : 'negative');
            }
            if (data.total_trades !== undefined) document.getElementById('totalTrades').textContent = data.total_trades;
            if (data.win_rate !== undefined) document.getElementById('winRate').textContent = (data.win_rate * 100).toFixed(1) + '%';
            if (data.mode !== undefined) { document.getElementById('tradingMode').textContent = data.mode; }
            if (data.regime !== undefined) { document.getElementById('marketRegime').textContent = data.regime; document.getElementById('regimeBadge').textContent = data.regime; }
            if (data.ai_metrics) {
                document.getElementById('aiAction').textContent = data.ai_metrics.action || 'HOLD';
                document.getElementById('aiConfidence').textContent = ((data.ai_metrics.confidence || 0) * 100).toFixed(0) + '%';
            }
            if (data.open_positions !== undefined) {
                document.getElementById('openPositionsCount').textContent = data.open_positions.length;
                const tbody = document.getElementById('openPositionsTable');
                if (data.open_positions.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="5" class="text-center text-secondary py-4">No open positions</td></tr>';
                } else {
                    tbody.innerHTML = data.open_positions.map(p => `<tr>
                        <td>${p.symbol}</td>
                        <td><span class="badge ${p.direction === 'BUY' ? 'bg-success' : 'bg-danger'}">${p.direction}</span></td>
                        <td>${p.volume}</td>
                        <td>${p.entry_price?.toFixed(5)}</td>
                        <td class="${p.unrealized_pnl >= 0 ? 'text-success' : 'text-danger'}">$${p.unrealized_pnl?.toFixed(2)}</td>
                    </tr>`).join('');
                }
            }
            if (data.trade_history !== undefined) {
                const tbody = document.getElementById('tradeHistoryTable');
                if (data.trade_history.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="6" class="text-center text-secondary py-4">No trades yet</td></tr>';
                } else {
                    tbody.innerHTML = data.trade_history.slice(-20).reverse().map(t => `<tr>
                        <td>${new Date(t.timestamp * 1000).toLocaleTimeString()}</td>
                        <td>${t.symbol}</td>
                        <td><span class="badge ${t.direction === 'BUY' ? 'bg-success' : 'bg-danger'}">${t.direction}</span></td>
                        <td>${t.entry?.toFixed(5)}</td>
                        <td>${t.exit?.toFixed(5)}</td>
                        <td class="${t.pnl >= 0 ? 'text-success' : 'text-danger'}">$${t.pnl?.toFixed(2)}</td>
                    </tr>`).join('');
                }
            }
            if (data.market_data) {
                document.getElementById('livePrice').textContent = (data.market_data.bid || 0).toFixed(5);
                document.getElementById('bidPrice').textContent = (data.market_data.bid || 0).toFixed(5);
                document.getElementById('askPrice').textContent = (data.market_data.ask || 0).toFixed(5);
                document.getElementById('spread').textContent = ((data.market_data.spread || 0) * 10000).toFixed(1) + ' pip';
            }
            if (data.equity !== undefined) {
                equityData.labels.push(new Date());
                equityData.datasets[0].data.push(data.equity);
                if (equityData.labels.length > 100) { equityData.labels.shift(); equityData.datasets[0].data.shift(); }
                equityChart.update('none');
            }
        }
        const ctx = document.getElementById('equityChart').getContext('2d');
        const equityChart = new Chart(ctx, {
            type: 'line', data: equityData,
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false }, tooltip: { mode: 'index', callbacks: { label: (ctx) => '$' + ctx.parsed.y.toFixed(2) } } },
                scales: {
                    x: { grid: { color: '#1f2937' }, ticks: { color: '#6b7280', maxTicksLimit: 10 } },
                    y: { grid: { color: '#1f2937' }, ticks: { color: '#6b7280', callback: (v) => '$' + v.toFixed(0) } }
                }
            }
        });
        window.addEventListener('load', connectWebSocket);
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html)


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
    global latest_state
    latest_state.update(kwargs)
    latest_state["timestamp"] = time.time()


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}
