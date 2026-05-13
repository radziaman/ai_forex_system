# RTS: Agentic Moneybot System Elite v4.0

Autonomous multi-agent AI forex trading system with 15 self-aware agents collaborating through an event-driven architecture. Each agent has consciousness, memory, emotions, and communicates via a priority-message bus with capability routing.

## Architecture

### Agentic Core Framework

```
src/agentic/
├── main_agentic.py           # Entry point: boots 15 autonomous agents
├── core/
│   ├── agent_message.py       # Typed protocol: payload schemas, ACK, SHA256 checksums
│   ├── agent_consciousness.py # Self-awareness: identity, emotions, cycle metrics, resource governance
│   ├── agent_memory.py        # Three-tier: episodic + semantic + working, cross-agent queries
│   ├── agent_bus.py           # Priority queues, parallel workers, capability routing, payload validation
│   ├── agent_registry.py      # Directory: heartbeat health, capability discovery, supervisor hierarchy
│   ├── world_state.py         # Shared reality: versioned updates, integrity checks, change observers
│   └── base_agent.py          # Foundation: perceive→reason→act→reflect lifecycle
└── agents/
    ├── data_agent.py          # Market data ingestion, gap healing, TICK_RECEIVED handler
    ├── feature_agent.py       # 55+ feature engineering, z-score normalization, caching
    ├── regime_agent.py        # HMM 4-state detector (trending/ranging/volatile/crisis)
    ├── signal_agent.py        # MoE ensemble (PPO + LSTM-CNN + rule), online learning, calibration
    ├── risk_agent.py          # Kelly/VaR gatekeeper, circuit breakers, adaptive risk integration
    ├── adaptive_risk_agent.py # Dynamic Kelly/risk adjustment by volatility, drawdown, win rate
    ├── execution_agent.py     # Order execution, tick wiring, position publishing
    ├── position_agent.py      # Trailing stops, partial closes, correlation monitoring
    ├── performance_agent.py   # PnL analytics, Sharpe, profit factor, per-symbol/regime breakdowns
    ├── master_agent.py        # System orchestrator, error escalation, human-in-loop halt
    ├── validation_agent.py    # Walk-forward CPCV, Monte Carlo, A/B testing, autonomous data fetching
    ├── monitoring_agent.py    # Telegram alerts, health dashboard, reliable delivery with retry
    ├── connection_agent.py    # Broker connectivity, auto-reconnect with exponential backoff
    ├── learning_agent.py      # Drift-triggered retraining, model registry, curriculum management
    └── memory_agent.py        # State persistence, SHA256 checkpoint integrity, crash recovery
```

### AI/ML Stack
| Component | Technology | Purpose |
|-----------|------------|---------|
| **PPO Reinforcement Learning** | PyTorch | 4 regime-specialist agents (719K params trending, 195K ranging/volatile, 57K crisis) |
| **LSTM-CNN Hybrid** | TensorFlow/Keras | Price regression: 30-bar lookback, 51 features, dual-branch fusion |
| **Profitability Classifier** | TensorFlow/Keras | Direction prediction (binary) with CORRECT future-label construction |
| **Mixture-of-Experts Ensemble** | NumPy | Regime-gated expert weighting via HMM posterior + Elo + Sharpe |
| **HMM Regime Detector** | hmmlearn | 4-state — learns transition probabilities from data |
| **MC Dropout Uncertainty** | TensorFlow | 50 forward passes → prediction variance filters low-confidence trades |
| **ADWIN Concept Drift** | River | Adaptive windowing — triggers retraining on distribution shifts |

### Feature Engineering
- **Multi-timeframe**: 15m / 1h / 4h / 1d parallel processing with z-score normalization
- **55+ features**: RSI(14,21), MACD, ATR(14,21), Bollinger Bands, ADX, Stochastic, EMA ratios, SMA distances
- **Cyclical encoding**: sin/cos for hour/day/month
- **Hurst exponent**: Rolling 50-bar estimate for mean-reversion vs trending
- **Market microstructure**: CVD, bid-ask imbalance, volatility regimes

### Risk Management
| Feature | Implementation |
|---------|---------------|
| **Kelly Criterion** | Fractional (25%), adaptive win-rate tracking, R-multiple averaging |
| **ATR-based SL/TP** | Configurable multipliers, regime-adjusted |
| **Value-at-Risk** | Historical simulation, 95% confidence |
| **Conditional VaR** | Expected shortfall beyond VaR threshold |
| **Daily drawdown** | 5% max daily loss, 10% total max drawdown |
| **Correlation filter** | Blocks trades >0.80 correlation with open positions |
| **Circuit breaker** | Flash crash, liquidity drought, volume anomaly, volatility spike |
| **Trailing stop** | 3-tier (30%/30%/40% partial close) |
| **Adaptive sizing** | Dynamic Kelly adjustment by volatility/drawdown/win-rate |

### Data Sources
- **cTrader Open API** (IC Markets): SSL+Protobuf real-time connection, Level II DOM, async-safe
- **Yahoo Finance**: Historical data with caching, multi-source fallback chain
- **Dukascopy BI5**: Cached tick data for EURUSD

### Agent Communication Protocol
Every inter-agent message carries:
- **Typed schema**: 13 validated message types with required field checking
- **Priority queue**: 5 levels (DEBUG→CRITICAL), higher priority processed first
- **Delivery confirmation**: Optional ACK with timeout for guaranteed delivery
- **Capability routing**: Messages can target any agent by capability, not just by name
- **SHA256 checksums**: End-to-end payload integrity verification
- **Causal chain**: conversation_id, hop_count, causal_parent_id for full traceability

## Quick Start

### Prerequisites
- Python 3.9+
- Docker (optional, for containerized deployment)

### Local Setup
```bash
# Clone and setup
git clone https://github.com/radziaman/ai_forex_system.git
cd ai_forex_system

# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure credentials (see .env.example)
cp .env.example .env
# Edit .env with your cTrader credentials

# Run agentic system
python -m agentic.main_agentic --mode paper     # 15 agents, paper trading
python -m agentic.main_agentic --mode live       # with cTrader broker
python -m agentic.main_agentic --simulate        # agents skip real I/O
python -m agentic.main_agentic --status          # print all 15 agents
```

### Docker Setup
```bash
cp .env.example .env
# Edit .env with your credentials
docker compose up -d
```

### OAuth for cTrader
1. Visit: `https://id.ctrader.com/my/settings/openapi/grantingaccess/?client_id=YOUR_APP_ID&redirect_uri=https://spotware.com&scope=trading&product=web`
2. Copy the authorization code from redirect URL
3. Exchange for tokens:
```bash
curl -X GET "https://openapi.ctrader.com/apps/token?grant_type=authorization_code&code=YOUR_CODE&redirect_uri=https://spotware.com&client_id=YOUR_APP_ID&client_secret=YOUR_SECRET"
```
4. Add `CTRADER_ACCESS_TOKEN` and `CTRADER_REFRESH_TOKEN` to `.env`

## Agent Lifecycle

Every agent runs an independent perceive→reason→act→reflect cycle:

```
PERCEIVE → collect signals from world state + message inbox
    ↓
REASON → analyze, plan, form decisions
    ↓
ACT → execute, send messages, modify world state
    ↓
REFLECT → evaluate outcomes, update memory, consolidate knowledge
    ↓
SLEEP → wait for next tick interval
```

Each agent has:
- **Consciousness**: identity, role, purpose, state, health, emotional model (fatigue/stress/engagement/confidence/curiosity)
- **Memory**: episodic ring buffer (experiences), semantic store (facts), working scratchpad (current context)
- **Emotions**: 5-dimensional state updated per cycle, affects decision confidence
- **Error escalation**: consecutive failures reported to master_agent with severity levels

## 27 G-Fixes Applied

| Fix | Description | Component |
|-----|-------------|-----------|
| G1 | Live tick wiring — cTrader→data_agent via TICK_RECEIVED | execution_agent, data_agent |
| G2 | Adaptive risk values read by risk_agent from world state | risk_agent |
| G3 | Position state published to world state every cycle | execution_agent |
| G4 | Priority queues — 5 levels, CRITICAL processed first | agent_bus |
| G5 | Delivery ACK — optional guaranteed delivery with timeout | agent_message, agent_bus |
| G6 | Capability routing — resolve targets by capability | agent_message, agent_bus, agent_registry |
| G7 | Halted symbol alerts actually sent | risk_agent |
| G8 | Online learning — expert Elo/Sharpe updated from outcomes | signal_agent |
| G9 | Human-in-loop halt approval with timeout | master_agent, risk_agent |
| G10 | Parallel workers — 2 async workers processing bus | agent_bus |
| G11 | Adaptive cycle governance — skip when overrunning | base_agent, agent_consciousness |
| G12 | Cross-agent memory queries + external knowledge | agent_memory |
| G13 | Payload schema validation on every publish | agent_message, agent_bus |
| G14 | Autonomous data fetching for validation | validation_agent |
| G15 | 5-dimensional emotional state model | agent_consciousness |
| G16 | Confidence calibration bins tracking | signal_agent |
| G17 | Supervisor hierarchy in registry | agent_registry, main_agentic |
| G18 | Simulation mode for all agents | base_agent, main_agentic |
| G19 | Human-readable explanations on messages | agent_message |
| G20 | A/B testing framework | validation_agent |
| G21 | Memory consolidation + knowledge broadcast | agent_memory |
| G22 | Error escalation protocol to master | base_agent, master_agent |
| G23 | N/A (dashboard separate concern) | — |
| G24 | Metrics counters in bus stats | agent_bus |
| G25 | Cycle budget + overrun governance | agent_consciousness, base_agent |
| G26 | SHA256 checksums on messages + checkpoints | agent_message, memory_agent, world_state |
| G27 | Telegram delivery with retry + health tracking | monitoring_agent |

## Development
```bash
make test      # Run tests
make lint     # Flake8 linting
make format   # Black formatting
make check    # All checks (lint + type-check + test)
```

## Deployment
```bash
# Docker (recommended for production)
docker compose up -d

# The dashboard is available at http://localhost:8000
# Health check: http://localhost:8000/health
```

## CI/CD
- **GitHub Actions**: Multi-Python (3.9, 3.10, 3.11) testing + linting on push
- **GitHub Pages**: Auto-deploys dashboard to `https://radziaman.github.io/ai_forex_system/`
- **Pre-commit**: Black, flake8, mypy enforced locally

## Environment Variables
See `.env.example` for all configuration. Never commit `.env` to git.

| Variable | Required | Description |
|----------|----------|-------------|
| `CTRADER_APP_ID` | Yes | cTrader application ID |
| `CTRADER_APP_SECRET` | Yes | cTrader application secret |
| `CTRADER_ACCESS_TOKEN` | Yes | OAuth2 access token |
| `CTRADER_REFRESH_TOKEN` | For refresh | OAuth2 refresh token |
| `CTRADER_ACCOUNT_ID` | Yes | cTrader account ID |
| `TELEGRAM_BOT_TOKEN` | No | Telegram alerting bot token |
| `TELEGRAM_CHAT_ID` | No | Telegram alerting chat ID |
| `LOG_LEVEL` | No | Logging level (default: INFO) |

## Performance Targets
| Metric | Target |
|--------|--------|
| Sharpe Ratio | > 2.0 |
| Win Rate | > 60% |
| Max Drawdown | < 10% |
| Monthly Return | 2-5% |
| Risk per Trade | 2% (adaptive) |

## Disclaimer
Trading involves substantial risk. This software is for educational/research purposes. Always test thoroughly on demo accounts before using real money. Past performance does not guarantee future results.

## Branches
- `main` — Stable release
- `develop` — Active development
- `production` — Live trading
- `gh-pages` — Dashboard deployment
