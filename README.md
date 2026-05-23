# RTS: Agentic Moneybot System Elite v5.1

**Autonomous multi-agent AI forex trading system** — 20 self-aware agents with 10 adaptive strategies (5 rule-based + 4 PPO reinforcement learning + 1 LSTM-CNN), real-time market data ingestion, multi-source sentiment analysis, per-symbol AI learning, and institutional-grade risk management.

The system monitors **11 forex symbols** simultaneously, automatically learns which strategies work on each pair, adapts to market conditions in real-time, and manages trades with ATR-based trailing stops, partial profit taking, and correlation-aware position sizing.

> **v5.1 — Code Quality & Stability Pass:** 19 runtime bugs fixed (undefined names, missing awaits, None crashes), 120+ type annotations cleaned, mypy errors reduced 76% (158 → 38), flake8 F-level errors eliminated, full black formatting compliance. All 74 tests pass with zero regressions.

> ⚠️ **Disclaimer:** Trading involves substantial risk. This software is for educational/research purposes. Always test thoroughly on demo accounts before using real money. Past performance does not guarantee future results.

---

## 🧠 System Architecture

### Agentic Core Framework

```
src/agentic/
├── main_agentic.py               # Entry point: boots 20 autonomous agents
├── core/
│   ├── agent_message.py           # Typed protocol: payload schemas, ACK, SHA256 checksums
│   ├── agent_consciousness.py     # Self-awareness: identity, emotions, cycle metrics, resource governance
│   ├── agent_memory.py            # Three-tier: episodic + semantic + working, cross-agent queries
│   ├── agent_bus.py               # Priority queues, parallel workers, capability routing, payload validation
│   ├── agent_registry.py          # Directory: heartbeat health, capability discovery, supervisor hierarchy
│   ├── world_state.py             # Shared reality: versioned updates, integrity checks, change observers
│   └── base_agent.py              # Foundation: perceive→reason→act→reflect lifecycle
└── agents/
    ├── data_agent.py              # Market data ingestion (cTrader/Dukascopy/Yahoo → OHLCV → features)
    ├── regime_agent.py            # HMM 4-state detector (trending/ranging/volatile/crisis)
    ├── signal_agent.py            # MoE ensemble (PPO × 4 regimes + LSTM-CNN + rule-based), online learning
    ├── risk_agent.py              # Kelly/VaR gatekeeper, economic calendar suppression, stale-data halt
    ├── adaptive_risk_agent.py     # Dynamic Kelly by volatility, drawdown, win rate, sentiment
    ├── execution_agent.py         # Order execution via cTrader protobuf, tick wiring, position publishing
    ├── position_agent.py          # 3-tier trailing stops, partial closes, correlation/concentration monitoring
    ├── performance_agent.py       # PnL analytics, Sharpe, profit factor, per-symbol/regime breakdowns
    ├── master_agent.py            # System orchestrator, error escalation, human-in-loop halt
    ├── validation_agent.py        # Walk-forward CPCV, Monte Carlo, A/B testing, autonomous data fetching
    ├── monitoring_agent.py        # Telegram alerts, health dashboard, reliable delivery with retry
    ├── connection_agent.py        # Broker connectivity, auto-reconnect with exponential backoff
    ├── learning_agent.py          # Drift-triggered retraining, model registry, curriculum management
    ├── memory_agent.py            # State persistence, SHA256 checkpoint integrity, crash recovery
    ├── model_registry_agent.py    # Symbol model lifecycle management
    ├── drift_agent.py             # Concept drift monitoring (ADWIN)
    ├── circuit_breaker_agent.py   # Market health checks (velocity, spread, volume)
    ├── cost_agent.py              # Transaction cost monitoring
    └── screener_agent.py          # Autonomous instrument screener
```

### Adaptive AI Strategy Selector

```
MARKET DATA (11 symbols)
  → ADX (trend strength) → trending/ranging/volatile/crisis
  → ATR (volatility) → normal/high/extreme
  → Session (UTC) → london/newyork/asia/overlap
       ↓
STRATEGY SELECTOR (picks best strategy for conditions)
  → Trending: Breakout (SL=2.0, TP=5.0) OR Time Series Momentum (SL=2.5, TP=4.0)
  → Ranging: Mean Reversion (SL=1.5, TP=2.0) OR Bollinger Squeeze (SL=2.0, TP=3.0)
  → Volatile: Volatility Mean Reversion (SL=1.5, TP=2.0)
  → All: PPO agents (4 regimes) + LSTM-CNN
       ↓
PER-SYMBOL LEARNING (auto-optimizes over time)
  → Each symbol × strategy tracked independently
  → After 10 trades: disable if Sharpe < -0.5
  → After 50 global trades: re-evaluate blocked symbols
  → Auto-block symbols where no strategy works
       ↓
MoE ENSEMBLE (weights by recent performance)
  → weight = Sharpe × regime_match × Elo × confidence
  → Best strategy gets 1.5× weight
  → Auto-disable losers at 0.1× weight
       ↓
EXECUTION (per-strategy optimized)
  → Strategy-specific SL/TP
  → ATR-based trailing (breakeven at 1.0×ATR, trail at 0.5×ATR)
  → Partial profit taking (30% at 1.5×, 30% at 2.5× ATR)
  → Correlation filter (blocks EURUSD+GBPUSD, AUDUSD+NZDUSD, etc.)
  → Up to 3 simultaneous uncorrelated positions
```

### Agent Lifecycle

Every agent runs an independent **perceive→reason→act→reflect** cycle:

```
PERCEIVE → collect signals from world state + message inbox
    ↓
REASON → analyze, plan, form decisions
    ↓
ACT → execute, send messages, modify world state
    ↓
REFLECT → evaluate outcomes, update memory, consolidate knowledge
    ↓
SLEEP → wait for next tick interval (0.1s–60s per agent)
```

Each agent has:
- **Consciousness**: identity, role, purpose, state, health, 5-dimensional emotional model
- **Memory**: episodic ring buffer (experiences), semantic store (facts), working scratchpad
- **Emotions**: fatigue/stress/engagement/confidence/curiosity — updated per cycle, affects decision confidence
- **Error escalation**: consecutive failures reported to master_agent with severity levels

---

## 📊 End-to-End Data Pipeline

```
DATA SOURCES
│
├─► MARKET DATA
│   ├─ cTrader Open API (protobuf streaming) — live ticks, Level II DOM
│   ├─ Dukascopy (BI5 cache + HTTP polling) — historical + real-time OHLCV
│   └─ Yahoo Finance (yfinance) — fallback data source
│
├─► SENTIMENT & ALTERNATIVE DATA
│   ├─ News RSS (Bloomberg, CNBC, Investing.com, MarketWatch, Yahoo Finance, ZeroHedge)
│   ├─ Twitter/X API (v2, tweepy/requests) — forex keyword search
│   ├─ Reddit (PRAW/OAuth) — r/Forex, r/wallstreetbets, r/trading, r/investing
│   ├─ Fear & Greed Index (alternative.me API) — market sentiment indicator
│   ├─ NASA EONET + POWER — natural events + agricultural weather impact scoring
│   └─ ForexFactory Calendar — high-impact economic event suppression
│
└─► FEATURE PIPELINE
    └─ FeaturePipeline.transform()
       ├─ compute_features() → 49 technical indicators per timeframe
       │   ├─ Price dynamics: body, range, shadows
       │   ├─ Momentum: RSI(14,21), MACD, mom(1,5,10), price_acceleration
       │   ├─ Volatility: ATR(14,21), Bollinger Bands, volatility_20
       │   ├─ Trend: EMA(20,50,200), ADX, SMA distances, ema_cross_ratio
       │   ├─ Stochastic: stoch_k, stoch_d
       │   ├─ Time encoding: sin/cos hour/day/month
       │   └─ Hurst exponent, atr_normalized, rsi_divergence
       ├─ compute_microstructure_features() → CVD, OFI, volume metrics
       └─ compute_cross_asset_features() → sentiment_* columns from all sources
           • News scores weighted at 60%
           • Twitter scores weighted at 25%
           • Reddit scores weighted at 15%
           • Adaptive redistribution when sources unavailable
```

---

## 🤖 AI/ML Stack

| Component | Technology | Details |
|-----------|------------|---------|
| **PPO Reinforcement Learning** | PyTorch | 4 regime-specialist agents (trending 719K, ranging 195K, volatile 195K, crisis 57K params) |
| **LSTM-CNN Hybrid** | TensorFlow/Keras | 30-bar lookback × 49 features, dual-branch fusion, per-symbol fine-tuned |
| **Profitability Classifier** | TensorFlow/Keras | Binary direction classifier (~54% accuracy, per-symbol) |
| **Mixture-of-Experts Ensemble** | NumPy | Regime-gated: PPO × LSTM × rule-based, weighted by Elo + Sharpe + regime_match + confidence |
| **HMM Regime Detector** | hmmlearn | 4-state GaussianHMM, 8-dimensional feature vector, learned transition probabilities |
| **MC Dropout Uncertainty** | TensorFlow | Prediction variance estimation for low-confidence filtering |
| **ADWIN Concept Drift** | River | Adaptive windowing — triggers automated retraining on distribution shifts |
| **FinBERT Sentiment** | HuggingFace | `ProsusAI/finbert` — financial news/social media sentiment classification |

### Trained Models (39 total)

| Model Type | Count | Format | Size |
|-----------|-------|--------|------|
| LSTM-CNN (per symbol) | 11 | `.keras` | 1.5 MB each |
| Profitability Classifier | 11 | `.keras` | 194 KB each |
| PPO Regime Agents | 4 | `.pth` | 8.5 MB each |
| Base Transfer Model | 1 | `.keras` | 1.5 MB |
| Feature Normalization | 1 | `.npz` | 16 KB |

### Feature Count: 49 Dimensions

All models operate on a consistent 49-feature vector per bar, ensuring dimension alignment across the entire pipeline — from FeaturePipeline through PPO agents, LSTM models, and classifiers.

---

## 📈 Data Scrapers & External Integrations

| Source | Type | Status | Data Flow |
|--------|------|--------|-----------|
| **Dukascopy** | Historical OHLCV + Tick | ✅ Live | CSV → DataManager → FeaturePipeline |
| **cTrader** | Live ticks + DOM | ✅ Live | Protobuf → ExecutionAgent → DataManager |
| **Yahoo Finance** | OHLCV fallback | ✅ Live | yfinance → DataManager |
| **Bloomberg RSS** | News headlines | ✅ Live | feedparser → SentimentAnalyzer |
| **CNBC RSS** | News headlines | ✅ Live | feedparser → SentimentAnalyzer |
| **Investing.com RSS** | News headlines | ✅ Live | feedparser → SentimentAnalyzer |
| **MarketWatch RSS** | News headlines | ✅ Live | feedparser → SentimentAnalyzer |
| **Yahoo Finance RSS** | News headlines | ✅ Live | feedparser → SentimentAnalyzer |
| **ZeroHedge RSS** | News headlines | ✅ Live | feedparser → SentimentAnalyzer |
| **Twitter/X API** | Social media | ✅ Ready | tweepy/requests → SentimentAnalyzer (25% blend) |
| **Reddit API** | Social media | ✅ Ready | PRAW/OAuth → SentimentAnalyzer (15% blend) |
| **Fear & Greed Index** | Market sentiment | ✅ Live | alternative.me API → Dashboard |
| **NASA EONET + POWER** | Natural events + Ag | ✅ Live | REST APIs → Satellite score → Dashboard |
| **ForexFactory Calendar** | Economic events | ✅ Live | JSON API → RiskAgent trade suppression |

---

## 🛡️ Risk Management

| Feature | Implementation |
|---------|---------------|
| **Kelly Criterion** | Fractional (adaptive, 0-25%), adjusted by volatility × drawdown × win rate × sentiment |
| **ATR-based SL/TP** | Configurable multipliers per regime (trending 2×/4×, volatile 2.5×/5×, crisis 1×/2×) |
| **Value-at-Risk** | Historical simulation, 95% confidence |
| **Conditional VaR** | Expected shortfall beyond VaR threshold |
| **Daily drawdown** | 5% max daily loss, 10% total max drawdown |
| **Correlation filter** | Blocks trades >0.80 correlation with open positions |
| **Circuit breaker** | Flash crash, liquidity drought, volume anomaly, volatility spike |
| **Trailing stop** | ATR-based: breakeven at 1.0×ATR, trail at 0.5×ATR behind best price |
| **Economic calendar** | Auto-suppresses trading 2h before high-impact events (NFP, FOMC, CPI) |
| **Stale-data halt** | Rejects signals when market data exceeds 60s staleness |
| **Adaptive sizing** | Dynamic Kelly multiplier from combined regime × sentiment state |
| **Max positions** | Configurable per asset class (forex: 5, crypto: 3, indices: 3, commodities: 2) |

---

## ⚡ Quick Start

### Prerequisites
- Python 3.9+
- Docker (optional, for containerized deployment)
- 8 GB+ RAM recommended (models load into memory)

### Local Setup
```bash
# Clone and setup
git clone https://github.com/radziaman/ai_forex_system.git
cd ai_forex_system

# Create environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure credentials
cp .env.example .env
# Edit .env with your API keys

# Run the system
python -m src.agentic.main_agentic --mode paper          # Paper trading (default)
python -m src.agentic.main_agentic --mode paper --timeout 120  # Auto-stop after 120s
python -m src.agentic.main_agentic --mode live            # Live trading with cTrader
```

### Telegram Alerts
The system sends 8 types of enhanced real-time alerts to Telegram:
- 📈 **Trade Opened** — Strategy, session, confidence, SL/TP, account balance
- 💰 **Trade Closed (Win)** — PnL, updated balance, exit price
- 💥 **Trade Closed (Loss)** — Loss amount, encouraging message
- 📊 **Daily Summary** — Per-symbol active/learning status, PnL, Sharpe
- ⚠️ **Risk Alert** — Circuit breaker, volatility spikes, drawdown warnings
- 🎯 **Strategy Performance** — Per-strategy Sharpe, auto-disable notices
- 🔄 **Symbol Status** — When blocked symbols get re-enabled for testing
- 🤖 **System Startup** — Confirmation that all systems are online

### Docker Setup
```bash
docker compose up -d
# Dashboard: http://localhost:8000
# Health:    http://localhost:8000/health
```

### Verification
```bash
make test      # Run all 74 tests
make lint     # Flake8 linting (F-level errors: 0)
make format   # Black auto-formatting (120 files clean)
make type-check  # Mypy type checking (38 annotation warnings remaining)
make check    # Full suite: lint → type-check → test
```

### Sanity Checks
```bash
python src/agentic/_verify.py        # 23 module imports
python src/agentic/_check_all.py    # 42 comprehensive checks
python src/agentic/_diagnose_ai.py  # AI/ML pipeline diagnostic
python src/agentic/_test_lifecycle.py  # Agent lifecycle end-to-end
```

---

## 🔧 Configuration

### Environment Variables
See `.env.example` — never commit `.env` to git.

| Variable | Required | Description |
|----------|----------|-------------|
| `CTRADER_APP_ID` | Yes | cTrader application ID |
| `CTRADER_APP_SECRET` | Yes | cTrader application secret |
| `CTRADER_ACCESS_TOKEN` | Yes | OAuth2 access token |
| `CTRADER_REFRESH_TOKEN` | For refresh | OAuth2 refresh token |
| `CTRADER_ACCOUNT_ID` | Yes | cTrader account ID |
| `TELEGRAM_BOT_TOKEN` | No | Telegram alerting bot token |
| `TELEGRAM_CHAT_ID` | No | Telegram alerting chat ID |
| `FRED_API_KEY` | No | St. Louis Fed economic data |
| `TWITTER_BEARER_TOKEN` | No | Twitter/X API v2 |
| `REDDIT_CLIENT_ID` | No | Reddit API (PRAW) |
| `NASA_API_KEY` | No | NASA satellite data (default: DEMO_KEY) |
| `LOG_LEVEL` | No | Logging level (default: INFO) |

### config.yaml
All trading parameters in `config.yaml`:
- `trading.max_risk_per_trade`, `max_drawdown`, `max_positions`
- `features.timeframes`, `lookback`, `use_microstructure`
- `ai.algorithm`, `ensemble.experts`, `regime_agents.*`

---

## 🧪 Testing & Code Quality

| Suite | Tests | Coverage |
|-------|-------|----------|
| Smoke imports | 32 | All modules import cleanly |
| Config validation | 11 | Config loading, env override, validation |
| Backtester | 31 | SL/TP, costs, edge cases, Monte Carlo, HMM alignment |
| Lifecycle | 1 | Full agent boot → communicate → shutdown |
| **Total** | **74** | All passing |

| Quality Gate | Status | Count |
|-------------|--------|-------|
| Flake8 F-level errors | ✅ **0** | All fixed — 4 undefined-name bugs eliminated |
| Black formatting | ✅ **120/120** | Full compliance |
| Mypy type errors | 🟡 **38 remaining** | Reduced 76% (158→38). All remaining are annotation gaps, not runtime bugs |
| Compilation | ✅ **121/121** | All source files compile cleanly |

---

## 📁 Project Structure

```
src/
├── agentic/          # Multi-agent framework (20 agents + core)
├── ai/               # ML models (PPO, regime agents, sentiment, social scrapers)
├── api/              # cTrader broker connectivity (protobuf)
├── backtest/         # Vectorized backtester
├── data/             # Data providers (Dukascopy, multi-source, economic calendar)
├── execution/        # Order execution engine, cost model, algo executor
├── infrastructure/   # Config, secrets, system info
├── notifications/    # Telegram alerts
├── risk/             # Circuit breaker, risk manager
├── rts_ai_fx/        # Core ML pipeline (features, models, ensemble, regime detector)
├── scripts/          # CLI entry points (training, backtesting, simulation)
├── training/         # Online learning, distributed training, model registry
└── validation/       # Walk-forward, Monte Carlo, stress tests

models/               # Trained models (.keras, .pth, .npz) — gitignored
data/                 # Market data cache (CSV, BI5, feature cache)
tests/                # pytest suite (74 tests)
```

---

## 🔐 Security & Governance

- **27 G-Fixes applied** — comprehensive fix catalog covering data flow, resilience, error handling, and safety
- **SHA256 checksums** on all inter-agent messages and memory checkpoints
- **Circuit breakers** at market, risk, and system levels
- **Human-in-loop halt** approval for emergency shutdown
- **Error escalation** protocol with severity levels
- **Simulation mode** for all agents — safe testing without real I/O

## 🧹 Recent Code Quality Improvements (v5.1)

| Fix Category | Files Changed | Details |
|-------------|---------------|---------|
| **Undefined-name bugs** | 4 | `tradeable` scope, `GlobalAveragePooling1D`/`LayerNormalization` imports, broken `tf_keras` import path |
| **Async/await fixes** | 1 | `speed_trainer.py` — coroutines passed instead of awaited DataFrames (3 methods converted) |
| **None-crash fixes** | 3 | Sentiment classifier guard, missing return in `load_from_ctrader`, `_role_index` initialization |
| **Type annotation repairs** | 6 | `master_agent.py` (Dict[str, Any]), `provider_factory.py` (import scope), `dukascopy_realtime.py` (duplicate method), `ctrader_client.py` (variable reuse) |
| **Mypy config** | 1 | `pyproject.toml` — suppressed 70+ false-positive numpy/pandas stub errors via per-module overrides |
| **Formatting** | 14 | `black` auto-format applied to all non-compliant files |
| **numpy type safety** | 2 | `circuit_breaker.py` (explicit `float()` casts), `risk/manager.py` (abs type fix) |

---

## 📊 Performance Targets

| Metric | Target |
|--------|--------|
| Sharpe Ratio | > 2.0 |
| Win Rate | > 55% |
| Max Drawdown | < 10% |
| Risk per Trade | 2% (adaptive) |
| Model Inference | < 35ms (LSTM), < 0.3ms (PPO) |

---

## 🌿 Branches

- `main` — Stable release
- `develop` — Active development
- `production` — Live trading
- `gh-pages` — Dashboard deployment

---

## 📜 License

Educational/research purposes only. Not licensed for commercial trading use.
