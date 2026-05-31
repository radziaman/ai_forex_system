# RTS AI Forex Trading System v8.0

**Multi-strategy algorithmic FX trading system** — 5-module EventBus pipeline with Mixture-of-Experts ensemble AI (28 experts), HMM regime detection, real-time market data ingestion, institutional-grade risk management, and performance attribution.

The system monitors **11 forex symbols** simultaneously across 4 market regimes (trending/ranging/volatile/crisis), combines 28 expert predictions via Elo-weighted MoE voting, executes trades with ATR-based trailing stops and partial profit taking, and continuously adapts through online learning and concept drift detection.

> ⚠️ **Disclaimer:** Trading involves substantial risk. This software is for educational/research purposes. Always test thoroughly on demo accounts before using real money. Past performance does not guarantee future results.

---

## 🏗️ Architecture

### Pipeline Architecture (Current — Canonical)

```
src/pipeline/ (5-module EventBus system)
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Orchestrator ──→ EventBus ←── SignalEngine                        │
│       │                  │          │                               │
│       │                  ├── "tick" ◄── (from DataManager)           │
│       │                  ├── "signal_generated" ──→ RiskManager      │
│       │                  ├── "risk_approved/rejected" ──→ ExecMgr   │
│       │                  ├── "position_opened/closed" ──→ LearnMgr  │
│       │                  ├── "config_changed" ◄── ConfigWatcher     │
│       │                  ├── "health_check" ──→ Dashboard           │
│       │                  └── ...                                    │
└─────────────────────────────────────────────────────────────────────┘
```

### Event Flow

```
DataManager.update_tick(symbol, bid, ask, volume)
       │ bus.emit("tick")
       ▼
┌────────────────────────────────────────────────────────────────┐
│ SignalEngine._on_tick()                                        │
│  1. FeaturePipeline.transform()  → 49-dim feature vector       │
│  2. HMMRegimeDetector.detect()  → "trending"/"ranging"/...    │
│  3. MoEEnsemble.predict()       → EnsemblePrediction           │
│  4. ATR-based threshold gate    → dynamic per-symbol           │
│  5. bus.emit("signal_generated")                               │
└────────────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────────────┐
│ RiskManager._on_signal()                                       │
│  1. Pre-trade checks (drawdown, daily loss, consecutive)       │
│  2. Circuit breaker (velocity, spread, volume, volatility)     │
│  3. Kelly sizing with VaR/CVaR adjustment                      │
│  4. Correlation risk (regime-dependent matrix)                 │
│  → bus.emit("risk_approved/rejected")                          │
└────────────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────────────┐
│ ExecutionManager._on_risk_approved()                            │
│  1. Order placement (paper or cTrader live)                    │
│  2. ATR-based trailing stops (30%/30%/40% partial close)       │
│  3. Execution quality tracking (slippage, fill rate)           │
│  → bus.emit("position_opened/closed", "execution_result")       │
└────────────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────────────┐
│ LearningManager (listens to position_closed)                    │
│  1. DriftMonitor.update()  → ADWIN concept drift               │
│  2. PerformanceTracker     → Sharpe, win rate, profit factor    │
│  3. ModelRegistry          → champion/challenger                │
│  4. CheckpointManager      → SHA256-verified state              │
└────────────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────────────┐
│ AttributionManager (listens to position_closed)                 │
│  1. StrategyAttributionEngine → decompose P&L                  │
│  2. Alpha decay detection → auto-disable decaying strategies   │
│  → bus.emit("trade_attributed", "strategy_disable")            │
└────────────────────────────────────────────────────────────────┘
```

### Config Hot-Reload

The `ConfigWatcher` polls `config.yaml` every 10 seconds and emits `config_changed` events on the EventBus when the file is modified — no restart required for parameter changes.

---

## 🤖 AI/ML Stack

| Component | Technology | Details |
|-----------|------------|---------|
| **PPO Reinforcement Learning** | PyTorch 2.0+ | 4 regime-specialist agents (trending 719K, ranging 195K, volatile 195K, crisis 57K params) |
| **LSTM-CNN Hybrid** | TensorFlow 2.15 | 30-bar lookback × 49 features, dual-branch fusion, per-symbol fine-tuned |
| **Profitability Classifier** | TensorFlow 2.15 | Binary direction classifier (~54% accuracy, per-symbol) |
| **Mixture-of-Experts Ensemble** | NumPy | 28 experts: weight = regime × Elo × Sharpe × confidence × tracker |
| **HMM Regime Detector** | hmmlearn | 4-state GaussianHMM, 8-dim feature vector, learned transitions |
| **MC Dropout Uncertainty** | TensorFlow | Prediction variance estimation for low-confidence filtering |
| **ADWIN Concept Drift** | River (ported) | Adaptive windowing — triggers automated retraining |
| **FinBERT Sentiment** | HuggingFace | `ProsusAI/finbert` — financial news sentiment classification |
| **MAML Meta-Learning** | PyTorch | (Planned) — model-agnostic meta-learning for fast adaptation |

### Ensemble Weighting Formula

```
weight = regime_weight × elo_weight × sharpe_weight × conf_weight × tracker_weight
```

- **regime_weight**: How well the expert matches the current HMM regime state
- **elo_weight**: Expert Elo rating (updated after every trade, k-factor decays with experience)
- **sharpe_weight**: Rolling Sharpe ratio of expert predictions
- **conf_weight**: Expert's self-reported confidence score
- **tracker_weight**: Strategy-tracker dynamic weight (per-symbol)

### Expert Lockout System

Experts with consecutive losses are automatically disabled with exponential backoff:
- 3 consecutive losses → 5 min cooldown
- 5 consecutive losses → 30 min cooldown  
- 7+ consecutive losses → disabled for 24 hours

### State Persistence

Ensemble state (Elo ratings, Sharpe ratios, lockout timestamps, win/loss counts) is persisted to `models/ensemble_state.json` and survives restarts.

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
└─► FEATURE PIPELINE (49-dimension invariant)
    └─ FeaturePipeline.transform()
       ├─ compute_features() → 45 technical indicators per timeframe
       │   ├─ Price dynamics: body, range, shadows
       │   ├─ Momentum: RSI(14,21), MACD, mom(1,5,10), price_acceleration
       │   ├─ Volatility: ATR(14,21), Bollinger Bands, volatility_20
       │   ├─ Trend: EMA(20,50,200), ADX, SMA distances, ema_cross_ratio
       │   ├─ Stochastic: stoch_k, stoch_d
       │   ├─ Time encoding: sin/cos hour/day/month
       │   └─ Hurst exponent, atr_normalized, rsi_divergence
       ├─ compute_microstructure_features() → CVD, OFI, volume metrics
       └─ compute_cross_asset_features() → sentiment columns
```

---

## 🛡️ Risk Management

| Feature | Implementation |
|---------|---------------|
| **Kelly Criterion** | Fractional (adaptive, 0-25%), adjusted by volatility × drawdown × win rate × sentiment |
| **ATR-based SL/TP** | Configurable multipliers per regime (trending 2×/4×, volatile 2.5×/5×, crisis 1×/2×) |
| **Value-at-Risk** | Historical simulation, 95% confidence |
| **Conditional VaR** | Expected shortfall beyond VaR threshold |
| **Daily drawdown** | 5% max daily loss, 10% total max drawdown |
| **Correlation filter** | Blocks trades >0.80 correlation with open positions (regime-dependent) |
| **Circuit breaker** | 4 detectors: price velocity, spread widening, volume anomaly, volatility spike |
| **Graceful degradation** | NORMAL → DEGRADED → HALTED with confidence threshold auto-adjustment |
| **Trailing stop** | ATR-based: breakeven at 1.0×ATR, trail at 0.5×ATR behind best price |
| **Partial profit taking** | 30% at 1.5× ATR, 30% at 2.5× ATR, 40% at final target |
| **Economic calendar** | Auto-suppresses trading 2h before high-impact events (NFP, FOMC, CPI) |
| **Stale-data halt** | Rejects signals when market data exceeds 60s staleness |
| **Adaptive sizing** | Dynamic Kelly multiplier from combined regime × sentiment state |
| **Performance attribution** | Alpha/execution/slippage/luck decomposition per trade, alpha decay auto-disable |

### Circuit Breaker Details

- **Price velocity**: Triggers halt on >0.5% move in a single tick
- **Spread widening**: Triggers at 5× normal spread
- **Volume anomaly**: Triggers at 10× normal volume
- **Volatility spike**: Bollinger Band breakout on 20-bar lookback
- **Warm-up period**: 50 observations before detectors activate
- **Cooldown**: 5-minute cooldown after halt before auto-recovery

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
pip install -e .

# Configure credentials
cp .env.example .env
# Edit .env with your API keys

# Run the system
python -m pipeline.main --mode paper              # Paper trading (default)
python -m pipeline.main --mode paper --timeout 120 # Auto-stop after 120s
python -m pipeline.main --mode live                # Live trading with cTrader
```

### Docker Setup
```bash
docker compose up -d
# Dashboard: http://localhost:8000
# Health:    http://localhost:8000/health
```

### Verification
```bash
make test       # Run all 708 tests
make lint       # Flake8 linting (0 fatal errors)
make format     # Black auto-formatting
make type-check # Mypy type checking (0 errors in 211 source files)
make check      # Full suite: lint → type-check → test
```

---

## 📈 Feature Count: 49 Dimensions

All models operate on a consistent 49-feature vector per bar, ensuring dimension alignment across the entire pipeline — from FeaturePipeline through PPO agents, LSTM models, and classifiers.

The feature pipeline enforces this contract at runtime with padding/trimming, allowing independent model versioning.

---

## 🔧 Configuration

### Environment Variables
See `.env.example` — never commit `.env` to git.

| Variable | Required | Description |
|----------|----------|-------------|
| `CTRADER_APP_CLIENT_ID` | Yes | cTrader application client ID |
| `CTRADER_APP_CLIENT_SECRET` | Yes | cTrader application secret |
| `CTRADER_APP_ID` | Yes | cTrader application ID |
| `CTRADER_APP_SECRET` | Yes | cTrader application secret |
| `CTRADER_ACCOUNT_ID` | Yes | cTrader account ID |
| `CTRADER_ACCESS_TOKEN` | Yes | OAuth2 access token |
| `CTRADER_REFRESH_TOKEN` | For refresh | OAuth2 refresh token |
| `CTRADER_DEMO` | No | Use demo account (`true`/`false`) |
| `TRADING_PROVIDER` | No | Data provider: `ctrader` (default) or `dukascopy` |
| `REDIS_URL` | No | Redis connection URL |
| `TELEGRAM_BOT_TOKEN` | No | Telegram alerting bot token |
| `TELEGRAM_CHAT_ID` | No | Telegram alerting chat ID |
| `DASHBOARD_PORT` | No | Dashboard port (default: 8000) |
| `LOG_LEVEL` | No | Logging level (default: INFO) |
| `HF_TOKEN` | No | HuggingFace model hub token |
| `FRED_API_KEY` | No | St. Louis Fed economic data |

### config.yaml
All trading parameters in `config.yaml` — hot-reloadable without restart:
- `trading.max_risk_per_trade`, `max_drawdown`, `max_positions`
- `trading.atr_threshold_multiplier` (dynamic prediction threshold)
- `features.timeframes`, `lookback`, `use_microstructure`
- `ai.algorithm`, `ensemble.experts`, `regime_agents.*`

---

## 🧪 Testing & Code Quality

| Suite | Tests | Coverage |
|-------|-------|----------|
| Pipeline modules | 14 | EventBus, Execution, Risk, Signal, Learning |
| Ensemble & models | 61+ | MoE, predict, should_trade, edge cases |
| Risk management | 5+ | Kelly sizing, circuit breaker, correlation |
| Data providers | 8+ | Dukascopy, tick, microstructure, session |
| Integration | 9+ | Full signal pipeline end-to-end |
| Infrastructure | 4 | Config watcher, hot-reload |
| Validation | 8+ | Walk-forward, attribution, Monte Carlo |
| Execution | 5+ | Almgren-Chriss, broker health, reconciler |
| **Total** | **708** | All passing |

| Quality Gate | Status |
|-------------|--------|
| Flake8 fatal errors (E9/F63/F7/F82) | ✅ **0** |
| Flake8 style warnings | ✅ Minimal (pre-existing) |
| Black formatting | ✅ Full compliance |
| Mypy type errors | ✅ **0 errors in 211 source files** |
| Tests | ✅ **708/708 passing** |

---

## 📁 Project Structure

```
src/
├── pipeline/           # 🟢 ACTIVE — 5-module EventBus architecture (3,172 LOC)
│   ├── event_bus.py           Pub/sub with priority, once(), wait_for()
│   ├── signal_engine.py       Feature → HMM → MoE → Signal (555 LOC)
│   ├── risk_manager.py        Pre-trade checks, Kelly, circuit breaker
│   ├── execution_manager.py   Orders, positions, ATR trailing stops
│   ├── learning_manager.py    Drift, registry, checkpoint, online learning
│   ├── expert_registry.py     28 expert registration and tracking
│   ├── attribution_manager.py Performance attribution (alpha decay)
│   ├── orchestrator.py        Lifecycle + health checks
│   ├── pipeline_context.py    DI container
│   └── main.py                Entry point
│
├── rts_ai_fx/         # Core ML/AI (14 files)
│   ├── ensemble.py            MoE ensemble with Elo/Sharpe/lockout
│   ├── model.py               LSTM-CNN hybrid + classifier
│   ├── features_unified.py    49-dim feature pipeline
│   ├── regime_detector.py     HMM + simple regime detection
│   ├── drift_detector.py      ADWIN concept drift
│   ├── adversarial.py         PGD adversarial training
│   └── ...
│
├── data/              # Data ingestion (17 files)
│   ├── data_manager.py        Multi-source orchestrator (519 LOC)
│   ├── tick_ingester.py       Tick validation, batching
│   ├── historical_loader.py   BI5/CSV historical loading
│   ├── feature_cache.py       Hash-based feature caching
│   ├── dukascopy_provider.py
│   ├── dukascopy_realtime.py
│   └── ...
│
├── risk/              # Risk management (6 files)
│   ├── manager.py             Core RiskManager
│   ├── enhanced_manager.py    MAE/MFE, CVaR-Kelly
│   ├── circuit_breaker.py     4-detector market health
│   └── portfolio_optimizer.py HRP, Mean-Variance, Risk Parity
│
├── execution/         # Execution engine (8 files)
│   ├── engine.py              Execution engine
│   ├── almgren_chriss.py      IS execution for large orders
│   ├── broker_health.py       Connection monitoring
│   └── ...
│
├── validation/        # Model validation (8 files)
│   ├── smart_walk_forward.py  CPCV walk-forward
│   ├── monte_carlo.py         Permutation significance
│   ├── attribution.py         StrategyAttributionEngine
│   └── ...
│
├── infrastructure/    # Config, logging, secrets
│   ├── config.py             Typed config from config.yaml
│   ├── config_watcher.py     Hot-reload file watcher
│   └── ...
│
├── dashboard/         # FastAPI web dashboard
├── api/               # cTrader broker connectivity
├── notifications/     # Telegram alerts
├── training/          # Online learning, validation gate
└── scripts/           # CLI entry points

tests/                # 708 tests across all modules
models/               # Trained models (.keras, .pth, .npz)
```

---

## 🎯 Performance Targets

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

---

## 📜 License

Educational/research purposes only. Not licensed for commercial trading use.

---

## 🏆 Key Achievements

- **25-item refactoring roadmap** completed in a single session across 5 phases
- **708 tests** (up from 74 — 9.6× increase)
- **Zero mypy errors** across 211 source files
- **EventBus** with priority ordering, one-shot listeners, and awaitable `wait_for()`
- **Performance attribution** with alpha decay detection and strategy auto-disable
- **Config hot-reload** — change parameters without restarting
- **Dynamic ATR-based prediction thresholds** per symbol
- **Circuit breaker** with warm-up period and graceful degradation
- **25,000+ LOC of dead code** removed (`_archive/`, `agentic/` stubs)
- **Zero `sys.path.insert()` hacks** — proper package installation
