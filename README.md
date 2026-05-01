# RTS AI Forex Trading System v4.0 (Elite)

Multi-timeframe AI forex trading system with HMM regime detection, Monte Carlo Dropout uncertainty quantification, concept drift detection, and institutional-grade risk management.

## Architecture

### AI/ML Stack
| Component | Technology | Purpose |
|-----------|------------|---------|
| **PPO Reinforcement Learning** | PyTorch | Meta-controller: 5 actions (HOLD/BUY/SELL/CLOSE_ALL/MODIFY), continuous SL/TP/size outputs |
| **LSTM-CNN Hybrid** | TensorFlow/Keras | Price regression: 30-bar lookback, 51 features, dual-branch fusion |
| **Profitability Classifier** | TensorFlow/Keras | Direction prediction (binary) with CORRECT future-label construction |
| **Mixture-of-Experts Ensemble** | NumPy | Regime-gated expert weighting via HMM posterior probabilities |
| **HMM Regime Detector** | hmmlearn | 4-state (trending/ranging/volatile/crisis) — learns transitions from data |
| **MC Dropout Uncertainty** | TensorFlow | 50 forward passes → prediction variance filters low-confidence trades |
| **ADWIN Concept Drift** | River | Adaptive windowing — triggers retraining on distribution shifts |

### Feature Engineering
- **Multi-timeframe**: 15m / 1h / 4h / 1d parallel processing with z-score normalization
- **55+ features**: RSI(14,21), MACD, ATR(14,21), Bollinger Bands, ADX, Stochastic, EMA ratios, SMA distances
- **Cyclical encoding**: sin/cos for hour/day/month (fixes integer discontinuity)
- **Hurst exponent**: Rolling 50-bar estimate for mean-reversion vs trending
- **Market microstructure**: CVD, bid-ask imbalance, volatility regimes

### Risk Management
| Feature | Implementation |
|---------|---------------|
| **Kelly Criterion** | Fractional (25%), adaptive win-rate tracking |
| **ATR-based SL/TP** | Configurable multipliers (default 2.0x SL, 4.0x TP) |
| **Value-at-Risk** | Historical simulation, 95% confidence |
| **Conditional VaR** | Expected shortfall beyond VaR threshold |
| **Daily drawdown** | 5% max daily loss, 10% total max drawdown |
| **Correlation filter** | Blocks trades >0.80 correlation with open positions |
| **Pre-trade checks** | Margin usage, consecutive losses, kill switch |
| **Trailing stop** | 3-tier (30%/30%/40% partial close) |

### Cost Model
Realistic transaction costs applied to every trade:
- **Spread**: Variable per pair (EURUSD 0.5 pips, GBPUSD 0.8, etc.), widens in high volatility
- **Commission**: $7/lot (IC Markets Raw Spread pricing)
- **Slippage**: Function of trade size relative to ATR

### Data Sources
- **cTrader Open API** (IC Markets): SSL+Protobuf real-time connection, async-safe IO with retry
- **Yahoo Finance**: Historical data with caching
- **OANDA** (optional): Real-time + historical OHLCV

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
make setup
source venv/bin/activate

# Configure credentials (see .env.example)
cp .env.example .env
# Edit .env with your cTrader credentials

# Install
pip install -r requirements.txt

# Run
python -m src.main
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

## Project Structure
```
src/
├── main.py                      # Entry point: async trading loop
├── api/
│   ├── ctrader_client.py        # Async-safe SSL+Protobuf client with retry
│   ├── ctrader_icmarkets.py     # Dashboard integration client
│   └── ctrader_env.py           # Environment-variable-based client
├── risk/
│   └── manager.py               # Kelly sizing, VaR/CVaR, pre-trade checks, stress tests
├── execution/
│   ├── engine.py                # Order execution with lifecycle tracking
│   └── cost_model.py            # Spread + commission + slippage per trade
├── rts_ai_fx/
│   ├── model.py                 # LSTM-CNN Hybrid + ProfitabilityClassifier (fixed labels)
│   ├── features_unified.py      # Multi-timeframe feature pipeline (55+ features)
│   ├── regime_detector.py       # HMM-based 4-state regime detector
│   ├── ensemble.py              # Mixture-of-Experts with HMM gating
│   ├── uncertainty.py           # Monte Carlo Dropout quantification
│   ├── drift_detector.py        # ADWIN concept drift monitor
│   └── rl_agent.py              # PPO reinforcement learning agent
├── ai/
│   └── rl_agent.py              # PPO agent with GAE and continuous action heads
├── data/
│   └── data_manager.py          # Multi-timeframe tick aggregation
├── dashboard/
│   └── app.py                   # FastAPI + WebSocket real-time dashboard
└── infrastructure/
    ├── config.py                # Typed configuration (YAML + env overrides)
    └── secrets.py               # .env-based credential management
```

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
| Risk per Trade | 2% (fixed fraction) |

## Disclaimer
Trading involves substantial risk. This software is for educational/research purposes. Always test thoroughly on demo accounts before using real money. Past performance does not guarantee future results.

## Branches
- `main` — Stable release
- `develop` — Active development
- `production` — Live trading
- `gh-pages` — Dashboard deployment
