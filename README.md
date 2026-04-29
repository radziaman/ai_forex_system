# AI Forex Trading System v3.0

Advanced AI-powered Forex trading system implementing **LSTM-CNN hybrid architecture** based on research of top-performing systems (Sentinel AI, Zenox EA, Aurum AI).

## Features

### Core AI Architecture (Based on Top Performers)
- **LSTM-CNN Hybrid Model** (Aurum AI architecture)
  - LSTM branch: Captures temporal dependencies (30 bars)
  - CNN branch: Extracts local patterns from feature matrix
  - Fusion layer: Combines both for price prediction
- **Profitability Classifier**: Secondary network for trade confidence scoring
- **25-year training capability** (like Zenox EA) with reinforcement learning
- **Out-of-sample validation** (Aurum AI methodology)

### 51+ Engineered Features
- **Price dynamics**: body_size, shadows, range
- **Momentum indicators**: RSI (14,21), MACD, momentum (3,5), ROC
- **Volatility features**: ATR (14,21), Yang-Zhang volatility, Garman-Klass
- **Trend indicators**: EMA (20,50,100,200), ADX, trend strength
- **Market regime**: Volatility regime, trend regime, crisis filter
- **Time features**: Hour, day of week, London/NY/Tokyo sessions
- **Cross-asset**: Gold correlation, USDX correlation

### Risk Management (Zenox EA Approach)
- **No martingale/grid** - Predefined SL/TP on every trade
- **Dynamic position sizing**: Kelly Criterion or volatility-based
- **ATR-based SL/TP**: Adaptive to market conditions
- **Correlation filtering**: Blocks correlated pairs (>0.80)
- **Daily drawdown protection**: Configurable limits
- **3-tier trailing stop**: Partial closes at 30%/30%/40%

### Multi-Pair Trading
- Trade 16+ pairs simultaneously (like Zenox EA)
- Single chart setup with portfolio context
- Real-time correlation monitoring

## Branches

- `main` - Stable release branch
- `develop` - Development branch (active)
- `production` - Live production branch
- `gh-pages` - GitHub Pages dashboard

## Installation

```bash
# Create virtual environment
make setup
# or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Train Model
```bash
python main.py train --symbol EURUSD=X --timeframe 1h --start-date 2015-01-01
```

### Run Backtest (with Out-of-Sample Validation)
```bash
python main.py backtest --symbol EURUSD=X --initial-balance 10000
```

### Live Trading (Simulation)
```bash
python main.py live --initial-balance 10000
```

### Jupyter Analysis
```bash
jupyter lab notebooks/analysis.ipynb
```

## Development Commands

```bash
make test          # Run tests
make lint          # Lint code
make format        # Format code
make check         # Run all checks
```

## Performance Targets (Based on Research)

| System | Monthly Return | Win Rate | Max Drawdown | Track Record |
|--------|---------------|----------|---------------|--------------|
| Sentinel AI | 2.05% | ~70% | 29% | 7+ years |
| Zenox EA | ~5-10% | 82% | 21% | 25-year backtest |
| Aurum AI | ~5% | ~70% | 8.35% | Verified on MyFxBook |

## Requirements

- Python 3.9+
- TensorFlow 2.15+ (for LSTM-CNN model)
- See `requirements.txt` for all dependencies

## Project Structure

```
src/ai_forex_system/
├── features.py      # 51+ feature engineering
├── model.py         # LSTM-CNN hybrid architecture
├── risk.py          # Risk management & position sizing
├── data.py          # Data fetching (yfinance/ccxt)
├── backtest.py      # Backtesting with out-of-sample validation
├── trader.py        # Main trading bot
└── dashboard.py     # Real-time monitoring
tests/              # Comprehensive test suite
notebooks/          # Jupyter analysis notebooks
```

## Automation

- **GitHub Pages**: Auto-deploys on push to `main`/`gh-pages`
- **CI/CD**: Multi-Python testing via GitHub Actions
- **Pre-commit hooks**: Black, flake8, mypy for code quality

## Research-Based Implementation

This system implements strategies from verified top performers:
- Sentinel AI: Deep learning filters, ChatGPT integration
- Zenox EA: 25-year reinforcement learning, multi-pair diversification
- Aurum AI: LSTM-CNN hybrid, 59,000+ hours training data

## Disclaimer

Trading involves substantial risk. This software is for educational/research purposes. Always test thoroughly on demo accounts before using real money. Past performance does not guarantee future results.
