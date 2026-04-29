# AI Forex Trading System v3.0

AI-powered Forex trading system with machine learning, risk management, backtesting, and real-time dashboard.

## Features

- Machine Learning-based trading signals
- Advanced risk management
- Backtesting engine
- Real-time GitHub Pages dashboard
- Automated CI/CD with GitHub Actions

## Branches

- `main` - Stable release branch
- `develop` - Development branch
- `production` - Live production branch
- `gh-pages` - GitHub Pages dashboard

## Setup

```bash
make setup
# or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Run tests
make test

# Lint code
make lint

# Format code
make format

# Run all checks
make check
```

## Requirements

- Python 3.9+
- See requirements.txt for Python dependencies

## Automation

- GitHub Pages auto-deploys on push to main/gh-pages
- CI/CD workflows in `.github/workflows/`
