"""
Live Trading with cTrader Integration
Connects the AI Forex Trading System to cTrader (IC Markets).
"""

import json
import sys
from pathlib import Path

# Add src to path (go up one level from scripts/ to project root, then to src/)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_forex_system.trader import AITrader
from ai_forex_system.data import DataFetcher
import tensorflow as tf


# flake8: noqa: E402


def load_ctrader_config():
    """Load cTrader configuration"""
    config_path = Path("config/ctrader_config.json")
    if not config_path.exists():
        print("cTrader config not found. Run: python scripts/setup_ctrader.py")
        return None

    with open(config_path) as f:
        return json.load(f)


def main():
    print("=" * 60)
    print("AI Forex Trading System - Live Trading (cTrader)")
    print("=" * 60 + "\n")

    # Load config
    config = load_ctrader_config()
    if not config:
        return

    # Initialize trader
    print("Initializing AI Trader...")
    trader = AITrader(
        initial_balance=10000, pairs=["EURUSD=X", "GBPUSD=X", "AUDUSD=X"], timeframe="1h"
    )

    # Load trained model
    print("Loading trained model...")
    try:
        trader.model = tf.keras.models.load_model("models/lstm_cnn_model.keras")
        trader.model_trained = True
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please train the model first: python main.py train")
        return

    # Initialize cTrader client
    print(f"\nConnecting to cTrader ({'Demo' if config['demo'] else 'Live'})...")
    # Note: cTrader client uses Twisted reactor, which needs special handling
    # For now, we'll run in simulation mode
    print("\n[SIMULATION MODE]")
    print("To enable live trading:")
    print("1. Complete cTrader OAuth2 setup")
    print("2. Uncomment cTrader client initialization in trader.py")
    print("3. Run this script again\n")

    # Simulation loop
    print("Starting simulation...")
    data_fetcher = DataFetcher(source="yfinance")

    for pair in trader.pairs:
        print(f"\nFetching data for {pair}...")
        df = data_fetcher.fetch_ohlcv(pair, trader.timeframe, "2024-01-01")
        df = trader.feature_engineer.generate_all_features(df)

        # Get last 30 bars for prediction
        if len(df) >= 30:
            window = df.iloc[-30:]
            signal = trader.generate_signal(window)
            current_price = window["close"].iloc[-1]

            print(f"{pair}: Current={current_price:.5f}, Signal={signal['signal']}")
            print(
                f"  Predicted: {signal['predicted_price']:.5f}, Confidence: {signal['confidence']:.2%}"
            )

    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
