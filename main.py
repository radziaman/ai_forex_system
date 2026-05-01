"""Main entry point for RTS - AI FX Trading System"""

import argparse
import sys
from pathlib import Path

import tensorflow as tf
from rts_ai_fx.backtest import BacktestEngine
from rts_ai_fx.dashboard import TradingDashboard
from rts_ai_fx.data import DataFetcher
from rts_ai_fx.features import FeatureEngineer
from rts_ai_fx.trader import AITrader

sys.path.insert(0, str(Path(__file__).parent / "src"))


def train_model(args):
    """Train the LSTM-CNN hybrid model"""
    print("Initializing AI Trader...")
    trader = AITrader(
        initial_balance=10000, pairs=[args.symbol], timeframe=args.timeframe
    )

    print(f"Training model on {args.symbol} from {args.start_date}...")
    trader.train_models(symbol=args.symbol, start=args.start_date)

    print("Saving model...")
    trader.model.save("models/lstm_cnn_model.h5")
    print("Model saved to models/lstm_cnn_model.h5")


def run_backtest(args):
    """Run backtest with out-of-sample validation"""

    print("Initializing Backtest Engine...")
    trader = AITrader(initial_balance=args.initial_balance, timeframe=args.timeframe)

    print("Loading trained model...")
    try:
        # Try loading the newer .keras format first
        trader.model = tf.keras.models.load_model("models/lstm_cnn_model.keras")
        trader.model_trained = True
        print("Model loaded successfully from models/lstm_cnn_model.keras")
    except Exception:
        print("No .keras model found, trying .h5...")
        try:
            trader.model = tf.keras.models.load_model("models/lstm_cnn_model.h5")
            trader.model_trained = True
            print("Model loaded successfully from models/lstm_cnn_model.h5")
        except Exception:
            print("No trained model found, training new model...")
            trader.train_models(symbol=args.symbol, start="2015-01-01")

    backtest = BacktestEngine(initial_balance=args.initial_balance, commission=0.0001)

    print(f"Fetching data for {args.symbol}...")
    data_fetcher = DataFetcher(source="yfinance")
    df = data_fetcher.fetch_ohlcv(args.symbol, args.timeframe, "2015-01-01")

    print("Engineering features...")
    feature_engineer = FeatureEngineer(lookback=30)
    df = feature_engineer.generate_all_features(df)

    print("Running backtest with out-of-sample validation...")
    results = backtest.run_backtest(
        df,
        trader.model,
        feature_engineer,
        trader.risk_manager,
        train_end="2020-01-01",
        out_of_sample=True,
    )

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    for key, value in results.items():
        print(f"{key}: {value}")
    print("=" * 60)


def run_live(args):
    """Run live trading (simulation mode)"""
    print("Initializing Live Trading...")
    trader = AITrader(
        initial_balance=args.initial_balance, pairs=["EURUSD=X", "GBPUSD=X", "XAUUSD=X"]
    )

    dashboard = TradingDashboard(trader)

    try:
        print("Loading trained model...")
        trader.model = trader.model.load("models/lstm_cnn_model.h5")
        trader.model_trained = True
    except Exception:
        print("No trained model found. Training new model...")
        trader.train_models()

    print("\nStarting live trading simulation...")
    dashboard.display_status()

    # Live trading loop would go here
    print("Live trading simulation active. Press Ctrl+C to stop.")


def main():
    parser = argparse.ArgumentParser(description="RTS - AI FX Trading System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--symbol", type=str, default="EURUSD=X", help="Trading symbol"
    )
    train_parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe")
    train_parser.add_argument(
        "--start-date", type=str, default="2015-01-01", help="Start date"
    )

    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument(
        "--symbol", type=str, default="EURUSD=X", help="Trading symbol"
    )
    backtest_parser.add_argument(
        "--timeframe", type=str, default="1h", help="Timeframe"
    )
    backtest_parser.add_argument(
        "--initial-balance", type=float, default=10000, help="Initial balance"
    )

    live_parser = subparsers.add_parser("live", help="Run live trading")
    live_parser.add_argument(
        "--initial-balance", type=float, default=10000, help="Initial balance"
    )

    args = parser.parse_args()

    if args.command == "train":
        train_model(args)
    elif args.command == "backtest":
        run_backtest(args)
    elif args.command == "live":
        run_live(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
