"""
Test script to verify dashboard and cTrader integration.
Run this to populate the dashboard with live data.
"""
import sys
import time
import json
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__)))

from api.ctrader_client import CtraderClient
from ai_forex_system.trader import AITrader
from ai_forex_system.data import DataFetcher
from ai_forex_system.features import FeatureEngineer
import tensorflow as tf

def test_dashboard():
    """Test the dashboard with live data."""
    print("=" * 60)
    print("AI Forex Trading System v3.0 - Dashboard Test")
    print("=" * 60)
    
    # Load config
    import json
    with open("config/ctrader_config.json") as f:
        config = json.load(f)
    
    print(f"\n✓ Config loaded for account: {config['account_id']}")
    print(f"  Demo mode: {config['demo']}")
    
    # Initialize trader
    print("\nInitializing AI Trader...")
    trader = AITrader(
        initial_balance=10000,
        pairs=["EURUSD=X", "GBPUSD=X", "AUDUSD=X"],
        timeframe="1h"
    )
    
    # Load model
    print("Loading trained model...")
    try:
        trader.model = tf.keras.models.load_model("models/lstm_cnn_model.keras")
        trader.n_features = 41
        trader.model_trained = True
        
        from ai_forex_system.model import ProfitabilityClassifier
        trader.classifier = ProfitabilityClassifier(lookback=30, n_features=41)
        trader.classifier.build()
        print("✓ Model and classifier loaded!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Initialize cTrader client
    print("\nConnecting to cTrader...")
    try:
        ct = CtraderClient(
            app_id=config["app_id"],
            app_secret=config["app_secret"],
            access_token=config["access_token"],
            account_id=config["account_id"],
            demo=config["demo"]
        )
        ct.start()
        print("✓ cTrader client started!")
        time.sleep(2)  # Wait for connection
    except Exception as e:
        print(f"✗ cTrader error: {e}")
        ct = None
    
    # Fetch data and generate signals
    print("\n" + "="*60)
    print("Generating Signals & Populating Dashboard")
    print("="*60)
    
    data_fetcher = DataFetcher(
        source="ctrader" if 'ct' in locals() else "yfinance",
        ctrader_client=ct if 'ct' in locals() else None
    )
    
    signals = []
    for pair in trader.pairs:
        print(f"\n{pair}:")
        try:
            # Fetch data
            df = data_fetcher.fetch_ohlcv(pair, "1d", "2024-06-01")
            df = trader.feature_engineer.generate_all_features(df)
            
            if len(df) >= 30:
                window = df.iloc[-30:]
                signal = trader.generate_signal(window)
                current_price = window["close"].iloc[-1]
                
                signal_data = {
                    "symbol": pair.replace("=X", ""),
                    "signal": signal['signal'],
                    "entry": float(current_price),
                    "predicted": float(signal['predicted_price']),
                    "confidence": float(signal['confidence']),
                    "sl": float(current_price * 0.995),
                    "tp": float(current_price * 1.01),
                }
                signals.append(signal_data)
                
                print(f"  Current: {current_price:.5f}")
                print(f"  Signal: {signal['signal']}")
                print(f"  Confidence: {signal['confidence']:.1%}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Update dashboard state
    print("\n" + "="*60)
    print("Dashboard Data Summary")
    print("="*60)
    print(f"✓ Generated {len(signals)} signals")
    print(f"✓ cTrader status: {'Connected' if ct else 'Disconnected'}")
    print(f"✓ Model status: Loaded")
    print(f"\nDashboard URL: http://localhost:8000")
    print(f"Run: source venv/bin/activate && python -m uvicorn src.dashboard.app:app --host 0.0.0.0 --port 8000")
    
    # Save signals for dashboard
    with open("signals_live.json", "w") as f:
        json.dump(signals, f, indent=2)
    print("\n✓ Signals saved to signals_live.json")
    
    print("\n" + "="*60)
    print("Next Steps:")
    print("1. Start dashboard: python -m uvicorn src.dashboard.app:app --host 0.0.0.0 --port 8000")
    print("2. Open browser: http://localhost:8000")
    print("3. Monitor live signals and cTrader data")
    print("="*60)

if __name__ == "__main__":
    test_dashboard()
