"""
Simulation with fine-tuned models (Jan-Apr 2026 data)
Uses V3 Fixed logic: threshold 0.7, $10k exposure, 1.5% SL/3% TP
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "src")
from rts_ai_fx.model_v2 import LSTMTransformerHybrid
from rts_ai_fx.model import ProfitabilityClassifier
from rts_ai_fx.features_v2 import add_technical_indicators

# Configuration
PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"]
MODELS_DIR = Path("models")
LOOKBACK = 30
INITIAL_BALANCE = 10000
SL_PERCENT = 0.015  # 1.5%
TP_PERCENT = 0.03   # 3% (2:1 reward-risk)
CONFIDENCE_THRESHOLD = 0.7

def fetch_test_data(pair, start="2026-03-01", end="2026-03-31"):
    """Fetch test data for March 2026 (known to have data)"""
    symbol = f"{pair}=X"
    try:
        df = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
        if df.empty:
            return None
        
        # Fix multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        
        df.columns = [c.lower() for c in df.columns]
        df = add_technical_indicators(df)
        return df
    except Exception as e:
        print(f"Error fetching {pair}: {e}")
        return None

def generate_signal(model, classifier, preprocessor, data, current_idx):
    """Generate trading signal using V3 Fixed logic"""
    if current_idx < LOOKBACK:
        return None, 0.0
    
    # Get recent data window
    window = data.iloc[current_idx - LOOKBACK:current_idx]
    features = preprocessor.normalize_features(window)
    
    if features is None or len(features) < LOOKBACK:
        return None, 0.0
    
    # Convert to numpy if needed
    if hasattr(features, 'values'):
        features = features.values
    
    X = np.array([features])  # Shape: (1, LOOKBACK, n_features)
    
    # Get prediction
    predicted_price = model.predict(X)[0]
    
    # Denormalize
    last_close = window['close'].iloc[-1]
    predicted_change_pct = (predicted_price - last_close) / last_close
    
    # Get classifier probability
    if classifier:
        prob_up = classifier.predict_proba(X)[0][1]
    else:
        prob_up = 0.5
    
    # V3 Fixed logic: trade if confident and prediction > threshold
    if prob_up > CONFIDENCE_THRESHOLD:
        return "BUY", prob_up
    elif prob_up < (1 - CONFIDENCE_THRESHOLD):
        return "SELL", 1 - prob_up
    else:
        return None, prob_up

def run_simulation():
    """Run simulation with fine-tuned models"""
    print("="*60)
    print("RTS - AI FX Trading System - Fine-Tuned Simulation")
    print("Testing: April 2026 (Out-of-Sample)")
    print("="*60)
    
    results = {}
    
    for pair in PAIRS:
        print(f"\n{'='*60}")
        print(f"Processing {pair}...")
        print(f"{'='*60}")
        
        pair_file = f"{pair}_X"
        
        # Load models
        try:
            model = LSTMTransformerHybrid.load(MODELS_DIR / f"{pair_file}_v2_lstm_transformer.keras")
            classifier = ProfitabilityClassifier.load(MODELS_DIR / f"{pair_file}_v2_classifier.keras")
            
            with open(MODELS_DIR / f"{pair_file}_v2_preprocessor.pkl", "rb") as f:
                preprocessor = pickle.load(f)
        except Exception as e:
            print(f"✗ Error loading models for {pair}: {e}")
            continue
        
        # Fetch test data (April 2026)
        test_data = fetch_test_data(pair)
        if test_data is None or len(test_data) < LOOKBACK:
            print(f"✗ Insufficient test data for {pair}")
            continue
        
        # Initialize portfolio
        balance = INITIAL_BALANCE
        position = None
        entry_price = 0
        sl_price = 0
        tp_price = 0
        trades = []
        
        # Simulate day by day
        for i in range(LOOKBACK, len(test_data)):
            current_price = test_data['close'].iloc[i]
            current_date = test_data.index[i]
            
            # Check if we have an open position
            if position:
                # Check SL/TP
                if position == "BUY":
                    if current_price <= sl_price:
                        pnl = (sl_price - entry_price) * (INITIAL_BALANCE / entry_price)
                        balance += pnl
                        trades.append({
                            'date': current_date,
                            'type': 'SL',
                            'pnl': pnl,
                            'balance': balance
                        })
                        position = None
                    elif current_price >= tp_price:
                        pnl = (tp_price - entry_price) * (INITIAL_BALANCE / entry_price)
                        balance += pnl
                        trades.append({
                            'date': current_date,
                            'type': 'TP',
                            'pnl': pnl,
                            'balance': balance
                        })
                        position = None
                elif position == "SELL":
                    # For shorts (if implemented)
                    pass
            else:
                # No position - check for signal
                signal, confidence = generate_signal(model, classifier, preprocessor, test_data, i)
                
                if signal == "BUY":
                    position = "BUY"
                    entry_price = current_price
                    sl_price = entry_price * (1 - SL_PERCENT)
                    tp_price = entry_price * (1 + TP_PERCENT)
                    trades.append({
                        'date': current_date,
                        'type': 'BUY',
                        'price': entry_price,
                        'confidence': confidence
                    })
        
        # Calculate results
        total_trades = len([t for t in trades if t.get('type') in ['SL', 'TP']])
        wins = len([t for t in trades if t.get('type') in ['SL', 'TP'] and t.get('pnl', 0) > 0])
        total_return = ((balance - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
        
        results[pair] = {
            'trades': total_trades,
            'wins': wins,
            'win_rate': (wins / total_trades * 100) if total_trades > 0 else 0,
            'return': total_return,
            'final_balance': balance
        }
        
        print(f"✓ {pair}: {total_trades} trades, {results[pair]['win_rate']:.1f}% win rate, {total_return:.2f}% return")
    
    # Print summary
    print(f"\n{'='*60}")
    print("FINE-TUNED SIMULATION RESULTS (April 2026)")
    print(f"{'='*60}")
    print(f"{'Pair':<10} {'Trades':>8} {'Win Rate':>10} {'Return':>10} {'Balance':>12}")
    print(f"{'-'*60}")
    
    total_return_all = 0
    for pair, res in results.items():
        print(f"{pair:<10} {res['trades']:>8} {res['win_rate']:>9.1f}% {res['return']:>9.2f}% ${res['final_balance']:>11.2f}")
        total_return_all += res['return']
    
    avg_return = total_return_all / len(results) if results else 0
    print(f"{'='*60}")
    print(f"{'AVERAGE':<10} {'':>8} {'':>10} {avg_return:>9.2f}%")
    print(f"{'='*60}")
    
    return results

if __name__ == "__main__":
    results = run_simulation()
    
    # Save results to JSON for dashboard
    import json
    output = {
        'simulation_date': '2026-04',
        'model_type': 'V2 Fine-Tuned (Jan-Apr 2026)',
        'results': results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('fine_tuned_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print("\n✓ Results saved to fine_tuned_results.json")
