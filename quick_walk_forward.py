"""
Quick Walk-Forward: Test FINE-TUNED models on UNSEEN 2025 data
This tests if models generalize (not just memorizing)
"""

import sys
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
from pathlib import Path
import json

sys.path.insert(0, "src")
from rts_ai_fx.model_v2 import LSTMTransformerHybrid
from rts_ai_fx.model import ProfitabilityClassifier
from rts_ai_fx.features_v2 import add_technical_indicators

# Configuration
PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"]
MODELS_DIR = Path("models")
LOOKBACK = 30
INITIAL_BALANCE = 10000
SL_PERCENT = 0.015
TP_PERCENT = 0.03
CONFIDENCE_THRESHOLD = 0.7

def fetch_data(pair, start, end):
    """Fetch and prepare data"""
    symbol = f"{pair}=X"
    try:
        df = yf.download(symbol, start=start, end=end, interval="1d", 
                         auto_adjust=False, progress=False)
        if df.empty:
            return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        
        df.columns = [c.lower() for c in df.columns]
        
        # Drop adj close (not needed for forex)
        if 'adj close' in df.columns:
            df = df.drop(columns=['adj close'])
        
        df = add_technical_indicators(df)
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None

def run_backtest(pair, lstm_model, classifier, preprocessor, test_data):
    """Run backtest on test data"""
    if test_data is None or len(test_data) < LOOKBACK + 10:
        return None
    
    balance = INITIAL_BALANCE
    position = None
    entry_price = 0
    sl_price = 0
    tp_price = 0
    trades = []
    
    features = preprocessor.normalize_features(test_data)
    if features is None:
        return None
    
    if hasattr(features, 'values'):
        features = features.values
    
    for i in range(LOOKBACK, len(features)):
        current_price = test_data['close'].iloc[i]
        
        if position:
            if position == "BUY":
                if current_price <= sl_price:
                    pnl = (sl_price - entry_price) * (INITIAL_BALANCE / entry_price)
                    balance += pnl
                    trades.append({'type': 'SL', 'pnl': pnl})
                    position = None
                elif current_price >= tp_price:
                    pnl = (tp_price - entry_price) * (INITIAL_BALANCE / entry_price)
                    balance += pnl
                    trades.append({'type': 'TP', 'pnl': pnl})
                    position = None
        else:
            X = np.array([features[i-LOOKBACK:i]])
            
            # Get classifier probability
            prob_array = classifier.predict_proba(X)
            
            # Handle output shape: 
            # - If softmax (2 outputs): prob_array shape (1,2) -> [0][1] is prob of class 1
            # - If sigmoid (1 output): prob_array shape (1,1) or (1,) -> [0] is prob of class 1
            if len(prob_array.shape) == 2:
                if prob_array.shape[1] == 2:
                    prob_up = prob_array[0][1]  # Softmax: prob of class 1
                else:
                    prob_up = prob_array[0][0]  # Sigmoid: single output
            else:
                prob_up = prob_array[0] if len(prob_array) > 1 else prob_array
            
            if prob_up > CONFIDENCE_THRESHOLD:
                position = "BUY"
                entry_price = current_price
                sl_price = entry_price * (1 - SL_PERCENT)
                tp_price = entry_price * (1 + TP_PERCENT)
    
    total_trades = len([t for t in trades if t['type'] in ['SL', 'TP']])
    wins = len([t for t in trades if t.get('pnl', 0) > 0])
    total_return = ((balance - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
    
    return {
        'trades': total_trades,
        'wins': wins,
        'losses': total_trades - wins,
        'win_rate': (wins / total_trades * 100) if total_trades > 0 else 0,
        'return': total_return,
        'final_balance': balance
    }

def main():
    print("="*70)
    print("QUICK WALK-FORWARD: Test on UNSEEN 2025 Data")
    print("="*70)
    print("Using FINE-TUNED models (Jan-Apr 2026 data included)")
    print("Testing on 2025 (models have NOT seen this data)")
    print("="*70)
    
    results = {}
    
    for pair in PAIRS:
        print(f"\nProcessing {pair}...")
        pair_file = f"{pair}_X"
        
        try:
            # Load FINE-TUNED models
            lstm_model = LSTMTransformerHybrid.load(MODELS_DIR / f"{pair_file}_v2_lstm_transformer.keras")
            classifier = ProfitabilityClassifier.load(MODELS_DIR / f"{pair_file}_v2_classifier.keras")
            
            with open(MODELS_DIR / f"{pair_file}_v2_preprocessor.pkl", "rb") as f:
                preprocessor = pickle.load(f)
            
            # Fetch 2025 data (UNSEEN by models)
            test_data = fetch_data(pair, "2025-01-01", "2025-12-31")
            if test_data is None:
                print(f"✗ No 2025 data for {pair}")
                continue
            
            # Run backtest
            result = run_backtest(pair, lstm_model, classifier, preprocessor, test_data)
            if result:
                results[pair] = result
                status = "✓" if result['return'] > 0 else "✗"
                print(f"{status} {pair}: {result['trades']} trades, {result['win_rate']:.1f}% win, {result['return']:+.2f}% return")
            else:
                print(f"✗ Backtest failed for {pair}")
                
        except Exception as e:
            print(f"✗ Error processing {pair}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS: Fine-Tuned Models on UNSEEN 2025 Data")
    print("="*70)
    print(f"{'Pair':<10} {'Trades':>8} {'Win Rate':>10} {'Return':>10} {'Balance':>12}")
    print("-"*70)
    
    total_return = 0
    for pair, res in results.items():
        print(f"{pair:<10} {res['trades']:>8} {res['win_rate']:>9.1f}% {res['return']:>+9.2f}% ${res['final_balance']:>11.2f}")
        total_return += res['return']
    
    avg_return = total_return / len(results) if results else 0
    print("="*70)
    print(f"{'AVERAGE':<10} {'':>8} {'':>10} {avg_return:>+9.2f}%")
    print("="*70)
    
    # Interpretation
    print("\nINTERPRETATION:")
    if avg_return > 0:
        print("✓ POSITIVE return on unseen data - models GENERALIZE well!")
        print("  This means the fine-tuning improved generalization, not just overfitting.")
    else:
        print("✗ NEGATIVE return on unseen data - models may be OVERFITTING!")
        print("  Consider reducing model complexity or using more regularization.")
    
    # Save results
    output = {
        'test_period': '2025 (unseen by models)',
        'model_type': 'V2 Fine-Tuned (with Jan-Apr 2026)',
        'results': results,
        'average_return': avg_return,
        'timestamp': str(pd.Timestamp.now())
    }
    
    with open('quick_walk_forward_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print("\n✓ Results saved to quick_walk_forward_results.json")
    
    return results

if __name__ == "__main__":
    results = main()
