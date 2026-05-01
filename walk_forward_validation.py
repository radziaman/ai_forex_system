"""
Walk-Forward Validation for RTS AI FX Trading System
Tests if models generalize to unseen data (out-of-sample)
"""

import sys
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
from datetime import datetime
from pathlib import Path
import json

sys.path.insert(0, "src")
from rts_ai_fx.model_v2 import LSTMTransformerHybrid
from rts_ai_fx.model import ProfitabilityClassifier
from rts_ai_fx.features_v2 import add_technical_indicators
from rts_ai_fx.data import DataPreprocessor

# Configuration
PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"]
MODELS_DIR = Path("models")
LOOKBACK = 30
INITIAL_BALANCE = 10000
SL_PERCENT = 0.015
TP_PERCENT = 0.03
CONFIDENCE_THRESHOLD = 0.7

def fetch_data(pair, start, end):
    """Fetch and prepare data for a date range"""
    symbol = f"{pair}=X"
    try:
        df = yf.download(symbol, start=start, end=end, interval="1d", 
                         auto_adjust=False, progress=False)
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

def train_models(pair, train_data):
    """Train LSTM-Transformer and Classifier on training data"""
    pair_file = f"{pair}_X"
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    features = preprocessor.normalize_features(train_data)
    
    if features is None or len(features) < LOOKBACK + 10:
        return None, None, None
    
    # Convert to numpy if needed
    if hasattr(features, 'values'):
        features = features.values
    
    # Prepare sequences
    X = []
    y_price = []
    for i in range(LOOKBACK, len(features)):
        X.append(features[i-LOOKBACK:i])
        y_price.append(features[i, 3])  # Close price
    
    X = np.array(X)
    y_price = np.array(y_price)
    
    if len(X) < 10:
        return None, None, None
    
    # Train LSTM model
    n_features = X.shape[2]
    lstm_model = LSTMTransformerHybrid(lookback=LOOKBACK, n_features=n_features)
    lstm_model.build()
    lstm_model.model.compile(optimizer="adam", loss="mse")
    lstm_model.model.fit(X, y_price, epochs=5, batch_size=32, verbose=0)
    
    # Train classifier
    # y_price[i] = close price at timestep i
    # We want to predict: will price go up from i to i+1?
    # y_class[i] = 1 if y_price[i] > y_price[i-1] else 0
    y_class = (y_price[1:] > y_price[:-1]).astype(int)  # Shape: (len-1,)
    X_class = X[1:]  # Align with y_class (drop first sample)
    
    classifier = ProfitabilityClassifier(n_features=n_features)
    classifier.build()
    classifier.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    classifier.model.fit(X_class, y_class, epochs=5, batch_size=32, verbose=0)
    
    return lstm_model, classifier, preprocessor

def run_backtest(pair, lstm_model, classifier, preprocessor, test_data):
    """Run backtest on test data"""
    if test_data is None or len(test_data) < LOOKBACK:
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
        
        # Check open position
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
            # Generate signal
            X = np.array([features[i-LOOKBACK:i]])
            predicted = lstm_model.predict(X)[0]
            
            # Get classifier probability (shape: (1, 2) for binary classifier)
            prob_array = classifier.predict_proba(X)
            # Extract probability of class 1 (price goes up)
            if len(prob_array.shape) > 1:
                prob_up = prob_array[0][1]  # Shape (1,2) -> prob of class 1
            else:
                prob_up = prob_array[1] if len(prob_array) > 1 else prob_array[0]
            
            if prob_up > CONFIDENCE_THRESHOLD:
                position = "BUY"
                entry_price = current_price
                sl_price = entry_price * (1 - SL_PERCENT)
                tp_price = entry_price * (1 + TP_PERCENT)
    
    # Calculate metrics
    total_trades = len([t for t in trades if t['type'] in ['SL', 'TP']])
    wins = len([t for t in trades if t.get('pnl', 0) > 0])
    total_return = ((balance - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
    
    return {
        'trades': total_trades,
        'wins': wins,
        'win_rate': (wins / total_trades * 100) if total_trades > 0 else 0,
        'return': total_return,
        'final_balance': balance
    }

def walk_forward_test():
    """Run walk-forward validation"""
    print("="*70)
    print("WALK-FORWARD VALIDATION - RTS AI FX Trading System")
    print("="*70)
    
    results = {
        'window_1': {},  # Train: 2019-2024, Test: 2025
        'window_2': {}   # Train: 2020-2025, Test: 2026 Q1
    }
    
    # Window 1: Train 2019-2024, Test 2025
    print("\n" + "="*70)
    print("WINDOW 1: Train 2019-2024, Test 2025")
    print("="*70)
    
    for pair in PAIRS:
        print(f"\nProcessing {pair}...")
        
        # Fetch training data (2019-2024)
        train_data = fetch_data(pair, "2019-01-01", "2024-12-31")
        if train_data is None or len(train_data) < 100:
            print(f"✗ Insufficient training data for {pair}")
            continue
        
        # Train models
        lstm_model, classifier, preprocessor = train_models(pair, train_data)
        if lstm_model is None:
            print(f"✗ Training failed for {pair}")
            continue
        
        # Fetch test data (2025)
        test_data = fetch_data(pair, "2025-01-01", "2025-12-31")
        if test_data is None:
            print(f"✗ No test data for {pair}")
            continue
        
        # Run backtest
        result = run_backtest(pair, lstm_model, classifier, preprocessor, test_data)
        if result:
            results['window_1'][pair] = result
            print(f"✓ {pair}: {result['trades']} trades, {result['win_rate']:.1f}% win, {result['return']:.2f}% return")
        else:
            print(f"✗ Backtest failed for {pair}")
    
    # Window 2: Train 2020-2025, Test 2026 Q1
    print("\n" + "="*70)
    print("WINDOW 2: Train 2020-2025, Test 2026 Q1 (Jan-Mar)")
    print("="*70)
    
    for pair in PAIRS:
        print(f"\nProcessing {pair}...")
        
        # Fetch training data (2020-2025)
        train_data = fetch_data(pair, "2020-01-01", "2025-12-31")
        if train_data is None or len(train_data) < 100:
            print(f"✗ Insufficient training data for {pair}")
            continue
        
        # Train models
        lstm_model, classifier, preprocessor = train_models(pair, train_data)
        if lstm_model is None:
            print(f"✗ Training failed for {pair}")
            continue
        
        # Fetch test data (2026 Q1)
        test_data = fetch_data(pair, "2026-01-01", "2026-03-31")
        if test_data is None:
            print(f"✗ No test data for {pair}")
            continue
        
        # Run backtest
        result = run_backtest(pair, lstm_model, classifier, preprocessor, test_data)
        if result:
            results['window_2'][pair] = result
            print(f"✓ {pair}: {result['trades']} trades, {result['win_rate']:.1f}% win, {result['return']:.2f}% return")
        else:
            print(f"✗ Backtest failed for {pair}")
    
    # Print summary
    print("\n" + "="*70)
    print("WALK-FORWARD VALIDATION SUMMARY")
    print("="*70)
    
    for window, label in [('window_1', '2019-2024 → 2025'), ('window_2', '2020-2025 → 2026 Q1')]:
        print(f"\n{label}:")
        print(f"{'Pair':<10} {'Trades':>8} {'Win Rate':>10} {'Return':>10}")
        print("-"*45)
        
        if window in results and results[window]:
            for pair, res in results[window].items():
                print(f"{pair:<10} {res['trades']:>8} {res['win_rate']:>9.1f}% {res['return']:>9.2f}%")
            
            # Average
            avg_return = np.mean([r['return'] for r in results[window].values()])
            avg_win = np.mean([r['win_rate'] for r in results[window].values()])
            print("-"*45)
            print(f"{'AVERAGE':<10} {'':>8} {avg_win:>9.1f}% {avg_return:>9.2f}%")
    
    # Save results
    with open('walk_forward_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n✓ Results saved to walk_forward_results.json")
    
    return results

if __name__ == "__main__":
    results = walk_forward_test()
