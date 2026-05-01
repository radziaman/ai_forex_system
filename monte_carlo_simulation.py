"""
Monte Carlo Simulation for RTS AI FX Trading System
Runs 1000+ simulations with randomized sequences to test robustness
"""

import sys
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, "src")
from rts_ai_fx.model_v2 import LSTMTransformerHybrid
from rts_ai_fx.model import ProfitabilityClassifier
from rts_ai_fx.features_v2 import add_technical_indicators

# Configuration
PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]
MODELS_DIR = Path("models")
LOOKBACK = 30
INITIAL_BALANCE = 10000
SL_PERCENT = 0.015
TP_PERCENT = 0.03
CONFIDENCE_THRESHOLD = 0.7
N_SIMULATIONS = 100  # Reduced from 1000 for speed

def fetch_data(pair, start="2026-01-01", end="2026-03-31"):
    """Fetch data for simulation"""
    symbol = f"{pair}=X"
    try:
        df = yf.download(symbol, start=start, end=end, interval="1d", 
                         auto_adjust=False, progress=False)
        if df.empty:
            return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        
        df.columns = [c.lower() for c in df.columns]
        
        if 'adj close' in df.columns:
            df = df.drop(columns=['adj close'])
        
        df = add_technical_indicators(df)
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None

def run_single_simulation(pair, lstm_model, classifier, preprocessor, data, seed):
    """Run a single simulation with randomized sequence"""
    np.random.seed(seed)
    
    # Randomize the order of days (with overlap for sequences)
    n_days = len(data)
    indices = np.arange(LOOKBACK, n_days)
    np.random.shuffle(indices)
    
    balance = INITIAL_BALANCE
    position = None
    entry_price = 0
    sl_price = 0
    tp_price = 0
    
    features = preprocessor.normalize_features(data)
    if features is None:
        return None
    
    if hasattr(features, 'values'):
        features = features.values
    
    trades = 0
    wins = 0
    
    for idx in indices:
        if idx < LOOKBACK or idx >= len(features):
            continue
        
        current_price = data['close'].iloc[idx]
        
        if position:
            if position == "BUY":
                if current_price <= sl_price:
                    pnl = (sl_price - entry_price) * (INITIAL_BALANCE / entry_price)
                    balance += pnl
                    trades += 1
                    position = None
                elif current_price >= tp_price:
                    pnl = (tp_price - entry_price) * (INITIAL_BALANCE / entry_price)
                    balance += pnl
                    trades += 1
                    wins += 1
                    position = None
        else:
            X = np.array([features[idx-LOOKBACK:idx]])
            predicted = lstm_model.predict(X)[0]
            
            prob_array = classifier.predict_proba(X)
            if len(prob_array.shape) == 2:
                if prob_array.shape[1] == 2:
                    prob_up = prob_array[0][1]
                else:
                    prob_up = prob_array[0][0]
            else:
                prob_up = prob_array[0] if len(prob_array) > 1 else prob_array
            
            if prob_up > CONFIDENCE_THRESHOLD:
                position = "BUY"
                entry_price = current_price
                sl_price = entry_price * (1 - SL_PERCENT)
                tp_price = entry_price * (1 + TP_PERCENT)
    
    return {
        'final_balance': balance,
        'return_pct': ((balance - INITIAL_BALANCE) / INITIAL_BALANCE) * 100,
        'trades': trades,
        'wins': wins
    }

def monte_carlo_simulation():
    """Run Monte Carlo simulation"""
    print("="*70)
    print("MONTE CARLO SIMULATION - RTS AI FX Trading System")
    print(f"Running {N_SIMULATIONS} simulations per pair")
    print("="*70)
    
    all_results = {}
    
    for pair in PAIRS:
        print(f"\nProcessing {pair}...")
        pair_file = f"{pair}_X"
        
        try:
            # Load models
            lstm_model = LSTMTransformerHybrid.load(MODELS_DIR / f"{pair_file}_v2_lstm_transformer.keras")
            classifier = ProfitabilityClassifier.load(MODELS_DIR / f"{pair_file}_v2_classifier.keras")
            
            with open(MODELS_DIR / f"{pair_file}_v2_preprocessor.pkl", "rb") as f:
                preprocessor = pickle.load(f)
            
            # Fetch data
            data = fetch_data(pair)
            if data is None or len(data) < LOOKBACK + 10:
                print(f"✗ Insufficient data for {pair}")
                continue
            
            # Run simulations
            print(f"Running {N_SIMULATIONS} simulations...")
            sim_results = []
            
            for i in range(N_SIMULATIONS):
                result = run_single_simulation(pair, lstm_model, classifier, preprocessor, data, seed=i)
                if result:
                    sim_results.append(result)
                
                if (i+1) % 100 == 0:
                    print(f"  Completed {i+1}/{N_SIMULATIONS}...")
            
            if not sim_results:
                print(f"✗ No valid simulations for {pair}")
                continue
            
            # Calculate statistics
            returns = [r['return_pct'] for r in sim_results]
            final_balances = [r['final_balance'] for r in sim_results]
            
            results = {
                'n_simulations': len(sim_results),
                'mean_return': float(np.mean(returns)),
                'median_return': float(np.median(returns)),
                'std_return': float(np.std(returns)),
                'min_return': float(np.min(returns)),
                'max_return': float(np.max(returns)),
                'pct_positive': float(np.sum(np.array(returns) > 0) / len(returns) * 100),
                'pct_negative': float(np.sum(np.array(returns) < 0) / len(returns) * 100),
                'var_95': float(np.percentile(returns, 5)),  # 95% VaR
                'cvar_95': float(np.mean([r for r in returns if r <= np.percentile(returns, 5)])),  # CVaR
                'profit_factor': float(np.sum([r for r in returns if r > 0]) / abs(np.sum([r for r in returns if r < 0]))) if np.sum([r for r in returns if r < 0]) != 0 else float('inf')
            }
            
            all_results[pair] = results
            
            print(f"✓ {pair} complete:")
            print(f"  Mean return: {results['mean_return']:.2f}%")
            print(f"  Median return: {results['median_return']:.2f}%")
            print(f"  Std Dev: {results['std_return']:.2f}%")
            print(f"  Min/Max: {results['min_return']:.2f}% / {results['max_return']:.2f}%")
            print(f"  % Positive: {results['pct_positive']:.1f}%")
            print(f"  95% VaR: {results['var_95']:.2f}%")
            
        except Exception as e:
            print(f"✗ Error processing {pair}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("MONTE CARLO SUMMARY")
    print("="*70)
    print(f"{'Pair':<10} {'Mean Ret':>10} {'Median':>10} {'Std Dev':>10} {'% Pos':>10}")
    print("-"*70)
    
    for pair, res in all_results.items():
        print(f"{pair:<10} {res['mean_return']:>9.2f}% {res['median_return']:>9.2f}% {res['std_return']:>9.2f}% {res['pct_positive']:>9.1f}%")
    
    print("="*70)
    
    # Interpretation
    print("\nINTERPRETATION:")
    for pair, res in all_results.items():
        if res['pct_positive'] > 60 and res['mean_return'] > 0:
            print(f"✓ {pair}: ROBUST - {res['pct_positive']:.1f}% of simulations profitable")
        elif res['pct_positive'] > 50:
            print(f"⚠ {pair}: MODERATE - {res['pct_positive']:.1f}% profitable, but high variance")
        else:
            print(f"✗ {pair}: RISKY - Only {res['pct_positive']:.1f}% profitable!")
    
    # Save results
    output = {
        'simulation_type': 'Monte Carlo',
        'n_simulations': N_SIMULATIONS,
        'results': all_results,
        'timestamp': str(datetime.now())
    }
    
    with open('monte_carlo_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print("\n✓ Results saved to monte_carlo_results.json")
    
    return all_results

if __name__ == "__main__":
    results = monte_carlo_simulation()
