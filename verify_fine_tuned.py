"""
Quick verification: Test fine-tuned models on March 2026 data
"""
import sys
import numpy as np
import yfinance as yf
import pickle
import pandas as pd
from pathlib import Path

sys.path.insert(0, "src")
from rts_ai_fx.model_v2 import LSTMTransformerHybrid
from rts_ai_fx.model import ProfitabilityClassifier
from rts_ai_fx.features_v2 import add_technical_indicators

PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"]
MODELS_DIR = Path("models")
LOOKBACK = 30

print("="*60)
print("Verifying Fine-Tuned Models")
print("="*60)

results = {}

for pair in PAIRS:
    print(f"\nProcessing {pair}...")
    pair_file = f"{pair}_X"
    
    try:
        # Load models
        model = LSTMTransformerHybrid.load(MODELS_DIR / f"{pair_file}_v2_lstm_transformer.keras")
        classifier = ProfitabilityClassifier.load(MODELS_DIR / f"{pair_file}_v2_classifier.keras")
        
        with open(MODELS_DIR / f"{pair_file}_v2_preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)
        
        # Fetch March 2026 data
        symbol = f"{pair}=X"
        df = yf.download(symbol, start="2026-03-01", end="2026-03-31", interval="1d", auto_adjust=False, progress=False)
        
        if df.empty or len(df) < LOOKBACK:
            print(f"✗ Insufficient data for {pair}")
            continue
        
        # Fix columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df.columns = [c.lower() for c in df.columns]
        
        # Add technical indicators
        df = add_technical_indicators(df)
        
        # Normalize features
        features = preprocessor.normalize_features(df)
        if features is None:
            print(f"✗ Feature normalization failed")
            continue
        
        if hasattr(features, 'values'):
            features = features.values
        
        # Test prediction for last day
        X = np.array([features[-LOOKBACK:]])
        predicted = model.predict(X)[0]
        prob_up = classifier.predict_proba(X)[0][1]
        
        actual_price = df['close'].iloc[-1]
        
        print(f"✓ {pair}: Predicted={predicted:.4f}, Actual={actual_price:.4f}, Prob_Up={prob_up:.2f}")
        
        results[pair] = {
            'predicted': float(predicted),
            'actual': float(actual_price),
            'prob_up': float(prob_up)
        }
        
    except Exception as e:
        print(f"✗ Error processing {pair}: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*60}")
print("VERIFICATION COMPLETE")
print(f"{'='*60}")
print(f"Successfully tested {len(results)}/{len(PAIRS)} pairs")
