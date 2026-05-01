"""
Fine-tune existing V2 models on Jan-Apr 2026 data
Preserves pre-trained knowledge, adapts to recent market conditions
"""
import sys
import os
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, "src")
from rts_ai_fx.data import DataPreprocessor
from rts_ai_fx.model_v2 import LSTMTransformerHybrid
from rts_ai_fx.model import ProfitabilityClassifier
from rts_ai_fx.features_v2 import add_technical_indicators

# Configuration
PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"]
MODELS_DIR = Path("models")
LOOKBACK = 30
N_FEATURES = 35
FINE_TUNE_EPOCHS = 2
LEARNING_RATE = 1e-5  # Low LR to preserve existing weights

def get_pair_filename(pair):
    """Convert pair name to filename format (add _X suffix)"""
    return f"{pair}_X"

def fetch_jan_apr_2026(pair):
    """Fetch Jan-Apr 2026 data for a pair"""
    symbol = f"{pair}=X"
    try:
        df = yf.download(
            symbol,
            start="2026-01-01",
            end="2026-04-30",
            interval="1d",
            auto_adjust=False,
            progress=False
        )
        if df.empty:
            print(f"⚠️ No data for {pair}")
            return None
        
        # Fix multi-level columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        
        # Keep only OHLCV and rename to lowercase for features_v2.py
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        
        df = add_technical_indicators(df)
        return df
    except Exception as e:
        print(f"✗ Error fetching {pair}: {e}")
        import traceback
        traceback.print_exc()
        return None

def fine_tune_pair(pair):
    """Fine-tune models for a single pair"""
    print(f"\n{'='*50}")
    print(f"Fine-tuning {pair}...")
    print(f"{'='*50}")
    
    pair_file = get_pair_filename(pair)
    
    # 1. Fetch new data
    new_data = fetch_jan_apr_2026(pair)
    if new_data is None or len(new_data) < LOOKBACK:
        print(f"✗ Insufficient data for {pair}")
        return False
    
    # 2. Load existing preprocessor
    preprocessor_path = MODELS_DIR / f"{pair_file}_v2_preprocessor.pkl"
    if not preprocessor_path.exists():
        print(f"✗ Preprocessor not found for {pair}")
        return False
    
    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)
    
    # 3. Normalize new data using existing preprocessor
    features = preprocessor.normalize_features(new_data)
    if features is None or len(features) < LOOKBACK:
        print(f"✗ Feature normalization failed for {pair}")
        return False
    
    # Convert to numpy array for easier indexing
    if hasattr(features, 'values'):
        features = features.values
    
    # 4. Prepare sequences
    X = []
    y = []
    close_idx = 3  # Close price index after OHLCV
    
    for i in range(LOOKBACK, len(features)):
        X.append(features[i-LOOKBACK:i])  # Shape: (LOOKBACK, n_features)
        y.append(features[i, close_idx])  # Close price at timestep i
    
    X = np.array(X)
    y = np.array(y)
    
    if len(X) == 0:
        print(f"✗ No sequences created for {pair}")
        return False
    
    print(f"✓ Prepared {len(X)} sequences for {pair}")
    
    # 5. Load existing LSTM-Transformer model
    model_path = MODELS_DIR / f"{pair_file}_v2_lstm_transformer.keras"
    if not model_path.exists():
        print(f"✗ Model not found for {pair}")
        return False
    
    model = LSTMTransformerHybrid.load(model_path)
    
    # 6. Fine-tune LSTM model (low LR)
    print(f"Fine-tuning LSTM model for {pair}...")
    model.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse"
    )
    model.model.fit(
        X, y,
        epochs=FINE_TUNE_EPOCHS,
        batch_size=32,
        verbose=0
    )
    
    # 7. Load and fine-tune classifier
    classifier_path = MODELS_DIR / f"{pair_file}_v2_classifier.keras"
    if classifier_path.exists():
        classifier = ProfitabilityClassifier.load(classifier_path)
        
        # Prepare classifier labels
        current_prices = y
        y_class = (y > current_prices).astype(int)
        
        print(f"Fine-tuning classifier for {pair}...")
        classifier.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        classifier.model.fit(
            X, y_class,
            epochs=FINE_TUNE_EPOCHS,
            batch_size=32,
            verbose=0
        )
        classifier.save(classifier_path)
        print(f"✓ Saved fine-tuned classifier for {pair}")
    
    # 8. Save fine-tuned LSTM model
    model.save(model_path)
    print(f"✓ Saved fine-tuned LSTM model for {pair}")
    
    return True

if __name__ == "__main__":
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    
    # Create models dir if needed
    MODELS_DIR.mkdir(exist_ok=True)
    
    # Fine-tune all pairs
    results = {}
    for pair in PAIRS:
        success = fine_tune_pair(pair)
        results[pair] = "✓ Success" if success else "✗ Failed"
    
    # Print summary
    print(f"\n{'='*50}")
    print("FINE-TUNING SUMMARY")
    print(f"{'='*50}")
    for pair, status in results.items():
        print(f"{pair}: {status}")
