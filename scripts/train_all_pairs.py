#!/usr/bin/env python3
"""
Automated Training Script for RTS - AI FX Trading System
Trains LSTM-CNN hybrid model on all major forex pairs with progress tracking.
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rts_ai_fx.trader import AITrader

# Major forex pairs to train on
MAJOR_PAIRS = [
    'EURUSD', 'GBPUSD', 'USDJPY',
    'AUDUSD', 'USDCAD', 'NZDUSD', 'USDCHF'
]

# Status file for dashboard monitoring
STATUS_FILE = Path(__file__).parent.parent / 'training_status.json'


class TrainingProgress:
    """Track and report training progress for dashboard."""

    def __init__(self):
        self.status = {
            'start_time': datetime.now().isoformat(),
            'current_pair': None,
            'completed_pairs': [],
            'failed_pairs': [],
            'total_pairs': 0,
            'progress_percent': 0,
            'current_epoch': 0,
            'total_epochs': 50,  # train_models uses 50 epochs
            'current_loss': None,
            'training_complete': False,
            'models_trained': 0,
            'backtest_results': {}
        }

    def save(self):
        """Save status to JSON file for dashboard."""
        with open(STATUS_FILE, 'w') as f:
            json.dump(self.status, f, indent=2)

    def update_pair(self, pair):
        """Update current training pair."""
        self.status['current_pair'] = pair
        self.status['total_pairs'] = len(MAJOR_PAIRS)
        self.save()

    def update_epoch(self, epoch, loss):
        """Update epoch progress."""
        self.status['current_epoch'] = epoch
        self.status['current_loss'] = float(loss) if loss else None
        pairs_done = len(self.status['completed_pairs'])
        total_pairs = self.status['total_pairs']
        epochs_done = epoch
        total_epochs = 50  # train_models uses 50 epochs
        progress = int(((pairs_done / total_pairs) * 100) +
                      ((epochs_done / total_epochs) * (100 / total_pairs)))
        self.status['progress_percent'] = min(progress, 99)
        self.save()

    def complete_pair(self, pair, backtest_result=None):
        """Mark pair as completed."""
        self.status['completed_pairs'].append(pair)
        self.status['models_trained'] += 1
        if backtest_result:
            self.status['backtest_results'][pair] = backtest_result
        self.save()

    def fail_pair(self, pair, error):
        """Mark pair as failed."""
        self.status['failed_pairs'].append({'pair': pair, 'error': str(error)})
        self.save()

    def complete_all(self):
        """Mark all training as complete."""
        self.status['training_complete'] = True
        self.status['progress_percent'] = 100
        self.status['end_time'] = datetime.now().isoformat()
        self.save()


def train_pair(pair, args, progress):
    """Train model for a single forex pair."""
    print(f"\n{'='*60}")
    print(f"Training model for {pair}")
    print(f"{'='*60}\n")

    try:
        # Use AITrader which handles everything
        trader = AITrader(
            pairs=[pair],
            initial_balance=args.initial_balance,
            timeframe='1d'
        )

        # Train the model (train_models uses 50 epochs internally)
        print(f"1. Training model for {pair} (50 epochs)...")
        progress.update_pair(pair)


        # Suppress TensorFlow output
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # Train with custom callback for progress
        import tensorflow as tf

        class ProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress.update_epoch(epoch + 1, logs.get('loss') if logs else None)

        # Run training using the correct method
        # train_models(symbol, start) - symbol should be without =X
        model, history = trader.train_models(
            symbol=pair + '=X',
            start=args.start_date
        )

        print(f"   ✓ Model trained successfully")

        # Run backtest
        print(f"2. Running backtest for {pair}...")
        backtest_result = trader.backtest(
            initial_balance=args.initial_balance
        )

        result_summary = {
            'total_return': float(backtest_result.get('total_return', 0)),
            'win_rate': float(backtest_result.get('win_rate', 0)),
            'sharpe_ratio': float(backtest_result.get('sharpe_ratio', 0)),
            'max_drawdown': float(backtest_result.get('max_drawdown', 0))
        }
        print(f"   ✓ Backtest complete: {result_summary}")

        progress.complete_pair(pair, result_summary)
        print(f"\n✅ {pair} training completed!\n")

        return True

    except Exception as e:
        print(f"\n❌ Error training {pair}: {e}\n")
        progress.fail_pair(pair, e)
        return False


def main():
    """Main training loop for all pairs."""
    parser = argparse.ArgumentParser(description='Train RTS models on all major pairs')
    parser.add_argument('--pairs', nargs='+', help='Specific pairs to train (default: all major)')
    parser.add_argument('--start-date', default='2015-01-01', help='Start date (default: 2015-01-01)')
    parser.add_argument('--initial-balance', type=float, default=10000, help='Initial balance')
    args = parser.parse_args()

    # Determine pairs to train
    pairs = args.pairs if args.pairs else MAJOR_PAIRS

    print("="*60)
    print("RTS - AI FX Trading System: Automated Training")
    print("="*60)
    print(f"Pairs to train: {len(pairs)}")
    print(f"Epochs per pair: 50 (fixed in train_models)")
    print(f"Start date: {args.start_date}")
    print("="*60)

    # Initialize progress tracking
    progress = TrainingProgress()
    progress.status['total_pairs'] = len(pairs)
    progress.save()

    # Train each pair
    for i, pair in enumerate(pairs, 1):
        print(f"\nProgress: [{i}/{len(pairs)}] {pair}")
        train_pair(pair, args, progress)
        time.sleep(2)

    # Mark all complete
    progress.complete_all()

    print("\n" + "="*60)
    print("✅ ALL TRAINING COMPLETE!")
    print("="*60)
    print(f"Completed: {len(progress.status['completed_pairs'])} pairs")
    print(f"Failed: {len(progress.status['failed_pairs'])} pairs")
    print(f"Models saved to: models/")
    print(f"Status file: {STATUS_FILE}")
    print("="*60)


if __name__ == "__main__":
    main()
