#!/usr/bin/env python3
"""Train only the meta-learner (regime prediction classifier).

This is a focused version of the meta-learner step from train_all_models.py.
Trains a MetaNet: 16 feature dims → 128 → 64 → 32 → 4 regime outputs.
Saves checkpoint to models/meta_learner.pt on best validation loss.
"""
import sys
import time
from pathlib import Path

_src_path = str(Path(__file__).resolve().parent.parent)
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger

from rts_ai_fx.features_unified import compute_features
from rts_ai_fx.regime_detector import HMMRegimeDetector

# ── Constants ──────────────────────────────────────────────────────────
SYMBOLS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "USDCHF",
    "NZDUSD",
    "XAUUSD",
    "XTIUSD",
    "US500",
    "BTCUSD",
]
TIMEFRAMES = ["1h"]
LOOKBACK = 30
TRAIN_RATIO = 0.8
MODEL_DIR = Path("models")
DATA_DIR = Path("data/historical")

REGIME_NAMES = ["trending", "ranging", "volatile", "crisis"]


def load_csv_data(symbol: str) -> dict:
    path = DATA_DIR / f"{symbol}_1h.csv"
    if not path.exists():
        return {}
    return {"1h": pd.read_csv(path)}


def _get_hmm_regime_labels(detector, df, prices):
    """Get per-bar HMM regime labels aligned with price array length.

    Returns array of int regime labels (0=trending, 1=ranging, 2=volatile, 3=crisis).
    """
    from rts_ai_fx.regime_detector import HMM_AVAILABLE

    if not HMM_AVAILABLE or detector.model is None:
        returns = np.abs(np.diff(prices) / (prices[:-1] + 1e-10))
        vol = pd.Series(prices).rolling(20).std().values[1:]
        fallback_labels = np.zeros(len(returns), dtype=int)
        if len(vol) > 0:
            vol_sorted = np.sort(vol)
            for i in range(len(vol)):
                pct = np.searchsorted(vol_sorted, vol[i]) / len(vol_sorted)
                if pct > 0.95:
                    fallback_labels[i] = 3
                elif pct > 0.70:
                    fallback_labels[i] = 2
                elif pct > 0.30:
                    fallback_labels[i] = 1
        return np.concatenate([[0], fallback_labels])

    features = detector._extract_features(df)
    if len(features) < 5:
        return np.zeros(len(prices), dtype=int)

    hidden_states = detector.model.predict(features)

    if detector.model.means_ is not None:
        mean_returns = detector.model.means_[:, 0]
        state_order = np.argsort(mean_returns)
        label_to_name = {
            i: REGIME_NAMES[state_order[i]] for i in range(detector.n_regimes)
        }
        regime_names_states = np.array([label_to_name[s] for s in hidden_states])
        regime_labels = np.array(
            [REGIME_NAMES.index(name) for name in regime_names_states]
        )
    else:
        regime_labels = hidden_states % len(REGIME_NAMES)

    if len(regime_labels) < len(prices):
        regime_labels = np.concatenate([[int(regime_labels[0])], regime_labels])

    return regime_labels.astype(int)


class MetaNet(nn.Module):
    def __init__(self, input_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 4),
        )

    def forward(self, x):
        return self.net(x)


def train_meta_learner():
    logger.info("=" * 60)
    logger.info("  META-LEARNER TRAINING")
    logger.info("=" * 60)

    # Load primary symbol data
    symbol = "EURUSD"
    data = load_csv_data(symbol)
    if not data:
        logger.error("Cannot load EURUSD data")
        return False

    df = data[TIMEFRAMES[0]]
    logger.info(f"Loaded {symbol}: {len(df)} bars")

    # Compute features
    df_features = compute_features(df)
    prices = df["close"].values.astype(np.float64)
    logger.info(f"Features computed: {df_features.shape[1]} columns")

    # HMM regime detection
    detector = HMMRegimeDetector(n_regimes=4)
    detector.fit(df)
    regime_labels = _get_hmm_regime_labels(detector, df, prices)
    logger.info(f"Regime distribution: {np.bincount(regime_labels)}")

    # Build meta-features: 16 key indicators
    meta_feature_cols = [
        "rsi_14",
        "adx_14",
        "atr_14",
        "volatility_20",
        "bb_width",
        "mom_5",
        "mom_10",
        "vol_ratio",
        "hurst",
        "stoch_k",
    ]
    # Fill remaining slots with available features
    available = [c for c in meta_feature_cols if c in df_features.columns]
    extra_cols = [
        c
        for c in df_features.columns
        if c
        not in {
            "open",
            "high",
            "low",
            "close",
            "volume",
            "timestamp",
            "hour",
            "day_of_week",
            "month",
        }
        and c not in available
    ]
    while len(available) < 16 and extra_cols:
        available.append(extra_cols.pop(0))
    available = available[:16]

    meta_features = df_features[available].values.astype(np.float32)
    meta_features = np.nan_to_num(meta_features, nan=0.0)

    # Align
    min_len = min(len(meta_features), len(regime_labels))
    meta_features = meta_features[:min_len]
    regime_labels = regime_labels[:min_len]

    # Train/val split
    split = int(min_len * TRAIN_RATIO)
    X_tr = torch.FloatTensor(meta_features[:split])
    y_tr = torch.LongTensor(regime_labels[:split])
    X_val = torch.FloatTensor(meta_features[split:])
    y_val = torch.LongTensor(regime_labels[split:])

    logger.info(f"Meta features ({len(available)}): {', '.join(available)}")
    logger.info(f"Train: {len(X_tr)}, Val: {len(X_val)}")

    # Build model
    model = MetaNet(input_dim=len(available))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    t0 = time.time()
    epochs = 200
    batch_size = 64
    best_val_loss = float("inf")
    patience = 20
    patience_counter = 0
    final_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr))
        total_loss = 0.0
        n_batches = 0

        for start in range(0, len(X_tr), batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            out = model(X_tr[idx])
            loss = criterion(out, y_tr[idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_loss = criterion(val_out, y_val).item()
            val_pred = val_out.argmax(dim=1)
            val_acc = (val_pred == y_val).float().mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            final_val_acc = val_acc
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "epoch": epoch,
                    "training_step": epoch + 1,
                    "system_sharpe": 0.0,  # will be updated at runtime
                },
                str(MODEL_DIR / "meta_learner.pt"),
            )
        else:
            patience_counter += 1

        if (epoch + 1) % 25 == 0:
            avg_loss = total_loss / max(n_batches, 1)
            logger.info(
                f"  Epoch {epoch+1:3d}/200: "
                f"train_loss={avg_loss:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

        if patience_counter >= patience:
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break

    elapsed = time.time() - t0
    ckpt = torch.load(str(MODEL_DIR / "meta_learner.pt"), map_location="cpu")
    logger.success(
        f"Meta-learner trained: {elapsed:.0f}s | "
        f"val_loss={ckpt['val_loss']:.4f} | "
        f"val_acc={ckpt['val_acc']:.4f} | "
        f"epochs={ckpt['epoch']+1}"
    )
    return True


if __name__ == "__main__":
    success = train_meta_learner()
    sys.exit(0 if success else 1)
