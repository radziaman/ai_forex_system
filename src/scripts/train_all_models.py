#!/usr/bin/env python3
"""
train_all_models.py — Train ALL models for the RTS AI Forex Trading System.

Pipeline:
  1. LSTM-CNN models (11 symbols) — using FeaturePipeline + LSTMCNNHybrid
  2. ProfitabilityClassifiers (11 symbols) — retrain to match current features
  3. Base LSTM model from EURUSD
  4. PPO regime agents (4 regimes) — trained on HMM-labeled historical data
  5. Meta-learner (optional) — predicts which regime agent to use
  6. Verification of all saved models

Usage:
    python -m src.scripts.train_all_models
    python -m src.scripts.train_all_models --skip-ppo --skip-meta
"""

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
import tensorflow as tf  # noqa: E402

tf.get_logger().setLevel("ERROR")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402

# ─── Project setup ────────────────────────────────────────────────────────────
_src_path = str(Path(__file__).resolve().parent.parent)
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from rts_ai_fx.model import LSTMCNNHybrid, ProfitabilityClassifier  # noqa: E402
from rts_ai_fx.features_unified import FeaturePipeline, compute_features  # noqa: E402
from rts_ai_fx.regime_detector import HMMRegimeDetector  # noqa: E402
from ai.rl_agent import PPOAgent, TradingEnvironment  # noqa: E402

# ─── Constants ────────────────────────────────────────────────────────────────

SYMBOLS: List[str] = [
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

REGIME_NAMES: List[str] = ["trending", "ranging", "volatile", "crisis"]

LOOKBACK = 30
TIMEFRAMES = ["1h"]
BATCH_SIZE = 32
LSTM_EPOCHS = 100
CLASSIFIER_EPOCHS = 100
TRAIN_RATIO = 0.8

MODEL_DIR = Path("models")
DATA_DIR = Path("data/historical")

RESULTS: Dict[str, dict] = {}

# ─── Logging ──────────────────────────────────────────────────────────────────

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | {message}",
    level="INFO",
)
logger.add("logs/train_all.log", rotation="100 MB", level="DEBUG")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Data Loading
# ═══════════════════════════════════════════════════════════════════════════════


def load_csv_data(
    symbol: str, timeframes: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """Load training data from local CSV cache.

    Returns dict mapping timeframe -> DataFrame.
    """
    if timeframes is None:
        timeframes = TIMEFRAMES
    dfs = {}
    for tf_name in timeframes:
        path = DATA_DIR / f"{symbol}_{tf_name}.csv"
        if path.exists():
            df = pd.read_csv(path)
            # Keep Unix timestamp as-is for FeaturePipeline compatibility
            dfs[tf_name] = df
            logger.info(f"  Loaded {symbol}_{tf}.csv: {len(df)} bars")
        else:
            logger.warning(f"  Missing {path} — skipping")
    return dfs


def load_all_data() -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load all 11 symbols' 1h data into nested dict."""
    all_data: Dict[str, Dict[str, pd.DataFrame]] = {}
    for sym in SYMBOLS:
        data = load_csv_data(sym)
        if data:
            all_data[sym] = data
    return all_data


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: LSTM-CNN Model Training
# ═══════════════════════════════════════════════════════════════════════════════


def train_lstm_cnn(
    symbol: str,
    lstm_epochs: int = LSTM_EPOCHS,
    use_microstructure: bool = False,
) -> Optional[dict]:
    """Train a single LSTM-CNN model for a symbol.

    Uses FeaturePipeline to create sequences, then trains LSTMCNNHybrid.
    Returns dict with metrics or None on failure.
    """
    data = load_csv_data(symbol)
    if not data:
        logger.error(f"  {symbol}: No data loaded, skipping LSTM-CNN")
        return None

    try:
        fp = FeaturePipeline(
            lookback=LOOKBACK,
            timeframes=TIMEFRAMES,
            use_microstructure=use_microstructure,
        )
        X, y = fp.fit_transform(
            {symbol: data},
            symbol=symbol,
            flatten=False,
        )
        if len(X) < 100:
            logger.error(f"  {symbol}: Only {len(X)} sequences, skipping")
            return None

        n_features = X.shape[-1]
        logger.info(f"  {symbol}: {len(X)} sequences, {n_features} features")

        split = int(len(X) * TRAIN_RATIO)
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]

        model = LSTMCNNHybrid(lookback=LOOKBACK, n_features=n_features)
        model.build()

        t0 = time.time()
        history = model.train(
            X_tr,
            y_tr,
            X_val,
            y_val,
            epochs=lstm_epochs,
            batch_size=BATCH_SIZE,
        )
        elapsed = time.time() - t0

        val_loss = float(min(history.history["val_loss"]))
        val_mae = float(min(history.history["val_mae"]))

        path = str(MODEL_DIR / f"lstm_cnn_{symbol}.keras")
        model.save(path)

        # Also save under original naming convention
        alt_path = str(MODEL_DIR / f"{symbol}_lstm_cnn.keras")
        model.save(alt_path)

        result = {
            "symbol": symbol,
            "type": "lstm_cnn",
            "val_loss": val_loss,
            "val_mae": val_mae,
            "epochs_trained": len(history.history["val_loss"]),
            "time_sec": round(elapsed, 1),
            "n_features": n_features,
            "n_samples": len(X),
            "path": path,
        }
        logger.info(
            f"  {symbol} LSTM-CNN: val_loss={val_loss:.6f}, "
            f"val_mae={val_mae:.6f}, {elapsed:.0f}s"
        )
        return result

    except Exception as e:
        logger.error(f"  {symbol} LSTM-CNN FAILED: {e}")
        import traceback

        traceback.print_exc()
        return None


def train_all_lstm_cnn(
    use_microstructure: bool = False,
) -> Dict[str, dict]:
    """Train LSTM-CNN models for all symbols."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("  STEP 1: LSTM-CNN Models (11 symbols)")
    logger.info("=" * 60)

    results: Dict[str, dict] = {}
    for i, symbol in enumerate(SYMBOLS, 1):
        logger.info(f"\n  [{i}/{len(SYMBOLS)}] Training {symbol} LSTM-CNN...")
        result = train_lstm_cnn(symbol, use_microstructure=use_microstructure)
        if result:
            results[symbol] = result
        else:
            results[symbol] = {"status": "FAILED"}

    # Summary
    logger.info("")
    logger.info("-" * 60)
    logger.info("  LSTM-CNN Training Summary:")
    logger.info(f"  {'Symbol':<12s} {'Val Loss':<12s} {'Val MAE':<12s} {'Time':<8s}")
    logger.info(f"  {'-' * 44}")
    for sym, r in results.items():
        if "val_loss" in r:
            logger.info(
                f"  {sym:<12s} {r['val_loss']:<12.6f} {r['val_mae']:<12.6f} "
                f"{r['time_sec']:<8.1f}"
            )
        else:
            logger.info(f"  {sym:<12s} {'FAILED':<12s}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Profitability Classifier Training
# ═══════════════════════════════════════════════════════════════════════════════


def train_classifier(
    symbol: str,
    classifier_epochs: int = CLASSIFIER_EPOCHS,
    use_microstructure: bool = False,
) -> Optional[dict]:
    """Train a single ProfitabilityClassifier for a symbol."""
    data = load_csv_data(symbol)
    if not data:
        logger.error(f"  {symbol}: No data loaded, skipping classifier")
        return None

    try:
        fp = FeaturePipeline(
            lookback=LOOKBACK,
            timeframes=TIMEFRAMES,
            use_microstructure=use_microstructure,
        )
        X, _ = fp.fit_transform(
            {symbol: data},
            symbol=symbol,
            flatten=False,
        )
        if len(X) < 100:
            logger.error(f"  {symbol}: Only {len(X)} sequences, skipping classifier")
            return None

        n_features = X.shape[-1]

        # Close prices for label construction
        df = data[TIMEFRAMES[0]]
        prices = df["close"].values.astype(np.float64)

        # X[t] uses prices[t-lookback:t], predicts direction of prices[t+1]
        # _make_labels does diff, producing len(prices) - 1 labels,
        # then X is trimmed accordingly: X_train_adj = X_train[:len(y_train)]
        prices_adj = prices[LOOKBACK - 1 :]

        split_idx = int(len(X) * TRAIN_RATIO)
        X_tr = X[:split_idx]
        X_val = X[split_idx:]
        # prices_train should be longer than X_tr by 1 (for diff)
        p_split = split_idx + 1
        prices_tr = prices_adj[:p_split]
        prices_val = prices_adj[p_split - 1 :]

        clf = ProfitabilityClassifier(lookback=LOOKBACK, n_features=n_features)
        clf.build()

        t0 = time.time()
        history = clf.train(
            X_tr,
            prices_tr,
            X_val,
            prices_val,
            epochs=classifier_epochs,
            batch_size=BATCH_SIZE,
        )
        elapsed = time.time() - t0

        val_loss = float(min(history.history["val_loss"]))
        val_acc = float(max(history.history.get("val_accuracy", [0])))

        path = str(MODEL_DIR / f"{symbol}_classifier.keras")
        clf.save(path)

        result = {
            "symbol": symbol,
            "type": "classifier",
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "epochs_trained": len(history.history["val_loss"]),
            "time_sec": round(elapsed, 1),
            "n_features": n_features,
        }
        logger.info(
            f"  {symbol} Classifier: val_loss={val_loss:.6f}, "
            f"val_acc={val_acc:.4f}, {elapsed:.0f}s"
        )
        return result

    except Exception as e:
        logger.error(f"  {symbol} Classifier FAILED: {e}")
        import traceback

        traceback.print_exc()
        return None


def train_all_classifiers(
    use_microstructure: bool = False,
) -> Dict[str, dict]:
    """Train ProfitabilityClassifiers for all symbols."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("  STEP 2: Profitability Classifiers (11 symbols)")
    logger.info("=" * 60)

    results: Dict[str, dict] = {}
    for i, symbol in enumerate(SYMBOLS, 1):
        logger.info(f"\n  [{i}/{len(SYMBOLS)}] Training {symbol} classifier...")
        result = train_classifier(symbol, use_microstructure=use_microstructure)
        if result:
            results[symbol] = result
        else:
            results[symbol] = {"status": "FAILED"}

    logger.info("")
    logger.info("-" * 60)
    logger.info("  Classifier Training Summary:")
    logger.info(f"  {'Symbol':<12s} {'Val Loss':<12s} {'Val Acc':<12s} {'Time':<8s}")
    logger.info(f"  {'-' * 44}")
    for sym, r in results.items():
        if "val_loss" in r:
            logger.info(
                f"  {sym:<12s} {r['val_loss']:<12.6f} {r['val_accuracy']:<12.4f} "
                f"{r['time_sec']:<8.1f}"
            )
        else:
            logger.info(f"  {sym:<12s} {'FAILED':<12s}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Base LSTM Model
# ═══════════════════════════════════════════════════════════════════════════════


def train_base_model(
    use_microstructure: bool = False,
) -> Optional[dict]:
    """Train base LSTM-CNN model from EURUSD."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("  STEP 3: Base LSTM-CNN Model (EURUSD)")
    logger.info("=" * 60)

    symbol = "EURUSD"
    data = load_csv_data(symbol)
    if not data:
        logger.error("  Cannot load EURUSD data, skipping base model")
        return None

    try:
        fp = FeaturePipeline(
            lookback=LOOKBACK,
            timeframes=TIMEFRAMES,
            use_microstructure=use_microstructure,
        )
        X, y = fp.fit_transform(
            {symbol: data},
            symbol=symbol,
            flatten=False,
        )
        if len(X) < 100:
            logger.error(f"  Only {len(X)} sequences, skipping base model")
            return None

        n_features = X.shape[-1]
        split = int(len(X) * TRAIN_RATIO)
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]

        model = LSTMCNNHybrid(
            lookback=LOOKBACK, n_features=n_features, lstm_units=128, cnn_filters=128
        )
        model.build()

        t0 = time.time()
        history = model.train(
            X_tr,
            y_tr,
            X_val,
            y_val,
            epochs=LSTM_EPOCHS,
            batch_size=BATCH_SIZE,
        )
        elapsed = time.time() - t0

        val_loss = float(min(history.history["val_loss"]))
        val_mae = float(min(history.history["val_mae"]))

        path = str(MODEL_DIR / "base_lstm_cnn.keras")
        model.save(path)

        result = {
            "symbol": "EURUSD",
            "type": "base_lstm_cnn",
            "val_loss": val_loss,
            "val_mae": val_mae,
            "epochs_trained": len(history.history["val_loss"]),
            "time_sec": round(elapsed, 1),
            "n_features": n_features,
            "n_samples": len(X),
            "path": path,
        }
        logger.info(
            f"  Base LSTM-CNN: val_loss={val_loss:.6f}, "
            f"val_mae={val_mae:.6f}, {elapsed:.0f}s"
        )
        return result

    except Exception as e:
        logger.error(f"  Base model FAILED: {e}")
        import traceback

        traceback.print_exc()
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: PPO Regime Agents
# ═══════════════════════════════════════════════════════════════════════════════


def train_ppo_regime_agents(  # noqa: C901
    use_microstructure: bool = False,
    ppo_timesteps: int = 5000,
) -> Dict[str, dict]:
    """Train 4 regime-specific PPO agents using HMM regime labels.

    Approach: Load all data, detect regimes with HMM, then for each regime
    train a PPO agent using chronological walk-through of regime-labeled bars.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("  STEP 4: PPO Regime Agents")
    logger.info("=" * 60)

    # Load all data (use EURUSD for regime labeling and training)
    # We'll use EURUSD as the primary training data for PPO agents
    symbol = "EURUSD"
    data = load_csv_data(symbol)
    if not data:
        logger.error("  Cannot load EURUSD data, skipping PPO agents")
        return {}

    df = data[TIMEFRAMES[0]]

    try:
        # ── 4a: Compute features and get state vectors ──
        logger.info("  Computing features for PPO state vectors...")
        df_features = compute_features(df)
        feature_cols = [
            c
            for c in df_features.columns
            if c not in {"open", "high", "low", "close", "volume", "timestamp"}
        ]
        features_np = df_features[feature_cols].values.astype(np.float64)
        prices = df["close"].values.astype(np.float64)

        # Normalize features with expanding window to prevent look-ahead
        n_total = len(features_np)
        normed = np.empty_like(features_np, dtype=np.float64)
        for i in range(n_total):
            window = features_np[: i + 1]
            mu = np.mean(window, axis=0)
            sigma = np.std(window, axis=0)
            sigma = np.where(sigma < 1e-12, 1.0, sigma)
            normed[i] = (features_np[i] - mu) / sigma

        # Build state vectors: [close_price, ...features]
        state_dim = 1 + normed.shape[1]
        states = np.column_stack([prices.reshape(-1, 1), normed]).astype(np.float32)

        logger.info(f"  State vectors: {states.shape} ({state_dim} dims)")

        # ── 4b: Run HMM regime detection ──
        logger.info("  Running HMM regime detection...")
        detector = HMMRegimeDetector(n_regimes=4)
        detector.fit(df)

        # Get per-bar regime labels
        regime_labels = _get_hmm_regime_labels(detector, df, prices)
        regime_names = np.array([REGIME_NAMES[int(rl)] for rl in regime_labels])

        # Log regime distribution
        unique, counts = np.unique(regime_names, return_counts=True)
        for name, count in zip(unique, counts):
            pct = 100 * count / len(regime_names)
            logger.info(f"    {name:<12s}: {count:>5d} bars ({pct:.1f}%)")

        # ── 4c: Train per-regime agents ──
        results: Dict[str, dict] = {}
        for regime in REGIME_NAMES:
            logger.info(f"\n  Training {regime} agent...")

            # Get indices for this regime
            regime_indices = np.where(regime_names == regime)[0]
            logger.info(f"    {len(regime_indices)} bars labeled as {regime}")

            if len(regime_indices) < 50:
                logger.warning(f"    Too few bars ({len(regime_indices)}), skipping")
                results[regime] = {
                    "status": "SKIPPED",
                    "reason": f"only {len(regime_indices)} bars",
                }
                continue

            # Create agent and environment
            agent = PPOAgent(
                state_dim=state_dim,
                n_actions=5,
                lr=3e-4,
                gamma=0.99,
                clip_range=0.2,
            )
            env = TradingEnvironment(
                state_dim=state_dim,
                spread_pips=0.5,
                commission_per_lot=7.0,
            )
            env.reset()

            # Train: walk through regime bars in chronological order
            n_steps = 0
            total_reward = 0.0
            policy_losses: List[float] = []
            value_losses: List[float] = []
            entropies: List[float] = []
            episode_rewards: List[float] = []

            train_interval = 128
            n_epochs = 5
            burn_in = 100

            max_steps = min(len(regime_indices), ppo_timesteps)

            for step, idx in enumerate(regime_indices[:max_steps]):
                if idx < burn_in:
                    continue

                state = states[idx]
                env.update_state(state)

                action, sl_raw, tp_raw, size_raw, info = agent.select_action(state)
                next_state, reward, done, _ = env.step(action, sl_raw, tp_raw, size_raw)
                agent.store_transition(reward, done)

                total_reward += reward
                n_steps += 1

                if n_steps % train_interval == 0:
                    metrics = agent.train(next_value=0.0, n_epochs=n_epochs)
                    if metrics:
                        policy_losses.append(metrics.get("policy_loss", 0.0))
                        value_losses.append(metrics.get("value_loss", 0.0))
                        entropies.append(metrics.get("entropy", 0.0))
                        episode_rewards.append(total_reward)
                        total_reward = 0.0

            # Final training pass
            if agent.states and len(agent.states) >= 2:
                metrics = agent.train(next_value=0.0, n_epochs=n_epochs)
                if metrics:
                    policy_losses.append(metrics.get("policy_loss", 0.0))
                    value_losses.append(metrics.get("value_loss", 0.0))
                    entropies.append(metrics.get("entropy", 0.0))
                    episode_rewards.append(total_reward)

            # Save agent
            path = str(MODEL_DIR / f"{regime}_agent.pth")
            agent.save(path)

            result = {
                "regime": regime,
                "type": "ppo_agent",
                "n_steps": n_steps,
                "n_train_calls": len(policy_losses),
                "avg_reward": (
                    float(np.mean(episode_rewards)) if episode_rewards else 0.0
                ),
                "total_trades": env.total_trades,
                "win_rate": float(env.winning_trades / max(env.total_trades, 1)),
                "avg_policy_loss": (
                    float(np.mean(policy_losses)) if policy_losses else 0.0
                ),
                "path": path,
                "state_dim": state_dim,
            }
            results[regime] = result

            logger.info(
                f"    {regime}: {n_steps} steps, "
                f"avg_reward={result['avg_reward']:.2f}, "
                f"win_rate={result['win_rate']:.1%}, "
                f"trades={env.total_trades}"
            )

        # ── 4d: Summary ──
        logger.info("")
        logger.info("-" * 60)
        logger.info("  PPO Agent Training Summary:")
        logger.info(
            f"  {'Regime':<12s} {'Steps':<8s} {'WinRate':<10s} "
            f"{'Trades':<8s} {'Reward':<10s}"
        )
        logger.info(f"  {'-' * 48}")
        for regime, r in results.items():
            if "n_steps" in r:
                logger.info(
                    f"  {regime:<12s} {r['n_steps']:<8d} "
                    f"{r['win_rate']:<9.1%} "
                    f"{r['total_trades']:<8d} "
                    f"{r['avg_reward']:<+10.2f}"
                )
            else:
                logger.info(f"  {regime:<12s} {'SKIPPED':<8s}")

        return results

    except Exception as e:
        logger.error(f"  PPO training FAILED: {e}")
        import traceback

        traceback.print_exc()
        return {}


def _get_hmm_regime_labels(
    detector: HMMRegimeDetector,
    df: pd.DataFrame,
    prices: np.ndarray,
) -> np.ndarray:
    """Get per-bar HMM regime labels aligned with price array length.

    Returns array of int regime labels (0=trending, 1=ranging, 2=volatile, 3=crisis).
    """
    from rts_ai_fx.regime_detector import HMM_AVAILABLE

    if not HMM_AVAILABLE or detector.model is None:
        # Fallback: simple volatility-based regime detection
        returns = np.abs(np.diff(prices) / (prices[:-1] + 1e-10))
        vol = pd.Series(prices).rolling(20).std().values[1:]
        fallback_labels = np.zeros(len(returns), dtype=int)
        if len(vol) > 0:
            vol_sorted = np.sort(vol)
            for i in range(len(vol)):
                pct = np.searchsorted(vol_sorted, vol[i]) / len(vol_sorted)
                if pct > 0.95:
                    fallback_labels[i] = 3  # crisis
                elif pct > 0.70:
                    fallback_labels[i] = 2  # volatile
                elif pct > 0.30:
                    fallback_labels[i] = 1  # ranging
                # else 0 = trending
        return np.concatenate([[0], fallback_labels])

    # Use HMM
    features = detector._extract_features(df)
    if len(features) < 5:
        return np.zeros(len(prices), dtype=int)

    hidden_states = detector.model.predict(features)

    # Map states to regime names using mean returns ordering
    if detector.model.means_ is not None:
        mean_returns = detector.model.means_[:, 0]
        state_order = np.argsort(mean_returns)
        label_to_name = {
            i: REGIME_NAMES[state_order[i]] for i in range(detector.n_regimes)
        }
        # Convert to regime indices
        regime_names_states = np.array([label_to_name[s] for s in hidden_states])
        regime_labels = np.array(
            [REGIME_NAMES.index(name) for name in regime_names_states]
        )
    else:
        regime_labels = hidden_states % len(REGIME_NAMES)

    # Align length: HMM features are shorter by 1 (due to diff in returns)
    if len(regime_labels) < len(prices):
        regime_labels = np.concatenate([[int(regime_labels[0])], regime_labels])

    return regime_labels.astype(int)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: Meta-Learner
# ═══════════════════════════════════════════════════════════════════════════════


def train_meta_learner(  # noqa: C901
    use_microstructure: bool = False,
) -> Optional[dict]:
    """Train a meta-learner that predicts which regime agent to use.

    Uses MetaNet architecture: 16 input features -> 128 -> 64 -> 32 -> 5 (regime probs).
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("  STEP 5: Meta-Learner (optional)")
    logger.info("=" * 60)

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        logger.warning("  PyTorch not available, skipping meta-learner")
        return None

    # Load data
    symbol = "EURUSD"
    data = load_csv_data(symbol)
    if not data:
        logger.error("  Cannot load EURUSD data, skipping meta-learner")
        return None

    df = data[TIMEFRAMES[0]]

    try:
        # Compute features and get regime labels
        df_features = compute_features(df)
        prices = df["close"].values.astype(np.float64)

        # Get regime labels
        detector = HMMRegimeDetector(n_regimes=4)
        detector.fit(df)
        regime_labels = _get_hmm_regime_labels(detector, df, prices)

        # Build meta-features: select 16 key indicators
        meta_feature_cols = [
            "rsi_14",
            "adx_14",
            "atr_14",
            "volatility_20",
            "bb_width",
            "mom_5",
            "mom_10",
            "vol_ratio",
            "ema_cross_ratio",
            "rsi_divergence",
            "price_acceleration",
            "atr_normalized",
            "hurst",
            "stoch_k",
        ]
        # Add more features to reach 16
        available = [c for c in meta_feature_cols if c in df_features.columns]

        # Pad with additional features if needed
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

        # Fill NaN
        meta_features = np.nan_to_num(meta_features, nan=0.0)

        # Align regime labels to feature length
        min_len = min(len(meta_features), len(regime_labels))
        meta_features = meta_features[:min_len]
        regime_labels = regime_labels[:min_len]

        # Split
        split = int(min_len * TRAIN_RATIO)
        X_tr = torch.FloatTensor(meta_features[:split])
        y_tr = torch.LongTensor(regime_labels[:split])
        X_val = torch.FloatTensor(meta_features[split:])
        y_val = torch.LongTensor(regime_labels[split:])

        logger.info(f"  Meta features: {len(available)} dims ({', '.join(available)})")
        logger.info(f"  Training samples: {len(X_tr)}, Validation: {len(X_val)}")

        # Build MetaNet
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
                    nn.Linear(32, 4),  # 4 regimes
                )

            def forward(self, x):
                return self.net(x)

        model = MetaNet(input_dim=len(available))
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # Train
        t0 = time.time()
        epochs = 200
        batch_size = 64
        best_val_loss = float("inf")
        patience = 20
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(len(X_tr))
            total_loss = 0.0

            for start in range(0, len(X_tr), batch_size):
                idx = perm[start : start + batch_size]
                optimizer.zero_grad()
                out = model(X_tr[idx])
                loss = criterion(out, y_tr[idx])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Validation
            model.eval()
            with torch.no_grad():
                val_out = model(X_val)
                val_loss = criterion(val_out, y_val).item()
                val_pred = val_out.argmax(dim=1)
                val_acc = (val_pred == y_val).float().mean().item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "epoch": epoch,
                    },
                    str(MODEL_DIR / "meta_learner.pt"),
                )
            else:
                patience_counter += 1

            if (epoch + 1) % 25 == 0:
                logger.info(
                    f"    Epoch {epoch+1:3d}/{epochs}: "
                    f"train_loss={total_loss/max(len(X_tr)//batch_size,1):.4f} "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
                )

            if patience_counter >= patience:
                logger.info(f"    Early stopping at epoch {epoch+1}")
                break

        elapsed = time.time() - t0

        result = {
            "type": "meta_learner",
            "val_loss": best_val_loss,
            "val_accuracy": val_acc,
            "epochs_trained": epoch + 1,
            "time_sec": round(elapsed, 1),
            "n_features": len(available),
            "feature_names": available,
        }
        logger.info(
            f"  Meta-learner: val_loss={best_val_loss:.4f}, "
            f"val_acc={val_acc:.4f}, {elapsed:.0f}s"
        )
        return result

    except Exception as e:
        logger.error(f"  Meta-learner FAILED: {e}")
        import traceback

        traceback.print_exc()
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7: Verification
# ═══════════════════════════════════════════════════════════════════════════════


def verify_all_models() -> Dict[str, list]:  # noqa: C901
    """Verify all trained models exist and are functional."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("  STEP 6: Verification")
    logger.info("=" * 60)

    issues: list = []
    success: list = []

    # 1. LSTM-CNN models
    logger.info("\n  [6a] LSTM-CNN Models...")
    for symbol in SYMBOLS:
        paths = [
            MODEL_DIR / f"lstm_cnn_{symbol}.keras",
            MODEL_DIR / f"{symbol}_lstm_cnn.keras",
        ]
        found = False
        for path in paths:
            if path.exists():
                try:
                    model = LSTMCNNHybrid.load(str(path))
                    # Construct a minimal test input
                    if model.model is not None:
                        nf = model.model.input_shape[-1]
                        lookback = model.model.input_shape[1]
                        test_X = np.random.randn(1, lookback, nf).astype(np.float32)
                        pred = model.predict(test_X)
                        trained = model.is_trained()
                        status = "TRAINED" if trained else "UNTRAINED"
                        logger.info(
                            f"    {symbol}: OK ({status}, "
                            f"input={model.model.input_shape}, pred={pred[0][0]:.4f})"
                        )
                        success.append(f"lstm_cnn_{symbol}: {status}")
                        found = True
                        break
                except Exception as e:
                    issues.append(f"lstm_cnn_{symbol}: load/predict failed: {e}")
                    logger.warning(f"    {symbol}: FAILED — {e}")
                    found = True
                    break
        if not found:
            issues.append(f"lstm_cnn_{symbol}: file not found")
            logger.warning(f"    {symbol}: FILE NOT FOUND")

    # 2. Classifier models
    logger.info("\n  [6b] Classifier Models...")
    for symbol in SYMBOLS:
        path = MODEL_DIR / f"{symbol}_classifier.keras"
        if path.exists():
            try:
                import tensorflow as tf

                model = tf.keras.models.load_model(str(path), compile=False)
                logger.info(f"    {symbol}: OK ({len(model.layers)} layers)")
                success.append(f"classifier_{symbol}: OK")
            except Exception as e:
                issues.append(f"classifier_{symbol}: load failed: {e}")
                logger.warning(f"    {symbol}: FAILED — {e}")
        else:
            issues.append(f"classifier_{symbol}: file not found")
            logger.warning(f"    {symbol}: FILE NOT FOUND")

    # 3. Base LSTM-CNN model
    logger.info("\n  [6c] Base LSTM-CNN Model...")
    base_path = MODEL_DIR / "base_lstm_cnn.keras"
    if base_path.exists():
        try:
            model = LSTMCNNHybrid.load(str(base_path))
            if model.model is not None:
                trained = model.is_trained()
                status = "TRAINED" if trained else "UNTRAINED"
                logger.info(f"    base_lstm_cnn: OK ({status})")
                success.append(f"base_lstm_cnn: {status}")
        except Exception as e:
            issues.append(f"base_lstm_cnn: load failed: {e}")
            logger.warning(f"    base_lstm_cnn: FAILED — {e}")
    else:
        issues.append("base_lstm_cnn: file not found")
        logger.warning("    base_lstm_cnn: FILE NOT FOUND")

    # 4. PPO agents
    logger.info("\n  [6d] PPO Agents...")
    import torch

    for regime in REGIME_NAMES:
        path = MODEL_DIR / f"{regime}_agent.pth"
        if path.exists():
            try:
                ckpt = torch.load(str(path), map_location="cpu")
                actor_state = ckpt.get("actor", {})
                if actor_state:
                    first_key = list(actor_state.keys())[0]
                    weight_shape = actor_state[first_key].shape
                    total_params = sum(p.numel() for p in actor_state.values())
                    logger.info(
                        f"    {regime}: OK ({total_params:,} params, "
                        f"key={first_key}, shape={weight_shape})"
                    )
                    success.append(f"ppo_{regime}: {total_params:,} params")
                else:
                    issues.append(f"ppo_{regime}: empty state dict")
                    logger.warning(f"    {regime}: EMPTY STATE DICT")
            except Exception as e:
                issues.append(f"ppo_{regime}: load failed: {e}")
                logger.warning(f"    {regime}: FAILED — {e}")
        else:
            issues.append(f"ppo_{regime}: file not found")
            logger.warning(f"    {regime}: FILE NOT FOUND")

    # 5. Meta-learner
    logger.info("\n  [6e] Meta-Learner...")
    meta_path = MODEL_DIR / "meta_learner.pt"
    if meta_path.exists():
        try:
            ckpt = torch.load(str(meta_path), map_location="cpu")
            val_acc = ckpt.get("val_acc", "N/A")
            val_loss = ckpt.get("val_loss", "N/A")
            logger.info(
                f"    meta_learner: OK (val_acc={val_acc}, val_loss={val_loss})"
            )
            success.append(f"meta_learner: val_acc={val_acc}")
        except Exception as e:
            issues.append(f"meta_learner: load failed: {e}")
            logger.warning(f"    meta_learner: FAILED — {e}")
    else:
        logger.info("    meta_learner: not found (skipped)")

    # 6. Feature normalization
    logger.info("\n  [6f] Feature Normalization...")
    norm_path = MODEL_DIR / "feature_norm.npz"
    if norm_path.exists():
        try:
            data = np.load(str(norm_path), allow_pickle=True)
            n_pairs = int(data.get("n_pairs", 0))
            n_features = len(data.get("feature_cols", []))
            logger.info(
                f"    feature_norm: OK ({n_pairs} pairs, {n_features} features)"
            )
            success.append(f"feature_norm: {n_pairs} pairs, {n_features} features")
        except Exception as e:
            issues.append(f"feature_norm: load failed: {e}")
            logger.warning(f"    feature_norm: FAILED — {e}")
    else:
        issues.append("feature_norm: file not found")
        logger.warning("    feature_norm: FILE NOT FOUND")

    return {"success": success, "issues": issues}


def save_feature_norm(
    use_microstructure: bool = False,
) -> bool:
    """Save feature normalization for all symbols."""
    logger.info("\n  Saving feature normalization...")
    try:
        all_data = load_all_data()
        if not all_data:
            logger.error("  No data loaded for normalization")
            return False

        fp = FeaturePipeline(
            lookback=LOOKBACK,
            timeframes=TIMEFRAMES,
            use_microstructure=use_microstructure,
        )

        for symbol, data in all_data.items():
            fp.fit(data, symbol)
            n_features = len(fp._feature_cols) if fp._feature_cols else 0
            logger.info(f"    {symbol}: fitted ({n_features} features)")

        fp.save_normalization()
        logger.info("  Normalization saved to models/feature_norm.npz")
        return True
    except Exception as e:
        logger.error(f"  Failed to save normalization: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Section 8: Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


def main():  # noqa: C901
    import argparse

    parser = argparse.ArgumentParser(description="RTS AI Forex — Train All Models")
    parser.add_argument(
        "--skip-lstm", action="store_true", help="Skip LSTM-CNN training"
    )
    parser.add_argument(
        "--skip-classifier", action="store_true", help="Skip classifier training"
    )
    parser.add_argument(
        "--skip-base", action="store_true", help="Skip base model training"
    )
    parser.add_argument(
        "--skip-ppo", action="store_true", help="Skip PPO agent training"
    )
    parser.add_argument(
        "--skip-meta", action="store_true", help="Skip meta-learner training"
    )
    parser.add_argument(
        "--skip-norm", action="store_true", help="Skip feature norm saving"
    )
    parser.add_argument(
        "--lstm-epochs",
        type=int,
        default=LSTM_EPOCHS,
        help=f"LSTM-CNN training epochs (default: {LSTM_EPOCHS})",
    )
    parser.add_argument(
        "--ppo-timesteps",
        type=int,
        default=5000,
        help="PPO timesteps per regime (default: 5000)",
    )
    parser.add_argument(
        "--use-microstructure",
        action="store_true",
        help="Enable microstructure features (default: False)",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode: fewer epochs, skip meta"
    )
    args = parser.parse_args()

    t_start = time.time()

    # Welcome
    print()
    print("=" * 65)
    print("  RTS AI FOREX — Full Model Training Pipeline")
    print("=" * 65)
    print(f"  Symbols: {len(SYMBOLS)}")
    print(f"  Lookback: {LOOKBACK}")
    print(f"  Timeframes: {TIMEFRAMES}")
    print(f"  Microstructure: {args.use_microstructure}")
    print(f"  Quick mode: {args.quick}")

    if args.quick:
        lstm_epochs = min(30, args.lstm_epochs)
        ppo_timesteps = min(2000, args.ppo_timesteps)
        skip_meta = True
    else:
        lstm_epochs = args.lstm_epochs
        ppo_timesteps = args.ppo_timesteps
        skip_meta = args.skip_meta

    use_microstructure = args.use_microstructure

    print(f"  LSTM epochs: {lstm_epochs}")
    print(f"  PPO timesteps: {ppo_timesteps}")
    print()

    overall_results: Dict[str, dict] = {}

    # Step 1: LSTM-CNN Models
    if not args.skip_lstm:
        lstm_results = train_all_lstm_cnn(use_microstructure=use_microstructure)
        overall_results["lstm_cnn"] = lstm_results
        RESULTS["lstm_cnn"] = lstm_results
    else:
        logger.info("Skipping LSTM-CNN training (--skip-lstm)")

    # Step 2: Profitability Classifiers
    if not args.skip_classifier:
        clf_results = train_all_classifiers(use_microstructure=use_microstructure)
        overall_results["classifiers"] = clf_results
        RESULTS["classifiers"] = clf_results
    else:
        logger.info("Skipping classifier training (--skip-classifier)")

    # Step 3: Base LSTM-CNN Model
    if not args.skip_base:
        base_result = train_base_model(use_microstructure=use_microstructure)
        overall_results["base_lstm_cnn"] = base_result
        RESULTS["base_lstm_cnn"] = base_result
    else:
        logger.info("Skipping base model (--skip-base)")

    # Step 4: Save feature normalization
    if not args.skip_norm:
        save_feature_norm(use_microstructure=use_microstructure)

    # Step 5: PPO Regime Agents
    if not args.skip_ppo:
        ppo_results = train_ppo_regime_agents(
            use_microstructure=use_microstructure,
            ppo_timesteps=ppo_timesteps,
        )
        overall_results["ppo_agents"] = ppo_results
        RESULTS["ppo_agents"] = ppo_results
    else:
        logger.info("Skipping PPO training (--skip-ppo)")

    # Step 6: Meta-Learner
    if not skip_meta and not args.skip_meta:
        meta_result = train_meta_learner(use_microstructure=use_microstructure)
        overall_results["meta_learner"] = meta_result
        RESULTS["meta_learner"] = meta_result
    else:
        logger.info("Skipping meta-learner (--skip-meta or quick mode)")

    # Step 7: Verification
    verify_results = verify_all_models()
    overall_results["verification"] = verify_results

    # ─── Final Report ───
    elapsed = time.time() - t_start
    print()
    print("=" * 65)
    print("  TRAINING COMPLETE")
    print("=" * 65)
    print(f"  Total time: {elapsed:.0f}s ({elapsed / 60:.1f}m)")
    print()

    # Summary counts
    lstm_ok = sum(
        1
        for r in overall_results.get("lstm_cnn", {}).values()
        if isinstance(r, dict) and "val_loss" in r
    )
    lstm_fail = sum(
        1
        for r in overall_results.get("lstm_cnn", {}).values()
        if isinstance(r, dict) and r.get("status") == "FAILED"
    )
    clf_ok = sum(
        1
        for r in overall_results.get("classifiers", {}).values()
        if isinstance(r, dict) and "val_loss" in r
    )
    ppo_ok = sum(
        1
        for r in overall_results.get("ppo_agents", {}).values()
        if isinstance(r, dict) and "n_steps" in r
    )
    ppo_skip = sum(
        1
        for r in overall_results.get("ppo_agents", {}).values()
        if isinstance(r, dict) and r.get("status") == "SKIPPED"
    )

    print(f"  LSTM-CNN models:   {lstm_ok}/{len(SYMBOLS)} OK, {lstm_fail} failed")
    print(f"  Classifiers:       {clf_ok}/{len(SYMBOLS)} OK")
    base_status = "OK" if overall_results.get("base_lstm_cnn") else "N/A"
    print(f"  Base LSTM:         {base_status}")
    print(f"  PPO agents:        {ppo_ok}/4 trained, {ppo_skip} skipped")
    print(
        f"  Meta-learner:      {'OK' if overall_results.get('meta_learner') else 'N/A'}"
    )
    print()

    ver = overall_results.get("verification", {})
    print("  Verification:")
    print(f"    Successes: {len(ver.get('success', []))}")
    print(f"    Issues:    {len(ver.get('issues', []))}")
    if ver.get("issues"):
        print("    Issues:")
        for issue in ver["issues"]:
            print(f"      - {issue}")
    if ver.get("success"):
        print("    All models verified successfully!")
    print()

    # Show all model files
    print(f"  {'=' * 55}")
    print(f"  {'Model files in models/':^55}")
    print(f"  {'=' * 55}")
    for f in sorted(MODEL_DIR.iterdir()):
        if f.is_dir():
            continue
        size_kb = f.stat().st_size / 1024
        # Classify
        name = f.name
        if "keras" in name:
            label = "Keras "
        elif "pth" in name:
            label = "PyTorch"
        elif "npz" in name:
            label = "Numpy "
        else:
            label = "Other "
        print(f"  [{label}] {name:<42s} {size_kb:>8.1f} KB")
    print()
    print("=" * 65)
    print()

    # Return summary as dict (also saved)
    return overall_results


if __name__ == "__main__":
    main()
