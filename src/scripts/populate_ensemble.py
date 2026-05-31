#!/usr/bin/env python3
"""Populate the MoEEnsemble with all trained models and save config."""
import json
import sys
import numpy as np
from pathlib import Path

# Run with: pip install -e . && python -m src.scripts.populate_ensemble


from loguru import logger  # noqa: E402
from rts_ai_fx.ensemble import MoEEnsemble  # noqa: E402

MODEL_DIR = Path("models")

# ── Lazy-loading wrappers ──────────────────────────────────────────────


def _align_features(X: np.ndarray, n_expected: int) -> np.ndarray:
    """Truncate or pad the feature dimension of X to match n_expected.

    Handles X shapes (1, lookback, n_features) or (lookback, n_features).
    """
    if X.ndim == 3:
        curr = X.shape[-1]
        if curr > n_expected:
            return X[:, :, :n_expected]
        elif curr < n_expected:
            pad = np.zeros((X.shape[0], X.shape[1], n_expected - curr), dtype=X.dtype)
            return np.concatenate([X, pad], axis=-1)
        return X
    elif X.ndim == 2:
        curr = X.shape[-1]
        if curr > n_expected:
            return X[:, :n_expected]
        elif curr < n_expected:
            pad = np.zeros((X.shape[0], n_expected - curr), dtype=X.dtype)
            return np.concatenate([X, pad], axis=-1)
        return X
    return X


def _make_lstm_predictor(symbol: str):
    """Lazy-load LSTM-CNN model and return predict function."""
    _model = None
    _n_features = None

    def predict(X: np.ndarray) -> float:
        nonlocal _model, _n_features
        if _model is None:
            import tensorflow as tf

            path = str(MODEL_DIR / f"{symbol}_lstm_cnn.keras")
            if not Path(path).exists():
                path = str(MODEL_DIR / f"lstm_cnn_{symbol}.keras")
            _model = tf.keras.models.load_model(path, compile=False)
            _n_features = _model.input_shape[-1]
        X_aligned = _align_features(X, _n_features)
        pred = _model.predict(X_aligned, verbose=0)
        return float(pred[0, 0]) if pred.ndim > 1 else float(pred[0])

    return predict


def _make_classifier_predictor(symbol: str):
    """Lazy-load classifier model."""
    _model = None
    _n_features = None

    def predict(X: np.ndarray) -> float:
        nonlocal _model, _n_features
        if _model is None:
            import tensorflow as tf

            path = str(MODEL_DIR / f"{symbol}_classifier.keras")
            _model = tf.keras.models.load_model(path, compile=False)
            _n_features = _model.input_shape[-1]
        X_aligned = _align_features(X, _n_features)
        prob = _model.predict(X_aligned, verbose=0)
        # Return confidence-scaled direction: (prob - 0.5) * 2
        return (
            float((prob[0, 0] - 0.5) * 2)
            if prob.ndim > 1
            else float((prob[0] - 0.5) * 2)
        )

    return predict


def _make_xgboost_predictor(symbol: str):
    """Lazy-load XGBoost model. Expects flattened input."""
    _model = None
    _n_features_flat = None

    def predict(X: np.ndarray) -> float:
        nonlocal _model, _n_features_flat
        if _model is None:
            import joblib

            _model = joblib.load(str(MODEL_DIR / f"xgboost_{symbol}.pkl"))
            _n_features_flat = _model.n_features_in_
        # Align feature dimension before flattening
        seq_len = X.shape[-2] if X.ndim == 3 else X.shape[-2]
        n_per_step = _n_features_flat // seq_len
        X_aligned = _align_features(X, n_per_step)
        flat = X_aligned.reshape(1, -1)
        return float(_model.predict(flat)[0])

    return predict


def _make_ppo_predictor(regime: str):
    """Lazy-load PPO regime agent.

    Reads the saved checkpoint to determine the correct state_dim,
    then creates and loads the PPOAgent with matching dimensions.
    """
    _agent = None
    _state_dim = None

    def predict(X: np.ndarray) -> float:
        nonlocal _agent, _state_dim
        if _agent is None:
            import torch
            from ai.rl_agent import PPOAgent

            # Read state_dim from saved checkpoint to avoid shape mismatch
            path = str(MODEL_DIR / f"{regime}_agent.pth")
            ckpt = torch.load(path, map_location="cpu")
            first_key = next(k for k in ckpt["actor"] if "weight" in k)
            input_dim = ckpt["actor"][first_key].shape[1]
            _state_dim = input_dim

            _agent = PPOAgent(state_dim=_state_dim, n_actions=5)
            _agent.load(path)

        # Extract features from X and align to expected dimension
        if X.ndim == 3:
            last_features = X[0, -1, :]
        else:
            last_features = X[-1, :]

        n_expected = _state_dim - 1  # features minus the price placeholder
        if last_features.shape[0] > n_expected:
            last_features = last_features[:n_expected]
        elif last_features.shape[0] < n_expected:
            last_features = np.pad(
                last_features, (0, n_expected - last_features.shape[0])
            )

        price_approx = 1.0  # placeholder
        state = np.concatenate([[price_approx], last_features]).astype(np.float32)
        action, sl, tp, size, info = _agent.select_action(state)
        # Map action index to prediction direction
        action_map = {0: -0.02, 1: -0.01, 2: 0.0, 3: 0.01, 4: 0.02}
        return action_map.get(action, 0.0)

    return predict


# ── Default confidence functions ───────────────────────────────────────


def _confidence_high(X):  # noqa: E731
    return 0.7


def _confidence_medium(X):  # noqa: E731
    return 0.55


def _confidence_low(X):  # noqa: E731
    return 0.4


# ── Main ───────────────────────────────────────────────────────────────


def main():  # noqa: C901
    ensemble = MoEEnsemble()
    ensemble.use_sharpe_weighting = True

    # Major forex pairs + XAUUSD (gold) only — per user requirement
    SYMBOLS = [
        "EURUSD",
        "GBPUSD",
        "USDJPY",
        "AUDUSD",
        "USDCAD",
        "USDCHF",
        "NZDUSD",
        "XAUUSD",
    ]

    # 1. Register PPO regime agents (4 experts)
    REGIMES = ["trending", "ranging", "volatile", "crisis"]
    for regime in REGIMES:
        path = MODEL_DIR / f"{regime}_agent.pth"
        if not path.exists():
            logger.warning(f"Missing {path}, skipping {regime} PPO")
            continue
        ensemble.add_expert(
            name=f"ppo_{regime}",
            predict_fn=_make_ppo_predictor(regime),
            confidence_fn=_confidence_high,
            regime=regime,
        )
        ensemble.elo_ratings[f"ppo_{regime}"] = 1400.0
        logger.info(f"Registered ppo_{regime} (Elo=1400, regime={regime})")

    # 2. Register LSTM-CNN models (11 experts)
    for symbol in SYMBOLS:
        path = MODEL_DIR / f"{symbol}_lstm_cnn.keras"
        alt_path = MODEL_DIR / f"lstm_cnn_{symbol}.keras"
        if not path.exists() and not alt_path.exists():
            logger.warning(f"Missing LSTM for {symbol}, skipping")
            continue
        ensemble.add_expert(
            name=f"lstm_{symbol}",
            predict_fn=_make_lstm_predictor(symbol),
            confidence_fn=_confidence_medium,
            regime="trending",
        )
        ensemble.elo_ratings[f"lstm_{symbol}"] = 1300.0
        logger.info(f"Registered lstm_{symbol} (Elo=1300)")

    # 3. Register XGBoost models (11 experts)
    for symbol in SYMBOLS:
        path = MODEL_DIR / f"xgboost_{symbol}.pkl"
        if not path.exists():
            continue
        ensemble.add_expert(
            name=f"xgb_{symbol}",
            predict_fn=_make_xgboost_predictor(symbol),
            confidence_fn=_confidence_medium,
            regime="ranging",
        )
        ensemble.elo_ratings[f"xgb_{symbol}"] = 1250.0
        logger.info(f"Registered xgb_{symbol} (Elo=1250)")

    # 4. Register ProfitabilityClassifiers (11 experts)
    for symbol in SYMBOLS:
        path = MODEL_DIR / f"{symbol}_classifier.keras"
        if not path.exists():
            continue
        ensemble.add_expert(
            name=f"clf_{symbol}",
            predict_fn=_make_classifier_predictor(symbol),
            confidence_fn=_confidence_low,
            regime="ranging",
        )
        ensemble.elo_ratings[f"clf_{symbol}"] = 1150.0
        logger.info(f"Registered clf_{symbol} (Elo=1150)")

    # 5. Inference test
    logger.info(f"\nTotal experts registered: {len(ensemble.experts)}")
    # Models were trained with 45 features (lookback=30)
    dummy_X = np.random.randn(1, 30, 49).astype(np.float32)
    pred = ensemble.predict(dummy_X, regime="ranging")
    logger.info(
        f"Test prediction: price={pred.price:.6f}, conf={pred.confidence:.4f},"
        f" dir={pred.direction}"
    )
    logger.info(f"Expert outputs: {len(pred.expert_outputs)}")
    for name, out in pred.expert_outputs.items():
        logger.info(
            f"  {name}: pred={out['prediction']:.6f}, weight={out['weight']:.4f}"
        )

    # Sync Expert.elo with elo_ratings dict for consistent config output
    for expert in ensemble.experts:
        expert.elo = ensemble.elo_ratings.get(expert.name, 1200.0)

    # 6. Save config
    config = {
        "n_experts": len(ensemble.experts),
        "experts": [
            {"name": e.name, "regime": e.regime, "elo": e.elo} for e in ensemble.experts
        ],
        "elo_ratings": dict(ensemble.elo_ratings),
    }
    config_path = MODEL_DIR / "ensemble_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved ensemble config to {config_path}")
    logger.success("Ensemble populated successfully!")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
