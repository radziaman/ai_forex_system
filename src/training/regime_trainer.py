"""
Regime-Dependent Training Pipeline.
Trains separate expert models per HMM regime for specialized expertise.
"""
import numpy as np
import pandas as pd
import os
import json
import pickle
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from loguru import logger

from rts_ai_fx.regime_detector import HMMRegimeDetector
from rts_ai_fx.model import LSTMCNNHybrid, ProfitabilityClassifier
from rts_ai_fx.features_unified import FeaturePipeline


@dataclass
class RegimeModelBundle:
    regime: str
    lstm_cnn: Optional[LSTMCNNHybrid] = None
    classifier: Optional[ProfitabilityClassifier] = None
    feature_pipeline: Optional[FeaturePipeline] = None
    train_metrics: Dict = field(default_factory=lambda: {
        "mse": 0.0, "accuracy": 0.0, "n_samples": 0,
    })
    creation_timestamp: float = 0.0


class RegimeTrainer:
    def __init__(
        self,
        n_regimes: int = 4,
        lookback: int = 30,
        model_dir: str = "models/regime_models",
        min_samples: int = 200,
    ):
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.model_dir = model_dir
        self.min_samples = min_samples
        self.regime_detector = HMMRegimeDetector(n_regimes=n_regimes)
        self.models: Dict[str, RegimeModelBundle] = {}

        os.makedirs(model_dir, exist_ok=True)

        for regime in HMMRegimeDetector.REGIME_NAMES[:n_regimes]:
            self.models[regime] = RegimeModelBundle(regime=regime)

    def assign_regime_labels(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        self.regime_detector.fit(df)
        features = self.regime_detector._extract_features(df)
        if self.regime_detector.model is not None:
            states = self.regime_detector.model.predict(features)
            mean_returns = self.regime_detector.model.means_[:, 0]
            state_order = np.argsort(mean_returns)
            regime_map = {s: HMMRegimeDetector.REGIME_NAMES[i] for i, s in enumerate(state_order)}
            regimes = np.array([regime_map.get(s, "ranging") for s in states])
        else:
            regimes = np.array(["ranging"] * len(features))
        return df.iloc[-len(regimes):].copy(), regimes

    def train_regime_models(
        self,
        dfs: Dict[str, pd.DataFrame],
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
    ) -> Dict[str, Dict]:
        primary_tf = list(dfs.keys())[0]
        df = dfs[primary_tf]
        df_labeled, regimes = self.assign_regime_labels(df)
        results = {}

        for regime in HMMRegimeDetector.REGIME_NAMES[:self.n_regimes]:
            mask = regimes == regime
            n_found = int(np.sum(mask))
            if n_found < self.min_samples:
                logger.info(f"Regime '{regime}': only {n_found} samples (need {self.min_samples}), skipping")
                continue

            logger.info(f"Training models for regime '{regime}' ({n_found} samples)")

            regime_idx = np.where(mask)[0]
            start_idx = max(0, regime_idx[0] - self.lookback)
            end_idx = min(len(df_labeled), regime_idx[-1] + 1)
            regime_df = df_labeled.iloc[start_idx:end_idx]

            regime_dfs = {tf: regime_df if tf == primary_tf else dfs.get(tf, pd.DataFrame()) for tf in dfs}

            fp = FeaturePipeline(lookback=self.lookback, timeframes=list(dfs.keys()))
            X, y = fp.fit_transform(regime_dfs)

            if len(X) < self.min_samples:
                logger.info(f"Regime '{regime}': only {len(X)} sequences, skipping")
                continue

            split = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]
            prices_train, prices_val = y_train, y_val

            n_features = X.shape[-1]
            lstm_cnn = LSTMCNNHybrid(lookback=self.lookback, n_features=n_features)
            lstm_cnn.build()
            hist = lstm_cnn.train(
                X_train, y_train, X_val, y_val,
                epochs=epochs, batch_size=batch_size,
            )

            classifier = ProfitabilityClassifier(lookback=self.lookback, n_features=n_features)
            classifier.build()
            classifier.train(
                X_train, prices_train, X_val, prices_val,
                epochs=epochs, batch_size=batch_size,
            )

            bundle = RegimeModelBundle(
                regime=regime,
                lstm_cnn=lstm_cnn,
                classifier=classifier,
                feature_pipeline=fp,
                train_metrics={
                    "mse": float(min(hist.history.get("val_loss", [0]))),
                    "accuracy": float(max(
                        hist.history.get("val_accuracy", [0])
                    )) if hist and hist.history.get("val_accuracy") else 0.0,
                    "n_samples": n_found,
                },
                creation_timestamp=pd.Timestamp.now().timestamp(),
            )
            self.models[regime] = bundle
            self._save_regime_models(regime, bundle)

            results[regime] = bundle.train_metrics
            logger.info(f"Regime '{regime}' trained: MSE={bundle.train_metrics['mse']:.6f}")

        return results

    def predict_regime_models(
        self,
        dfs: Dict[str, pd.DataFrame],
        current_regime: str,
        X: np.ndarray,
    ) -> Tuple[float, float]:
        bundle = self.models.get(current_regime)
        if bundle is None or bundle.lstm_cnn is None:
            return 0.0, 0.0
        price_pred = bundle.lstm_cnn.predict(X)
        direction_prob = bundle.classifier.predict_proba(X) if bundle.classifier else np.array([[0.5]])
        return float(price_pred[0, 0]), float(direction_prob[0, 0])

    def get_regime_ensemble_predictions(
        self,
        dfs: Dict[str, pd.DataFrame],
        X: np.ndarray,
        regime_posteriors: np.ndarray,
    ) -> Dict[str, Tuple[float, float]]:
        results = {}
        for i, (regime, bundle) in enumerate(self.models.items()):
            if bundle.lstm_cnn is None:
                continue
            price_pred = bundle.lstm_cnn.predict(X)
            direction_prob = bundle.classifier.predict_proba(X) if bundle.classifier else np.array([[0.5]])
            results[regime] = (float(price_pred[0, 0]), float(direction_prob[0, 0]))
        return results

    def _save_regime_models(self, regime: str, bundle: RegimeModelBundle):
        regime_dir = os.path.join(self.model_dir, regime)
        os.makedirs(regime_dir, exist_ok=True)
        if bundle.lstm_cnn and bundle.lstm_cnn.model:
            bundle.lstm_cnn.save(os.path.join(regime_dir, "lstm_cnn.keras"))
        if bundle.classifier and bundle.classifier.model:
            bundle.classifier.save(os.path.join(regime_dir, "classifier.keras"))
        with open(os.path.join(regime_dir, "metrics.json"), "w") as f:
            json.dump(bundle.train_metrics, f, indent=2)

    def load_regime_models(self) -> Dict[str, bool]:
        loaded = {}
        for regime in HMMRegimeDetector.REGIME_NAMES[:self.n_regimes]:
            regime_dir = os.path.join(self.model_dir, regime)
            lstm_path = os.path.join(regime_dir, "lstm_cnn.keras")
            clf_path = os.path.join(regime_dir, "classifier.keras")
            metrics_path = os.path.join(regime_dir, "metrics.json")
            if not os.path.exists(lstm_path):
                loaded[regime] = False
                continue
            try:
                lstm = LSTMCNNHybrid.load(lstm_path)
                clf = ProfitabilityClassifier.load(clf_path) if os.path.exists(clf_path) else None
                metrics = {}
                if os.path.exists(metrics_path):
                    with open(metrics_path) as f:
                        metrics = json.load(f)
                self.models[regime] = RegimeModelBundle(
                    regime=regime,
                    lstm_cnn=lstm,
                    classifier=clf,
                    train_metrics=metrics,
                    creation_timestamp=os.path.getmtime(lstm_path),
                )
                loaded[regime] = True
                logger.info(f"Loaded regime model '{regime}'")
            except Exception as e:
                logger.error(f"Failed to load regime '{regime}': {e}")
                loaded[regime] = False
        return loaded
