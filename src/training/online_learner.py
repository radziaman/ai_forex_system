"""
Online Learning Engine — drift-triggered retraining with model validation.
Monitors ADWIN drift signals, retrains in background, auto-deploys if better.
"""
import os
import time
import json
import threading
import warnings
from typing import Dict, Optional, List, Callable
from dataclasses import dataclass, field
from loguru import logger

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd

@dataclass
class ModelSnapshot:
    pair: str
    path_lstm: str
    path_clf: str
    val_loss: float
    val_mae: float
    val_accuracy: float
    trained_bars: int
    timestamp: float = field(default_factory=time.time)

    def is_better_than(self, other: Optional['ModelSnapshot']) -> bool:
        if other is None:
            return True
        if self.val_loss < other.val_loss * 0.95:
            return True
        if self.val_accuracy > other.val_accuracy * 1.05:
            return True
        return False


class OnlineLearner:
    def __init__(
        self,
        model_dir: str = "models",
        retrain_cooldown_hours: float = 4.0,
        min_trades_before_retrain: int = 50,
        val_split: float = 0.2,
        lstm_epochs: int = 30,
        clf_epochs: int = 20,
        notify_callback: Optional[Callable] = None,
    ):
        self.model_dir = model_dir
        self.retrain_cooldown = retrain_cooldown_hours * 3600
        self.min_trades = min_trades_before_retrain
        self.val_split = val_split
        self.lstm_epochs = lstm_epochs
        self.clf_epochs = clf_epochs
        self._notify = notify_callback

        self._last_retrain: Dict[str, float] = {}
        self._deployed: Dict[str, ModelSnapshot] = {}
        self._staging: Dict[str, ModelSnapshot] = {}
        self._running: Dict[str, bool] = {}
        self._lock = threading.Lock()
        self._drift_signals: Dict[str, int] = {}

        self._load_deployed_models()

    def _load_deployed_models(self):
        snap_path = os.path.join(self.model_dir, "deployed_snapshots.json")
        if os.path.exists(snap_path):
            try:
                with open(snap_path) as f:
                    data = json.load(f)
                for pair, snap in data.items():
                    if os.path.exists(snap["path_lstm"]):
                        self._deployed[pair] = ModelSnapshot(**snap)
                logger.info(f"Loaded {len(self._deployed)} deployed model snapshots")
            except Exception as e:
                logger.warning(f"Failed to load snapshots: {e}")

    def _save_deployed_models(self):
        snap_path = os.path.join(self.model_dir, "deployed_snapshots.json")
        try:
            data = {
                pair: {
                    "pair": snap.pair,
                    "path_lstm": snap.path_lstm,
                    "path_clf": snap.path_clf,
                    "val_loss": snap.val_loss,
                    "val_mae": snap.val_mae,
                    "val_accuracy": snap.val_accuracy,
                    "trained_bars": snap.trained_bars,
                    "timestamp": snap.timestamp,
                }
                for pair, snap in self._deployed.items()
            }
            with open(snap_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save snapshots: {e}")

    def on_drift_detected(self, pair: str, drift_count: int):
        with self._lock:
            self._drift_signals[pair] = drift_count

    def should_retrain(self, pair: str, total_trades: int) -> bool:
        with self._lock:
            if self._running.get(pair, False):
                return False
            last = self._last_retrain.get(pair, 0)
            if time.time() - last < self.retrain_cooldown:
                return False
            drift = self._drift_signals.get(pair, 0)
            if drift > 0:
                return True
            if total_trades >= self.min_trades and self._deployed.get(pair) is None:
                return True
            return False

    def request_retrain(
        self,
        pair: str,
        fetch_data_fn: Callable,
        feature_pipeline,
    ):
        with self._lock:
            if self._running.get(pair, False):
                return
            self._running[pair] = True

        thread = threading.Thread(
            target=self._retrain_worker,
            args=(pair, fetch_data_fn, feature_pipeline),
            daemon=True,
        )
        thread.start()
        logger.info(f"Online learner: retrain started for {pair}")

    def _retrain_worker(self, pair: str, fetch_fn, fp):
        try:
            self._do_retrain(pair, fetch_fn, fp)
        except Exception as e:
            logger.error(f"Online learner failed for {pair}: {e}")
        finally:
            with self._lock:
                self._running[pair] = False
                self._last_retrain[pair] = time.time()
                self._drift_signals[pair] = 0

    def _do_retrain(self, pair: str, fetch_fn, fp):
        logger.info(f"Online learner: fetching data for {pair}")
        df = fetch_fn(pair)
        if df is None or len(df) < 200:
            logger.warning(f"Online learner: insufficient data for {pair}")
            return

        # Build 3D sequences
        try:
            X, y = fp.fit_transform(
                {pair: {"1h": df}}, symbol=pair, flatten=False
            )
            if len(X) < 200:
                logger.warning(f"Online learner: only {len(X)} sequences for {pair}")
                return
        except Exception as e:
            logger.error(f"Online learner: feature pipeline failed: {e}")
            return

        nf = X.shape[-1]
        split = int(len(X) * (1 - self.val_split))
        X_tr, X_v = X[:split], X[split:]
        y_tr, y_v = y[:split], y[split:]

        # Import here to avoid TF overhead at module level
        from rts_ai_fx.model import LSTMCNNHybrid, ProfitabilityClassifier

        # Train LSTM-CNN
        logger.info(f"Online learner: training LSTM-CNN for {pair}")
        lstm = LSTMCNNHybrid(lookback=30, n_features=nf, lstm_units=128, cnn_filters=128)
        lstm.build()
        h = lstm.train(X_tr, y_tr, X_v, y_v, epochs=self.lstm_epochs, batch_size=32)
        vl = float(min(h.history.get("val_loss", [999])))
        vm = float(min(h.history.get("val_mae", [999])))

        # Train classifier
        logger.info(f"Online learner: training classifier for {pair}")
        clf = ProfitabilityClassifier(lookback=30, n_features=nf)
        clf.build()
        ch = clf.train(X_tr, y_tr, X_v, y_v, epochs=self.clf_epochs, batch_size=32)
        va = float(max(ch.history.get("val_accuracy", [0])))

        # Stage new model
        ts = int(time.time())
        lstm_path = os.path.join(self.model_dir, f"{pair}_lstm_cnn_staging_{ts}.keras")
        clf_path = os.path.join(self.model_dir, f"{pair}_classifier_staging_{ts}.keras")
        lstm.save(lstm_path)
        clf.save(clf_path)

        new_snap = ModelSnapshot(
            pair=pair, path_lstm=lstm_path, path_clf=clf_path,
            val_loss=vl, val_mae=vm, val_accuracy=va,
            trained_bars=len(df),
        )

        # Compare with deployed
        deployed = self._deployed.get(pair)
        if new_snap.is_better_than(deployed):
            # Promote to deployment
            live_lstm = os.path.join(self.model_dir, f"{pair}_lstm_cnn.keras")
            live_clf = os.path.join(self.model_dir, f"{pair}_classifier.keras")

            # Backup old model
            if os.path.exists(live_lstm):
                backup_lstm = live_lstm.replace(".keras", "_backup.keras")
                os.rename(live_lstm, backup_lstm) if os.path.exists(live_lstm) else None
            if os.path.exists(live_clf):
                backup_clf = live_clf.replace(".keras", "_backup.keras")
                os.rename(live_clf, backup_clf) if os.path.exists(live_clf) else None

            # Deploy new
            os.rename(lstm_path, live_lstm)
            os.rename(clf_path, live_clf)

            new_snap.path_lstm = live_lstm
            new_snap.path_clf = live_clf
            self._deployed[pair] = new_snap
            self._save_deployed_models()

            improvement = ""
            if deployed:
                dl = (deployed.val_loss - vl) / deployed.val_loss * 100
                da = (va - deployed.val_accuracy) * 100
                improvement = f" (loss -{dl:.1f}%, acc +{da:.1f}%)"
            logger.success(f"Online learner: deployed new {pair} model{improvement}")

            msg = (
                f"\U0001F4A1 <b>Model Upgraded: {pair}</b>\n"
                f"Loss: {vl:.6f} \u2192 {vm:.6f}\n"
                f"Accuracy: {va:.2%}\n"
                f"Trained on: {len(df)} bars{improvement}"
            )
            if self._notify:
                self._notify(msg)

            # Clean up staging files
            for f in [lstm_path, clf_path]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except Exception:
                        pass
        else:
            logger.info(f"Online learner: new {pair} model not better, discarding")
            for f in [lstm_path, clf_path]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except Exception:
                        pass
