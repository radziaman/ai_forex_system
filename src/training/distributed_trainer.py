"""
Distributed Training Pipeline for scaling model training.
Supports Ray-based parallel training, hyperparameter sweeps,
and experiment tracking with Weights & Biases.
"""
import numpy as np
import os
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from loguru import logger

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


@dataclass
class TrialConfig:
    lstm_units: int = 128
    cnn_filters: int = 128
    learning_rate: float = 0.001
    dropout: float = 0.2
    batch_size: int = 32
    n_lstm_layers: int = 1
    n_dense_units: int = 64
    lookback: int = 30
    epochs: int = 100
    early_stop_patience: int = 10


@dataclass
class TrialResult:
    config: TrialConfig
    val_loss: float
    val_mae: float
    val_accuracy: float
    train_time: float
    params_count: int


class DistributedTrainer:
    def __init__(
        self,
        num_workers: int = 4,
        use_ray: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "rts_forex",
        results_dir: str = "models/hp_sweeps",
    ):
        self.num_workers = num_workers
        self.use_ray = use_ray and RAY_AVAILABLE
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_project = wandb_project
        self.results_dir = results_dir

        if self.use_ray:
            try:
                if not ray.is_initialized():
                    ray.init(ignore_reinit_error=True)
                logger.info(f"Ray initialized with {num_workers} workers")
            except Exception as e:
                logger.warning(f"Ray init failed, using local: {e}")
                self.use_ray = False

        os.makedirs(results_dir, exist_ok=True)

    def hyperparameter_sweep(
        self,
        param_grid: Dict[str, List[Any]],
        train_fn: Callable,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 50,
    ) -> List[TrialResult]:
        if self.use_ray and RAY_AVAILABLE:
            return self._ray_sweep(
                param_grid, train_fn, X_train, y_train, X_val, y_val, n_trials
            )
        elif OPTUNA_AVAILABLE:
            return self._optuna_sweep(
                param_grid, train_fn, X_train, y_train, X_val, y_val, n_trials
            )
        else:
            return self._local_sweep(
                param_grid, train_fn, X_train, y_train, X_val, y_val, n_trials
            )

    def _ray_sweep(
        self,
        param_grid: Dict[str, List[Any]],
        train_fn: Callable,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int,
    ) -> List[TrialResult]:
        if not RAY_AVAILABLE:
            return self._local_sweep(param_grid, train_fn, X_train, y_train, X_val, y_val, n_trials)

        @ray.remote(num_gpus=0.5)
        def _train_trial(config_dict: Dict) -> Dict:
            import tensorflow as tf
            tf.get_logger().setLevel("ERROR")
            cfg = TrialConfig(**config_dict)
            start = time.time()
            metrics = train_fn(X_train, y_train, X_val, y_val, cfg)
            elapsed = time.time() - start
            return {
                "config": config_dict,
                "val_loss": float(metrics.get("val_loss", 999)),
                "val_mae": float(metrics.get("val_mae", 999)),
                "val_accuracy": float(metrics.get("val_accuracy", 0)),
                "train_time": elapsed,
                "params_count": int(metrics.get("params", 0)),
            }

        all_params = self._generate_param_combinations(param_grid, n_trials)
        logger.info(f"Launching {len(all_params)} Ray trials across {self.num_workers} workers")

        futures = [_train_trial.remote(p) for p in all_params]
        results = []
        batch_size = self.num_workers
        for i in range(0, len(futures), batch_size):
            batch = futures[i : i + batch_size]
            ready, _ = ray.wait(batch, num_returns=len(batch), timeout=300)
            for ref in ready:
                try:
                    result = ray.get(ref)
                    results.append(TrialResult(
                        config=TrialConfig(**result["config"]),
                        val_loss=result["val_loss"],
                        val_mae=result["val_mae"],
                        val_accuracy=result["val_accuracy"],
                        train_time=result["train_time"],
                        params_count=result["params_count"],
                    ))
                    self._log_progress(result, i + len(results), len(all_params))
                except Exception as e:
                    logger.error(f"Trial failed: {e}")

        results.sort(key=lambda r: r.val_loss)
        self._save_sweep_results(results)
        return results

    def _local_sweep(
        self,
        param_grid: Dict[str, List[Any]],
        train_fn: Callable,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int,
    ) -> List[TrialResult]:
        all_params = self._generate_param_combinations(param_grid, n_trials)
        results = []
        for i, params in enumerate(all_params):
            cfg = TrialConfig(**params)
            start = time.time()
            metrics = train_fn(X_train, y_train, X_val, y_val, cfg)
            elapsed = time.time() - start
            result = TrialResult(
                config=cfg,
                val_loss=float(metrics.get("val_loss", 999)),
                val_mae=float(metrics.get("val_mae", 999)),
                val_accuracy=float(metrics.get("val_accuracy", 0)),
                train_time=elapsed,
                params_count=int(metrics.get("params", 0)),
            )
            results.append(result)
            self._log_progress(
                {"config": params, "val_loss": result.val_loss}, i + 1, len(all_params)
            )

        results.sort(key=lambda r: r.val_loss)
        self._save_sweep_results(results)
        return results

    def _optuna_sweep(
        self,
        param_grid: Dict[str, List[Any]],
        train_fn: Callable,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int,
    ) -> List[TrialResult]:
        if not OPTUNA_AVAILABLE:
            return self._local_sweep(param_grid, train_fn, X_train, y_train, X_val, y_val, n_trials)

        def objective(trial):
            cfg = TrialConfig(
                lstm_units=trial.suggest_int("lstm_units", *param_grid.get("lstm_units", [32, 256])),
                cnn_filters=trial.suggest_int("cnn_filters", *param_grid.get("cnn_filters", [32, 256])),
                learning_rate=trial.suggest_float("learning_rate", *param_grid.get("learning_rate", [1e-5, 1e-2]), log=True),
                dropout=trial.suggest_float("dropout", *param_grid.get("dropout", [0.1, 0.5])),
                batch_size=trial.suggest_categorical("batch_size", param_grid.get("batch_size", [16, 32, 64])),
                n_lstm_layers=trial.suggest_int("n_lstm_layers", *param_grid.get("n_lstm_layers", [1, 3])),
                n_dense_units=trial.suggest_int("n_dense_units", *param_grid.get("n_dense_units", [32, 256])),
            )
            metrics = train_fn(X_train, y_train, X_val, y_val, cfg)
            val_mae = metrics.get("val_mae", 999)
            if self.use_wandb:
                wandb.log({"val_mae": val_mae, **cfg.__dict__})
            return val_mae

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        results = []
        for t in study.trials:
            if t.value is not None:
                results.append(TrialResult(
                    config=TrialConfig(**t.params),
                    val_loss=t.value,
                    val_mae=t.value,
                    val_accuracy=0.0,
                    train_time=0.0,
                    params_count=0,
                ))
        return results

    def train_distributed(
        self,
        train_fn: Callable,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: TrialConfig,
    ) -> Dict:
        if self.use_wandb:
            wandb.init(project=self.wandb_project, config=config.__dict__)

        if self.use_ray and RAY_AVAILABLE:
            return self._ray_train(train_fn, X_train, y_train, X_val, y_val, config)
        return train_fn(X_train, y_train, X_val, y_val, config)

    def _ray_train(
        self,
        train_fn: Callable,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: TrialConfig,
    ) -> Dict:
        if not RAY_AVAILABLE:
            return train_fn(X_train, y_train, X_val, y_val, config)

        @ray.remote(num_gpus=0.5)
        def _train(config_dict: Dict) -> Dict:
            import tensorflow as tf
            tf.get_logger().setLevel("ERROR")
            return train_fn(X_train, y_train, X_val, y_val, TrialConfig(**config_dict))

        future = _train.remote(config.__dict__)
        result = ray.get(future, timeout=3600)
        if self.use_wandb:
            wandb.log(result)
            wandb.finish()
        return result

    def _generate_param_combinations(
        self, param_grid: Dict[str, List[Any]], n_trials: int
    ) -> List[Dict]:
        import itertools
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        all_combs = list(itertools.product(*values))
        np.random.shuffle(all_combs)
        selected = all_combs[:n_trials]
        return [dict(zip(keys, comb)) for comb in selected]

    def _log_progress(self, result: Dict, current: int, total: int):
        loss = result.get("val_loss", "?")
        logger.info(f"Sweep [{current}/{total}] val_loss={loss:.6f}")

    def _save_sweep_results(self, results: List[TrialResult]):
        ts = int(time.time())
        path = os.path.join(self.results_dir, f"sweep_{ts}.json")
        data = [
            {
                "config": r.config.__dict__,
                "val_loss": r.val_loss,
                "val_mae": r.val_mae,
                "val_accuracy": r.val_accuracy,
                "train_time": r.train_time,
                "params_count": r.params_count,
            }
            for r in results
        ]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Sweep results saved to {path}")
        if results:
            best = results[0]
            logger.info(f"Best: val_loss={best.val_loss:.6f}, "
                        f"lstm={best.config.lstm_units}, "
                        f"lr={best.config.learning_rate}")
