"""LearningManager — drift detection, model registry, retraining, performance tracking.

Replaces: LearningAgent + DriftAgent + ModelRegistryAgent + ValidationAgent
         + MemoryAgent + PerformanceAgent.

Responsibilities:
- Concept drift detection via ADWIN (from drift_agent.py)
- Model registry with champion/challenger (from model_registry_agent.py)
- Performance tracking with Sharpe, profit factor, win rate (from performance_agent.py)
- Walk-forward validation (from validation_agent.py)
- Periodic retraining triggers (from learning_agent.py)
- State persistence / checkpoint management (from memory_agent.py)
"""

import json
import time
import hashlib
import numpy as np
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque
from pathlib import Path
from loguru import logger

from .pipeline_context import PipelineContext


class DriftMonitor:
    """ADWIN-based drift detection wrapper (from drift_agent.py logic)."""

    def __init__(self, error_threshold: float = 0.02):
        self.error_detector = self._adwin()
        self.feature_detector = self._adwin(delta=0.01, min_window=50)
        self.error_threshold = error_threshold
        self.retrain_triggered = False

    @staticmethod
    def _adwin(delta: float = 0.05, min_window: int = 30) -> Any:
        """Create an ADWIN drift detector."""
        from rts_ai_fx.drift_detector import ADWIN

        return ADWIN(delta=delta, min_window=min_window)

    def update(self, prediction: float, actual: float) -> bool:
        """Update monitors with prediction outcome.
        Returns True if retraining recommended.
        """
        error = abs(prediction - actual)
        error_drift = self.error_detector.update(error)
        sustained_bad = self.error_detector.mean > self.error_threshold
        self.retrain_triggered = error_drift or sustained_bad
        return self.retrain_triggered

    def reset(self) -> None:
        """Reset all detectors."""
        self.error_detector = self._adwin()
        self.feature_detector = self._adwin(delta=0.01, min_window=50)
        self.retrain_triggered = False


class PerformanceTracker:
    """Trade performance analytics (from performance_agent.py)."""

    def __init__(self, max_trades: int = 1000):
        self._trades: List[Dict] = []
        self._max_trades = max_trades
        self._pnl_series: deque = deque(maxlen=500)
        self._by_symbol: Dict[str, Dict] = defaultdict(
            lambda: {"trades": 0, "wins": 0, "pnl": 0.0}
        )
        self._by_regime: Dict[str, Dict] = defaultdict(
            lambda: {"trades": 0, "wins": 0, "pnl": 0.0}
        )

    def record_trade(self, trade: Dict) -> None:
        """Record a completed trade."""
        self._trades.append(trade)
        if len(self._trades) > self._max_trades:
            self._trades = self._trades[-self._max_trades :]
        pnl = trade.get("pnl", 0.0)
        self._pnl_series.append(pnl)
        symbol = trade.get("symbol", "unknown")
        regime = trade.get("regime", "unknown")
        self._by_symbol[symbol]["trades"] += 1
        self._by_symbol[symbol]["pnl"] += pnl
        if pnl > 0:
            self._by_symbol[symbol]["wins"] += 1
        self._by_regime[regime]["trades"] += 1
        self._by_regime[regime]["pnl"] += pnl
        if pnl > 0:
            self._by_regime[regime]["wins"] += 1

    @property
    def total_trades(self) -> int:
        return len(self._trades)

    @property
    def win_rate(self) -> float:
        if not self._trades:
            return 0.0
        wins = sum(1 for t in self._trades if t.get("pnl", 0) > 0)
        return wins / len(self._trades)

    @property
    def sharpe_ratio(self) -> float:
        """Annualized Sharpe ratio from PnL series."""
        if len(self._pnl_series) < 10:
            return 0.0
        returns = np.array(self._pnl_series)
        if returns.std() == 0:
            return 0.0
        return float(np.sqrt(252) * returns.mean() / returns.std())

    @property
    def profit_factor(self) -> float:
        """Ratio of gross profit to gross loss."""
        gross_profit = sum(t.get("pnl", 0) for t in self._trades if t.get("pnl", 0) > 0)
        gross_loss = abs(
            sum(t.get("pnl", 0) for t in self._trades if t.get("pnl", 0) < 0)
        )
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "sharpe": self.sharpe_ratio,
            "profit_factor": self.profit_factor,
            "by_symbol": dict(self._by_symbol),
            "by_regime": dict(self._by_regime),
        }


class CheckpointManager:
    """State persistence with integrity verification (from memory_agent.py)."""

    def __init__(self, base_path: str = "data/agent_memory"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._last_save_time: float = 0.0
        self._save_count: int = 0
        self._checkpoint_interval: float = 300.0

    def save_checkpoint(self, state: Dict[str, Any], name: str = "pipeline") -> bool:
        """Save a state checkpoint with integrity hash."""
        try:
            now = time.time()
            if now - self._last_save_time < self._checkpoint_interval:
                return False
            state["_timestamp"] = now
            content = json.dumps(state, sort_keys=True, default=str)
            checksum = hashlib.sha256(content.encode()).hexdigest()
            filepath = self.base_path / f"{name}_checkpoint.json"
            with open(filepath, "w") as f:
                f.write(content)
            hashpath = self.base_path / f"{name}_checkpoint.sha256"
            with open(hashpath, "w") as f:
                f.write(checksum)
            self._last_save_time = now
            self._save_count += 1
            return True
        except Exception as e:
            logger.warning(f"[checkpoint] Save failed: {e}")
            return False

    def load_checkpoint(self, name: str = "pipeline") -> Optional[Dict]:
        """Load and verify a state checkpoint."""
        try:
            filepath = self.base_path / f"{name}_checkpoint.json"
            hashpath = self.base_path / f"{name}_checkpoint.sha256"
            if not filepath.exists() or not hashpath.exists():
                return None
            with open(filepath) as f:
                content = f.read()
            with open(hashpath) as f:
                expected = f.read().strip()
            checksum = hashlib.sha256(content.encode()).hexdigest()
            if checksum != expected:
                logger.warning("[checkpoint] Integrity check FAILED")
                return None
            state = json.loads(content)
            state.pop("_timestamp", None)
            return state
        except Exception as e:
            logger.warning(f"[checkpoint] Load failed: {e}")
            return None

    def verify_integrity(self, name: str = "pipeline") -> bool:
        """Verify checkpoint integrity."""
        try:
            filepath = self.base_path / f"{name}_checkpoint.json"
            hashpath = self.base_path / f"{name}_checkpoint.sha256"
            if not filepath.exists() or not hashpath.exists():
                return True  # no checkpoint to verify
            with open(filepath) as f:
                content = f.read()
            with open(hashpath) as f:
                expected = f.read().strip()
            checksum = hashlib.sha256(content.encode()).hexdigest()
            return checksum == expected
        except Exception:
            return False


class LearningManager:
    """Central learning pipeline — drift, retraining, registry,
    performance, persistence.
    """

    ACTIVE_SYMBOLS = [
        "EURUSD",
        "GBPUSD",
        "USDJPY",
        "AUDUSD",
        "USDCAD",
        "USDCHF",
        "NZDUSD",
        "XAUUSD",
    ]

    def __init__(
        self,
        ctx: PipelineContext,
        model_registry: Optional[Any] = None,
        online_learner: Optional[Any] = None,
    ):
        self.ctx = ctx
        self.config = ctx.config
        self.bus = ctx.bus
        self._model_registry = model_registry
        self._online_learner = online_learner
        self._initialized = False
        self._retraining_count = 0
        self._last_training_time = 0.0
        self._feature_pipeline: Any = None

        # Drift monitors per symbol (from DriftAgent)
        self._drift_monitors: Dict[str, DriftMonitor] = {}

        # Performance tracker (from PerformanceAgent)
        self._performance = PerformanceTracker()

        # Checkpoint manager (from MemoryAgent)
        self._checkpointer = CheckpointManager()

        # Validation state
        self._last_validation_time = 0.0

        # Walk-forward optimization state
        self._best_sharpe: float = 0.0
        self._last_walk_forward_time: float = 0.0
        self._walk_forward_interval: float = 14400.0  # 4 hours

        # Subscribe to events
        self.bus.on("execution_result", self._on_execution_result)
        self.bus.on("position_closed", self._on_position_closed)

    async def start(self) -> None:
        """Initialize the learning manager."""
        # Initialize drift monitors for all active symbols
        for sym in self.ACTIVE_SYMBOLS:
            self._drift_monitors[sym] = DriftMonitor()

        # Initialize model registry
        if self._model_registry is None:
            try:
                from training.model_registry import ModelRegistry

                self._model_registry = ModelRegistry(registry_path="models/registry")
            except Exception:
                logger.warning("[learning_manager] ModelRegistry not available")

        # Initialize OnlineLearner
        if self._online_learner is None:
            try:
                from training.online_learner import OnlineLearner

                self._online_learner = OnlineLearner(
                    retrain_cooldown_hours=4.0,
                    min_trades_before_retrain=50,
                    notify_callback=self._log_training_notification,
                )
            except Exception:
                logger.warning("[learning_manager] OnlineLearner not available")

        # Initialize feature pipeline for retraining
        try:
            from rts_ai_fx.features_unified import FeaturePipeline

            self._feature_pipeline = FeaturePipeline(
                lookback=30,
                timeframes=["1h", "4h"],
                use_microstructure=True,
            )
        except Exception:
            logger.warning("[learning_manager] Feature pipeline not loaded")

        # Load checkpoint if available
        saved_state = self._checkpointer.load_checkpoint("learning")
        if saved_state:
            self._retraining_count = saved_state.get("retraining_count", 0)
            logger.info(
                f"[learning_manager] Restored state: "
                f"{self._retraining_count} past retrainings"
            )

        self._initialized = True
        logger.info(
            f"[learning_manager] Ready — monitoring {len(self._drift_monitors)} symbols"
        )

    async def _on_execution_result(self, **data: Any) -> None:
        """Track prediction errors for drift detection."""
        symbol = data.get("symbol", "")
        predicted_price = data.get("signal_price", 0.0)
        filled_price = data.get("filled_price", 0.0)

        if symbol in self._drift_monitors and predicted_price > 0 and filled_price > 0:
            drift_detected = self._drift_monitors[symbol].update(
                predicted_price, filled_price
            )
            if drift_detected:
                metric = self._drift_monitors[symbol].error_detector.mean
                await self.bus.emit(
                    "drift_detected",
                    symbol=symbol,
                    metric="prediction_error",
                    score=float(metric),
                )
                logger.warning(
                    f"[learning_manager] Drift detected for {symbol}: "
                    f"error={metric:.4f}"
                )

    async def _on_position_closed(self, **data: Any) -> None:
        """Record completed trade for performance tracking."""
        trade = {
            "symbol": data.get("symbol", ""),
            "direction": data.get("direction", ""),
            "volume": data.get("volume", 0.0),
            "entry_price": data.get("entry_price", 0.0),
            "exit_price": data.get("exit_price", 0.0),
            "pnl": data.get("pnl", 0.0),
            "reason": data.get("reason", ""),
            "regime": data.get("regime", "unknown"),
            "timestamp": time.time(),
        }
        self._performance.record_trade(trade)

    async def _run_walk_forward_if_due(self) -> None:
        """Run walk-forward optimization if enough time has passed."""
        now = time.time()
        if now - self._last_walk_forward_time < self._walk_forward_interval:
            return
        await self._run_walk_forward()

    async def _run_walk_forward(self) -> Dict[str, Any]:
        """
        Automatic walk-forward optimization loop.
        Runs every N hours (configurable) on accumulated data.

        1. Split data into training/validation windows
        2. Train model on training window
        3. Evaluate on validation window
        4. If Sharpe improves > threshold, promote to champion
        5. Log results via event bus
        """
        self._last_walk_forward_time = time.time()

        if self._performance.total_trades < 20:
            logger.info(
                "[learning_manager] Walk-forward skipped — " "insufficient trade data"
            )
            return {"skipped": True, "reason": "insufficient_trades"}

        try:
            prices = self._get_training_prices()
            if prices is None or len(prices) < 100:
                logger.warning(
                    "[learning_manager] Walk-forward skipped — "
                    "insufficient price data"
                )
                return {"skipped": True, "reason": "insufficient_price_data"}

            from validation.smart_walk_forward import SmartWalkForward

            swf = SmartWalkForward(
                n_folds=6,
                min_train_window=min(500, len(prices) // 2),
                test_window=min(100, len(prices) // 4),
            )

            # Build a simple features function that computes returns
            def _returns_fn(p: np.ndarray, f: Any, r: Any) -> List[float]:
                if p is None or len(p) < 2:
                    return []
                rets = np.diff(np.log(p + 1e-10))
                return rets.tolist()

            features = prices  # use prices as features for baseline
            result = swf.run(
                prices=prices,
                features_fn=_returns_fn,
                features=features,
            )

            if result.total_folds == 0:
                logger.warning("[learning_manager] Walk-forward returned no folds")
                return {"skipped": True, "reason": "no_folds"}

            logger.info(
                f"[learning_manager] Walk-forward: "
                f"train_sharpe={result.avg_train_sharpe:.3f}, "
                f"test_sharpe={result.avg_test_sharpe:.3f}, "
                f"degradation={result.avg_degradation:.3f}, "
                f"stability={result.stability_score:.3f}, "
                f"passed={result.passed}"
            )

            if result.avg_test_sharpe > self._best_sharpe + 0.05:
                self._best_sharpe = result.avg_test_sharpe
                logger.info(
                    f"[learning_manager] New champion: "
                    f"Sharpe {result.avg_test_sharpe:.3f}"
                )
                await self.bus.emit(
                    "model_promoted",
                    model_id=f"walk_forward_{int(time.time())}",
                    sharpe=result.avg_test_sharpe,
                    train_sharpe=result.avg_train_sharpe,
                    degradation=result.avg_degradation,
                    stability=result.stability_score,
                    passed=result.passed,
                )
            else:
                logger.info(
                    f"[learning_manager] Walk-forward: "
                    f"Sharpe {result.avg_test_sharpe:.3f} "
                    f"≤ best {self._best_sharpe:.3f} — no promotion"
                )

            return {
                "skipped": False,
                "avg_train_sharpe": result.avg_train_sharpe,
                "avg_test_sharpe": result.avg_test_sharpe,
                "avg_degradation": result.avg_degradation,
                "stability": result.stability_score,
                "passed": result.passed,
                "total_folds": result.total_folds,
            }

        except Exception as e:
            logger.warning(f"[learning_manager] Walk-forward failed: {e}")
            return {"skipped": True, "error": str(e)}

    def _get_training_prices(self) -> Optional[np.ndarray]:
        """Get synthetic price data for walk-forward optimization.

        Falls back to generating price-like series from trade performance
        if historical prices are not available.
        """
        try:
            # Try to fetch actual price data
            if self._feature_pipeline is not None:
                from data.data_manager import DataManager

                dm = DataManager()
                data = dm.get_historical_data("EURUSD", timeframe="1h", bars=1000)
                if data is not None and hasattr(data, "close"):
                    return data["close"].values.astype(np.float64)
        except Exception:
            logger.debug(
                "[learning_manager] Cannot fetch real prices "
                "— using synthetic from trade PnL"
            )

        # Fallback: generate synthetic price from trade PnL
        pnl_series = list(self._performance._pnl_series)
        if len(pnl_series) < 10:
            return None
        synthetic = np.cumsum(np.array(pnl_series, dtype=np.float64))
        synthetic = synthetic - synthetic.min() + 100.0  # positive baseline
        return synthetic

    def _log_training_notification(self, message: str) -> None:
        """Callback for OnlineLearner notifications."""
        logger.info(f"[learning_manager] Training notification: {message}")

    async def check_retraining_needed(self) -> List[str]:
        """Check if retraining is needed for any symbol.

        Returns list of symbol names needing retraining.
        """
        if not self._initialized:
            return []

        symbols_needing_retrain: List[str] = []

        # Check drift monitors
        for sym, monitor in self._drift_monitors.items():
            if monitor.retrain_triggered:
                symbols_needing_retrain.append(sym)

        # Check OnlineLearner
        if self._online_learner and self._feature_pipeline:
            for sym in self.ACTIVE_SYMBOLS:
                if self._online_learner.should_retrain(
                    sym, self._performance.total_trades
                ):
                    symbols_needing_retrain.append(sym)

        # Check performance-based triggers
        stats = self._performance.get_stats()
        if (
            stats["total_trades"] > 20
            and stats["sharpe"] < 0.3
            and time.time() - self._last_training_time > 600
        ):
            # Performance degradation — retrain all symbols
            symbols_needing_retrain = list(self.ACTIVE_SYMBOLS)

        # Run walk-forward optimization if due (non-blocking)
        await self._run_walk_forward_if_due()

        return list(set(symbols_needing_retrain))

    async def trigger_retraining(self, symbols: List[str]) -> None:
        """Trigger retraining for specified symbols."""
        if not symbols:
            return

        self._retraining_count += 1
        self._last_training_time = time.time()

        logger.info(
            f"[learning_manager] Retraining #{self._retraining_count} "
            f"for {len(symbols)} symbols: {symbols}"
        )

        if self._online_learner and self._feature_pipeline:
            for sym in symbols:
                self._online_learner.request_retrain(
                    sym,
                    fetch_data_fn=self._fetch_ohlcv_data,
                    feature_pipeline=self._feature_pipeline,
                )
        else:
            # Fallback: emit event for external retraining
            await self.bus.emit(
                "retraining_requested",
                symbols=symbols,
                count=self._retraining_count,
                reason="drift_or_performance",
            )

    async def _fetch_ohlcv_data(self, symbol: str) -> Any:
        """Fetch OHLCV data for retraining."""
        try:
            from data.data_manager import DataManager

            dm = DataManager()
            return dm.get_historical_data(symbol, timeframe="1h", bars=1000)
        except Exception:
            logger.warning(f"[learning_manager] Cannot fetch data for {symbol}")
            return None

    async def run_validation(self) -> Dict[str, Any]:
        """Run walk-forward validation cycle."""
        if time.time() - self._last_validation_time < 3600:
            return {"skipped": True}
        self._last_validation_time = time.time()

        try:
            from training.validation_gate import ValidationGate, GateConfig

            gate = ValidationGate(GateConfig(min_sharpe=1.0, max_drawdown_pct=0.10))
            # Champion/challenger evaluation would go here
            # Simplified: just validate current performance
            stats = self._performance.get_stats()
            is_valid = gate.validate(
                stats.get("sharpe", 0),
                stats.get("max_drawdown", 0),
            )
            return {
                "validated": is_valid,
                "sharpe": stats.get("sharpe", 0),
                "trades": stats.get("total_trades", 0),
            }
        except Exception as e:
            logger.warning(f"[learning_manager] Validation failed: {e}")
            return {"validated": False, "error": str(e)}

    async def save_state(self) -> bool:
        """Persist current learning state."""
        state = {
            "retraining_count": self._retraining_count,
            "last_training_time": self._last_training_time,
            "performance": self._performance.get_stats(),
        }
        return self._checkpointer.save_checkpoint(state, "learning")

    async def stop(self) -> None:
        """Clean shutdown — save state."""
        await self.save_state()
        performance_stats = self._performance.get_stats()
        logger.info(
            f"[learning_manager] Stopped — "
            f"{performance_stats['total_trades']} trades tracked, "
            f"{self._retraining_count} retrainings"
        )

    @property
    def is_alive(self) -> bool:
        """Whether the learning manager is initialized."""
        return self._initialized

    @property
    def performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return self._performance.get_stats()

    @property
    def drifted_symbols(self) -> List[str]:
        """Get list of symbols where drift has been detected."""
        return [
            sym
            for sym, monitor in self._drift_monitors.items()
            if monitor.retrain_triggered
        ]
