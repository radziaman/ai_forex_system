"""CheckpointManager — periodic state persistence for crash recovery.

Saves ensemble state, open positions, and module state to disk on a
configurable interval. On restart, loads previous state to recover.

Events consumed:
    position_opened -> {position: dict}
    position_closed -> {position: dict}
    module_heartbeat -> {module, status}

Events emitted:
    checkpoint_saved -> {path, timestamp, size_bytes}
    checkpoint_loaded -> {path, timestamp, has_positions, has_ensemble}
"""

import asyncio
import json
import os
import time
from typing import Dict, List, Optional
from loguru import logger


class CheckpointManager:
    """Periodically persists pipeline state to disk for crash recovery.

    Saves to two files:
      models/ensemble_state.json  — MoE ensemble state (via ensemble.save_state)
      data/checkpoints/positions.json  — open positions
      data/checkpoints/pipeline_state.json  — module metadata

    On startup, loads previous state and emits checkpoint_loaded.
    """

    def __init__(
        self,
        event_bus,
        pipeline_ctx=None,
        ensemble=None,
        checkpoint_dir: str = "data/checkpoints",
        save_interval: float = 60.0,
        max_checkpoints: int = 5,
        auto_load: bool = True,
    ):
        self._bus = event_bus
        self._ctx = pipeline_ctx  # PipelineContext for lazy ensemble resolution
        self._ensemble = ensemble
        self._checkpoint_dir = checkpoint_dir
        self._save_interval = save_interval
        self._max_checkpoints = max_checkpoints
        self._auto_load = auto_load

        # Open positions tracking
        self._open_positions: Dict[int, Dict] = {}

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_save_time: float = 0.0

    async def start(self):
        """Subscribe to events and start periodic save loop."""
        self._bus.on("position_opened", self._on_position_opened)
        self._bus.on("position_closed", self._on_position_closed)
        self._bus.on("save_checkpoint", self._on_save_request)

        os.makedirs(self._checkpoint_dir, exist_ok=True)

        # Auto-load previous state on startup
        if self._auto_load:
            await self._load_state()

        self._running = True
        self._task = asyncio.create_task(self._save_loop())
        logger.info(
            f"CheckpointManager started: saving every {self._save_interval}s "
            f"to {self._checkpoint_dir}"
        )

    async def stop(self):
        """Save final state and stop."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        # Save one final time
        await self._save_all()
        self._bus.off("position_opened", self._on_position_opened)
        self._bus.off("position_closed", self._on_position_closed)
        self._bus.off("save_checkpoint", self._on_save_request)
        logger.info("CheckpointManager stopped")

    async def _on_position_opened(self, **data):
        """Track newly opened positions."""
        position = data.get("position", data)
        if isinstance(position, dict):
            pos_id = position.get("position_id", hash(str(position)))
            self._open_positions[pos_id] = {
                "position_id": pos_id,
                "symbol": position.get("symbol", "unknown"),
                "direction": position.get("direction", "unknown"),
                "volume": position.get("volume", 0.0),
                "entry_price": position.get("entry_price", 0.0),
                "open_time": position.get("open_time", time.time()),
                "sl": position.get("sl", 0.0),
                "tp": position.get("tp", 0.0),
            }

    async def _on_position_closed(self, **data):
        """Remove closed positions from tracking."""
        position = data.get("position", data)
        if isinstance(position, dict):
            pos_id = position.get("position_id", 0)
            self._open_positions.pop(pos_id, None)

    async def _on_save_request(self, **data):
        """Handle manual save checkpoint event."""
        await self._save_all()

    async def _save_loop(self):
        """Periodic save loop."""
        while self._running:
            await asyncio.sleep(self._save_interval)
            await self._save_all()

    def _get_ensemble(self):
        """Resolve ensemble from context if not directly set."""
        if self._ensemble is not None:
            return self._ensemble
        if self._ctx is not None and hasattr(self._ctx, "ensemble"):
            return self._ctx.ensemble
        return None

    async def _save_all(self):
        """Save all state to disk."""
        now = time.time()
        results = []

        # Save ensemble state
        ensemble_obj = self._get_ensemble()
        if ensemble_obj and hasattr(ensemble_obj, "save_state"):
            try:
                result = ensemble_obj.save_state()
                results.append(("ensemble", result))
            except Exception as e:
                logger.warning(f"CheckpointManager: ensemble save failed: {e}")
                results.append(("ensemble", False))

        # Save open positions
        if self._open_positions:
            try:
                path = os.path.join(self._checkpoint_dir, "positions.json")
                with open(path, "w") as f:
                    json.dump(
                        {
                            "timestamp": now,
                            "positions": list(self._open_positions.values()),
                        },
                        f,
                        indent=2,
                    )
                results.append(("positions", True))
            except Exception as e:
                logger.warning(f"CheckpointManager: positions save failed: {e}")
                results.append(("positions", False))

        # Save pipeline metadata
        try:
            path = os.path.join(self._checkpoint_dir, "pipeline_state.json")
            metadata = {
                "timestamp": now,
                "open_position_count": len(self._open_positions),
                "checkpoint_version": 1,
            }
            with open(path, "w") as f:
                json.dump(metadata, f, indent=2)
            results.append(("metadata", True))
        except Exception as e:
            logger.warning(f"CheckpointManager: metadata save failed: {e}")
            results.append(("metadata", False))

        self._last_save_time = now

        # Emit checkpoint event
        all_ok = all(r[1] for r in results)
        await self._bus.emit(
            "checkpoint_saved",
            timestamp=now,
            open_positions=len(self._open_positions),
            results={r[0]: r[1] for r in results},
        )

        if not all_ok:
            logger.warning(f"CheckpointManager: some saves failed: {results}")

    async def _load_state(self):
        """Load previous state from disk."""
        results = []

        # Load ensemble state
        ensemble_obj = self._get_ensemble()
        if ensemble_obj and hasattr(ensemble_obj, "load_state"):
            try:
                result = ensemble_obj.load_state()
                results.append(("ensemble", result))
            except Exception as e:
                logger.warning(f"CheckpointManager: ensemble load failed: {e}")
                results.append(("ensemble", False))

        # Load open positions
        path = os.path.join(self._checkpoint_dir, "positions.json")
        positions_loaded = 0
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                for pos in data.get("positions", []):
                    pos_id = pos.get("position_id", hash(str(pos)))
                    self._open_positions[pos_id] = pos
                    positions_loaded += 1
                results.append(("positions", True))
            except Exception as e:
                logger.warning(f"CheckpointManager: positions load failed: {e}")
                results.append(("positions", False))

        await self._bus.emit(
            "checkpoint_loaded",
            timestamp=time.time(),
            has_positions=positions_loaded > 0,
            has_ensemble=any(r[1] for r in results if r[0] == "ensemble"),
            positions_loaded=positions_loaded,
        )

        if positions_loaded > 0:
            logger.info(
                f"CheckpointManager: loaded {positions_loaded} open positions "
                f"from {path}"
            )

    def get_open_positions(self) -> List[Dict]:
        """Return list of currently tracked open positions."""
        return list(self._open_positions.values())

    @property
    def open_position_count(self) -> int:
        return len(self._open_positions)
