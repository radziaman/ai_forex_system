import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ReconciliationDiff:
    missing: List[Dict[str, Any]] = field(default_factory=list)
    extra: List[Dict[str, Any]] = field(default_factory=list)
    mismatched: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = 0.0


class PositionReconciler:
    def __init__(
        self,
        get_internal_positions: Callable[[], List[Dict[str, Any]]],
        get_broker_positions: Callable[[], List[Dict[str, Any]]],
        on_mismatch: Optional[Callable[[ReconciliationDiff], None]] = None,
    ):
        self.get_internal_positions = get_internal_positions
        self.get_broker_positions = get_broker_positions
        self.on_mismatch = on_mismatch
        self._last_diff: Optional[ReconciliationDiff] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def reconcile(
        self,
        internal: List[Dict[str, Any]],
        broker: List[Dict[str, Any]],
    ) -> ReconciliationDiff:
        missing: List[Dict[str, Any]] = []
        extra: List[Dict[str, Any]] = []
        mismatched: List[Dict[str, Any]] = []

        internal_by_key: Dict[tuple, Dict[str, Any]] = {}
        for pos in internal:
            key = (pos.get("symbol"), pos.get("direction"))
            internal_by_key[key] = pos

        broker_by_key: Dict[tuple, Dict[str, Any]] = {}
        for pos in broker:
            key = (pos.get("symbol"), pos.get("direction"))
            broker_by_key[key] = pos

        for key, i_pos in internal_by_key.items():
            b_pos = broker_by_key.get(key)
            if b_pos is None:
                missing.append(i_pos)
            elif not self._positions_match(i_pos, b_pos):
                mismatched.append({"internal": i_pos, "broker": b_pos})

        for key, b_pos in broker_by_key.items():
            if key not in internal_by_key:
                extra.append(b_pos)

        diff = ReconciliationDiff(
            missing=missing,
            extra=extra,
            mismatched=mismatched,
            timestamp=time.time(),
        )
        self._last_diff = diff
        if (missing or extra or mismatched) and self.on_mismatch:
            self.on_mismatch(diff)
        return diff

    def _positions_match(
        self,
        internal: Dict[str, Any],
        broker: Dict[str, Any],
    ) -> bool:
        sym_ok = internal.get("symbol") == broker.get("symbol")
        dir_ok = internal.get("direction") == broker.get("direction")
        vol_int = float(internal.get("volume", 0))
        vol_brk = float(broker.get("volume", 0))
        if vol_brk == 0:
            vol_ok = vol_int == 0
        else:
            vol_ok = abs(vol_int - vol_brk) / vol_brk <= 0.01
        return sym_ok and dir_ok and vol_ok

    async def reconcile_loop(self) -> None:
        self._running = True
        while self._running:
            try:
                internal = self.get_internal_positions()
                if asyncio.iscoroutine(internal) or hasattr(internal, "__await__"):
                    internal = await internal  # type: ignore[assignment]
                broker = self.get_broker_positions()
                if asyncio.iscoroutine(broker) or hasattr(broker, "__await__"):
                    broker = await broker  # type: ignore[assignment]
                self.reconcile(internal, broker)
            except Exception:
                pass
            await asyncio.sleep(30)

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    def get_last_diff(self) -> Optional[ReconciliationDiff]:
        return self._last_diff
