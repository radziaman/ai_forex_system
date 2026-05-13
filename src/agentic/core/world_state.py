"""
World State — shared reality layer with versioned updates, change observers, and G26 integrity checks.
"""

from __future__ import annotations
import time
import threading
import json
import hashlib
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class StateVariable:
    key: str
    value: Any
    version: int = 0
    updated_at: float = field(default_factory=time.time)
    updated_by: str = ""
    ttl_seconds: float = 0.0
    checksum: str = ""  # G26

    def __post_init__(self):
        self._recompute_checksum()

    def _recompute_checksum(self):
        try:
            raw = json.dumps(self.value, sort_keys=True, default=str)
            self.checksum = hashlib.sha256(raw.encode()).hexdigest()[:8]
        except Exception:
            self.checksum = ""

    @property
    def is_expired(self) -> bool:
        if self.ttl_seconds <= 0:
            return False
        return time.time() - self.updated_at > self.ttl_seconds


class WorldState:
    """
    Thread-safe shared state with:
    - Versioned updates (monotonic counter)
    - Change notification
    - TTL expiry
    - Update provenance
    - G26: Checksum validation
    """

    def __init__(self):
        self._state: Dict[str, StateVariable] = {}
        self._lock = threading.RLock()
        self._global_version: int = 0
        self._observers: Dict[str, List[Callable]] = {}
        self._update_history: List[Dict[str, Any]] = []
        self._max_history: int = 10000
        self._integrity_errors: int = 0
        logger.info("WorldState initialized")

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            sv = self._state.get(key)
            if sv is None or sv.is_expired:
                return default
            return sv.value

    def set(self, key: str, value: Any, updated_by: str = "", ttl: float = 0.0):
        with self._lock:
            self._global_version += 1
            if key in self._state:
                sv = self._state[key]
                sv.value = value
                sv.version = self._global_version
                sv.updated_at = time.time()
                sv.updated_by = updated_by
                sv.ttl_seconds = ttl
                sv._recompute_checksum()
            else:
                self._state[key] = StateVariable(
                    key=key, value=value, version=self._global_version,
                    updated_by=updated_by, ttl_seconds=ttl,
                )
            self._update_history.append({
                "key": key, "version": self._global_version,
                "timestamp": time.time(), "updated_by": updated_by,
            })
            if len(self._update_history) > self._max_history:
                self._update_history = self._update_history[-self._max_history:]

            for cb in self._observers.get(key, []):
                try:
                    cb(key, value, updated_by)
                except Exception:
                    pass

    def update(self, key: str, **kwargs):
        current = self.get(key, {})
        if isinstance(current, dict):
            current.update(kwargs)
            self.set(key, current)

    def observe(self, key: str, callback: Callable):
        self._observers.setdefault(key, []).append(callback)

    def unobserve(self, key: str, callback: Callable):
        if callback in self._observers.get(key, []):
            self._observers[key].remove(callback)

    def exists(self, key: str) -> bool:
        with self._lock:
            return key in self._state and not self._state[key].is_expired

    def delete(self, key: str):
        with self._lock:
            self._state.pop(key, None)

    # G26: Verify integrity of a key
    def verify_integrity(self, key: str) -> bool:
        with self._lock:
            sv = self._state.get(key)
            if sv is None:
                return True
            old_checksum = sv.checksum
            sv._recompute_checksum()
            if old_checksum and old_checksum != sv.checksum:
                self._integrity_errors += 1
                logger.warning(f"WorldState integrity error for '{key}': "
                               f"{old_checksum} != {sv.checksum}")
                return False
            return True

    def verify_all(self) -> Dict[str, bool]:
        results = {}
        with self._lock:
            for key in list(self._state.keys()):
                results[key] = self.verify_integrity(key)
        return results

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {k: sv.value for k, sv in self._state.items()
                    if not sv.is_expired}

    def version_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {k: {"value": sv.value, "version": sv.version,
                        "updated_at": sv.updated_at, "updated_by": sv.updated_by,
                        "checksum": sv.checksum}
                    for k, sv in self._state.items() if not sv.is_expired}

    def get_version(self, key: str) -> int:
        with self._lock:
            sv = self._state.get(key)
            return sv.version if sv else 0

    def get_domain_state(self, prefix: str) -> Dict[str, Any]:
        return {k: v for k, v in self.snapshot().items()
                if k.startswith(prefix)}

    def summary(self) -> Dict:
        return {
            "variables": len(self._state),
            "global_version": self._global_version,
            "observers": sum(len(v) for v in self._observers.values()),
            "history_size": len(self._update_history),
            "integrity_errors": self._integrity_errors,
        }


_world: Optional[WorldState] = None


def get_world_state() -> WorldState:
    global _world
    if _world is None:
        _world = WorldState()
    return _world
