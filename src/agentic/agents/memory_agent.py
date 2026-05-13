"""
Memory Agent — G26: checkpoint integrity via SHA256 content verification.
"""

from __future__ import annotations
import time
import json
import hashlib
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
from loguru import logger

from agentic.core.base_agent import BaseAgent
from agentic.core.agent_message import MessageType, MessagePriority, AgentIntention
from agentic.core.agent_consciousness import ConsciousnessLevel


class MemoryAgent(BaseAgent):
    def __init__(self, base_path: str = "data/agent_memory"):
        super().__init__(
            name="memory_agent",
            role="State Persistence Manager",
            purpose="Persist system state with integrity verification for crash recovery",
            domain="persistence",
            capabilities={
                "state_persistence", "crash_recovery",
                "checkpoint_management", "audit_trail",
                "versioned_backups", "integrity_verification",  # G26
            },
            tick_interval=30.0,
            consciousness_level=ConsciousnessLevel.REFLECTIVE,
        )
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._last_save_time = 0.0
        self._save_count = 0
        self._checkpoint_interval = 300.0
        self._integrity_errors = 0  # G26

        self.subscribe(MessageType.POSITION_OPENED)
        self.subscribe(MessageType.POSITION_CLOSED)
        self.subscribe(MessageType.RISK_ALERT)
        self.subscribe(MessageType.DIAGNOSTIC_REQUEST)

    async def _on_start(self):
        self.set_world("persistence.status", "active")
        self.set_world("persistence.path", str(self.base_path))
        # G26: Verify existing checkpoints
        self._verify_existing()
        self.log_state(f"Persistence active at {self.base_path}")

    async def perceive(self) -> Dict[str, Any]:
        time_since_save = time.time() - self._last_save_time
        return {"should_save": time_since_save >= self._checkpoint_interval,
                "time_since_save": time_since_save}

    async def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        return {"save": perception.get("should_save", False),
                "verify": self.consciousness.cycle_count % 10 == 0}

    async def act(self, decision: Dict[str, Any]):
        if decision.get("save"):
            await self._save_checkpoint()
        # G26: Periodic integrity verification
        if decision.get("verify"):
            self._verify_existing()

    async def reflect(self, outcome: Dict[str, Any]):
        self.memory.know("persistence.last_save", self._last_save_time, ttl=3600)
        self.memory.know("persistence.save_count", self._save_count, ttl=3600)
        self.set_world("persistence.integrity_errors", self._integrity_errors)

    async def on_message(self, message: AgentMessage):
        if message.msg_type in (MessageType.POSITION_OPENED, MessageType.POSITION_CLOSED):
            await self._save_checkpoint()
        elif message.msg_type == MessageType.RISK_ALERT:
            payload = message.payload if isinstance(message.payload, dict) else {}
            if payload.get("type") in ("halt", "kill_switch"):
                await self._save_checkpoint()
        elif message.msg_type == MessageType.DIAGNOSTIC_REQUEST:
            await self.send(MessageType.DIAGNOSTIC_RESULT, payload={
                "agent": self.name, "save_count": self._save_count,
                "last_save": self._last_save_time,
                "integrity_errors": self._integrity_errors,
                "path": str(self.base_path),
            }, target=message.source_agent)

    def _compute_checksum(self, data: Dict) -> str:
        """G26: SHA256 checksum of serialized state."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _verify_checkpoint(self, path: Path) -> bool:
        """G26: Verify a single checkpoint file's integrity."""
        try:
            with open(path) as f:
                data = json.load(f)
            stored_checksum = data.get("_checksum", "")
            data.pop("_checksum", None)
            computed = self._compute_checksum(data)
            if stored_checksum and stored_checksum != computed:
                self._integrity_errors += 1
                logger.warning(f"[{self.name}] INTEGRITY ERROR: {path.name} "
                               f"({stored_checksum[:8]} != {computed[:8]})")
                return False
            return True
        except Exception as e:
            self._integrity_errors += 1
            logger.warning(f"[{self.name}] Cannot verify {path.name}: {e}")
            return False

    def _verify_existing(self):
        """G26: Verify all existing checkpoint files."""
        files = sorted(self.base_path.glob("checkpoint_*.json"))
        if not files:
            return
        bad = 0
        for f in files:
            if not self._verify_checkpoint(f):
                bad += 1
        if bad > 0:
            self.log_state(f"Integrity: {bad}/{len(files)} checkpoints corrupted", "warning")
        else:
            logger.debug(f"[{self.name}] Integrity: {len(files)} checkpoints verified OK")

    async def _save_checkpoint(self):
        self.consciousness.current_intention = "saving system state checkpoint with integrity"

        # Gather world state snapshot
        snapshot = self.world.snapshot()
        checkpoint = {
            "timestamp": time.time(),
            "version": "4.0.0",
            "world_state": snapshot,
            "integrity": {
                "n_variables": len(snapshot),
                "verified": True,
            },
        }

        # G26: Compute and attach checksum
        checksum = self._compute_checksum({k: v for k, v in checkpoint.items()
                                            if k != "_checksum"})
        checkpoint["_checksum"] = checksum

        path = self.base_path / f"checkpoint_{int(time.time())}.json"
        try:
            with open(path, "w") as f:
                json.dump(checkpoint, f, indent=2, default=str)
            self._save_count += 1
            self._last_save_time = time.time()
            self._cleanup_old()
        except Exception as e:
            logger.warning(f"[{self.name}] Save failed: {e}")

    def _cleanup_old(self, max_files: int = 50):
        files = sorted(self.base_path.glob("checkpoint_*.json"))
        for f in files[:-max_files]:
            try:
                f.unlink()
            except Exception:
                pass
