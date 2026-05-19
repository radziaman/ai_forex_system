"""
Agent Memory — episodic, semantic, working memory + G12 cross-agent queries + G21 consolidation.
"""

from __future__ import annotations
import time
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from collections import defaultdict, deque
from pathlib import Path
from loguru import logger


@dataclass
class EpisodicMemoryEntry:
    timestamp: float
    cycle_id: int
    agent_name: str
    event_type: str
    description: str
    data: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    emotion: str = "neutral"
    related_entries: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "ts": self.timestamp,
            "cycle": self.cycle_id,
            "agent": self.agent_name,
            "type": self.event_type,
            "desc": self.description[:200],
            "importance": self.importance,
            "emotion": self.emotion,
        }


@dataclass
class SemanticMemoryEntry:
    key: str
    value: Any
    confidence: float = 1.0
    source: str = ""
    source_agent: str = ""  # G12: which agent provided this knowledge
    updated_at: float = field(default_factory=time.time)
    ttl_seconds: float = 86400

    @property
    def is_expired(self) -> bool:
        return time.time() - self.updated_at > self.ttl_seconds


class AgentMemory:
    """
    Three-tier memory system.

    Episodic: ring buffer of experiences
    Semantic: key-value store of learned facts
    Working: transient context for current cycle

    G12: Supports cross-agent memory queries (query/respond protocol)
    G21: Periodic consolidation promotes important patterns to semantic
    """

    def __init__(
        self,
        agent_name: str,
        max_episodic: int = 1000,
        persist_path: Optional[str] = None,
    ):
        self.agent_name = agent_name
        self.persist_path = Path(persist_path) if persist_path else None

        self._episodic: deque = deque(maxlen=max_episodic)
        self._episodic_count = 0
        self._semantic: Dict[str, SemanticMemoryEntry] = {}
        self._working: Dict[str, Any] = {}
        self._working_timestamp: float = time.time()

        # G12: Cross-agent knowledge cache (agent_name -> {key: value})
        self._external_knowledge: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._external_knowledge_timestamps: Dict[str, float] = {}

        self.consolidation_threshold: float = 0.7
        self._type_index: Dict[str, List[int]] = defaultdict(list)

        logger.info(f"[{agent_name}] Memory initialized ({max_episodic} slots)")

    # ------------------------------------------------------------------
    # Episodic Memory
    # ------------------------------------------------------------------

    def remember(
        self,
        event_type: str,
        description: str,
        data: Dict[str, Any] = None,
        importance: float = 0.5,
        emotion: str = "neutral",
    ) -> EpisodicMemoryEntry:
        entry = EpisodicMemoryEntry(
            timestamp=time.time(),
            cycle_id=0,
            agent_name=self.agent_name,
            event_type=event_type,
            description=description,
            data=data or {},
            importance=importance,
            emotion=emotion,
        )
        self._episodic.append(entry)
        self._episodic_count += 1
        idx = len(self._episodic) - 1
        self._type_index[event_type].append(idx)

        if importance >= self.consolidation_threshold:
            self._consolidate(entry)

        return entry

    def recall(
        self,
        event_type: Optional[str] = None,
        n: int = 10,
        min_importance: float = 0.0,
    ) -> List[EpisodicMemoryEntry]:
        if event_type:
            indices = self._type_index.get(event_type, [])
            entries = [self._episodic[i] for i in indices if i < len(self._episodic)]
        else:
            entries = list(self._episodic)
        entries = [e for e in entries if e.importance >= min_importance]
        return sorted(entries, key=lambda e: e.timestamp, reverse=True)[:n]

    def recall_recent(self, n: int = 5) -> List[EpisodicMemoryEntry]:
        return list(self._episodic)[-n:] if self._episodic else []

    def count_events(self, event_type: str) -> int:
        return len(self._type_index.get(event_type, []))

    # ------------------------------------------------------------------
    # Semantic Memory
    # ------------------------------------------------------------------

    def know(
        self,
        key: str,
        value: Any,
        confidence: float = 1.0,
        source: str = "",
        ttl: float = 86400,
    ):
        self._semantic[key] = SemanticMemoryEntry(
            key=key,
            value=value,
            confidence=confidence,
            source=source,
            ttl_seconds=ttl,
        )

    def recall_knowledge(self, key: str, default: Any = None) -> Any:
        entry = self._semantic.get(key)
        if entry is None or entry.is_expired:
            return default
        return entry.value

    def forget(self, key: str):
        self._semantic.pop(key, None)

    def query(self, query: str, top_k: int = 5) -> List[Dict]:
        """G12: Query memory by keyword matching across episodic + semantic."""
        results = []
        query_lower = query.lower()

        # Search episodic
        for entry in self._episodic:
            if (
                query_lower in entry.event_type.lower()
                or query_lower in entry.description.lower()
            ):
                results.append(
                    {
                        "type": "episodic",
                        "data": entry.to_dict(),
                        "score": entry.importance,
                    }
                )
                if len(results) >= top_k:
                    break

        # Search semantic
        for key, entry in self._semantic.items():
            if query_lower in key.lower():
                results.append(
                    {
                        "type": "semantic",
                        "data": {
                            "key": key,
                            "value": str(entry.value)[:200],
                            "confidence": entry.confidence,
                            "source": entry.source,
                        },
                        "score": entry.confidence,
                    }
                )
                if len(results) >= top_k:
                    break

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    # G12: Store knowledge from other agents
    def learn_from_agent(self, source_agent: str, key: str, value: Any):
        self._external_knowledge[source_agent][key] = value
        self._external_knowledge_timestamps[f"{source_agent}:{key}"] = time.time()

    def recall_external(self, source_agent: str, key: str, default: Any = None) -> Any:
        return self._external_knowledge.get(source_agent, {}).get(key, default)

    def get_all_external(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._external_knowledge)

    # ------------------------------------------------------------------
    # Working Memory
    # ------------------------------------------------------------------

    def focus(self, key: str, value: Any):
        self._working[key] = value
        self._working_timestamp = time.time()

    def recall_focus(self, key: str, default: Any = None) -> Any:
        return self._working.get(key, default)

    def clear_focus(self):
        self._working.clear()

    # ------------------------------------------------------------------
    # G21: Consolidation
    # ------------------------------------------------------------------

    def _consolidate(self, entry: EpisodicMemoryEntry):
        key = f"memory:{entry.event_type}:{int(entry.timestamp)}"
        existing = self._semantic.get(key)
        if existing:
            existing.updated_at = time.time()
            existing.confidence = min(1.0, existing.confidence + 0.1)
        else:
            self._semantic[key] = SemanticMemoryEntry(
                key=key,
                value=entry.to_dict(),
                confidence=entry.importance,
                source="episodic_consolidation",
            )

    def consolidate_semantic(self):
        """G21: Promote stable patterns to semantic memory."""
        recent = self.recall(n=100)
        if len(recent) < 10:
            return
        from collections import Counter

        type_counts = Counter(e.event_type for e in recent)
        for event_type, count in type_counts.most_common(5):
            if count >= 5:
                avg_importance = float(
                    np.mean(
                        [e.importance for e in recent if e.event_type == event_type]
                    )
                )
                self.know(
                    key=f"pattern:{event_type}",
                    value={"count": count, "avg_importance": round(avg_importance, 3)},
                    confidence=min(1.0, count / 20),
                    source="memory_consolidation",
                    ttl=86400 * 7,
                )

    def get_knowledge_broadcast(self) -> Dict[str, Any]:
        """G21: Get high-confidence semantic knowledge to share with other agents."""
        return {
            k: {"value": v.value, "confidence": v.confidence}
            for k, v in self._semantic.items()
            if v.confidence > 0.7
            and not k.startswith("memory:")  # share only learned facts
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self):
        if not self.persist_path:
            return
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "episodic": [e.to_dict() for e in self._episodic],
                "semantic": {
                    k: {
                        "value": v.value,
                        "confidence": v.confidence,
                        "source": v.source,
                        "updated_at": v.updated_at,
                    }
                    for k, v in self._semantic.items()
                },
                "external": {k: dict(v) for k, v in self._external_knowledge.items()},
            }
            with open(self.persist_path.with_suffix(".mem.json"), "w") as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"[{self.agent_name}] Memory save failed: {e}")

    def load(self):
        if not self.persist_path:
            return
        try:
            path = self.persist_path.with_suffix(".mem.json")
            if not path.exists():
                return
            with open(path) as f:
                state = json.load(f)
            for entry_data in state.get("episodic", []):
                self.remember(
                    event_type=entry_data.get("type", "unknown"),
                    description=entry_data.get("desc", ""),
                    importance=entry_data.get("importance", 0.5),
                    emotion=entry_data.get("emotion", "neutral"),
                )
            for key, data in state.get("semantic", {}).items():
                self._semantic[key] = SemanticMemoryEntry(
                    key=key,
                    value=data["value"],
                    confidence=data["confidence"],
                    source=data.get("source", ""),
                    updated_at=data.get("updated_at", time.time()),
                )
            for agent, knowledge in state.get("external", {}).items():
                self._external_knowledge[agent] = knowledge
        except Exception as e:
            logger.warning(f"[{self.agent_name}] Memory load failed: {e}")

    def summary(self) -> Dict:
        return {
            "episodic_count": len(self._episodic),
            "semantic_count": len(self._semantic),
            "external_sources": len(self._external_knowledge),
            "event_types": dict(self._type_index),
            "working_keys": list(self._working.keys()),
        }
