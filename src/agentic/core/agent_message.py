"""
Agent Message Protocol — mathematically typed communication between agents.
Every message carries identity, intent, context, causal chain, delivery guarantees, and payload schemas.  # noqa: E501
"""

from __future__ import annotations
import time
import uuid
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Tuple, Union
from enum import Enum, auto


class MessagePriority(Enum):
    DEBUG = 0
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class MessageType(Enum):
    # Lifecycle
    AGENT_HEARTBEAT = auto()
    AGENT_STATE_CHANGE = auto()
    AGENT_ERROR = auto()
    AGENT_REQUEST = auto()
    AGENT_RESPONSE = auto()

    # Market data
    TICK_RECEIVED = auto()
    BAR_CLOSED = auto()
    FEATURES_READY = auto()
    REGIME_CHANGED = auto()

    # Signals & Trading
    SIGNAL_GENERATED = auto()
    TRADE_DECISION = auto()
    EXECUTION_REQUEST = auto()
    EXECUTION_RESULT = auto()
    POSITION_OPENED = auto()
    POSITION_CLOSED = auto()
    POSITION_MODIFIED = auto()

    # Risk
    RISK_CHECK = auto()
    RISK_APPROVED = auto()
    RISK_REJECTED = auto()
    RISK_ALERT = auto()
    CIRCUIT_BREAKER = auto()
    KILL_SWITCH = auto()

    # System
    SYSTEM_STATE_CHANGE = auto()
    RECONFIGURATION = auto()
    TRAINING_REQUEST = auto()
    VALIDATION_REQUEST = auto()
    VALIDATION_RESULT = auto()
    MODEL_UPDATE = auto()
    PERSISTENCE_REQUEST = auto()
    DIAGNOSTIC_REQUEST = auto()
    DIAGNOSTIC_RESULT = auto()

    # Agent communication
    AGENT_QUERY = auto()
    AGENT_DIRECTIVE = auto()
    AGENT_COLLABORATE = auto()
    AGENT_LEARN = auto()

    # Instrument Screening
    INSTRUMENTS_UPDATED = auto()
    SCREENING_REQUEST = auto()
    SCREENING_RESULT = auto()

    # Delivery
    MESSAGE_ACK = auto()
    MEMORY_QUERY = auto()
    MEMORY_RESPONSE = auto()
    METRICS_REPORT = auto()


# --- Payload Schemas ---
# Defines required fields and types for each MessageType payload

PAYLOAD_SCHEMAS: Dict[MessageType, Dict[str, Union[type, Tuple[type, ...]]]] = {
    MessageType.SIGNAL_GENERATED: {
        "symbol": str,
        "direction": str,
        "confidence": (int, float),
        "regime": str,
        "price": (int, float),
        "timestamp": (int, float),
    },
    MessageType.RISK_APPROVED: {
        "signal": dict,
        "volume": (int, float),
        "sl_price": (int, float),
        "tp_price": (int, float),
        "timestamp": (int, float),
    },
    MessageType.RISK_REJECTED: {
        "signal": dict,
        "reason": str,
        "timestamp": (int, float),
    },
    MessageType.EXECUTION_RESULT: {
        "success": bool,
        "symbol": str,
        "timestamp": (int, float),
    },
    MessageType.POSITION_OPENED: {
        "position_id": (int, str),
        "symbol": str,
        "direction": str,
        "volume": (int, float),
    },
    MessageType.POSITION_CLOSED: {
        "position_id": (int, str),
        "reason": str,
        "timestamp": (int, float),
    },
    MessageType.POSITION_MODIFIED: {
        "position_id": (int, str),
    },
    MessageType.TICK_RECEIVED: {
        "symbol": str,
        "bid": (int, float),
        "ask": (int, float),
        "volume": (int, float),
        "timestamp": (int, float),
    },
    MessageType.FEATURES_READY: {
        "symbol": str,
        "timestamp": (int, float),
    },
    MessageType.REGIME_CHANGED: {
        "from": str,
        "to": str,
        "timestamp": (int, float),
    },
    MessageType.RISK_ALERT: {
        "type": str,
        "reason": str,
    },
    MessageType.AGENT_DIRECTIVE: {
        "action": str,
        "reason": str,
    },
    MessageType.AGENT_ERROR: {
        "error": str,
        "source": str,
        "timestamp": (int, float),
    },
    MessageType.MEMORY_QUERY: {
        "query_type": str,
        "query": str,
    },
    MessageType.MEMORY_RESPONSE: {
        "query_id": str,
        "results": list,
    },
}


def validate_payload(msg_type: MessageType, payload: Any) -> Tuple[bool, str]:
    if msg_type not in PAYLOAD_SCHEMAS:
        return True, ""
    if not isinstance(payload, dict):
        return False, f"payload must be dict for {msg_type.name}"
    schema = PAYLOAD_SCHEMAS[msg_type]
    for field_name, expected_type in schema.items():
        if field_name not in payload:
            return False, f"missing required field '{field_name}' in {msg_type.name}"
        val = payload[field_name]
        if not isinstance(val, expected_type):
            return False, (
                f"field '{field_name}' in {msg_type.name}: "
                f"expected {expected_type}, got {type(val).__name__}"
            )
    return True, ""


@dataclass
class AgentIntention:
    """Why the agent sent this message — its reasoning chain."""

    primary_goal: str
    reasoning: str
    expected_outcome: str
    confidence: float = 1.0
    context_window: List[str] = field(default_factory=list)

    def explain(self) -> str:
        return (
            f"I want to {self.primary_goal}. "
            f"My reasoning: {self.reasoning}. "
            f"I expect: {self.expected_outcome} "
            f"(confidence: {self.confidence:.0%})."
        )


@dataclass
class AgentMessage:
    """
    Universal message protocol.
    Every inter-agent communication carries full context for autonomous decision-making.
    """

    msg_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    msg_type: MessageType = MessageType.AGENT_HEARTBEAT
    priority: MessagePriority = MessagePriority.NORMAL

    # Who
    source_agent: str = ""
    target_agent: str = ""  # "" = broadcast
    target_capability: str = ""  # G6: resolve via registry

    # What
    payload: Any = None
    payload_schema: str = ""

    # When
    created_at: float = field(default_factory=time.time)
    ttl_seconds: float = 60.0

    # Why
    intention: Optional[AgentIntention] = None

    # Traceability
    causal_parent_id: str = ""
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    hop_count: int = 0

    # Delivery (G5)
    requires_ack: bool = False
    ack_message_type: Optional[MessageType] = None
    delivery_ack: bool = False
    retry_count: int = 0
    max_retries: int = 3

    # Context
    agent_state_snapshot: Dict[str, Any] = field(default_factory=dict)
    world_state_snapshot: Dict[str, Any] = field(default_factory=dict)

    # Consistency (G26)
    checksum: str = ""

    def __post_init__(self):
        if self.payload is not None:
            self._compute_checksum()

    def _compute_checksum(self):
        try:
            raw = json.dumps(self.payload, sort_keys=True, default=str)
            self.checksum = hashlib.sha256(raw.encode()).hexdigest()[:16]
        except Exception:
            self.checksum = self.msg_id[:16]

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl_seconds

    @property
    def is_broadcast(self) -> bool:
        return not self.target_agent and not self.target_capability

    def explain(self) -> str:
        intention_text = (
            self.intention.explain() if self.intention else "no stated intention"
        )
        return (
            f"[{self.msg_type.name}] {self.source_agent} → "
            f"{self.target_agent or self.target_capability or '*'} "
            f"(#{self.msg_id[:8]}): {intention_text}"
        )

    def reply(
        self,
        payload: Any,
        msg_type: Optional[MessageType] = None,
        requires_ack: bool = False,
    ) -> AgentMessage:
        return AgentMessage(
            msg_type=msg_type or self.msg_type,
            source_agent=self.target_agent or "system",
            target_agent=self.source_agent,
            payload=payload,
            causal_parent_id=self.msg_id,
            conversation_id=self.conversation_id,
            hop_count=self.hop_count + 1,
            requires_ack=requires_ack,
        )

    def forward(self, target: str) -> AgentMessage:
        m = self.reply(self.payload)
        m.source_agent = self.source_agent
        m.target_agent = target
        m.hop_count = self.hop_count + 1
        return m

    def ack_message(self) -> AgentMessage:
        return AgentMessage(
            msg_type=MessageType.MESSAGE_ACK,
            source_agent="",
            target_agent=self.source_agent,
            payload={
                "ack_for": self.msg_id,
                "original_type": self.msg_type.name,
                "checksum": self.checksum,
                "timestamp": time.time(),
            },
            causal_parent_id=self.msg_id,
            conversation_id=self.conversation_id,
            priority=MessagePriority.DEBUG,
        )

    def __repr__(self) -> str:
        return (
            f"<{self.msg_type.name} #{self.msg_id[:8]} "
            f"{self.source_agent}→{self.target_agent or self.target_capability or '*'} "
            f"hop={self.hop_count} "
            f"{'ACK' if self.requires_ack else 'NOACK'}>"
        )
