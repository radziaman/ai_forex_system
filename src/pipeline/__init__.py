"""Pipeline architecture for RTS AI Forex Trading System.
Replaces the 20-agent swarm with 5 focused modules."""

from .orchestrator import Orchestrator
from .signal_engine import SignalEngine
from .expert_registry import ExpertRegistry
from .risk_manager import RiskManager
from .execution_manager import ExecutionManager
from .learning_manager import LearningManager
from .event_bus import EventBus
from .pipeline_context import PipelineContext

__all__ = [
    "Orchestrator",
    "SignalEngine",
    "ExpertRegistry",
    "RiskManager",
    "ExecutionManager",
    "LearningManager",
    "EventBus",
    "PipelineContext",
]
