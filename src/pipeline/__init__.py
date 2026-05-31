"""Pipeline architecture for RTS AI Forex Trading System.
Replaces the 20-agent swarm with a clean multi-module EventBus pipeline."""

from .orchestrator import Orchestrator
from .signal_engine import SignalEngine
from .expert_registry import ExpertRegistry
from .risk_manager import RiskManager
from .execution_manager import ExecutionManager
from .learning_manager import LearningManager
from .event_bus import EventBus
from .health_monitor import HealthMonitor
from .symbol_discovery import SymbolDiscovery
from .strategy_discovery import StrategyDiscovery
from .pipeline_context import PipelineContext

__all__ = [
    "Orchestrator",
    "SignalEngine",
    "ExpertRegistry",
    "RiskManager",
    "ExecutionManager",
    "LearningManager",
    "EventBus",
    "HealthMonitor",
    "SymbolDiscovery",
    "StrategyDiscovery",
    "PipelineContext",
]
