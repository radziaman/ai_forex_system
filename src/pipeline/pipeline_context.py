"""PipelineContext — DI container for pipeline modules.

Instead of global singletons (old approach), each pipeline module
receives a PipelineContext with all its dependencies.
"""

from dataclasses import dataclass, field
from typing import Optional, Any

from .event_bus import EventBus
from infrastructure.config import AppConfig
from infrastructure.secrets import Secrets
from data.data_manager import DataManager


@dataclass
class PipelineContext:
    """Shared context for all pipeline modules."""

    config: AppConfig
    secrets: Secrets
    bus: EventBus = field(default_factory=EventBus)
    data_manager: Optional[DataManager] = None

    # Lazy-loaded services (populated on first use)
    ensemble: Optional[Any] = None
    feature_pipeline: Optional[Any] = None
    regime_detector: Optional[Any] = None
    risk_manager_service: Optional[Any] = None
    execution_engine: Optional[Any] = None
    attribution_manager: Optional[Any] = None
