"""Dashboard package for RTS AI Forex Trading System.

Exposes the unified dashboard FastAPI app with AgentBus integration.
"""

from .unified_dashboard import app, update_state, dashboard_state

__all__ = ["app", "update_state", "dashboard_state"]
