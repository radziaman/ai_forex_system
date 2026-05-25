"""
Service Container — lightweight dependency injection for the agentic system.

Allows agents to request dependencies without knowing how they're created,
making them testable without full system bootstrap.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, Optional, TypeVar

T = TypeVar("T")


class ServiceNotFoundError(Exception):
    pass


class ServiceContainer:
    """Lightweight DI container.

    Services are registered by type/name and lazily constructed.
    Supports singleton and factory lifetimes.
    """

    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, bool] = {}

    def register(
        self,
        name: str,
        instance: Any = None,
        factory: Optional[Callable] = None,
        singleton: bool = True,
    ):
        """Register a service by name.

        Args:
            name: Service identifier (e.g. "data_manager", "ensemble")
            instance: Pre-created instance (for tests)
            factory: Factory function that creates the service
            singleton: If True, factory is called once and cached
        """
        if instance is not None:
            self._services[name] = instance
            self._singletons[name] = True
        elif factory is not None:
            self._factories[name] = factory
            self._singletons[name] = singleton
        else:
            raise ValueError(f"Must provide instance or factory for '{name}'")

    def get(self, name: str, default: Any = None) -> Any:
        """Get a service by name.

        If registered as singleton with factory, creates once and caches.
        If registered as factory-only, creates each time.
        If not found, returns default or raises ServiceNotFoundError.
        """
        # Return cached singleton
        if name in self._services:
            return self._services[name]

        # Create from factory
        if name in self._factories:
            instance = self._factories[name]()
            if self._singletons.get(name, True):
                self._services[name] = instance
            return instance

        if default is not None:
            return default

        raise ServiceNotFoundError(f"Service '{name}' not registered")

    def has(self, name: str) -> bool:
        return name in self._services or name in self._factories

    def clear(self):
        """Clear all services (useful in tests)."""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()


# Global singleton
_container: Optional[ServiceContainer] = None


def get_container() -> ServiceContainer:
    global _container
    if _container is None:
        _container = ServiceContainer()
    return _container


def reset_container():
    """Reset the container (useful in tests)."""
    global _container
    _container = None
