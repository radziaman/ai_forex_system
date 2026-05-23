"""Alpha strategies package for the RTS AI Forex Trading System."""

from typing import Dict, List, Type


class StrategyRegistry:
    """Central registry for alpha strategy classes.

    Usage:
        registry = StrategyRegistry()
        registry.register("stat_arb", StatArbStrategy)
        strategy = registry.get("stat_arb")("EURUSD", {})
    """

    _instance: "StrategyRegistry" = None  # type: ignore[assignment]
    _strategies: Dict[str, Type]
    _regime_map: Dict[str, List[str]]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._strategies = {}
            cls._instance._regime_map = {}
        return cls._instance

    def register(self, name: str, strategy_class: Type):
        """Register a strategy class by name."""
        self._strategies[name] = strategy_class

    def get(self, name: str):
        """Return a strategy class or None."""
        return self._strategies.get(name)

    def get_for_regime(self, regime: str) -> List[str]:
        """Return names of strategies suitable for a given regime."""
        return self._regime_map.get(regime, list(self._strategies.keys()))

    def get_all(self) -> Dict[str, Type]:
        """Return all registered strategies."""
        return dict(self._strategies)

    def map_regime(self, regime: str, strategy_names: List[str]):
        """Associate strategies with a regime."""
        self._regime_map[regime] = strategy_names

    def clear(self):
        """Clear all registrations (mainly for tests)."""
        self._strategies.clear()
        self._regime_map.clear()
