from typing import Dict, List
from .service_base import TradingService
from loguru import logger


class ServiceRegistry:
    """Manages lifecycle of all services."""

    def __init__(self):
        self._services: Dict[str, TradingService] = {}

    def register(self, service: TradingService) -> None:
        self._services[service.name] = service

    def get(self, name: str) -> TradingService:
        return self._services[name]

    @property
    def all(self) -> List[TradingService]:
        return list(self._services.values())

    async def start_all(self) -> None:
        for svc in self._services.values():
            logger.info(f"Starting {svc.name}...")
            await svc.start()
            logger.info(f"{svc.name} started")

    async def stop_all(self) -> None:
        for svc in reversed(list(self._services.values())):
            try:
                await svc.stop()
            except Exception as e:
                logger.error(f"Error stopping {svc.name}: {e}")
