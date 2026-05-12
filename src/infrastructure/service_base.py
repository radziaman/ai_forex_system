from abc import ABC, abstractmethod
from loguru import logger


class TradingService(ABC):
    """Base class for all trading system services."""

    def __init__(self, name: str):
        self.name = name
        self._running = False

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @property
    def is_running(self) -> bool:
        return self._running

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *args):
        await self.stop()
