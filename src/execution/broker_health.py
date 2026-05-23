import asyncio
import time
from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass
class ConnectionQuality:
    score: float = 100.0
    latency_ms: float = 0.0
    error_rate: float = 0.0
    fill_rate: float = 1.0
    consecutive_slow: int = 0


class BrokerHealthMonitor:
    def __init__(
        self,
        failover_threshold: int = 3,
        on_failover: Optional[Callable[[], None]] = None,
        slow_threshold_ms: float = 500.0,
    ):
        self.failover_threshold = failover_threshold
        self.on_failover = on_failover
        self.slow_threshold_ms = slow_threshold_ms
        self._latencies: List[float] = []
        self._errors = 0
        self._fills = 0
        self._misses = 0
        self._consecutive_slow = 0
        self._score = 100.0
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def record_heartbeat(self, latency_ms: float) -> None:
        self._latencies.append(latency_ms)
        if len(self._latencies) > 100:
            self._latencies.pop(0)
        if latency_ms > self.slow_threshold_ms:
            self._consecutive_slow += 1
        else:
            self._consecutive_slow = 0
        self._recalculate_score()

    def record_error(self) -> None:
        self._errors += 1
        self._recalculate_score()

    def record_fill(self, success: bool) -> None:
        if success:
            self._fills += 1
        else:
            self._misses += 1
        self._recalculate_score()

    def _recalculate_score(self) -> None:
        latency_penalty = 0.0
        if self._latencies:
            avg_latency = sum(self._latencies) / len(self._latencies)
            latency_penalty = min(avg_latency / 10.0, 30.0)

        total_events = self._errors + self._fills + self._misses
        error_rate = self._errors / total_events if total_events > 0 else 0.0
        error_penalty = min(error_rate * 50.0, 40.0)

        total_fills = self._fills + self._misses
        fill_rate = self._fills / total_fills if total_fills > 0 else 1.0
        fill_bonus = fill_rate * 10.0

        score = 100.0 - latency_penalty - error_penalty + fill_bonus
        self._score = max(0.0, min(100.0, score))

    def get_health_score(self) -> float:
        return self._score

    def should_failover(self) -> bool:
        return self._consecutive_slow >= self.failover_threshold

    def get_quality(self) -> ConnectionQuality:
        total_events = self._errors + self._fills + self._misses
        error_rate = self._errors / total_events if total_events > 0 else 0.0
        total_fills = self._fills + self._misses
        fill_rate = self._fills / total_fills if total_fills > 0 else 1.0
        avg_latency = (
            sum(self._latencies) / len(self._latencies) if self._latencies else 0.0
        )
        return ConnectionQuality(
            score=self._score,
            latency_ms=avg_latency,
            error_rate=error_rate,
            fill_rate=fill_rate,
            consecutive_slow=self._consecutive_slow,
        )

    async def monitor_loop(self) -> None:
        self._running = True
        while self._running:
            try:
                if self.should_failover():
                    if self.on_failover:
                        self.on_failover()
                    self._consecutive_slow = 0
            except Exception:
                pass
            await asyncio.sleep(1.0)

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
