"""
Flash Crash Detection & Circuit Breakers.
Prevents trading during disorderly market conditions to avoid catastrophic losses.
Enhanced with API recovery, confidence thresholds, and graceful degradation modes.
"""
import numpy as np
import time
import asyncio
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class DegradationMode(Enum):
    """System degradation modes for graceful degradation."""
    NORMAL = "normal"
    DEGRADED = "degraded"  # Some features disabled
    MINIMAL = "minimal"  # Only essential features
    HALTED = "halted"  # Full stop


@dataclass
class MarketStressSnapshot:
    timestamp: float = field(default_factory=time.time)
    is_healthy: bool = True
    should_halt: bool = False
    stress_level: str = "normal"  # normal | elevated | high | extreme
    halt_reason: str = ""
    price_velocity: float = 0.0
    spread_ratio: float = 1.0
    volume_anomaly: float = 0.0
    volatility_spike: bool = False
    degradation_mode: DegradationMode = DegradationMode.NORMAL
    confidence_threshold: float = 0.5  # Minimum confidence to trade
    api_health: Dict[str, bool] = field(default_factory=dict)
    recovery_attempts: int = 0


class CircuitBreaker:
    """
    Detect market stress and halt trading.
    Multiple detectors: price velocity, liquidity drought, volume anomalies.
    Enhanced with API recovery, confidence thresholds, and graceful degradation.
    """

    def __init__(
        self,
        price_velocity_threshold: float = 0.005,  # 0.5% in one tick
        spread_multiplier_threshold: float = 5.0,  # 5x normal spread
        volume_spike_multiplier: float = 10.0,  # 10x normal volume
        volatility_lookback: int = 20,
        cooldown_seconds: int = 300,  # 5 min cooldown after halt
        api_timeout_seconds: int = 30,  # API timeout before degradation
        max_api_retries: int = 3,
    ):
        self.price_velocity_threshold = price_velocity_threshold
        self.spread_multiplier_threshold = spread_multiplier_threshold
        self.volume_spike_multiplier = volume_spike_multiplier
        self.volatility_lookback = volatility_lookback
        self.cooldown_seconds = cooldown_seconds
        self.api_timeout_seconds = api_timeout_seconds
        self.max_api_retries = max_api_retries

        self.price_history: Dict[str, List[float]] = {}
        self.spread_history: Dict[str, List[float]] = {}
        self.volume_history: Dict[str, List[float]] = {}
        self.normal_spreads: Dict[str, float] = {}
        self.normal_volume: Dict[str, float] = {}
        self.last_halt_time: Dict[str, float] = {}
        self.volatility_history: Dict[str, List[float]] = {}

        # API health tracking
        self.api_health: Dict[str, Dict] = {}  # api_name -> {status, last_success, failures}
        self.current_degradation_mode: DegradationMode = DegradationMode.NORMAL
        self._degradation_history: List[Tuple[float, DegradationMode, str]] = []

        # Confidence threshold adjustment based on market stress
        self.base_confidence_threshold: float = 0.65
        self.current_confidence_threshold: float = self.base_confidence_threshold

    def check_market_health(
        self, symbol: str, tick: Dict
    ) -> Tuple[bool, str, MarketStressSnapshot]:
        """
        Return (should_halt, reason, snapshot).
        Checks multiple stress indicators.
        Enhanced with degradation mode and confidence adjustment.
        """
        snapshot = MarketStressSnapshot()
        snapshot.timestamp = time.time()
        snapshot.degradation_mode = self.current_degradation_mode
        snapshot.confidence_threshold = self.current_confidence_threshold
        self._last_snapshot = snapshot

        bid = tick.get("bid",0)
        ask = tick.get("ask",0)
        price = tick.get("price", (bid + ask) / 2.0)
        volume = tick.get("volume",0)
        spread = ask - bid if ask > bid else 0.0

        # Initialize history if needed
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            self.spread_history[symbol] = []
            self.volume_history[symbol] = []
            self.volatility_history[symbol] = []

        # Update histories
        self.price_history[symbol].append(price)
        self.spread_history[symbol].append(spread)
        self.volume_history[symbol].append(volume)

        # Trim histories
        max_len = max(100, self.volatility_lookback * 2)
        for hist in [self.price_history[symbol], self.spread_history[symbol],
                     self.volume_history[symbol], self.volatility_history[symbol]]:
            if len(hist) > max_len:
                hist = hist[-max_len:]

        # Update normal levels (rolling average, excluding extremes)
        self._update_normal_levels(symbol)

        # Check 1: Price velocity (flash crash detection)
        halt, reason = self._check_price_velocity(symbol, snapshot)
        if halt:
            snapshot.should_halt = True
            snapshot.halt_reason = reason
            self._update_degradation(DegradationMode.HALTED, reason)
            return True, reason, snapshot

        # Check 2: Liquidity drought (spread widening)
        halt, reason = self._check_liquidity(symbol, spread, snapshot)
        if halt:
            snapshot.should_halt = True
            snapshot.halt_reason = reason
            self._update_degradation(DegradationMode.DEGRADED, reason)
            return True, reason, snapshot

        # Check 3: Volume anomaly (potential manipulation)
        halt, reason = self._check_volume_anomaly(symbol, volume, snapshot)
        if halt:
            snapshot.should_halt = True
            snapshot.halt_reason = reason
            self._update_degradation(DegradationMode.DEGRADED, reason)
            return True, reason, snapshot

        # Check 4: Volatility spike
        halt, reason = self._check_volatility_spike(symbol, snapshot)
        if halt:
            snapshot.should_halt = True
            snapshot.halt_reason = reason
            self._update_degradation(DegradationMode.DEGRADED, reason)
            return True, reason, snapshot

        # Check cooldown period
        if symbol in self.last_halt_time:
            elapsed = time.time() - self.last_halt_time[symbol]
            if elapsed < self.cooldown_seconds:
                return True, f"cooldown_{elapsed:.0f}s_remaining", snapshot

        # Update degradation mode based on overall health
        if self.current_degradation_mode != DegradationMode.NORMAL:
            self._attempt_recovery()

        snapshot.is_healthy = True
        return False, "market_healthy", snapshot

    def update_api_health(self, api_name: str, success: bool):
        """Track API health for circuit breaker decisions."""
        if api_name not in self.api_health:
            self.api_health[api_name] = {
                "failures": 0,
                "last_success": time.time(),
                "status": "healthy",
            }

        if success:
            self.api_health[api_name]["failures"] = 0
            self.api_health[api_name]["last_success"] = time.time()
            self.api_health[api_name]["status"] = "healthy"
        else:
            self.api_health[api_name]["failures"] += 1
            if self.api_health[api_name]["failures"] >= self.max_api_retries:
                self.api_health[api_name]["status"] = "unhealthy"
                self._update_degradation(
                    DegradationMode.DEGRADED,
                    f"api_{api_name}_unhealthy"
                )

    def _update_degradation(self, mode: DegradationMode, reason: str):
        """Update system degradation mode."""
        if mode.value == "halted":
            self.current_degradation_mode = DegradationMode.HALTED
        elif mode.value == "degraded" and self.current_degradation_mode == DegradationMode.NORMAL:
            self.current_degradation_mode = DegradationMode.DEGRADED
            # Increase confidence threshold in degraded mode
            self.current_confidence_threshold = min(self.base_confidence_threshold * 1.5, 0.95)
        elif mode.value == "normal":
            self.current_degradation_mode = DegradationMode.NORMAL
            self.current_confidence_threshold = self.base_confidence_threshold

        self._degradation_history.append((time.time(), mode, reason))
        if len(self._degradation_history) > 100:
            self._degradation_history = self._degradation_history[-100:]

        logger.warning(f"Degradation mode: {mode.value} - {reason}")

    def _attempt_recovery(self):
        """Attempt to recover from degraded state."""
        all_healthy = all(
            info["status"] == "healthy"
            for info in self.api_health.values()
        )

        if all_healthy and self.current_degradation_mode != DegradationMode.HALTED:
            # Check if we've been healthy for long enough
            recent_history = [h for h in self._degradation_history if time.time() - h[0] < 300]
            if not recent_history or all(h[1] == DegradationMode.NORMAL for h in recent_history):
                self._update_degradation(DegradationMode.NORMAL, "recovery_success")
                logger.info("System recovered from degraded state")

    def _check_price_velocity(self, symbol: str, snapshot: MarketStressSnapshot) -> Tuple[bool, str]:
        prices = self.price_history[symbol]
        if len(prices) < 2:
            return False, ""

        # Calculate recent returns
        returns = np.diff(prices[-10:]) / prices[-11:-1]
        if len(returns) == 0:
            return False, ""

        max_return = np.max(np.abs(returns))
        snapshot.price_velocity = max_return

        if max_return > self.price_velocity_threshold:
            reason = f"flash_crash_velocity_{max_return:.4f}"
            snapshot.stress_level = "extreme"
            logger.error(f"CIRCUIT BREAKER: {symbol} flash crash detected! Velocity={max_return:.4f}")
            self.last_halt_time[symbol] = time.time()
            return True, reason

        return False, ""

    def _check_liquidity(self, symbol: str, spread: float, snapshot: MarketStressSnapshot) -> Tuple[bool, str]:
        normal = self.normal_spreads.get(symbol, spread)
        if normal <= 0:
            normal = spread

        ratio = spread / normal
        snapshot.spread_ratio = ratio

        if ratio > self.spread_multiplier_threshold:
            reason = f"liquidity_drought_spread_{ratio:.1f}x_normal"
            snapshot.stress_level = "high" if ratio > 10 else "elevated"
            logger.warning(f"CIRCUIT BREAKER: {symbol} liquidity drought! Spread {ratio:.1f}x normal")
            self.last_halt_time[symbol] = time.time()
            return True, reason

        return False, ""

    def _check_volume_anomaly(self, symbol: str, volume: float, snapshot: MarketStressSnapshot) -> Tuple[bool, str]:
        normal = self.normal_volume.get(symbol, volume)
        if normal <= 0:
            normal = volume

        ratio = volume / normal if normal > 0 else 1.0
        snapshot.volume_anomaly = ratio

        if ratio > self.volume_spike_multiplier:
            reason = f"volume_anomaly_{ratio:.1f}x_normal"
            snapshot.stress_level = "high"
            logger.warning(f"CIRCUIT BREAKER: {symbol} volume anomaly! {ratio:.1f}x normal")
            self.last_halt_time[symbol] = time.time()
            return True, reason

        return False, ""

    def _check_volatility_spike(self, symbol: str, snapshot: MarketStressSnapshot) -> Tuple[bool, str]:
        prices = self.price_history[symbol]
        if len(prices) < self.volatility_lookback + 1:
            return False, ""

        recent = prices[-self.volatility_lookback:]
        returns = np.diff(recent) / recent[:-1]
        current_vol = np.std(returns)

        self.volatility_history[symbol].append(current_vol)

        if len(self.volatility_history[symbol]) < 10:
            return False, ""

        avg_vol = np.mean(self.volatility_history[symbol][-20:])
        if avg_vol <= 0:
            return False, ""

        vol_ratio = current_vol / avg_vol
        snapshot.volatility_spike = vol_ratio > 3.0

        if vol_ratio > 5.0:
            reason = f"volatility_spike_{vol_ratio:.1f}x_normal"
            snapshot.stress_level = "extreme"
            logger.error(f"CIRCUIT BREAKER: {symbol} volatility spike! {vol_ratio:.1f}x normal")
            self.last_halt_time[symbol] = time.time()
            return True, reason

        return False, ""

    def _update_normal_levels(self, symbol: str):
        """Update normal spread/volume using median (robust to outliers)."""
        spreads = self.spread_history[symbol]
        if len(spreads) > 20:
            self.normal_spreads[symbol] = np.median(spreads[-50:])

        volumes = self.volume_history[symbol]
        if len(volumes) > 20:
            self.normal_volume[symbol] = np.median(volumes[-50:])

    def force_resume(self, symbol: str):
        """Manually resume trading after halt."""
        if symbol in self.last_halt_time:
            del self.last_halt_time[symbol]
        logger.info(f"CIRCUIT BREAKER: {symbol} manually resumed")

    def get_stress_summary(self) -> Dict:
        """Get summary of market stress across all symbols."""
        summary = {}
        for symbol in self.price_history.keys():
            if symbol in self.last_halt_time:
                elapsed = time.time() - self.last_halt_time[symbol]
                summary[symbol] = {
                    "halted": elapsed < self.cooldown_seconds,
                    "cooldown_remaining": max(0, self.cooldown_seconds - elapsed),
                }
        return summary

    def get_snapshot(self):
        """Return last market health snapshot for dashboard."""
        return getattr(self, '_last_snapshot', MarketStressSnapshot())
