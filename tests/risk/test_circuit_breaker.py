"""Tests for CircuitBreaker — safety system that halts trading during market stress."""

import time

import numpy as np

from risk.circuit_breaker import CircuitBreaker, DegradationMode


def _send_tick(cb, symbol="EURUSD", bid=1.12, ask=1.1201, volume=100):
    """Send a tick to the circuit breaker and return (halt, reason, snapshot)."""
    tick = {"bid": bid, "ask": ask, "price": (bid + ask) / 2.0, "volume": volume}
    return cb.check_market_health(symbol, tick)


class TestCircuitBreakerInit:
    """Default and custom parameter initialization."""

    def test_init_defaults(self):
        cb = CircuitBreaker()
        assert cb.price_velocity_threshold == 0.005
        assert cb.spread_multiplier_threshold == 5.0
        assert cb.volume_spike_multiplier == 10.0
        assert cb.volatility_lookback == 20
        assert cb.cooldown_seconds == 300
        assert cb.api_timeout_seconds == 30
        assert cb.max_api_retries == 3
        assert cb.current_degradation_mode == DegradationMode.NORMAL
        assert cb.current_confidence_threshold == 0.65
        assert cb._warmup_period == 50
        assert cb._observation_count == 0
        assert cb.is_warmed_up is False

    def test_init_custom(self):
        cb = CircuitBreaker(
            price_velocity_threshold=0.01,
            spread_multiplier_threshold=10.0,
            volume_spike_multiplier=20.0,
            volatility_lookback=50,
            cooldown_seconds=600,
            api_timeout_seconds=60,
            max_api_retries=5,
            warmup_period=100,
        )
        assert cb.price_velocity_threshold == 0.01
        assert cb.spread_multiplier_threshold == 10.0
        assert cb.volume_spike_multiplier == 20.0
        assert cb.volatility_lookback == 50
        assert cb.cooldown_seconds == 600
        assert cb.api_timeout_seconds == 60
        assert cb.max_api_retries == 5
        assert cb._warmup_period == 100


class TestPriceVelocity:
    """Flash crash detection via sudden price movements."""

    def test_price_velocity_insufficient_data(self):
        cb = CircuitBreaker(warmup_period=0)
        # Need 11 prices for velocity check — 10 should be safe
        for _ in range(10):
            halt, reason, _ = _send_tick(cb)
            assert not halt, f"Unexpected halt at tick: {reason}"

    def test_price_velocity_normal(self):
        cb = CircuitBreaker(warmup_period=0)
        # 15 normal ticks should not trigger velocity halt
        for _ in range(15):
            halt, reason, _ = _send_tick(cb)
            assert not halt, f"Unexpected halt at tick: {reason}"

    def test_price_velocity_flash_crash(self):
        cb = CircuitBreaker(warmup_period=0)
        # 10 normal ticks to establish history
        for _ in range(10):
            _send_tick(cb)
        # 11th tick with sudden large drop >> 0.5%
        halt, reason, snapshot = _send_tick(cb, bid=1.11, ask=1.1101)
        assert halt, "Expected flash crash halt"
        assert "flash_crash" in reason
        assert snapshot.stress_level == "extreme"
        assert snapshot.price_velocity > cb.price_velocity_threshold


class TestLiquidityDrought:
    """Spread widening detection."""

    def test_liquidity_normal(self):
        cb = CircuitBreaker(warmup_period=0)
        for _ in range(5):
            halt, reason, _ = _send_tick(cb, bid=1.12, ask=1.1201)
            assert not halt, f"Unexpected halt: {reason}"

    def test_liquidity_widening(self):
        cb = CircuitBreaker(warmup_period=0)
        # Establish baseline with 50 normal ticks (spread=0.0001)
        for _ in range(50):
            _send_tick(cb, bid=1.12, ask=1.1201)
        # Spread 6x normal (ask widened to 1.1206)
        halt, reason, snapshot = _send_tick(cb, bid=1.12, ask=1.1206)
        assert halt, "Expected liquidity drought halt"
        assert "liquidity_drought" in reason
        assert snapshot.spread_ratio > 5.0
        assert snapshot.stress_level in ("elevated", "high")

    def test_liquidity_first_tick_no_baseline(self):
        cb = CircuitBreaker(warmup_period=0)
        # First tick with extreme spread — no baseline yet, ratio=1
        halt, reason, snapshot = _send_tick(cb, bid=1.12, ask=1.1210)
        assert not halt, "First tick should not trigger liquidity halt"
        assert snapshot.spread_ratio == 1.0


class TestVolumeAnomaly:
    """Volume spike detection."""

    def test_volume_normal(self):
        cb = CircuitBreaker(warmup_period=0)
        for _ in range(5):
            halt, reason, _ = _send_tick(cb, volume=100)
            assert not halt, f"Unexpected halt: {reason}"

    def test_volume_spike(self):
        cb = CircuitBreaker(warmup_period=0)
        # Establish baseline with 50 normal ticks
        for _ in range(50):
            _send_tick(cb, volume=100)
        # Volume 20x normal
        halt, reason, snapshot = _send_tick(cb, volume=2000)
        assert halt, "Expected volume anomaly halt"
        assert "volume_anomaly" in reason
        assert snapshot.volume_anomaly > 10.0


class TestVolatilitySpike:
    """Volatility regime change detection."""

    def test_volatility_normal(self):
        cb = CircuitBreaker(warmup_period=0)
        for _ in range(10):
            halt, reason, _ = _send_tick(cb)
            assert not halt, f"Unexpected halt: {reason}"

    def test_volatility_spike(self):
        cb = CircuitBreaker(warmup_period=0)
        # 30 baseline ticks with tiny noise to establish low-vol history.
        # Use a sine pattern so returns are non-zero but very small.
        for i in range(30):
            bid = 1.12 + 0.00005 * np.sin(i * 0.3)
            _send_tick(cb, bid=bid, ask=bid + 0.0001, volume=100)

        # Spike: 10 ticks alternating ±0.0005 around 1.12.
        # Each individual return is ~0.09% (< 0.5% flash-crash threshold),
        # but the rolling std of returns (~0.09%) dwarfs the baseline vol
        # (~4.5e-5), so vol_ratio >> 5.
        for i in range(10):
            bid = 1.12 + 0.0005 * (1 if i % 2 == 0 else -1)
            halt, reason, snapshot = _send_tick(
                cb, bid=bid, ask=bid + 0.0001, volume=100
            )
            if halt:
                assert "volatility_spike" in reason
                assert snapshot.volatility_spike
                return
        assert False, "Volatility spike not triggered within 10 high-vol ticks"


class TestCooldown:
    """Cooldown period after a halt."""

    def test_cooldown_active(self):
        cb = CircuitBreaker(cooldown_seconds=300, warmup_period=0)
        # Directly simulate a recent halt so we don't rely on detector ordering
        cb.last_halt_time["EURUSD"] = time.time()

        # Next tick should immediately hit the cooldown gate
        halt, reason, _ = _send_tick(cb)
        assert halt, "Expected cooldown halt"
        assert "cooldown" in reason

    def test_cooldown_expired(self):
        cb = CircuitBreaker(cooldown_seconds=300, warmup_period=0)
        # Place halt far enough in the past that cooldown has elapsed
        cb.last_halt_time["EURUSD"] = time.time() - 301

        halt, reason, _ = _send_tick(cb)
        assert not halt, "Cooldown should have expired"
        assert reason == "market_healthy"


class TestDegradation:
    """Graceful degradation and confidence threshold adjustments."""

    def test_degradation_normal_start(self):
        cb = CircuitBreaker(warmup_period=0)
        assert cb.current_degradation_mode == DegradationMode.NORMAL
        assert cb.current_confidence_threshold == 0.65

    def test_degradation_confidence_increases(self):
        cb = CircuitBreaker(warmup_period=0)
        # Establish baseline for liquidity check
        for _ in range(50):
            _send_tick(cb)
        # Trigger liquidity drought → sets DEGRADED, confidence 0.95
        _send_tick(cb, bid=1.12, ask=1.1206)
        assert cb.current_confidence_threshold == 0.95
        assert cb.current_degradation_mode == DegradationMode.DEGRADED


class TestWarmupPeriod:
    """Circuit breaker warm-up before detectors activate."""

    def test_default_warmup_period(self):
        cb = CircuitBreaker()
        assert cb._warmup_period == 50
        assert cb._observation_count == 0
        assert cb.is_warmed_up is False

    def test_returns_warmup_during_warmup(self):
        cb = CircuitBreaker(warmup_period=5)
        for i in range(4):
            halt, reason, snapshot = _send_tick(cb)
            assert not halt, f"Tick {i}: unexpected halt: {reason}"
            assert reason == "warmup", f"Tick {i}: expected 'warmup', got '{reason}'"
            assert snapshot.is_healthy

    def test_transitions_after_warmup(self):
        cb = CircuitBreaker(warmup_period=3)
        # Ticks 1-2: warmup (observation_count < warmup_period)
        for _ in range(2):
            halt, reason, _ = _send_tick(cb)
            assert reason == "warmup"
        # Tick 3: observation_count reaches warmup_period, logs completion,
        #          exits warm-up and runs detectors on same tick
        halt, reason, _ = _send_tick(cb)
        assert (
            reason != "warmup"
        ), "Should exit warm-up when observation_count >= warmup_period"
        assert cb._observation_count == 3
        assert cb.is_warmed_up is True

    def test_observation_counting(self):
        cb = CircuitBreaker(warmup_period=10)
        for i in range(5):
            _send_tick(cb)
            assert cb._observation_count == i + 1
            assert cb.is_warmed_up is False

        for i in range(5):
            _send_tick(cb)
        assert cb._observation_count == 10
        assert cb.is_warmed_up is True

    def test_invalid_data_not_counted(self):
        cb = CircuitBreaker(warmup_period=5)
        # Send ticks with bid=0, ask=0 → price becomes 0
        for _ in range(3):
            cb.check_market_health("EURUSD", {"bid": 0, "ask": 0, "volume": 0})
        assert cb._observation_count == 0, "Zero-price ticks should not count"


class TestApiHealth:
    """API health tracking for circuit breaker decisions."""

    def test_api_health_init(self):
        cb = CircuitBreaker()
        assert cb.api_health == {}

    def test_api_health_success(self):
        cb = CircuitBreaker()
        cb.update_api_health("oanda", True)
        assert cb.api_health["oanda"]["status"] == "healthy"
        assert cb.api_health["oanda"]["failures"] == 0

    def test_api_health_failures(self):
        cb = CircuitBreaker(max_api_retries=3)
        # One success to register the API
        cb.update_api_health("oanda", True)
        assert cb.api_health["oanda"]["status"] == "healthy"

        cb.update_api_health("oanda", False)  # failure 1
        cb.update_api_health("oanda", False)  # failure 2
        assert cb.api_health["oanda"]["status"] == "healthy"

        cb.update_api_health("oanda", False)  # failure 3 → unhealthy
        assert cb.api_health["oanda"]["status"] == "unhealthy"
        assert cb.api_health["oanda"]["failures"] == 3


class TestForceResume:
    """Manual resume after halt."""

    def test_force_resume_symbol(self):
        cb = CircuitBreaker(warmup_period=0)
        # Trigger a halt on EURUSD
        for _ in range(50):
            _send_tick(cb)
        _send_tick(cb, bid=1.12, ask=1.1206)
        assert "EURUSD" in cb.last_halt_time

        cb.force_resume("EURUSD")
        assert "EURUSD" not in cb.last_halt_time
        assert cb.current_degradation_mode == DegradationMode.NORMAL
        assert cb.current_confidence_threshold == 0.65

    def test_force_resume_all(self):
        cb = CircuitBreaker(warmup_period=0)
        # Trigger halt on EURUSD
        for _ in range(50):
            _send_tick(cb)
        _send_tick(cb, bid=1.12, ask=1.1206)

        # Also halt GBPUSD
        for _ in range(50):
            _send_tick(cb, symbol="GBPUSD")
        _send_tick(cb, symbol="GBPUSD", bid=1.12, ask=1.1206)
        assert "EURUSD" in cb.last_halt_time
        assert "GBPUSD" in cb.last_halt_time

        cb.force_resume("")  # empty string resumes all
        assert cb.last_halt_time == {}
        assert cb.current_degradation_mode == DegradationMode.NORMAL
        assert cb.current_confidence_threshold == 0.65
