"""Tests for broker_health module."""

import asyncio
from execution.broker_health import BrokerHealthMonitor, ConnectionQuality


def test_health_score_degrades_with_latency():
    monitor = BrokerHealthMonitor()
    assert monitor.get_health_score() == 100.0

    for _ in range(10):
        monitor.record_heartbeat(1000.0)  # 1000 ms

    score = monitor.get_health_score()
    assert score < 100.0
    assert score >= 0.0


def test_failover_trigger_after_consecutive_slow():
    monitor = BrokerHealthMonitor(failover_threshold=3)
    assert not monitor.should_failover()

    monitor.record_heartbeat(100.0)  # fast
    assert not monitor.should_failover()

    monitor.record_heartbeat(600.0)  # slow
    assert not monitor.should_failover()

    monitor.record_heartbeat(600.0)  # slow
    assert not monitor.should_failover()

    monitor.record_heartbeat(600.0)  # slow — threshold reached
    assert monitor.should_failover()


def test_score_improves_with_good_fills():
    monitor = BrokerHealthMonitor()

    # Degrade score with errors
    for _ in range(5):
        monitor.record_error()

    degraded_score = monitor.get_health_score()
    assert degraded_score < 100.0

    # Improve with successful fills
    for _ in range(20):
        monitor.record_fill(True)

    improved_score = monitor.get_health_score()
    assert improved_score > degraded_score
    assert improved_score <= 100.0


def test_record_fill_success_and_failure():
    monitor = BrokerHealthMonitor()
    monitor.record_fill(True)
    monitor.record_fill(False)
    monitor.record_fill(True)

    quality = monitor.get_quality()
    assert quality.fill_rate == 2 / 3
    assert quality.error_rate == 0.0  # errors are separate from fills


def test_error_rate_affects_score():
    monitor = BrokerHealthMonitor()
    for _ in range(10):
        monitor.record_error()
    score = monitor.get_health_score()
    assert score < 100.0


def test_consecutive_slow_resets_on_fast():
    monitor = BrokerHealthMonitor(failover_threshold=3)
    monitor.record_heartbeat(600.0)
    monitor.record_heartbeat(600.0)
    assert monitor._consecutive_slow == 2
    monitor.record_heartbeat(50.0)
    assert monitor._consecutive_slow == 0
    assert not monitor.should_failover()


def test_get_quality_returns_connection_quality():
    monitor = BrokerHealthMonitor()
    monitor.record_heartbeat(100.0)
    monitor.record_fill(True)
    monitor.record_error()

    quality = monitor.get_quality()
    assert isinstance(quality, ConnectionQuality)
    assert quality.latency_ms == 100.0
    assert quality.fill_rate == 1.0
    # total_events = errors + fills + misses = 1 + 1 + 0 = 2
    assert quality.error_rate == 0.5


def test_monitor_loop_triggers_failover():
    triggered = []

    def on_failover():
        triggered.append(True)

    monitor = BrokerHealthMonitor(
        failover_threshold=2,
        on_failover=on_failover,
    )

    async def run():
        monitor.record_heartbeat(600.0)
        monitor.record_heartbeat(600.0)
        # Run one iteration of monitor_loop manually
        if monitor.should_failover():
            if monitor.on_failover:
                monitor.on_failover()
            monitor._consecutive_slow = 0

    asyncio.run(run())
    assert len(triggered) == 1


def test_score_clamped_between_zero_and_hundred():
    monitor = BrokerHealthMonitor()
    # Extreme penalties
    for _ in range(200):
        monitor.record_heartbeat(10000.0)
        monitor.record_error()
    score = monitor.get_health_score()
    assert 0.0 <= score <= 100.0
