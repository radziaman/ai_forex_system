"""Tests for position_reconciler module."""

import asyncio
from execution.position_reconciler import PositionReconciler, ReconciliationDiff


def test_reconcile_detects_missing_position():
    internal = [
        {"symbol": "EURUSD", "direction": "BUY", "volume": 1.0},
    ]
    broker = []

    reconciler = PositionReconciler(
        get_internal_positions=lambda: internal,
        get_broker_positions=lambda: broker,
    )
    diff = reconciler.reconcile(internal, broker)

    assert len(diff.missing) == 1
    assert diff.missing[0]["symbol"] == "EURUSD"
    assert len(diff.extra) == 0
    assert len(diff.mismatched) == 0


def test_reconcile_detects_mismatch():
    internal = [
        {"symbol": "EURUSD", "direction": "BUY", "volume": 1.0},
    ]
    broker = [
        {"symbol": "EURUSD", "direction": "BUY", "volume": 1.5},
    ]

    reconciler = PositionReconciler(
        get_internal_positions=lambda: internal,
        get_broker_positions=lambda: broker,
    )
    diff = reconciler.reconcile(internal, broker)

    assert len(diff.missing) == 0
    assert len(diff.extra) == 0
    assert len(diff.mismatched) == 1
    assert diff.mismatched[0]["internal"]["volume"] == 1.0
    assert diff.mismatched[0]["broker"]["volume"] == 1.5


def test_reconcile_aligned_positions():
    internal = [
        {"symbol": "EURUSD", "direction": "BUY", "volume": 1.0},
        {"symbol": "GBPUSD", "direction": "SELL", "volume": 2.0},
    ]
    broker = [
        {"symbol": "EURUSD", "direction": "BUY", "volume": 1.005},
        {"symbol": "GBPUSD", "direction": "SELL", "volume": 1.99},
    ]

    reconciler = PositionReconciler(
        get_internal_positions=lambda: internal,
        get_broker_positions=lambda: broker,
    )
    diff = reconciler.reconcile(internal, broker)

    assert len(diff.missing) == 0
    assert len(diff.extra) == 0
    assert len(diff.mismatched) == 0


def test_reconcile_volume_within_one_percent():
    internal = [
        {"symbol": "EURUSD", "direction": "BUY", "volume": 100.0},
    ]
    broker = [
        {"symbol": "EURUSD", "direction": "BUY", "volume": 100.99},
    ]

    reconciler = PositionReconciler(
        get_internal_positions=lambda: internal,
        get_broker_positions=lambda: broker,
    )
    diff = reconciler.reconcile(internal, broker)
    assert len(diff.mismatched) == 0


def test_reconcile_volume_just_over_one_percent():
    internal = [
        {"symbol": "EURUSD", "direction": "BUY", "volume": 100.0},
    ]
    broker = [
        {"symbol": "EURUSD", "direction": "BUY", "volume": 101.1},
    ]

    reconciler = PositionReconciler(
        get_internal_positions=lambda: internal,
        get_broker_positions=lambda: broker,
    )
    diff = reconciler.reconcile(internal, broker)
    assert len(diff.mismatched) == 1


def test_on_mismatch_callback_fires():
    calls = []

    def callback(diff: ReconciliationDiff):
        calls.append(diff)

    internal = [
        {"symbol": "EURUSD", "direction": "BUY", "volume": 1.0},
    ]
    broker = []

    reconciler = PositionReconciler(
        get_internal_positions=lambda: internal,
        get_broker_positions=lambda: broker,
        on_mismatch=callback,
    )
    reconciler.reconcile(internal, broker)

    assert len(calls) == 1
    assert calls[0].missing[0]["symbol"] == "EURUSD"


def test_get_last_diff():
    internal = [
        {"symbol": "EURUSD", "direction": "BUY", "volume": 1.0},
    ]
    broker = []

    reconciler = PositionReconciler(
        get_internal_positions=lambda: internal,
        get_broker_positions=lambda: broker,
    )
    assert reconciler.get_last_diff() is None
    diff = reconciler.reconcile(internal, broker)
    assert reconciler.get_last_diff() is diff


def test_reconcile_loop_runs_and_stops():
    calls = []

    def callback(diff: ReconciliationDiff):
        calls.append(diff)

    reconciler = PositionReconciler(
        get_internal_positions=lambda: [],
        get_broker_positions=lambda: [],
        on_mismatch=callback,
    )

    async def run_once():
        reconciler._running = True
        # reconcile_loop sleeps 30s; we can't wait that long.
        # Instead, manually run one iteration and stop.
        internal = reconciler.get_internal_positions()
        broker = reconciler.get_broker_positions()
        reconciler.reconcile(internal, broker)
        reconciler.stop()

    asyncio.run(run_once())
    assert len(calls) == 0  # No mismatch for empty lists
