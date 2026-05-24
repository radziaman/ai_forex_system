"""Tests for Implementation Shortfall Execution Engine."""

from execution.is_execution import (
    ISExecutionEngine,
    ExecutionPlan,
    ExecutionUrgency,
)


class TestISExecutionEngine:
    def test_plan_execution_returns_plan(self):
        engine = ISExecutionEngine()
        plan = engine.plan_execution(volume=100_000, price=1.10)
        assert isinstance(plan, ExecutionPlan)
        assert plan.total_volume == 100_000

    def test_plan_execution_slices_sum_to_volume(self):
        engine = ISExecutionEngine()
        plan = engine.plan_execution(volume=50_000, price=1.10)
        total = sum(s.volume for s in plan.slices)
        assert abs(total - 50_000) < 1.0

    def test_urgency_determination_high_confidence(self):
        engine = ISExecutionEngine()
        assert engine._determine_urgency(0.90) == ExecutionUrgency.URGENT
        assert engine._determine_urgency(0.75) == ExecutionUrgency.HIGH
        assert engine._determine_urgency(0.60) == ExecutionUrgency.MEDIUM
        assert engine._determine_urgency(0.40) == ExecutionUrgency.LOW

    def test_urgent_execution_fewer_slices(self):
        engine = ISExecutionEngine()
        plan_low = engine.plan_execution(100_000, 1.10, confidence=0.4)
        plan_urgent = engine.plan_execution(100_000, 1.10, confidence=0.9)
        assert len(plan_urgent.slices) < len(plan_low.slices)

    def test_high_urgency_more_aggressive_slices(self):
        engine = ISExecutionEngine()
        plan_low = engine.plan_execution(100_000, 1.10, urgency=ExecutionUrgency.LOW)
        plan_high = engine.plan_execution(100_000, 1.10, urgency=ExecutionUrgency.HIGH)
        aggressive_low = sum(1 for s in plan_low.slices if s.order_type == "AGGRESSIVE")
        aggressive_high = sum(
            1 for s in plan_high.slices if s.order_type == "AGGRESSIVE"
        )
        assert aggressive_high >= aggressive_low

    def test_execute_plan_with_market_order_fn(self):
        engine = ISExecutionEngine()
        plan = engine.plan_execution(10_000, 1.10, urgency=ExecutionUrgency.LOW)

        def market_order(volume):
            return 1.1001  # Simulated fill price

        executed = engine.execute_plan(plan, market_order_fn=market_order)
        assert len(executed) == len(plan.slices)
        filled = [s for s in executed if s.filled]
        assert len(filled) > 0

    def test_execute_plan_some_fails(self):
        engine = ISExecutionEngine()
        plan = engine.plan_execution(10_000, 1.10, n_slices=3)

        def market_order(volume):
            if volume > 5000:
                return None  # Simulate failure for large slices
            return 1.1001

        executed = engine.execute_plan(plan, market_order_fn=market_order)
        assert len(executed) == len(plan.slices)

    def test_execution_summary_empty(self):
        engine = ISExecutionEngine()
        summary = engine.get_execution_summary()
        assert "total_slices" in summary
        assert summary["total_slices"] == 0

    def test_expected_cost_increases_with_urgency(self):
        engine = ISExecutionEngine()
        plan_low = engine.plan_execution(100_000, 1.10, urgency=ExecutionUrgency.LOW)
        plan_urgent = engine.plan_execution(
            100_000, 1.10, urgency=ExecutionUrgency.URGENT
        )
        # Urgent execution should have higher expected cost
        assert plan_urgent.expected_cost_bps >= plan_low.expected_cost_bps

    def test_front_load_weight_decreasing(self):
        engine = ISExecutionEngine()
        w0 = engine._front_load_weight(0, 5)
        w4 = engine._front_load_weight(4, 5)
        assert w0 > w4
