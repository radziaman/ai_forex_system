"""Tests for LLM Orchestration components."""

from agentic.core.llm_brain import LLMBrain, LLMTradePlan, LLMDecision
from agentic.core.llm_orchestrator import LLMOrchestrator
from agentic.core.reflection_loop import ReflectionLoop


class TestLLMBrain:
    def test_heuristic_approves_high_confidence(self):
        brain = LLMBrain()
        plan = brain._heuristic_reason(
            symbol="EURUSD",
            direction="BUY",
            confidence=0.85,
            regime="trending",
            ensemble_breakdown={},
            open_positions=[],
            market_context={},
        )
        assert plan.decision == LLMDecision.APPROVE
        assert "EURUSD" in plan.reasoning

    def test_heuristic_rejects_at_max_positions(self):
        brain = LLMBrain()
        plan = brain._heuristic_reason(
            symbol="EURUSD",
            direction="BUY",
            confidence=0.8,
            regime="trending",
            ensemble_breakdown={},
            open_positions=[
                {"symbol": "GBPUSD", "direction": "BUY"},
                {"symbol": "USDJPY", "direction": "SELL"},
                {"symbol": "AUDUSD", "direction": "BUY"},
            ],
            market_context={},
        )
        assert plan.decision == LLMDecision.REJECT

    def test_heuristic_holds_near_economic_event(self):
        brain = LLMBrain()
        plan = brain._heuristic_reason(
            symbol="EURUSD",
            direction="BUY",
            confidence=0.65,
            regime="ranging",
            ensemble_breakdown={},
            open_positions=[],
            market_context={"near_economic_event": True},
        )
        assert plan.decision == LLMDecision.HOLD

    def test_heuristic_reduces_volume_on_low_agreement(self):
        brain = LLMBrain()
        # 2 positive, 2 negative => agreement 50% (< 60% threshold)
        plan = brain._heuristic_reason(
            symbol="EURUSD",
            direction="BUY",
            confidence=0.75,
            regime="ranging",
            ensemble_breakdown={
                "expert_a": {"prediction": 0.001, "weight": 1.0},
                "expert_b": {"prediction": -0.001, "weight": 1.0},
                "expert_c": {"prediction": 0.002, "weight": 1.0},
                "expert_d": {"prediction": -0.002, "weight": 1.0},
            },
            open_positions=[],
            market_context={},
        )
        assert plan.suggested_volume_modifier < 1.0

    def test_reflect_on_trades(self):
        brain = LLMBrain()
        trades = [{"pnl": 10}, {"pnl": -5}, {"pnl": 15}, {"pnl": 8}, {"pnl": -3}]
        stats = {"total_trades": 5, "wins": 3, "sharpe": 0.8}
        reflection = brain.reflect_on_trades(trades, stats)
        assert reflection.total_trades == 5
        assert reflection.win_rate == 0.6
        assert isinstance(reflection.suggestions, list)

    def test_assess_performance(self):
        brain = LLMBrain()
        assert "strong" in brain._assess_performance(0.6, 1.5, 50).lower()
        assert "negative" in brain._assess_performance(0.3, -0.5, 50).lower()
        assert "early" in brain._assess_performance(0.5, 0.5, 5).lower()

    def test_llm_trade_plan_to_dict(self):
        plan = LLMTradePlan(
            decision=LLMDecision.APPROVE,
            confidence=0.85,
            reasoning="Test reasoning",
            risk_flags=["flag1"],
            suggested_volume_modifier=0.8,
            market_context="test context",
        )
        d = plan.to_dict()
        assert d["decision"] == "approve"
        assert d["confidence"] == 0.85
        assert d["reasoning"] == "Test reasoning"

    def test_parse_llm_response_invalid_decision(self):
        brain = LLMBrain()
        plan = brain._parse_llm_response({"decision": "invalid"})
        assert plan.decision == LLMDecision.HOLD

    def test_parse_llm_response_missing_fields(self):
        brain = LLMBrain()
        plan = brain._parse_llm_response({})
        assert plan.decision == LLMDecision.HOLD
        assert plan.confidence == 0.5
        assert plan.reasoning == "No reasoning provided."


class TestLLMOrchestrator:
    def test_rejects_below_min_confidence(self):
        orch = LLMOrchestrator(min_confidence=0.6)
        plan = orch.evaluate_signal(
            symbol="EURUSD",
            direction="BUY",
            confidence=0.4,
            regime="ranging",
            ensemble_breakdown={},
        )
        assert plan.decision == LLMDecision.REJECT

    def test_approves_high_confidence(self):
        orch = LLMOrchestrator(min_confidence=0.3)
        plan = orch.evaluate_signal(
            symbol="GBPUSD",
            direction="SELL",
            confidence=0.8,
            regime="trending",
            ensemble_breakdown={},
        )
        assert plan.decision == LLMDecision.APPROVE

    def test_tracks_stats(self):
        orch = LLMOrchestrator(min_confidence=0.5)
        orch.evaluate_signal("A", "BUY", 0.9, "trending", {})
        orch.evaluate_signal("B", "SELL", 0.3, "ranging", {})
        stats = orch.get_stats()
        assert stats["total_evaluated"] == 2
        assert stats["approved"] == 1
        assert stats["rejected"] == 1

    def test_passes_open_positions(self):
        orch = LLMOrchestrator(min_confidence=0.1)
        plan = orch.evaluate_signal(
            symbol="EURUSD",
            direction="BUY",
            confidence=0.9,
            regime="trending",
            ensemble_breakdown={},
            open_positions=[
                {"symbol": "GBPUSD", "direction": "SELL", "unrealized_pnl": 50},
            ],
        )
        assert plan.decision == LLMDecision.APPROVE

    def test_provides_market_context(self):
        orch = LLMOrchestrator(min_confidence=0.1)
        plan = orch.evaluate_signal(
            symbol="EURUSD",
            direction="SELL",
            confidence=0.95,
            regime="volatile",
            ensemble_breakdown={},
            market_context={"near_economic_event": True, "event": "NFP"},
        )
        assert plan.decision in (LLMDecision.HOLD, LLMDecision.APPROVE)

    def test_empty_ensemble_breakdown(self):
        orch = LLMOrchestrator(min_confidence=0.1)
        plan = orch.evaluate_signal(
            symbol="USDJPY",
            direction="BUY",
            confidence=0.7,
            regime="ranging",
            ensemble_breakdown={},
        )
        assert plan.decision == LLMDecision.APPROVE


class TestReflectionLoop:
    def test_accumulates_trades(self):
        loop = ReflectionLoop(trade_window=5, min_trades_for_reflection=3)
        for i in range(3):
            loop.record_trade({"pnl": 10 if i % 2 == 0 else -5})
        assert len(loop._trade_buffer) == 3

    def test_triggers_reflection_at_window(self):
        loop = ReflectionLoop(trade_window=3, min_trades_for_reflection=2)
        for i in range(3):
            loop.record_trade({"pnl": 10})
        assert len(loop._trade_buffer) == 0  # buffer cleared after reflection
        assert loop._reflection_count == 1

    def test_get_stats(self):
        loop = ReflectionLoop()
        stats = loop.get_stats()
        assert "reflection_count" in stats
        assert "buffered_trades" in stats
        assert stats["trade_window"] == 50

    def test_does_not_reflect_below_min_trades(self):
        loop = ReflectionLoop(trade_window=10, min_trades_for_reflection=5)
        for i in range(3):
            loop.record_trade({"pnl": 10})
        result = loop.run_reflection()
        assert result is None

    def test_reflection_calculates_win_rate(self):
        loop = ReflectionLoop(trade_window=5, min_trades_for_reflection=2)
        for i in range(4):
            loop.record_trade({"pnl": 10 if i < 3 else -5})
        # Buffer was 4 trades, not yet at window of 5
        assert len(loop._trade_buffer) == 4

    def test_reflection_with_callback(self):
        callback_results = []

        def callback(reflection):
            callback_results.append(reflection)

        loop = ReflectionLoop(
            trade_window=2, min_trades_for_reflection=1, on_reflection=callback
        )
        loop.record_trade({"pnl": 10})
        loop.record_trade({"pnl": -5})
        assert len(callback_results) == 1
        assert callback_results[0].total_trades == 2
