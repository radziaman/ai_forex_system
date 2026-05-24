"""Tests for Validation Gate — mandatory checks before model promotion."""

from training.validation_gate import (
    GateDecision,
    ValidationGate,
)


class TestValidationGate:
    """Comprehensive tests for ValidationGate."""

    def test_gate_rejects_poor_sharpe(self):
        """Sharpe 0.5 < min 1.0 → REJECTED."""
        gate = ValidationGate()
        wf_results = {
            "folds": [{"sharpe": 0.5}] * 4,
            "sharpe": 0.5,
            "max_drawdown": 0.05,
        }
        stress_results = {
            "crisis_2008": {"max_loss": 0.10},
            "covid_crash": {"max_loss": 0.08},
        }
        result = gate.evaluate("test_model", wf_results, stress_results)
        assert result.decision == GateDecision.REJECTED
        assert "sharpe" in result.reason.lower()

    def test_gate_accepts_strong_performance(self):
        """Sharpe 1.5, dd 5% → APPROVED."""
        gate = ValidationGate()
        wf_results = {
            "folds": [{"sharpe": 1.5}] * 4,
            "sharpe": 1.5,
            "max_drawdown": 0.05,
        }
        stress_results = {
            "crisis_2008": {"max_loss": 0.10},
            "covid_crash": {"max_loss": 0.08},
        }
        result = gate.evaluate("test_model", wf_results, stress_results)
        assert result.decision == GateDecision.APPROVED
        assert result.sharpe == 1.5
        assert result.drawdown == 0.05

    def test_gate_rejects_excessive_drawdown(self):
        """Drawdown 15% > 10% → REJECTED."""
        gate = ValidationGate()
        wf_results = {
            "folds": [{"sharpe": 1.2}] * 4,
            "sharpe": 1.2,
            "max_drawdown": 0.15,
        }
        stress_results = {
            "crisis_2008": {"max_loss": 0.10},
        }
        result = gate.evaluate("test_model", wf_results, stress_results)
        assert result.decision == GateDecision.REJECTED
        assert "drawdown" in result.reason.lower()

    def test_gate_rejects_crisis_failure(self):
        """Crisis loses 20% > 15% → REJECTED."""
        gate = ValidationGate()
        wf_results = {
            "folds": [{"sharpe": 1.2}] * 4,
            "sharpe": 1.2,
            "max_drawdown": 0.05,
        }
        stress_results = {
            "crisis_2008": {"max_loss": 0.20},
        }
        result = gate.evaluate("test_model", wf_results, stress_results)
        assert result.decision == GateDecision.REJECTED
        assert "crisis" in result.reason.lower() or "stress" in result.reason.lower()

    def test_gate_requires_min_walk_forward_folds(self):
        """Only 2 folds < 4 → REJECTED."""
        gate = ValidationGate()
        wf_results = {
            "folds": [{"sharpe": 1.5}] * 2,
            "sharpe": 1.5,
            "max_drawdown": 0.05,
        }
        stress_results = {
            "crisis_2008": {"max_loss": 0.10},
        }
        result = gate.evaluate("test_model", wf_results, stress_results)
        assert result.decision == GateDecision.REJECTED
        assert "fold" in result.reason.lower()

    def test_gate_logs_result(self):
        """Result contains model_name and timestamp."""
        gate = ValidationGate()
        wf_results = {
            "folds": [{"sharpe": 1.5}] * 4,
            "sharpe": 1.5,
            "max_drawdown": 0.05,
        }
        stress_results = {
            "crisis_2008": {"max_loss": 0.10},
        }
        result = gate.evaluate("my_forex_model", wf_results, stress_results)
        assert result.model_name == "my_forex_model"
        assert result.timestamp > 0
        assert isinstance(result.decision, GateDecision)

    def test_gate_conditional_approval_with_warning(self):
        """Sharpe 0.9 (above conditional 0.8) + good stress → CONDITIONAL_APPROVED."""
        gate = ValidationGate()
        wf_results = {
            "folds": [{"sharpe": 0.9}] * 4,
            "sharpe": 0.9,
            "max_drawdown": 0.05,
        }
        stress_results = {
            "crisis_2008": {"max_loss": 0.10},
            "covid_crash": {"max_loss": 0.08},
        }
        result = gate.evaluate("test_model", wf_results, stress_results)
        assert result.decision == GateDecision.CONDITIONAL_APPROVED
        assert "conditional" in result.reason.lower()
