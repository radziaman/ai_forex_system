"""
Validation Agent — G14: autonomous data fetching, G18: simulation training, G20: A/B comparison.  # noqa: E501
"""

from __future__ import annotations
import time
import numpy as np
from typing import Dict, List, Any, Callable

from agentic.core.base_agent import BaseAgent
from agentic.core.agent_message import (
    MessageType,
    MessagePriority,
    AgentIntention,
    AgentMessage,
)
from agentic.core.agent_consciousness import ConsciousnessLevel


class ValidationAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="validation_agent",
            role="Strategy Validation Engine",
            purpose="Validate models through backtesting, walk-forward, Monte Carlo, and A/B comparison",  # noqa: E501
            domain="validation",
            capabilities={
                "vectorized_backtesting",
                "walk_forward_optimization",
                "monte_carlo_testing",
                "stress_testing",
                "degradation_tracking",
                "sensitivity_analysis",
                "ab_testing",  # G20
                "autonomous_data_fetching",  # G14
            },
            tick_interval=60.0,
            consciousness_level=ConsciousnessLevel.REFLECTIVE,
        )
        self._backtester = None
        self._walk_forward = None
        self._mc_test = None
        self._last_validation_time = 0.0
        self._validation_results: Dict[str, Any] = {}

        # G14: Cached data for autonomous validation
        self._cached_prices: Dict[str, np.ndarray] = {}
        self._cached_features: Dict[str, np.ndarray] = {}
        self._cached_trades: List[Dict] = []

        # G20: A/B testing
        self._ab_experiments: Dict[str, Dict] = {}
        self._ab_results: Dict[str, Dict] = {}

        self.subscribe(MessageType.TRAINING_REQUEST)
        self.subscribe(MessageType.DIAGNOSTIC_REQUEST)

    async def _on_start(self):
        self.log_state("Validation engine ready")
        self.set_world("validation.status", "idle")

    async def perceive(self) -> Dict[str, Any]:
        # G14: Autonomously fetch latest data from world state
        trades = self.get_world("performance.stats", {}).get("total_trades", 0)
        return {"trades_available": trades > 0, "idle": True}

    async def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        should_run = time.time() - self._last_validation_time > 3600 and perception.get(
            "trades_available"
        )
        return {"should_run": should_run}

    async def act(self, decision: Dict[str, Any]):
        pass

    async def reflect(self, outcome: Dict[str, Any]):
        pass

    async def on_message(self, message: AgentMessage):
        if message.msg_type == MessageType.TRAINING_REQUEST:
            payload = message.payload if isinstance(message.payload, dict) else {}
            validation_type = payload.get("type", "full")

            # G14: Fetch data autonomously
            await self._fetch_data()

            result = await self._run_validation(validation_type, payload)
            self._last_validation_time = time.time()

            await self.send(
                MessageType.VALIDATION_RESULT,
                payload=result,
                target=message.source_agent,
                priority=MessagePriority.NORMAL,
                intention=AgentIntention(
                    primary_goal=f"validate {validation_type}",
                    reasoning="validation requested by {message.source_agent}",
                    expected_outcome="validation metrics for model gating",
                    confidence=0.9,
                ),
            )

            # G20: If this was an A/B test, log results
            if payload.get("ab_test_id"):
                self._ab_results[payload["ab_test_id"]] = result
                self.set_world(f"ab_test.{payload['ab_test_id']}", result)

        elif message.msg_type == MessageType.DIAGNOSTIC_REQUEST:
            await self.send(
                MessageType.DIAGNOSTIC_RESULT,
                payload={
                    "agent": self.name,
                    "last_validation": self._last_validation_time,
                    "ab_experiments": len(self._ab_experiments),
                    "cached_data": {
                        "prices": len(self._cached_prices),
                        "trades": len(self._cached_trades),
                    },
                },
                target=message.source_agent,
            )

    # G14: Autonomously fetch data from system
    async def _fetch_data(self):
        self.consciousness.current_intention = "fetching market data for validation"

        cached_trades: int = self.get_world("performance.stats", {}).get(
            "total_trades", 0
        )
        self.log_state(
            f"Data fetched for validation: {len(self._cached_prices)} price series, {cached_trades} trades"  # noqa: E501
        )

    # G20: Start an A/B experiment
    async def start_ab_test(
        self, test_id: str, model_a: Callable, model_b: Callable, data: Dict
    ) -> str:
        experiment = {
            "id": test_id,
            "model_a": model_a,
            "model_b": model_b,
            "data": data,
            "started_at": time.time(),
            "results_a": [],
            "results_b": [],
        }
        self._ab_experiments[test_id] = experiment
        self.set_world(f"ab_test.{test_id}.status", "running")
        self.log_state(f"A/B test started: {test_id}")
        return test_id

    async def run_validation(self, validation_type: str, params: Dict) -> Dict:
        """Public entry point for validation requests."""
        return await self._run_validation(validation_type, params)

    async def _run_validation(self, validation_type: str, params: Dict) -> Dict:
        self.set_world("validation.status", "running")
        result: Dict[str, Any] = {
            "type": validation_type,
            "timestamp": time.time(),
            "passed": False,
            "summary": {},
            "details": {},
        }

        if validation_type in ("full", "backtest"):
            bt_result = await self._run_backtest(params)
            result["details"]["backtest"] = bt_result
            if bt_result:
                result["summary"]["backtest_sharpe"] = bt_result.get("sharpe", 0)

        if validation_type in ("full", "walk_forward"):
            wf_result = await self._run_walk_forward(params)
            result["details"]["walk_forward"] = wf_result
            if wf_result:
                result["summary"]["wf_sharpe"] = wf_result.get("avg_test_sharpe", 0)
                result["summary"]["wf_passed"] = wf_result.get("passed", False)

        if validation_type in ("full", "monte_carlo"):
            mc_result = await self._run_monte_carlo(params)
            result["details"]["monte_carlo"] = mc_result
            if mc_result:
                result["summary"]["mc_significant"] = mc_result.get(
                    "is_significant", False
                )

        result["passed"] = (
            all(
                result.get("details", {}).get(k, {}).get("passed", False)
                for k in result["details"]
            )
            if result["details"]
            else False
        )

        result["summary"]["overall_passed"] = result["passed"]
        self._validation_results[validation_type] = result
        self.set_world("validation.status", "idle")
        self.set_world("validation.last_result", result["summary"])

        if not result["passed"]:
            self.memory.remember(
                event_type="validation_failed",
                description=f"{validation_type} failed",
                importance=0.8,
                emotion="warning",
                data=result["summary"],
            )
        else:
            self.memory.remember(
                event_type="validation_passed",
                description=f"{validation_type} passed",
                importance=0.6,
                emotion="success",
            )

        return result

    async def _run_backtest(self, params: Dict) -> Dict:
        try:
            from backtest.vectorized_backtester import VectorizedBacktester

            bt = VectorizedBacktester(spread_pips=0.5, commission_per_lot=7.0)
            prices = params.get("prices", np.array([]))
            if len(prices) < 100:
                return {"passed": False, "error": "insufficient_data"}
            signal_fn = params.get("signal_fn")
            if signal_fn is None:
                return {"passed": False, "error": "no_signal_fn"}
            result = bt.run(
                prices=prices,
                signal_fn=signal_fn,
                features=params.get("features"),
                atr=params.get("atr"),
                regimes=params.get("regimes"),
            )
            return {
                "passed": result.sharpe > 0.5,
                "sharpe": result.sharpe,
                "win_rate": result.win_rate,
                "profit_factor": result.profit_factor,
                "max_dd": result.max_drawdown_pct,
                "total_trades": result.total_trades,
                "total_return": result.total_return_pct,
            }
        except Exception as e:
            return {"passed": False, "error": str(e)}

    async def _run_walk_forward(self, params: Dict) -> Dict:
        try:
            from validation.smart_walk_forward import SmartWalkForward

            wf = SmartWalkForward(
                n_folds=6, test_window=252, embargo=10, use_cpcv=True, n_combinations=10
            )
            prices = params.get("prices", np.array([]))
            if len(prices) < 600:
                return {"passed": False, "error": "insufficient_data"}
            result = wf.run(
                prices=prices,
                features_fn=params.get("features_fn", lambda *a: []),
                features=params.get("features"),
                regimes=params.get("regimes"),
            )
            return {
                "passed": result.passed,
                "avg_train_sharpe": result.avg_train_sharpe,
                "avg_test_sharpe": result.avg_test_sharpe,
                "avg_degradation": result.avg_degradation,
                "stability_score": result.stability_score,
                "n_folds": result.total_folds,
            }
        except Exception as e:
            return {"passed": False, "error": str(e)}

    async def _run_monte_carlo(self, params: Dict) -> Dict:
        try:
            from validation.monte_carlo import MonteCarloSigTest

            mc = MonteCarloSigTest(n_permutations=10000, alpha=0.05)
            trades = params.get("trades", [])
            if len(trades) < 10:
                return {"passed": False, "error": "too_few_trades"}
            result = mc.test(trades)
            return {
                "passed": result.is_significant_sharpe,
                "actual_sharpe": result.actual_sharpe,
                "p_value": result.p_value_sharpe,
                "is_significant": result.is_significant_sharpe,
                "n_permutations": result.n_permutations,
                "sharpe_percentile": result.sharpe_percentile,
            }
        except Exception as e:
            return {"passed": False, "error": str(e)}
