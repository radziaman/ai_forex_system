"""
LLM Orchestrator — synthesizes signals from all agents into trade plans.

Lies between SignalAgent and RiskAgent in the message flow:
  SignalAgent → LLMOrchestrator → RiskAgent

The orchestrator:
  1. Receives raw signals from MoE ensemble
  2. Consults LLM brain for strategic reasoning
  3. Checks market context (regime, calendar, sentiment)
  4. Produces structured LLMTradePlan
  5. Forwards approved plans to RiskAgent

This is OPTIONAL — skipping the orchestrator routes signals directly.
"""

from typing import Dict, List, Optional
from loguru import logger

from .llm_brain import LLMBrain, LLMTradePlan, LLMDecision


class LLMOrchestrator:
    """Orchestrates LLM reasoning in the signal→risk pipeline.

    Usage:
        orchestrator = LLMOrchestrator(llm_brain)
        plan = orchestrator.evaluate_signal(signal, context)
        if plan.decision == LLMDecision.APPROVE:
            risk_agent.handle(plan)
    """

    def __init__(
        self,
        llm_brain: Optional[LLMBrain] = None,
        min_confidence: float = 0.5,
    ):
        self.brain = llm_brain or LLMBrain()
        self.min_confidence = min_confidence
        self._approved_count = 0
        self._rejected_count = 0
        self._total_evaluated = 0

    def evaluate_signal(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        regime: str,
        ensemble_breakdown: Dict,
        open_positions: Optional[List[Dict]] = None,
        market_context: Optional[Dict] = None,
    ) -> LLMTradePlan:
        """Full signal evaluation pipeline.

        Returns LLMTradePlan with decision, reasoning, and volume modifier.
        """
        self._total_evaluated += 1

        # Fast reject: confidence too low
        if confidence < self.min_confidence:
            plan = LLMTradePlan(
                decision=LLMDecision.REJECT,
                confidence=confidence,
                reasoning=(
                    f"REJECT {symbol}: Confidence {confidence:.0%}"
                    f" below minimum {self.min_confidence:.0%}."
                ),
                market_context=f"Regime: {regime}",
            )
            self._rejected_count += 1
            return plan

        # LLM reasoning
        plan = self.brain.reason_about_signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            regime=regime,
            ensemble_breakdown=ensemble_breakdown,
            open_positions=open_positions or [],
            market_context=market_context,
        )

        if plan.decision == LLMDecision.APPROVE:
            self._approved_count += 1
            logger.info(
                "LLMOrchestrator: APPROVED %s %s (conf=%.0f%%)",
                symbol,
                direction,
                confidence * 100,
            )
        else:
            self._rejected_count += 1
            decision_str = plan.decision.value.upper()
            logger.info(
                "LLMOrchestrator: %s %s: %s",
                decision_str,
                symbol,
                plan.reasoning[:80],
            )

        return plan

    def get_stats(self) -> Dict:
        return {
            "total_evaluated": self._total_evaluated,
            "approved": self._approved_count,
            "rejected": self._rejected_count,
            "approval_rate": self._approved_count / max(self._total_evaluated, 1),
        }
