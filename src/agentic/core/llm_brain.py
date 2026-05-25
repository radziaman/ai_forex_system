"""
Signal Heuristics — lightweight heuristic reasoning for trading signals.

Replaces the LLMBrain abstraction — the LLM integration was never
implemented and this module only used its heuristic fallback.
Designed to be kept simple; if LLM integration is desired later,
add it as a separate service, not embedded here.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class LLMDecision(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    HOLD = "hold"
    ESCALATE = "escalate"


@dataclass
class LLMTradePlan:
    """Structured output from the reasoning engine."""

    decision: LLMDecision
    confidence: float  # 0.0 to 1.0
    reasoning: str  # Natural language explanation
    risk_flags: List[str] = field(default_factory=list)
    suggested_volume_modifier: float = 1.0  # 0.0 = skip, 1.0 = normal, 2.0 = double
    market_context: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "decision": self.decision.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "risk_flags": self.risk_flags,
            "suggested_volume_modifier": self.suggested_volume_modifier,
            "market_context": self.market_context,
            "timestamp": self.timestamp,
        }


@dataclass
class LLMReflection:
    """Output from the reflection loop — review of recent trades."""

    period_start: float = 0.0
    period_end: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    sharpe: float = 0.0
    assessment: str = ""
    suggestions: List[str] = field(default_factory=list)
    strategy_adjustments: Dict[str, Any] = field(default_factory=dict)


class SignalHeuristics:
    """Lightweight heuristic reasoning about trading signals.

    Replaces the LLMBrain abstraction — the LLM integration was never
    implemented and this module only used its heuristic fallback.
    Designed to be kept simple; if LLM integration is desired later,
    add it as a separate service, not embedded here.
    """

    def evaluate_signal(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        regime: str,
        ensemble_breakdown: Dict[str, Dict],
        open_positions: List[Dict],
        market_context: Optional[Dict] = None,
    ) -> LLMTradePlan:
        """Evaluate a trading signal using heuristic rules.

        Performs the following checks:
          - Conflicting expert signals -> reduce volume
          - Near economic events -> hold
          - Too many open positions -> reject
          - Otherwise -> approve with volume modifier
        """
        risk_flags = []
        volume_mod = 1.0

        # Check for conflicting experts
        expert_directions = [
            v.get("prediction", 0)
            for v in ensemble_breakdown.values()
            if isinstance(v, dict)
        ]
        if expert_directions:
            positive = sum(1 for p in expert_directions if p > 0.0003)
            negative = sum(1 for p in expert_directions if p < -0.0003)
            total = len([p for p in expert_directions if abs(p) > 0.0003])
            if total > 0:
                agreement = max(positive, negative) / total
                if agreement < 0.6:
                    risk_flags.append("low_expert_agreement")
                    volume_mod *= 0.5

        # Check market context
        if market_context:
            if market_context.get("near_economic_event", False):
                risk_flags.append("near_economic_event")
                volume_mod *= 0.3
                reasoning = (
                    f"HOLD {symbol}: Near economic calendar event. "
                    f"Confidence {confidence:.0%} too uncertain to "
                    f"trade through news."
                )
                return LLMTradePlan(
                    decision=LLMDecision.HOLD,
                    confidence=confidence * 0.5,
                    reasoning=reasoning,
                    risk_flags=risk_flags,
                    suggested_volume_modifier=volume_mod,
                    market_context=(
                        f"Regime: {regime}, Positions: {len(open_positions)}"
                    ),
                )

        # Check position count
        if len(open_positions) >= 3:
            risk_flags.append("max_positions_reached")
            return LLMTradePlan(
                decision=LLMDecision.REJECT,
                confidence=confidence,
                reasoning=(
                    f"REJECT {symbol}: Already at max positions"
                    f" ({len(open_positions)})."
                ),
                risk_flags=risk_flags,
                market_context=f"Regime: {regime}",
            )

        # Approve with reasoning
        expert_summary = ", ".join(
            f"{k}: {v.get('prediction', 0):+.4f}"
            for k, v in list(ensemble_breakdown.items())[:5]
        )
        reasoning = (
            f"{'BUY' if direction == 'BUY' else 'SELL'} {symbol}: "
            f"Confidence {confidence:.0%}, Regime: {regime}. "
            f"Experts: [{expert_summary}]. "
            f"Volume modifier: {volume_mod:.1f}x."
        )
        if risk_flags:
            reasoning += f" Risks: {', '.join(risk_flags)}."

        return LLMTradePlan(
            decision=LLMDecision.APPROVE,
            confidence=confidence * volume_mod,
            reasoning=reasoning,
            risk_flags=risk_flags,
            suggested_volume_modifier=volume_mod,
            market_context=f"Regime: {regime}, Positions: {len(open_positions)}",
        )

    def reflect_on_trades(
        self, recent_trades: List[Dict], performance_stats: Dict
    ) -> LLMReflection:
        """Review recent trades and produce improvement suggestions."""
        total = performance_stats.get("total_trades", 0)
        wins = performance_stats.get("wins", 0)
        win_rate = wins / max(total, 1)
        sharpe = performance_stats.get("sharpe", 0.0)

        suggestions = []
        if win_rate < 0.4:
            suggestions.append("Reduce confidence threshold — too many losing trades")
        if sharpe < 0.5:
            suggestions.append(
                "Review regime timing — Sharpe suggests poor risk adjustment"
            )
        if total < 10:
            suggestions.append("Insufficient trade data for meaningful reflection")

        return LLMReflection(
            total_trades=total,
            win_rate=win_rate,
            sharpe=sharpe,
            assessment=self._assess_performance(win_rate, sharpe, total),
            suggestions=suggestions,
        )

    def _assess_performance(self, win_rate: float, sharpe: float, trades: int) -> str:
        if trades < 10:
            return "Too early to assess — insufficient trade data"
        if win_rate > 0.55 and sharpe > 1.0:
            return "Strong performance — strategy is working well"
        elif win_rate > 0.45 and sharpe > 0.5:
            return "Acceptable performance — minor adjustments may help"
        elif sharpe < 0:
            return "Negative Sharpe — strategy losing money, consider pausing"
        else:
            return "Below expectations — review entry timing and risk management"
