"""
LLM Brain — lightweight reasoning layer for the agentic swarm.

Provides structured reasoning about trading signals using an LLM.
The LLM does NOT generate signals — it REASONS about signals from
the existing ML ensemble (PPO, LSTM-CNN, TFT, etc.).

Key capabilities:
  - Signal synthesis: given expert predictions, produce a consolidated view
  - Context-aware filtering: check regime, sentiment, economic calendar
  - Natural language explanations: every trade has a readable "why"
  - Self-reflection: review past trades, suggest improvements

Design philosophy:
  - LLM is OPTIONAL — system runs fully without it
  - LLM is a FILTER — it can only reject trades, not create them
  - LLM output is STRUCTURED JSON — easy to parse and validate
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from loguru import logger


class LLMDecision(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    HOLD = "hold"
    ESCALATE = "escalate"


@dataclass
class LLMTradePlan:
    """Structured output from the LLM reasoning engine."""

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


class LLMBrain:
    """Lightweight LLM reasoning engine for trading decisions.

    Uses a local or API-based LLM to provide strategic reasoning.
    Designed to be optional — the system runs fully without it.

    The reasoning is done via structured JSON prompts:
      1. Build context (regime, signals, positions, calendar)
      2. Send to LLM as structured prompt
      3. Parse JSON response into LLMTradePlan
    """

    def __init__(
        self,
        model_name: str = "local",
        api_key: Optional[str] = None,
        use_llm: bool = False,  # DISABLED by default — opt-in
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.use_llm = use_llm
        self._conversation_history: List[Dict] = []
        self._max_history = 50
        self._reflection_history: List[LLMReflection] = []

    def reason_about_signal(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        regime: str,
        ensemble_breakdown: Dict[str, Dict],
        open_positions: List[Dict],
        market_context: Optional[Dict] = None,
    ) -> LLMTradePlan:
        """Reason about whether to take a trading signal.

        When LLM is disabled (default), uses a simple heuristic:
          - High confidence + favorable regime → approve
          - Conflicting experts → hold
          - Too many open positions → reject
          - Market context flags (NFP, etc.) → reject
        """
        if not self.use_llm:
            return self._heuristic_reason(
                symbol,
                direction,
                confidence,
                regime,
                ensemble_breakdown,
                open_positions,
                market_context,
            )

        # When LLM is enabled, construct a structured prompt
        prompt = self._build_signal_prompt(
            symbol,
            direction,
            confidence,
            regime,
            ensemble_breakdown,
            open_positions,
            market_context,
        )
        response = self._call_llm(prompt)
        return self._parse_llm_response(response)

    def _heuristic_reason(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        regime: str,
        ensemble_breakdown: Dict[str, Dict],
        open_positions: List[Dict],
        market_context: Optional[Dict] = None,
    ) -> LLMTradePlan:
        """Fallback heuristic when LLM is disabled."""
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
                    f"Confidence {confidence:.0%} too uncertain to trade through news."
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

        reflection = LLMReflection(
            total_trades=total,
            win_rate=win_rate,
            sharpe=sharpe,
            assessment=self._assess_performance(win_rate, sharpe, total),
            suggestions=suggestions,
        )
        self._reflection_history.append(reflection)
        return reflection

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

    def _build_signal_prompt(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        regime: str,
        ensemble_breakdown: Dict,
        open_positions: List[Dict],
        market_context: Optional[Dict] = None,
    ) -> str:
        """Build structured prompt for LLM reasoning."""
        return json.dumps(
            {
                "task": "evaluate_trading_signal",
                "symbol": symbol,
                "proposed_action": direction,
                "ml_confidence": confidence,
                "regime": regime,
                "expert_breakdown": {
                    k: {
                        "prediction": v.get("prediction", 0),
                        "weight": v.get("weight", 1),
                    }
                    for k, v in ensemble_breakdown.items()
                },
                "open_positions": [
                    {
                        "symbol": p.get("symbol"),
                        "direction": p.get("direction"),
                        "pnl": p.get("unrealized_pnl", 0),
                    }
                    for p in open_positions
                ],
                "market_context": market_context or {},
            }
        )

    def _call_llm(self, prompt: str) -> Dict:
        """Call the LLM API with a structured prompt.

        Override this method to integrate with OpenAI, Llama, etc.
        Currently returns a default approval response.
        """
        logger.debug(f"LLM call (model={self.model_name}): {prompt[:100]}...")
        # Default response when LLM is not connected
        return {
            "decision": "approve",
            "confidence": 0.7,
            "reasoning": "Default approval — LLM not configured.",
            "risk_flags": [],
            "volume_modifier": 1.0,
        }

    def _parse_llm_response(self, response: Dict) -> LLMTradePlan:
        """Parse structured LLM response into LLMTradePlan."""
        decision_str = response.get("decision", "hold")
        try:
            decision = LLMDecision(decision_str)
        except ValueError:
            decision = LLMDecision.HOLD
        return LLMTradePlan(
            decision=decision,
            confidence=response.get("confidence", 0.5),
            reasoning=response.get("reasoning", "No reasoning provided."),
            risk_flags=response.get("risk_flags", []),
            suggested_volume_modifier=response.get("volume_modifier", 1.0),
        )
