"""
Signal Strategies — standalone prediction functions for the MoE ensemble.

Each strategy is a callable that takes features (np.ndarray) and returns
a prediction dict. Strategies delegate to SignalAgent's private methods
for stateful operations (accessing loaded models, world state, etc.).
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Callable, List, Optional, TYPE_CHECKING, Any
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from agentic.agents.signal_agent import SignalAgent


@dataclass
class StrategySpec:
    """Specification for registering a strategy with the ensemble."""

    name: str
    predict_fn: Callable
    confidence_fn: Callable
    regime: str
    sl_atr: float = 2.0
    tp_atr: float = 4.0


class StrategyRegistry:
    """Registry of all available strategies.

    Strategies are registered by name and can be looked up by regime.
    """

    def __init__(self):
        self._strategies: Dict[str, StrategySpec] = {}

    def register(self, spec: StrategySpec):
        """Register a strategy spec by name."""
        self._strategies[spec.name] = spec

    def get(self, name: str) -> Optional[StrategySpec]:
        """Get a strategy spec by name."""
        return self._strategies.get(name)

    def get_by_regime(self, regime: str) -> List[StrategySpec]:
        """Get all strategy specs for a given regime."""
        return [s for s in self._strategies.values() if s.regime == regime]

    def get_all(self) -> Dict[str, StrategySpec]:
        """Get all registered strategy specs."""
        return dict(self._strategies)

    def get_names(self) -> List[str]:
        """Get all registered strategy names."""
        return list(self._strategies.keys())


class StrategyExecutor:
    """Executes strategy predictions using loaded models and world state.

    SignalAgent creates one instance and passes it to strategies.
    Delegates to the SignalAgent's private prediction methods.
    """

    def __init__(self, agent: "SignalAgent"):
        self._agent = agent

    def get_all_specs(self) -> List[StrategySpec]:
        """Return all strategy specs for registration with the ensemble.

        Builds StrategySpec objects that delegate to the agent's prediction
        and confidence methods.
        """
        specs: List[StrategySpec] = []
        agent = self._agent

        # Phase 1: 4 regime-specific PPO experts
        if agent._regime_manager:
            for regime in ["trending", "ranging", "volatile", "crisis"]:
                specs.append(
                    StrategySpec(
                        name=f"ppo_{regime}",
                        predict_fn=lambda X, r=regime: agent._ppo_prediction(
                            X, regime=r
                        ),
                        confidence_fn=lambda X, r=regime: agent._ppo_confidence(
                            X, regime=r
                        ),
                        regime=regime,
                        sl_atr=STRATEGY_SL_TP.get(f"ppo_{regime}", (2.0, 4.0))[0],
                        tp_atr=STRATEGY_SL_TP.get(f"ppo_{regime}", (2.0, 4.0))[1],
                    )
                )

        # Phase 4: LSTM as a proper expert
        if len(agent._lstm_models) > 0:
            specs.append(
                StrategySpec(
                    name="lstm_cnn",
                    predict_fn=agent._lstm_ensemble_prediction,
                    confidence_fn=lambda X: 0.6,
                    regime="ranging",
                    sl_atr=STRATEGY_SL_TP.get("lstm_cnn", (2.0, 4.0))[0],
                    tp_atr=STRATEGY_SL_TP.get("lstm_cnn", (2.0, 4.0))[1],
                )
            )

        # Phase 2: Rule-based experts
        specs.append(
            StrategySpec(
                name="rule_breakout",
                predict_fn=agent._rule_breakout_prediction,
                confidence_fn=agent._rule_breakout_confidence,
                regime="trending",
                sl_atr=STRATEGY_SL_TP.get("rule_breakout", (2.0, 5.0))[0],
                tp_atr=STRATEGY_SL_TP.get("rule_breakout", (2.0, 5.0))[1],
            )
        )
        specs.append(
            StrategySpec(
                name="rule_mean_rev",
                predict_fn=agent._rule_mean_rev_prediction,
                confidence_fn=agent._rule_mean_rev_confidence,
                regime="ranging",
                sl_atr=STRATEGY_SL_TP.get("rule_mean_rev", (1.5, 2.0))[0],
                tp_atr=STRATEGY_SL_TP.get("rule_mean_rev", (1.5, 2.0))[1],
            )
        )

        # Phase 6: Research-backed forex strategies
        specs.append(
            StrategySpec(
                name="bb_squeeze",
                predict_fn=agent._bb_squeeze_prediction,
                confidence_fn=lambda X: 0.55,
                regime="volatile",
                sl_atr=STRATEGY_SL_TP.get("bb_squeeze", (2.0, 3.0))[0],
                tp_atr=STRATEGY_SL_TP.get("bb_squeeze", (2.0, 3.0))[1],
            )
        )
        specs.append(
            StrategySpec(
                name="ts_momentum",
                predict_fn=agent._ts_momentum_prediction,
                confidence_fn=lambda X: 0.6,
                regime="trending",
                sl_atr=STRATEGY_SL_TP.get("ts_momentum", (2.5, 4.0))[0],
                tp_atr=STRATEGY_SL_TP.get("ts_momentum", (2.5, 4.0))[1],
            )
        )
        specs.append(
            StrategySpec(
                name="vol_mean_rev",
                predict_fn=agent._vol_mean_rev_prediction,
                confidence_fn=lambda X: 0.55,
                regime="volatile",
                sl_atr=STRATEGY_SL_TP.get("vol_mean_rev", (1.5, 2.0))[0],
                tp_atr=STRATEGY_SL_TP.get("vol_mean_rev", (1.5, 2.0))[1],
            )
        )

        # Phase 7: Order flow / CVD expert
        specs.append(
            StrategySpec(
                name="orderflow",
                predict_fn=agent._orderflow_prediction,
                confidence_fn=agent._orderflow_confidence,
                regime="ranging",
                sl_atr=STRATEGY_SL_TP.get("orderflow", (1.5, 2.5))[0],
                tp_atr=STRATEGY_SL_TP.get("orderflow", (1.5, 2.5))[1],
            )
        )

        # Phase 8: Macro sentiment expert
        specs.append(
            StrategySpec(
                name="macro_sentiment",
                predict_fn=agent._macro_sentiment_prediction,
                confidence_fn=lambda X: 0.5,
                regime="trending",
            )
        )

        # Phase 9: Social sentiment expert
        specs.append(
            StrategySpec(
                name="social_sentiment",
                predict_fn=agent._social_sentiment_prediction,
                confidence_fn=lambda X: 0.35,
                regime="volatile",
            )
        )

        # Phase 10: XGBoost expert
        specs.append(
            StrategySpec(
                name="xgboost",
                predict_fn=agent._xgboost_prediction,
                confidence_fn=lambda X: 0.55,
                regime="ranging",
            )
        )

        # Phase 11: TFT primary expert
        if len(agent._tft_models) > 0:
            specs.append(
                StrategySpec(
                    name="tft_primary",
                    predict_fn=agent._tft_prediction,
                    confidence_fn=agent._tft_confidence,
                    regime="ranging",
                    sl_atr=STRATEGY_SL_TP.get("tft_primary", (2.0, 4.0))[0],
                    tp_atr=STRATEGY_SL_TP.get("tft_primary", (2.0, 4.0))[1],
                )
            )

        # Phase 12: Alpha strategies
        alpha_specs = self._get_alpha_specs()
        specs.extend(alpha_specs)

        # Phase 13: Cross-sectional alpha expert
        if (
            hasattr(agent, "_cross_sectional_alpha")
            and agent._cross_sectional_alpha is not None
        ):  # noqa: E501
            specs.append(
                StrategySpec(
                    name="cross_sectional",
                    predict_fn=agent._cross_sectional_predict,
                    confidence_fn=lambda X: 0.6,
                    regime="ranging",
                )
            )

        return specs

    def _get_alpha_specs(self) -> List[StrategySpec]:
        """Build specs for alpha strategies if loaded."""
        agent = self._agent
        specs: List[StrategySpec] = []
        alpha_names = list(agent._alpha_strategies.keys())
        if not alpha_names:
            return specs

        for name in alpha_names:
            regime = agent._alpha_regime_for(name)
            predict_fn = agent._make_alpha_predict_fn(name)
            conf_fn = agent._make_alpha_confidence_fn(name)
            sl_atr, tp_atr = STRATEGY_SL_TP.get(name, (1.5, 3.0))
            specs.append(
                StrategySpec(
                    name=name,
                    predict_fn=predict_fn,
                    confidence_fn=conf_fn,
                    regime=regime,
                    sl_atr=sl_atr,
                    tp_atr=tp_atr,
                )
            )
        return specs


# Strategy-specific SL/TP (ATR multipliers) — each strategy has its own optimal
# stop-loss and take-profit to improve profitability.
STRATEGY_SL_TP: Dict[str, tuple] = {
    "rule_breakout": (2.0, 5.0),  # Trending breakout
    "rule_mean_rev": (1.5, 2.0),  # Ranging mean reversion
    "bb_squeeze": (2.0, 3.0),  # Volatility breakout
    "ts_momentum": (2.5, 4.0),  # Trend momentum (wider SL)
    "vol_mean_rev": (1.5, 2.0),  # Volatility reversion
    "ppo_trending": (2.0, 4.0),  # PPO trending agent
    "ppo_ranging": (1.5, 3.0),  # PPO ranging agent
    "ppo_volatile": (2.5, 5.0),  # PPO volatile agent
    "ppo_crisis": (1.0, 2.0),  # PPO crisis agent
    "lstm_cnn": (2.0, 4.0),  # LSTM model
    "tft_primary": (2.0, 4.0),  # TFT model
    "orderflow": (1.5, 2.5),  # Order flow / CVD (tight SL, moderate TP)
    "macro_sentiment": (2.0, 4.0),  # Macroeconomic / sentiment (wider SL)
    "stat_arb": (1.5, 3.0),  # Statistical arbitrage
    "carry_trade": (2.0, 4.0),  # Carry trade
    "event_driven": (1.0, 2.0),  # Event-driven straddle
    "vol_expansion": (2.5, 5.0),  # Volatility breakout
    "order_flow_momentum": (1.5, 2.5),  # Order flow momentum
}
