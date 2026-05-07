"""
Master AI Orchestrator - Central AI Engine for the MoneyBot System.
Controls and manages all 18 enhancements, learns optimal configurations,
and makes high-level decisions about system behavior.
"""
import asyncio
import numpy as np
import time
import sys
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
from collections import defaultdict

# Import SYMBOLS from data_manager
try:
    from data.data_manager import SYMBOLS
except ImportError:
    # Fallback if not available
    SYMBOLS = [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
        "EURJPY", "GBPJPY", "EURGBP",
        "XAUUSD", "XAGUSD", "XTIUSD", "XBRUSD", "XNGUSD",
        "US500", "US30", "USTEC", "UK100", "DE40",
        "BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD",
    ]


class SystemState(str, Enum):
    """Overall system state."""
    OPTIMAL = "optimal"
    DEGRADED = "degraded"
    RECOVERING = "recovering"
    HALTED = "halted"
    LEARNING = "learning"
    BACKTESTING = "backtesting"


class EnhancementStatus(str, Enum):
    """Status of each enhancement."""
    ACTIVE = "active"
    DISABLED = "disabled"
    DEGRADED = "degraded"
    LEARNING = "learning"
    TESTING = "testing"


@dataclass
class EnhancementMetrics:
    """Performance metrics for each enhancement."""
    name: str
    status: EnhancementStatus = EnhancementStatus.ACTIVE
    confidence: float = 1.0
    performance_score: float = 0.0
    last_success: float = field(default_factory=time.time)
    failure_count: int = 0
    trades_influenced: int = 0
    pnl_contribution: float = 0.0
    sharpe_contribution: float = 0.0


@dataclass
class SystemDecision:
    """Decision made by the master AI."""
    timestamp: float = field(default_factory=time.time)
    action: str = ""  # continue | pause | reconfigure | halt
    reason: str = ""
    confidence: float = 0.0
    adjustments: Dict[str, Any] = field(default_factory=dict)
    enhancements_to_enable: List[str] = field(default_factory=list)
    enhancements_to_disable: List[str] = field(default_factory=list)


class MasterAIOrchestrator:
    """
    Central AI Engine that controls and manages ALL MoneyBot functions.
    
    Acts as the "brain" that:
    - FULLY CONTROLS all 18 enhancements (enable/disable/configure)
    - Makes meta-decisions about system configuration
    - Learns which enhancement combinations work best
    - Dynamically adjusts parameters based on market conditions
    - Provides unified control interface
    - AUTO-RECONFIGURES the entire system
    """

    # All 18 enhancements with their dependencies (class-level constant)
    ENHANCEMENTS = {
        # High Priority (1-3)
        "ppo_integration": {
            "name": "PPO Integration", "priority": 1, "dependencies": [],
        },
        "live_trading_loop": {
            "name": "Live Trading Loop", "priority": 1, "dependencies": ["ppo_integration"],
        },
        "model_versioning": {
            "name": "Model Versioning", "priority": 1, "dependencies": ["ppo_integration"],
        },
        # Medium Priority (4-12)
        "circuit_breaker": {
            "name": "Circuit Breaker", "priority": 2, "dependencies": ["live_trading_loop"],
        },
        "data_pipeline": {
            "name": "Data Pipeline", "priority": 2, "dependencies": [],
        },
        "error_recovery": {
            "name": "Error Recovery", "priority": 2, "dependencies": ["live_trading_loop"],
        },
        "feature_optimization": {
            "name": "Feature Optimization", "priority": 2, "dependencies": ["data_pipeline"],
        },
        "algo_execution": {
            "name": "Algo Execution", "priority": 2, "dependencies": ["live_trading_loop"],
        },
        "ensemble_weighting": {
            "name": "Ensemble Weighting", "priority": 2, "dependencies": ["ppo_integration"],
        },
        "var_sizing": {
            "name": "VaR-based Sizing", "priority": 2, "dependencies": [],
        },
        "sentiment_alpha": {
            "name": "Sentiment Alpha", "priority": 2, "dependencies": [],
        },
        "regime_transition": {
            "name": "Regime Transition Trading", "priority": 2, "dependencies": ["ppo_integration"],
        },
        # Low Priority (13-18)
        "stress_testing": {
            "name": "Stress Testing", "priority": 3, "dependencies": [],
        },
        "walk_forward": {
            "name": "Walk-Forward Optimization", "priority": 3, "dependencies": [],
        },
        "integration_tests": {
            "name": "Integration Tests", "priority": 3, "dependencies": [],
        },
        "trading_sessions": {
            "name": "Smart Trading Sessions", "priority": 3, "dependencies": [],
        },
        "enhanced_dashboard": {
            "name": "Enhanced Dashboard", "priority": 3, "dependencies": [],
            "module_name": "enhanced_dashboard",  # References smart_dashboard.py
        },
        "event_avoidance": {
            "name": "Event Avoidance", "priority": 3, "dependencies": ["sentiment_alpha"],
        },
    }

    def __init__(
        self,
        initial_balance: float = 100000.0,
        learning_rate: float = 0.01,
        adaptation_threshold: float = 0.6,
    ):
        self.initial_balance = initial_balance
        self.learning_rate = learning_rate
        self.adaptation_threshold = adaptation_threshold
        
        # System state
        self.system_state: SystemState = SystemState.OPTIMAL
        self.enhancements: Dict[str, EnhancementMetrics] = {}
        self._init_enhancements()
        
        # Performance tracking
        self.system_pnl: float = 0.0
        self.system_sharpe: float = 0.0
        self.trade_history = []  # List[Dict] - avoid 3.12 syntax
        self.enhancement_combinations = {}  # Dict[str, float] - combo -> performance
        
        # Decision making
        self.recent_decisions = []  # List[SystemDecision] - avoid 3.12 syntax
        self.last_reconfiguration: float = 0.0
        self.reconfiguration_cooldown: float = 300.0  # 5 min
        
        # AI models for meta-learning
        self.meta_weights: np.ndarray = np.ones(len(self.ENHANCEMENTS)) / len(self.ENHANCEMENTS)
        self.performance_history: List[float] = []
        
        logger.info("[MasterAI] Central AI Orchestrator initialized")
        logger.info(f"[MasterAI] Managing {len(self.ENHANCEMENTS)} enhancements")

    def _init_enhancements(self):
        """Initialize all enhancement metrics."""
        for key, config in self.ENHANCEMENTS.items():
            self.enhancements[key] = EnhancementMetrics(
                name=config["name"],
                status=EnhancementStatus.ACTIVE if config["priority"] == 1 else EnhancementStatus.LEARNING,
            )

    async def evaluate_system_state(
        self,
        bot_instance,
        market_data: Dict,
        account_info: Dict,
    ) -> SystemDecision:
        """
        Main entry point: Evaluate overall system state and make decisions.
        This is the master AI's primary function.
        NOW FULLY CONTROLS ALL 18 ENHANCEMENTS.
        """
        decision = SystemDecision()
        
        # Check all enhancement health
        health_scores = self._check_enhancement_health()
        avg_health = np.mean(list(health_scores.values())) if health_scores else 0.0
        
        # Check system performance
        performance_score = self._calculate_system_performance()
        
        # Update performance metrics for all enhancements
        if bot_instance:
            self._update_all_enhancement_metrics(bot_instance)
        
        # Check market conditions
        market_state = self._assess_market_conditions(market_data)
        
        # Master AI Decision Matrix (controls everything)
        # Priority 1: System health
        if avg_health < 0.3:
            decision.action = "halt"
            decision.reason = f"System health critical: {avg_health:.1%}"
            decision.confidence = 1.0 - avg_health
            # Disable ALL enhancements except circuit breaker
            decision.enhancements_to_disable = [
                e for e in self.ENHANCEMENTS.keys() if e != "circuit_breaker"
            ]
            
        # Priority 2: Performance degradation
        elif performance_score < -0.05:  # Losing money
            decision.action = "reconfigure"
            decision.reason = f"Poor performance: {performance_score:.2%}"
            decision.confidence = 0.8
            # Get Master AI's recommendation for improvements
            recommendation = self.get_market_recommendation(market_data)
            # Apply recommended configuration
            decision.enhancements_to_enable = recommendation["enhancements_to_trust"]
            # Disable underperforming enhancements
            self._identify_underperformers(decision)
            # Adjust parameters per Master AI
            decision.adjustments = recommendation["parameters_to_adjust"]
            
        # Priority 3: Market state-based control
        elif market_state == "volatile":
            decision.action = "reconfigure"
            decision.reason = f"Volatile market: adjusting enhancements"
            decision.confidence = 0.7
            # Disable enhancements that increase risk in volatility
            decision.enhancements_to_disable = [
                "regime_transition", "sentiment_alpha", "algo_execution"
            ]
            # Enable risk-focused enhancements
            decision.enhancements_to_enable = [
                "circuit_breaker", "var_sizing", "event_avoidance"
            ]
            decision.adjustments = {
                "position_multiplier": 0.5,
                "confidence_threshold": 0.8,
                "max_positions": 3,
            }
            
        # Priority 4: Optimal state - trust the system
        else:
            decision.action = "continue"
            decision.reason = "System optimal - all enhancements active"
            decision.confidence = avg_health
            # Ensure all high-priority enhancements are enabled
            decision.enhancements_to_enable = [
                e for e in self.ENHANCEMENTS.keys()
                if self.ENHANCEMENTS[e]["priority"] == 1
            ]
        
        # Record decision
        self.recent_decisions.append(decision)
        if len(self.recent_decisions) > 100:
            self.recent_decisions = self.recent_decisions[-100:]
        
        # Update system state
        self._update_system_state(decision)
        
        # Log Master AI's control actions
        if decision.enhancements_to_enable:
            logger.info(f"[MasterAI] Enabling: {decision.enhancements_to_enable}")
        if decision.enhancements_to_disable:
            logger.warning(f"[MasterAI] Disabling: {decision.enhancements_to_disable}")
        if decision.adjustments:
            logger.info(f"[MasterAI] Adjusting: {decision.adjustments}")
        
        return decision

    def _update_all_enhancement_metrics(self, bot_instance):
        """Update performance metrics for ALL enhancements from bot instance."""
        # Update PPO integration metrics
        if hasattr(bot_instance, 'regime_system') and bot_instance.regime_system:
            metrics = self.enhancements.get("ppo_integration")
            if metrics:
                # Check if regime system is working
                if bot_instance.regime_system.agents:
                    active_agents = sum(
                        1 for a in bot_instance.regime_system.agents.values() if a is not None
                    )
                    metrics.confidence = active_agents / len(bot_instance.regime_system.agents)
        
        # Update circuit breaker metrics
        if hasattr(bot_instance, 'circuit_breakers'):
            metrics = self.enhancements.get("circuit_breaker")
            if metrics:
                # Check if any circuit breakers are active
                any_halted = any(
                    cb.get_snapshot().should_halt if hasattr(cb, 'get_snapshot') else False
                    for cb in bot_instance.circuit_breakers.values()
                )
                metrics.confidence = 0.3 if any_halted else 1.0
        
        # Update data pipeline metrics
        if hasattr(bot_instance, 'data_manager'):
            metrics = self.enhancements.get("data_pipeline")
            if metrics:
                dm = bot_instance.data_manager
                # Check how many symbols have data
                with_data = sum(
                    1 for sym in SYMBOLS 
                    if dm.get_ohlcv(sym, "1h") is not None and len(dm.get_ohlcv(sym, "1h")) > 50
                )
                metrics.confidence = with_data / len(SYMBOLS)
        
        # Update ensemble weighting metrics
        if hasattr(bot_instance, 'ensemble'):
            metrics = self.enhancements.get("ensemble_weighting")
            if metrics:
                ensemble = bot_instance.ensemble
                if hasattr(ensemble, 'experts') and ensemble.experts:
                    metrics.confidence = len(ensemble.experts) / 3.0  # 3 = expected experts
        
        # Update VaR sizing metrics
        if hasattr(bot_instance, 'risk'):
            metrics = self.enhancements.get("var_sizing")
            if metrics:
                # Check if VaR is being used
                if hasattr(bot_instance.risk, 'calculate_kelly_size'):
                    metrics.confidence = 0.8  # Assume it's working
        
        # Update stress testing metrics
        if hasattr(bot_instance, 'stress_tester'):
            metrics = self.enhancements.get("stress_testing")
            if metrics:
                if bot_instance.stress_tester.results:
                    passed = sum(1 for r in bot_instance.stress_tester.results if r.passed)
                    total = len(bot_instance.stress_tester.results)
                    metrics.confidence = passed / total if total > 0 else 0.5
        
        # Update dashboard metrics
        if hasattr(bot_instance, 'master_ai'):
            metrics = self.enhancements.get("enhanced_dashboard")
            if metrics:
                metrics.confidence = 1.0 if self.system_state == SystemState.OPTIMAL else 0.5

    def _identify_underperformers(self, decision: SystemDecision):
        """Identify underperforming enhancements to disable."""
        for enh, metrics in self.enhancements.items():
            if metrics.pnl_contribution < -500:  # Lost more than $500
                if enh not in decision.enhancements_to_disable:
                    decision.enhancements_to_disable.append(enh)
            elif metrics.failure_count > 10:
                if enh not in decision.enhancements_to_disable:
                    decision.enhancements_to_disable.append(enh)

    def _check_enhancement_health(self) -> Dict[str, float]:
        """Check health of all enhancements."""
        health_scores = {}
        
        for key, metrics in self.enhancements.items():
            score = 1.0
            
            # Factor 1: Failure rate
            if metrics.failure_count > 10:
                score *= 0.5
            elif metrics.failure_count > 5:
                score *= 0.7
            
            # Factor 2: Recency of success
            time_since_success = time.time() - metrics.last_success
            if time_since_success > 3600:  # > 1 hour
                score *= 0.8
            elif time_since_success > 300:  # > 5 min
                score *= 0.9
            
            # Factor 3: Performance contribution
            if metrics.pnl_contribution < -1000:
                score *= 0.6
            elif metrics.pnl_contribution > 1000:
                score *= 1.2
            
            # Factor 4: Confidence
            score *= metrics.confidence
            
            health_scores[key] = min(max(score, 0.0), 1.0)
        
        return health_scores

    def _calculate_system_performance(self) -> float:
        """Calculate overall system performance."""
        if len(self.trade_history) < 5:
            return 0.0
        
        recent_trades = self.trade_history[-50:]
        wins = sum(1 for t in recent_trades if t.get("pnl", 0) > 0)
        total = len(recent_trades)
        
        if total == 0:
            return 0.0
        
        win_rate = wins / total
        
        # Calculate returns
        returns = [t.get("pnl", 0) / self.initial_balance for t in recent_trades]
        if len(returns) < 2:
            return win_rate - 0.5
        
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return > 0:
            self.system_sharpe = avg_return / std_return * np.sqrt(252)
        else:
            self.system_sharpe = 0.0
        
        # Combined score: win_rate (40%) + Sharpe (40%) + recent P&L (20%)
        recent_pnl = sum(returns[-10:]) if len(returns) >= 10 else sum(returns)
        performance = (win_rate - 0.5) * 0.4 + self.system_sharpe * 0.1 + recent_pnl * 2.0
        
        return float(performance)

    def _assess_market_conditions(self, market_data: Dict) -> str:
        """Assess current market conditions."""
        if not market_data:
            return "unknown"
        
        # Check volatility across symbols
        volatilities = []
        for sym, data in market_data.items():
            if isinstance(data, dict):
                if "atr" in data and "price" in data and data["price"] > 0:
                    vol = data["atr"] / data["price"]
                    volatilities.append(vol)
            elif isinstance(data, (int, float)) and data > 0:
                # data is just a price value - skip vol calc
                pass
        
        if not volatilities:
            return "normal"
        
        avg_vol = float(np.mean(volatilities))
        
        if avg_vol > 0.02:  # >2% volatility
            return "volatile"
        elif avg_vol < 0.005:  # <0.5% volatility
            return "calm"
        else:
            return "normal"

    def _suggest_configuration_change(self, decision: SystemDecision):
        """Suggest configuration changes based on meta-learning."""
        # Get current enhancement combination signature
        active = sorted([
            k for k, v in self.enhancements.items()
            if v.status == EnhancementStatus.ACTIVE
        ])
        sig = "|".join(active)
        
        # Find best performing combination
        best_sig = max(
            self.enhancement_combinations.items(),
            key=lambda x: x[1],
            default=(sig, 0.0)
        )
        
        if best_sig[0] != sig and best_sig[1] > self.enhancement_combinations.get(sig, 0.0) + 0.01:
            # Switch to better configuration
            new_active = set(best_sig[0].split("|"))
            current_active = set(active)
            
            # Enable new ones
            for enh in new_active - current_active:
                if enh in self.enhancements:
                    decision.enhancements_to_enable.append(enh)
            
            # Disable old ones
            for enh in current_active - new_active:
                if enh in self.enhancements:
                    decision.enhancements_to_disable.append(enh)
            
            decision.reason += f" | Switching to better config (perf: {best_sig[1]:.3f})"

    def _update_system_state(self, decision: SystemDecision):
        """Update system state based on decision."""
        if decision.action == "halt":
            self.system_state = SystemState.HALTED
        elif decision.action == "reconfigure":
            self.system_state = SystemState.DEGRADED
            self.last_reconfiguration = time.time()
        elif decision.action == "continue":
            if self.system_state == SystemState.HALTED:
                self.system_state = SystemState.RECOVERING
            elif self.system_state == SystemState.RECOVERING:
                self.system_state = SystemState.OPTIMAL
        
        # Apply enhancement changes
        for enh in decision.enhancements_to_enable:
            if enh in self.enhancements:
                self.enhancements[enh].status = EnhancementStatus.ACTIVE
                logger.info(f"[MasterAI] Enabled enhancement: {self.ENHANCEMENTS[enh]['name']}")
        
        for enh in decision.enhancements_to_disable:
            if enh in self.enhancements:
                self.enhancements[enh].status = EnhancementStatus.DISABLED
                logger.warning(f"[MasterAI] Disabled enhancement: {self.ENHANCEMENTS[enh]['name']}")

    def update_enhancement_performance(
        self,
        enhancement_name: str,
        pnl: float = 0.0,
        success: bool = True,
        confidence: float = 1.0,
        trades_count: int = 0,
    ):
        """Update performance metrics for an enhancement."""
        if enhancement_name not in self.enhancements:
            return
        
        metrics = self.enhancements[enhancement_name]
        
        if success:
            metrics.last_success = time.time()
            metrics.confidence = metrics.confidence * 0.9 + confidence * 0.1
        else:
            metrics.failure_count += 1
            metrics.confidence *= 0.95
        
        metrics.pnl_contribution += pnl
        metrics.trades_inflnced += trades_count
        
        # Update combination performance
        active = sorted([
            k for k, v in self.enhancements.items()
            if v.status == EnhancementStatus.ACTIVE
        ])
        sig = "|".join(active)
        
        if sig not in self.enhancement_combinations:
            self.enhancement_combinations[sig] = 0.0
        
        self.enhancement_combinations[sig] = (
            self.enhancement_combinations[sig] * 0.9 + pnl * 0.01
        )

    def record_trade(self, trade: Dict):
        """Record a trade for system-wide learning."""
        self.trade_history.append(trade)
        self.system_pnl += trade.get("pnl", 0.0)
        
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-500:]
        
        # Update performance history
        self.performance_history.append(trade.get("pnl", 0.0) / self.initial_balance)
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

    async def auto_reconfigure(self, bot_instance) -> Dict:
        """
        Automatically reconfigure system based on meta-learning.
        This is the key AI-driven adaptation.
        """
        if time.time() - self.last_reconfiguration < self.reconfiguration_cooldown:
            return {"action": "cooldown", "reason": "Too soon since last reconfiguration"}
        
        # Evaluate current configuration
        current_active = sorted([
            k for k, v in self.enhancements.items()
            if v.status == EnhancementStatus.ACTIVE
        ])
        current_sig = "|".join(current_active)
        current_perf = self.enhancement_combinations.get(current_sig, 0.0)
        
        # Generate candidate configurations
        candidates = self._generate_candidates(current_active)
        
        best_candidate = current_sig
        best_perf = current_perf
        
        for candidate in candidates:
            cand_sig = "|".join(sorted(candidate))
            cand_perf = self.enhancement_combinations.get(cand_sig, -999.0)
            if cand_perf > best_perf + 0.01:  # Need 1% improvement
                best_candidate = candidate
                best_perf = cand_perf
        
        if best_candidate != current_active:
            # Apply new configuration
            decision = SystemDecision(
                action="reconfigure",
                reason=f"Auto-reconfiguration: {current_perf:.3f} -> {best_perf:.3f}",
                confidence=0.8,
            )
            
            new_active = set(best_candidate)
            current_set = set(current_active)
            
            for enh in new_active - current_set:
                decision.enhancements_to_enable.append(enh)
            for enh in current_set - new_active:
                decision.enhancements_to_disable.append(enh)
            
            self._update_system_state(decision)
            
            return {
                "action": "reconfigured",
                "old_config": current_active,
                "new_config": best_candidate,
                "performance_improvement": best_perf - current_perf,
            }
        
        return {"action": "no_change", "reason": "Current configuration is optimal"}

    def _generate_candidates(self, current_active: List[str]) -> List[List[str]]:
        """Generate candidate configurations to try."""
        candidates = []
        
        # Strategy 1: Disable worst performing active enhancement
        if current_active:
            worst = min(
                current_active,
                key=lambda x: self.enhancements[x].pnl_contribution
            )
            candidate = [e for e in current_active if e != worst]
            candidates.append(candidate)
        
        # Strategy 2: Enable best disabled enhancement
        disabled = [
            k for k, v in self.enhancements.items()
            if v.status == EnhancementStatus.DISABLED and self.ENHANCEMENTS[k]["priority"] <= 2
        ]
        if disabled:
            best_disabled = max(
                disabled,
                key=lambda x: self.enhancements[x].pnl_contribution
            )
            candidate = current_active + [best_disabled]
            candidates.append(candidate)
        
        # Strategy 3: Toggle medium priority enhancements
        med_priority = [
            k for k, v in self.ENHANCEMENTS.items()
            if v["priority"] == 2
        ]
        for enh in med_priority:
            if enh in current_active:
                candidate = [e for e in current_active if e != enh]
            else:
                candidate = current_active + [enh]
            candidates.append(candidate)
        
        return candidates

    def get_system_status(self) -> Dict:
        """Get comprehensive system status."""
        return {
            "system_state": self.system_state.value,
            "system_pnl": self.system_pnl,
            "system_sharpe": self.system_sharpe,
            "total_trades": len(self.trade_history),
            "enhancements": {
                k: {
                    "name": v.name,
                    "status": v.status.value,
                    "confidence": v.confidence,
                    "pnl_contribution": v.pnl_contribution,
                    "failure_count": v.failure_count,
                    "priority": self.ENHANCEMENTS[k]["priority"],
                }
                for k, v in self.enhancements.items()
            },
            "recent_decisions": [
                {
                    "action": d.action,
                    "reason": d.reason,
                    "confidence": d.confidence,
                    "timestamp": d.timestamp,
                }
                for d in self.recent_decisions[-10:]
            ],
            "best_configurations": sorted(
                [{"config": k, "performance": v}
                for k, v in self.enhancement_combinations.items()
                if v != 0.0
            ],
            key=lambda x: x["performance"],
            reverse=True
        )[:5],
        }

    def get_market_recommendation(self, market_data: Dict) -> Dict:
        """
        Get AI recommendation for current market conditions.
        This is the Master AI providing trading guidance.
        """
        market_state = self._assess_market_conditions(market_data)
        
        recommendations = {
            "market_state": market_state,
            "recommended_action": "HOLD",
            "confidence": 0.5,
            "reasoning": "",
            "enhancements_to_trust": [],
            "parameters_to_adjust": {},
        }
        
        # Market-state specific recommendations
        if market_state == "volatile":
            recommendations["recommended_action"] = "REDUCE_SIZE"
            recommendations["reasoning"] = "High volatility detected"
            recommendations["enhancements_to_trust"] = ["circuit_breaker", "var_sizing"]
            recommendations["parameters_to_adjust"] = {
                "position_multiplier": 0.5,
                "confidence_threshold": 0.8,
            }
        elif market_state == "calm":
            recommendations["recommended_action"] = "NORMAL"
            recommendations["reasoning"] = "Low volatility, normal trading"
            recommendations["enhancements_to_trust"] = ["ppo_integration", "ensemble_weighting"]
            recommendations["parameters_to_adjust"] = {
                "position_multiplier": 1.0,
                "confidence_threshold": 0.65,
            }
        else:  # normal
            recommendations["recommended_action"] = "NORMAL"
            recommendations["reasoning"] = "Normal market conditions"
            recommendations["enhancements_to_trust"] = list(self.enhancements.keys())[:5]
            recommendations["parameters_to_adjust"] = {
                "position_multiplier": 0.8,
                "confidence_threshold": 0.7,
            }
        
        # Boost confidence if system is performing well
        if self.system_sharpe > 1.0:
            recommendations["confidence"] = 0.8
        elif self.system_sharpe < 0:
            recommendations["confidence"] = 0.3
        
        return recommendations


logger.info("[MasterAI] Master AI Orchestrator module loaded")
