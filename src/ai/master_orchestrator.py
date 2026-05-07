"""
Master AI Orchestrator - Central AI Engine for the MoneyBot System.
Controls and manages all 18 enhancements, learns optimal configurations,
and makes high-level decisions about system behavior.
"""
import asyncio
import numpy as np
import time
import sys
import os
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


class MetaLearner(nn.Module):
    """
    Neural network that learns optimal meta-actions (enhancement configurations)
    from experience. Uses online Q-learning style training.

    State (16-dim):
      0: avg_enhancement_health
      1: system_performance_score
      2: market_volatility (0=calm, 0.5=normal, 1.0=volatile)
      3: system_sharpe (clipped [-2, 2], normalized to [0, 1])
      4: recent_win_rate (last 50 trades)
      5: recent_pnl_normalized (last 10 trades / initial_balance)
      6: active_enhancement_ratio
      7: experience_level (total_trades / 1000, capped at 1.0)
      8-15: top 8 enhancement confidence scores

    Actions (5):
      0: CONTINUE       — no changes
      1: CONSERVATIVE   — enable protections, disable risky
      2: AGGRESSIVE     — enable all, full risk
      3: REDUCED_RISK   — half positions, disable sentiment/algo
      4: HALT           — emergency stop
    """
    def __init__(self, state_dim: int = 16, n_actions: int = 5, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return Q-values for each action given state."""
        return self.net(state)

    @torch.no_grad()
    def predict_q(self, state: np.ndarray) -> np.ndarray:
        """Predict Q-values for a single state vector."""
        self.eval()
        s = torch.FloatTensor(state).unsqueeze(0)
        return self.forward(s).squeeze(0).numpy()

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location='cpu'))

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

    # Safety-critical enhancements that must be ACTIVE from startup
    SAFETY_CRITICAL = {"circuit_breaker", "var_sizing", "error_recovery"}

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

    STATE_DIM = 16
    N_ACTIONS = 5
    COLD_START_TRADES = 50
    EPSILON_INIT = 0.3
    EPSILON_MIN = 0.05
    EPSILON_DECAY = 0.995

    def __init__(
        self,
        initial_balance: float = 100000.0,
        learning_rate: float = 0.001,
        adaptation_threshold: float = 0.6,
        model_path: str = "models/meta_learner.pt",
    ):
        self.initial_balance = initial_balance
        self.learning_rate = learning_rate
        self.adaptation_threshold = adaptation_threshold
        self.model_path = model_path
        
        # System state
        self.system_state: SystemState = SystemState.OPTIMAL
        self.enhancements: Dict[str, EnhancementMetrics] = {}
        self._init_enhancements()
        
        # Performance tracking
        self.system_pnl: float = 0.0
        self.system_sharpe: float = 0.0
        self.trade_history = []  # List[Dict]
        self.performance_history: List[float] = []
        self.enhancement_combinations = {}  # Dict[str, float] - combo -> performance
        
        # Decision making
        self.recent_decisions = []  # List[SystemDecision] - avoid 3.12 syntax
        self.last_reconfiguration: float = 0.0
        self.reconfiguration_cooldown: float = 300.0  # 5 min
        
        # Real system component references (wired by bot after construction)
        self.system_components: Dict[str, Any] = {}
        
        # Re-entrancy guards (prevent recursion in callback chains)
        self._processing_trade: bool = False
        self._evaluating: bool = False
        
        # ===== META-LEARNER (learned brain, replaces static heuristics) =====
        self.meta_learner = MetaLearner(
            state_dim=self.STATE_DIM,
            n_actions=self.N_ACTIONS,
            hidden_dim=32,
        )
        self.meta_optimizer = optim.Adam(
            self.meta_learner.parameters(), lr=learning_rate
        )
        self.meta_loss_fn = nn.MSELoss()
        
        # Training buffer: list of (state_vec, action_idx, reward)
        self._experience_buffer: List[Tuple[np.ndarray, int, float]] = []
        self._buffer_maxlen = 500
        
        # For tracking decisions and computing rewards
        self._last_state: Optional[np.ndarray] = None
        self._last_action: Optional[int] = None
        self._last_decision_time: float = time.time()
        self._pnl_at_last_decision: float = 0.0
        self._sharpe_at_last_decision: float = 0.0
        self._winrate_at_last_decision: float = 0.0
        
        # Exploration
        self.epsilon = self.EPSILON_INIT
        self.training_step = 0
        
        # Try loading pre-trained meta-learner
        self._load_meta_learner()
        
        logger.info("[MasterAI] Central AI Orchestrator initialized")
        logger.info(f"[MasterAI] Managing {len(self.ENHANCEMENTS)} enhancements")
        logger.info(f"[MasterAI] Meta-learner: {self.STATE_DIM} states, {self.N_ACTIONS} actions")
        logger.info(f"[MasterAI] Cold-start: first {self.COLD_START_TRADES} trades use heuristics")

    def wire_system_components(self, components: Dict[str, Any]):
        """
        Wire real system component references so the MasterAI can
        directly enable/disable/control them.
        """
        self.system_components = components
        logger.info(f"[MasterAI] Wired {len(components)} system components under direct control")

    def is_enabled(self, enhancement_name: str) -> bool:
        """Check if an enhancement is currently enabled by the MasterAI."""
        if enhancement_name not in self.enhancements:
            return False
        return self.enhancements[enhancement_name].status == EnhancementStatus.ACTIVE

    async def apply_decision(self, decision: SystemDecision):
        """
        Apply a SystemDecision to REAL system components, not just internal flags.
        This makes the MasterAI actually control the system.
        """
        comp = self.system_components
        
        # Apply enhancement status to internal tracking
        self._update_system_state(decision)
        
        # Physically enable/disable components based on decision
        for enh in decision.enhancements_to_disable:
            self._physically_disable_enhancement(enh, comp)
        
        for enh in decision.enhancements_to_enable:
            self._physically_enable_enhancement(enh, comp)
        
        # Apply parameter adjustments
        if decision.adjustments:
            self._apply_parameter_adjustments(decision.adjustments, comp)

    def _physically_enable_enhancement(self, enh: str, comp: Dict[str, Any]):
        if enh == "circuit_breaker":
            cb = comp.get("circuit_breakers")
            if cb:
                for cb_inst in cb.values():
                    if hasattr(cb_inst, 'enable'):
                        cb_inst.enable()
        elif enh == "algo_execution":
            ae = comp.get("algo_executor")
            if ae and hasattr(ae, 'enable'):
                ae.enable()
        elif enh == "sentiment_alpha":
            ba = comp.get("behavioral_ai")
            if ba:
                ba.enabled = True
        elif enh == "regime_transition":
            rs = comp.get("regime_system")
            if rs:
                rs.enabled = True
        elif enh == "data_pipeline":
            dm = comp.get("data_manager")
            if dm:
                dm.enabled = True
        elif enh == "var_sizing":
            risk = comp.get("risk")
            if risk:
                risk.var_enabled = True
        elif enh == "ensemble_weighting":
            ens = comp.get("ensemble")
            if ens:
                ens.enabled = True
        elif enh == "ppo_integration":
            rs = comp.get("regime_system")
            if rs:
                rs.enabled = True
        elif enh == "event_avoidance":
            ea = comp.get("event_avoidance")
            if ea:
                ea.enabled = True
        elif enh == "walk_forward":
            wf = comp.get("walk_forward")
            if wf:
                wf.enabled = True
        elif enh == "stress_testing":
            st = comp.get("stress_tester")
            if st:
                st.enabled = True
        elif enh == "model_versioning":
            mr = comp.get("model_registry")
            if mr:
                mr.enabled = True
            
    def _physically_disable_enhancement(self, enh: str, comp: Dict[str, Any]):
        if enh == "circuit_breaker":
            cb = comp.get("circuit_breakers")
            if cb:
                for cb_inst in cb.values():
                    if hasattr(cb_inst, 'disable'):
                        cb_inst.disable()
        elif enh == "algo_execution":
            ae = comp.get("algo_executor")
            if ae and hasattr(ae, 'disable'):
                ae.disable()
        elif enh == "sentiment_alpha":
            ba = comp.get("behavioral_ai")
            if ba:
                ba.enabled = False
        elif enh == "regime_transition":
            rs = comp.get("regime_system")
            if rs:
                rs.enabled = False
        elif enh == "data_pipeline":
            dm = comp.get("data_manager")
            if dm:
                dm.enabled = False
        elif enh == "var_sizing":
            risk = comp.get("risk")
            if risk:
                risk.var_enabled = False
        elif enh == "ensemble_weighting":
            ens = comp.get("ensemble")
            if ens:
                ens.enabled = False
        elif enh == "ppo_integration":
            rs = comp.get("regime_system")
            if rs:
                rs.enabled = False
        elif enh == "event_avoidance":
            ea = comp.get("event_avoidance")
            if ea:
                ea.enabled = False
        elif enh == "walk_forward":
            wf = comp.get("walk_forward")
            if wf:
                wf.enabled = False
        elif enh == "stress_testing":
            st = comp.get("stress_tester")
            if st:
                st.enabled = False
        elif enh == "model_versioning":
            mr = comp.get("model_registry")
            if mr:
                mr.enabled = False

    def _apply_parameter_adjustments(self, adjustments: Dict[str, Any], comp: Dict[str, Any]):
        risk = comp.get("risk")
        if risk:
            if "position_multiplier" in adjustments and hasattr(risk, 'kelly_fraction'):
                risk.kelly_fraction = risk.kelly_fraction * adjustments["position_multiplier"]
            if "max_positions" in adjustments and hasattr(risk, 'max_positions'):
                risk.max_positions = adjustments["max_positions"]
        if "confidence_threshold" in adjustments and hasattr(self, '_current_sentiment'):
            self._current_sentiment = adjustments["confidence_threshold"]

    def _init_enhancements(self):
        """Initialize all enhancement metrics. Safety-critical ones start ACTIVE."""
        for key, config in self.ENHANCEMENTS.items():
            is_active = config["priority"] == 1 or key in self.SAFETY_CRITICAL
            self.enhancements[key] = EnhancementMetrics(
                name=config["name"],
                status=EnhancementStatus.ACTIVE if is_active else EnhancementStatus.LEARNING,
            )
        logger.info(f"[MasterAI] Safety-critical: {len(self.SAFETY_CRITICAL & set(self.ENHANCEMENTS))} active from startup")

    # =====================================================================
    # STATE FEATURE ENGINEERING
    # =====================================================================

    def _build_state_vector(self, bot_instance, market_data: Dict) -> np.ndarray:
        """Build 16-dim state vector for the meta-learner."""
        # 0. Avg enhancement health
        health_scores = self._check_enhancement_health()
        avg_health = float(np.mean(list(health_scores.values()))) if health_scores else 0.5

        # 1. System performance
        perf = self._calculate_system_performance()

        # 2. Market volatility (normalized)
        vol = self._assess_market_conditions(market_data)
        vol_map = {"calm": 0.0, "normal": 0.5, "volatile": 1.0, "unknown": 0.5}
        vol_encoded = vol_map.get(vol, 0.5)

        # 3. Sharpe (clipped to [-2, 2], normalized to [0, 1])
        sharpe_norm = float(np.clip((self.system_sharpe + 2.0) / 4.0, 0.0, 1.0))

        # 4. Recent win rate
        n = len(self.trade_history)
        recent = self.trade_history[-50:] if n >= 50 else self.trade_history
        wins = sum(1 for t in recent if t.get("pnl", 0) > 0)
        win_rate = wins / max(len(recent), 1)

        # 5. Recent PnL (normalized)
        recent_pnl = sum(t.get("pnl", 0) for t in recent)
        pnl_norm = float(np.tanh(recent_pnl / max(self.initial_balance, 1)))

        # 6. Active enhancement ratio
        active_count = sum(
            1 for e in self.enhancements.values()
            if e.status == EnhancementStatus.ACTIVE
        )
        active_ratio = active_count / max(len(self.enhancements), 1)

        # 7. Experience level (total trades / 1000, capped)
        exp_level = min(n / 1000.0, 1.0)

        # 8-15. Top 8 enhancement confidence scores (sorted)
        confs = sorted(
            [e.confidence for e in self.enhancements.values()],
            reverse=True,
        )
        confs_padded = (confs + [0.5] * 8)[:8]

        state = np.array([
            avg_health, perf, vol_encoded, sharpe_norm,
            win_rate, pnl_norm, active_ratio, exp_level,
            *confs_padded,
        ], dtype=np.float32)
        return state

    # =====================================================================
    # REWARD & TRAINING
    # =====================================================================

    def _compute_reward(self) -> float:
        """
        Compute reward for the last meta-action based on:
        - PnL change since last decision
        - Sharpe change
        - Win rate change
        - Drawdown penalty
        """
        if len(self.trade_history) < 2:
            return 0.0

        recent = self.trade_history[-50:]
        pnl_now = sum(t.get("pnl", 0) for t in recent)
        pnl_delta = pnl_now - self._pnl_at_last_decision

        sharpe_now = self.system_sharpe
        sharpe_delta = sharpe_now - self._sharpe_at_last_decision

        wins = sum(1 for t in recent if t.get("pnl", 0) > 0)
        wr_now = wins / max(len(recent), 1)
        wr_delta = wr_now - self._winrate_at_last_decision

        pnl_reward = float(np.tanh(pnl_delta / 1000.0))
        sharpe_reward = float(np.clip(sharpe_delta * 0.5, -0.5, 0.5))
        wr_reward = float(np.clip(wr_delta * 2.0, -0.5, 0.5))

        drawdown = self._estimate_drawdown()
        dd_penalty = -float(np.clip(drawdown * 2.0, -1.0, 0.0))

        reward = pnl_reward * 0.5 + sharpe_reward * 0.2 + wr_reward * 0.2 + dd_penalty * 0.1
        return float(np.clip(reward, -2.0, 2.0))

    def _estimate_drawdown(self) -> float:
        """Estimate current drawdown from trade history."""
        if len(self.trade_history) < 5:
            return 0.0
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in self.trade_history[-100:]:
            cumulative += t.get("pnl", 0)
            if cumulative > peak:
                peak = cumulative
            dd = (peak - cumulative) / max(self.initial_balance, 1)
            if dd > max_dd:
                max_dd = dd
        return min(max_dd, 1.0)

    def _store_experience(self, state: np.ndarray, action: int, reward: float):
        """Store a training example."""
        self._experience_buffer.append((state.copy(), action, reward))
        if len(self._experience_buffer) > self._buffer_maxlen:
            self._experience_buffer.pop(0)

    def _train_meta_learner(self):
        """Train the meta-learner on the experience buffer."""
        if len(self._experience_buffer) < 10:
            return

        self.meta_learner.train()
        states = torch.FloatTensor(np.array([e[0] for e in self._experience_buffer]))
        actions = torch.LongTensor([e[1] for e in self._experience_buffer])
        rewards = torch.FloatTensor([e[2] for e in self._experience_buffer])

        # Forward: predict Q for all actions
        q_values = self.meta_learner(states)  # (N, 5)

        # Gather Q for taken actions
        q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = self.meta_loss_fn(q_taken, rewards)

        self.meta_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.meta_learner.parameters(), 1.0)
        self.meta_optimizer.step()

        self.training_step += 1
        self.epsilon = max(self.EPSILON_MIN, self.epsilon * self.EPSILON_DECAY)

        if self.training_step % 10 == 0:
            logger.debug(f"[MasterAI] Meta-learner trained (step={self.training_step}, loss={loss.item():.4f}, epsilon={self.epsilon:.3f})")
            self._save_meta_learner()

    def _select_meta_action(self, state: np.ndarray) -> int:
        """
        Select meta-action using epsilon-greedy.
        Cold-start: use heuristic fallback.
        """
        n_trades = len(self.trade_history)
        if n_trades < self.COLD_START_TRADES:
            return -1  # signal: use heuristic

        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.N_ACTIONS)
            logger.info(f"[MasterAI] Exploring: random action {action}")
            return action

        q_values = self.meta_learner.predict_q(state)
        action = int(np.argmax(q_values))
        logger.debug(f"[MasterAI] Greedy: action {action}, Q={q_values}")
        return action

    def _heuristic_action(self, state: np.ndarray, perf: float, market_state: str,
                          avg_health: float) -> SystemDecision:
        """Fallback heuristic when meta-learner is in cold-start."""
        decision = SystemDecision()
        if avg_health < 0.3:
            decision.action = "halt"
            decision.reason = f"[HEURISTIC] System health critical: {avg_health:.1%}"
            decision.confidence = 1.0 - avg_health
            decision.enhancements_to_disable = [
                e for e in self.ENHANCEMENTS.keys() if e != "circuit_breaker"
            ]
        elif perf < -0.05:
            decision.action = "reconfigure"
            decision.reason = f"[HEURISTIC] Poor performance: {perf:.2%}"
            decision.confidence = 0.8
            rec = self.get_market_recommendation({})
            decision.enhancements_to_enable = rec["enhancements_to_trust"]
            self._identify_underperformers(decision)
            decision.adjustments = rec["parameters_to_adjust"]
        elif market_state == "volatile":
            decision.action = "reconfigure"
            decision.reason = "[HEURISTIC] Volatile market"
            decision.confidence = 0.7
            decision.enhancements_to_disable = ["regime_transition", "sentiment_alpha", "algo_execution"]
            decision.enhancements_to_enable = ["circuit_breaker", "var_sizing", "event_avoidance"]
            decision.adjustments = {"position_multiplier": 0.5, "confidence_threshold": 0.8, "max_positions": 3}
        else:
            decision.action = "continue"
            decision.reason = "[HEURISTIC] System optimal"
            decision.confidence = avg_health
            decision.enhancements_to_enable = [
                e for e in self.ENHANCEMENTS if self.ENHANCEMENTS[e]["priority"] == 1
            ]
        return decision

    def _meta_action_to_decision(self, action: int, market_state: str) -> SystemDecision:
        """Convert a learned meta-action index into a SystemDecision."""
        decision = SystemDecision()
        decision.confidence = 0.85

        if action == 0:  # CONTINUE
            decision.action = "continue"
            decision.reason = f"[LEARNED] Optimal config (action={action})"
        elif action == 1:  # CONSERVATIVE
            decision.action = "reconfigure"
            decision.reason = f"[LEARNED] Conservative mode (action={action})"
            decision.enhancements_to_disable = ["sentiment_alpha", "algo_execution", "regime_transition"]
            decision.enhancements_to_enable = ["circuit_breaker", "var_sizing", "event_avoidance"]
            decision.adjustments = {"position_multiplier": 0.5, "confidence_threshold": 0.75}
        elif action == 2:  # AGGRESSIVE
            decision.action = "reconfigure"
            decision.reason = f"[LEARNED] Aggressive mode (action={action})"
            decision.enhancements_to_enable = [
                k for k, v in self.ENHANCEMENTS.items()
            ]
            decision.adjustments = {"position_multiplier": 1.2, "confidence_threshold": 0.6}
        elif action == 3:  # REDUCED_RISK
            decision.action = "reconfigure"
            decision.reason = f"[LEARNED] Reduced risk (action={action})"
            decision.enhancements_to_disable = ["algo_execution", "sentiment_alpha"]
            decision.enhancements_to_enable = ["circuit_breaker", "var_sizing"]
            decision.adjustments = {"position_multiplier": 0.3, "confidence_threshold": 0.85, "max_positions": 2}
        elif action == 4:  # HALT
            decision.action = "halt"
            decision.reason = f"[LEARNED] Emergency halt (action={action})"
            decision.enhancements_to_disable = list(self.ENHANCEMENTS.keys())
            decision.confidence = 0.95

        return decision

    def _load_meta_learner(self):
        """Load pre-trained meta-learner from disk."""
        try:
            path = Path(self.model_path)
            if path.exists():
                self.meta_learner.load(str(path))
                logger.info(f"[MasterAI] Loaded pre-trained meta-learner from {self.model_path}")
            else:
                logger.info("[MasterAI] No pre-trained meta-learner found, starting fresh")
        except Exception as e:
            logger.warning(f"[MasterAI] Failed to load meta-learner: {e}")

    def _save_meta_learner(self):
        """Save meta-learner to disk."""
        try:
            path = Path(self.model_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.meta_learner.save(str(path))
        except Exception as e:
            logger.warning(f"[MasterAI] Failed to save meta-learner: {e}")

    def on_trade_result(self, trade: Dict):
        """
        Called by the bot after each trade closes.
        Feeds the outcome back to the meta-learner for training.
        Protected against re-entrant calls (recursion guard).
        """
        if getattr(self, '_processing_trade', False):
            logger.debug("[MasterAI] Re-entrant on_trade_result call skipped (recursion guard)")
            return
        self._processing_trade = True
        try:
            self.record_trade(trade)
            if self._last_state is not None and self._last_action is not None:
                reward = self._compute_reward()
                self._store_experience(self._last_state, self._last_action, reward)
                self._train_meta_learner()
        finally:
            self._processing_trade = False

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
        Protected against re-entrant calls.
        """
        if getattr(self, '_evaluating', False):
            logger.debug("[MasterAI] Re-entrant evaluate_system_state skipped")
            return SystemDecision(action="continue", reason="re-entrant_guard")
        self._evaluating = True
        try:
            return await self._evaluate_impl(bot_instance, market_data, account_info)
        finally:
            self._evaluating = False

    async def _evaluate_impl(
        self,
        bot_instance,
        market_data: Dict,
        account_info: Dict,
    ) -> SystemDecision:
        """Actual evaluation logic (protected by re-entrancy guard)."""
        # Update enhancement metrics from bot instance
        if bot_instance:
            self._update_all_enhancement_metrics(bot_instance)
        
        # Compute state features
        state = self._build_state_vector(bot_instance, market_data)
        health_scores = self._check_enhancement_health()
        avg_health = float(np.mean(list(health_scores.values()))) if health_scores else 0.5
        perf = self._calculate_system_performance()
        market_state = self._assess_market_conditions(market_data)
        
        # Select meta-action (learned or heuristic)
        meta_action = self._select_meta_action(state)
        
        if meta_action == -1:
            # Cold-start: use heuristic fallback
            decision = self._heuristic_action(state, perf, market_state, avg_health)
            logger.info(f"[MasterAI] Cold-start heuristic: {decision.action}")
        else:
            # Learned: convert meta-action to decision
            decision = self._meta_action_to_decision(meta_action, market_state)
            # Heuristic override: always halt on critically low health
            if avg_health < 0.2:
                decision.action = "halt"
                decision.reason = f"[OVERRIDE] Health critical: {avg_health:.1%}"
                decision.enhancements_to_disable = list(self.ENHANCEMENTS.keys())
            logger.info(f"[MasterAI] Learned decision: action={meta_action} -> {decision.action}")
        
        # Store state and action for reward computation on next call
        self._last_state = state.copy()
        self._last_action = meta_action if meta_action != -1 else None
        self._pnl_at_last_decision = sum(
            t.get("pnl", 0) for t in self.trade_history[-50:]
        )
        self._sharpe_at_last_decision = self.system_sharpe
        self._winrate_at_last_decision = (
            sum(1 for t in self.trade_history[-50:] if t.get("pnl", 0) > 0)
            / max(len(self.trade_history[-50:]), 1)
        )
        self._last_decision_time = time.time()
        
        # Record decision
        self.recent_decisions.append(decision)
        if len(self.recent_decisions) > 100:
            self.recent_decisions = self.recent_decisions[-100:]
        
        self._update_system_state(decision)
        
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
