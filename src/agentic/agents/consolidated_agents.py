"""
DEPRECATED: This consolidated agent architecture is replaced by the 20-agent
full agentic system in agentic/core/ and agentic/agents/.
Kept for reference only — will be removed in next major version.
"""

"""
Consolidated Agent Architecture — 6 agents instead of 20.

Reduces AgentBus communication overhead while maintaining all functionality.

Agent responsibilities:
  1. Orchestrator    — scheduling, state management, human-in-loop, error escalation
  2. SignalEngine    — feature pipeline -> ensemble -> signal (pure function, stateless)
  3. RiskManager  — all risk checks (RiskAgent + AdaptiveRisk + CircuitBreaker)
  4. ExecutionManager — execution + position management + cost tracking + reconciliation
  5. LearningManager  — drift detection, retraining, model registry, champion/challenger
  6. DataManager      — market data ingestion, connectivity, health monitoring
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
from loguru import logger


class AgentRole(Enum):
    ORCHESTRATOR = "orchestrator"
    SIGNAL_ENGINE = "signal_engine"
    RISK_MANAGER = "risk_manager"
    EXECUTION_MANAGER = "execution_manager"
    LEARNING_MANAGER = "learning_manager"
    DATA_MANAGER = "data_manager"


class AgentStatus(Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    DEGRADED = "degraded"
    HALTED = "halted"


@dataclass
class AgentMessage:
    source: str
    target: str
    message_type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    urgent: bool = False


class BaseConsolidatedAgent:
    """Base class for all consolidated agents."""

    def __init__(self, name: str, role: AgentRole):
        self.name = name
        self.role = role
        self.status = AgentStatus.INITIALIZING
        self._message_queue: List[AgentMessage] = []
        self._callbacks: Dict[str, Callable] = {}

    def handle_message(self, msg: AgentMessage) -> Optional[AgentMessage]:
        """Process an incoming message. Override in subclasses."""
        raise NotImplementedError

    def send(self, msg: AgentMessage) -> bool:
        """Queue a message to another agent."""
        self._message_queue.append(msg)
        return True

    def drain_messages(self) -> List[AgentMessage]:
        """Get all queued outgoing messages."""
        msgs = list(self._message_queue)
        self._message_queue.clear()
        return msgs

    def register_callback(self, event: str, callback: Callable):
        self._callbacks[event] = callback

    def start(self):
        self.status = AgentStatus.RUNNING
        logger.info(f"Agent {self.name} started ({self.role.value})")

    def stop(self):
        self.status = AgentStatus.HALTED
        logger.info(f"Agent {self.name} stopped")


# --- Agent Implementations --------------------------------------------------


class OrchestratorAgent(BaseConsolidatedAgent):
    """System orchestrator -- scheduling, state, human-in-loop, error escalation.

    Consolidates: MasterAgent + MonitoringAgent (health dashboard)
    """

    def __init__(self):
        super().__init__("orchestrator", AgentRole.ORCHESTRATOR)
        self._trading_enabled = True
        self._error_count = 0
        self._max_errors = 10
        self._agent_status: Dict[str, AgentStatus] = {}
        self._system_start_time = time.time()
        self._schedules: Dict[str, float] = {}

    def handle_message(self, msg: AgentMessage) -> Optional[AgentMessage]:
        if msg.message_type == "error":
            self._error_count += 1
            if self._error_count >= self._max_errors:
                self._trading_enabled = False
                logger.error("Orchestrator: max errors reached, halting trading")
                return AgentMessage(
                    source=self.name,
                    target="*",
                    message_type="system_halt",
                    payload={"reason": f"{self._max_errors} consecutive errors"},
                    urgent=True,
                )
        elif msg.message_type == "health_check":
            return AgentMessage(
                source=self.name,
                target=msg.source,
                message_type="health_status",
                payload={
                    "status": self.status.value,
                    "trading_enabled": self._trading_enabled,
                    "uptime": time.time() - self._system_start_time,
                    "error_count": self._error_count,
                },
            )
        elif msg.message_type == "agent_status":
            agent_name = msg.payload.get("agent_name", "unknown")
            agent_status = msg.payload.get("status", "running")
            try:
                self._agent_status[agent_name] = AgentStatus(agent_status)
            except ValueError:
                pass
        return None

    def set_schedule(self, task_name: str, interval_seconds: float):
        self._schedules[task_name] = interval_seconds

    def is_trading_enabled(self) -> bool:
        return self._trading_enabled

    def get_system_state(self) -> Dict:
        return {
            "trading_enabled": self._trading_enabled,
            "uptime": time.time() - self._system_start_time,
            "error_count": self._error_count,
            "agent_status": {k: v.value for k, v in self._agent_status.items()},
            "schedules": self._schedules,
        }


class SignalEngineAgent(BaseConsolidatedAgent):
    """Signal generation -- feature pipeline -> ensemble -> signal.

    Consolidates: FeatureAgent + SignalAgent + RegimeAgent
    Stateless -- pure function composition. No I/O.
    """

    def __init__(self):
        super().__init__("signal_engine", AgentRole.SIGNAL_ENGINE)
        self._feature_pipeline = None
        self._ensemble = None
        self._regime_detector = None

    def set_feature_pipeline(self, pipeline):
        self._feature_pipeline = pipeline

    def set_ensemble(self, ensemble):
        self._ensemble = ensemble

    def set_regime_detector(self, detector):
        self._regime_detector = detector

    def handle_message(self, msg: AgentMessage) -> Optional[AgentMessage]:
        if msg.message_type == "generate_signal":
            symbol = msg.payload.get("symbol", "")
            data = msg.payload.get("data")
            if data is None:
                return AgentMessage(
                    source=self.name,
                    target=msg.source,
                    message_type="signal_error",
                    payload={"error": "no_data", "symbol": symbol},
                )
            try:
                features = None
                if self._feature_pipeline:
                    features = self._feature_pipeline.transform({symbol: {"1h": data}})

                regime = "ranging"
                if self._regime_detector and data is not None:
                    regime = self._regime_detector.detect_regime(data)

                signal = None
                if self._ensemble and features is not None:
                    signal = self._ensemble.predict(features, regime=regime)

                return AgentMessage(
                    source=self.name,
                    target=msg.source,
                    message_type="signal_result",
                    payload={
                        "symbol": symbol,
                        "regime": regime,
                        "signal": signal,
                        "features_shape": (
                            str(features.shape) if features is not None else None
                        ),
                    },
                )
            except Exception as e:
                logger.error(f"SignalEngine error for {symbol}: {e}")
                return AgentMessage(
                    source=self.name,
                    target=msg.source,
                    message_type="signal_error",
                    payload={"error": str(e), "symbol": symbol},
                )
        return None


class RiskManagerAgent(BaseConsolidatedAgent):
    """All risk management in one place.

    Consolidates: RiskAgent + AdaptiveRiskAgent + CircuitBreakerAgent + CostAgent
    """

    def __init__(self):
        super().__init__("risk_manager", AgentRole.RISK_MANAGER)
        self._risk_manager = None
        self._circuit_breaker = None
        self._cost_model = None
        self._correlation_matrix = None
        self._regime = "ranging"

    def set_risk_manager(self, rm):
        self._risk_manager = rm

    def set_circuit_breaker(self, cb):
        self._circuit_breaker = cb

    def set_cost_model(self, cm):
        self._cost_model = cm

    def handle_message(self, msg: AgentMessage) -> Optional[AgentMessage]:
        if msg.message_type == "pre_trade_check":
            trade_info = msg.payload
            symbol = trade_info.get("symbol", "")
            volume = trade_info.get("volume", 0)
            price = trade_info.get("price", 0)

            # Circuit breaker check
            if self._circuit_breaker:
                cb_ok = self._circuit_breaker.is_trading_allowed(symbol)
                if not cb_ok:
                    return AgentMessage(
                        source=self.name,
                        target=msg.source,
                        message_type="trade_rejected",
                        payload={"reason": "circuit_breaker", "symbol": symbol},
                    )

            # Pre-trade risk checks
            if self._risk_manager:
                balance = trade_info.get("balance", 100_000)
                equity = trade_info.get("equity", balance)
                approved, reason = self._risk_manager.pre_trade_checks(
                    balance, equity, 0, 0, symbol, trade_info.get("open_symbols", [])
                )
                if not approved:
                    return AgentMessage(
                        source=self.name,
                        target=msg.source,
                        message_type="trade_rejected",
                        payload={"reason": reason, "symbol": symbol},
                    )

            # Cost check
            if self._cost_model:
                cost = self._cost_model.calculate(symbol, "BUY", volume, price)
                if not cost.is_acceptable:
                    return AgentMessage(
                        source=self.name,
                        target=msg.source,
                        message_type="trade_rejected",
                        payload={"reason": cost.rejection_reason, "symbol": symbol},
                    )

            return AgentMessage(
                source=self.name,
                target=msg.source,
                message_type="trade_approved",
                payload={"symbol": symbol, "volume": volume},
            )

        elif msg.message_type == "update_state":
            self._regime = msg.payload.get("regime", self._regime)
            if self._circuit_breaker:
                self._circuit_breaker.update_regime(self._regime)
        return None


class ExecutionManagerAgent(BaseConsolidatedAgent):
    """Order execution + position management + cost tracking.

    Consolidates: ExecutionAgent + PositionAgent
    """

    def __init__(self):
        super().__init__("execution_manager", AgentRole.EXECUTION_MANAGER)
        self._execution_engine = None
        self._open_positions: Dict[int, dict] = {}
        self._execution_history: List[dict] = []

    def set_execution_engine(self, engine):
        self._execution_engine = engine

    def handle_message(self, msg: AgentMessage) -> Optional[AgentMessage]:
        if msg.message_type == "execute_trade":
            trade = msg.payload
            symbol = trade.get("symbol", "")
            direction = trade.get("direction", "BUY")
            volume = trade.get("volume", 0)
            sl = trade.get("sl", 0)
            tp = trade.get("tp", 0)

            if self._execution_engine:
                try:
                    result = self._execution_engine.open_position(
                        symbol,
                        direction,
                        volume,
                        sl,
                        tp,
                        reason=trade.get("reason", "signal"),
                    )
                    if result:
                        self._open_positions[result.position_id] = {
                            "position_id": result.position_id,
                            "symbol": symbol,
                            "direction": direction,
                            "volume": volume,
                            "entry_price": result.entry_price,
                        }
                        return AgentMessage(
                            source=self.name,
                            target=msg.source,
                            message_type="trade_executed",
                            payload={"position_id": result.position_id, **trade},
                        )
                except Exception as e:
                    logger.error(f"Execution error: {e}")

            return AgentMessage(
                source=self.name,
                target=msg.source,
                message_type="trade_failed",
                payload={"error": "execution_failed", **trade},
            )

        elif msg.message_type == "close_position":
            position_id = msg.payload.get("position_id")
            if position_id in self._open_positions and self._execution_engine:
                self._execution_engine.close_position(position_id, "signal_close")
                del self._open_positions[position_id]
        return None


class LearningManagerAgent(BaseConsolidatedAgent):
    """Model lifecycle management.

    Consolidates: LearningAgent + DriftAgent + ModelRegistryAgent + Screener
    """

    def __init__(self):
        super().__init__("learning_manager", AgentRole.LEARNING_MANAGER)
        self._online_learner = None
        self._model_registry = None
        self._drift_detector = None
        self._validation_gate = None

    def set_online_learner(self, learner):
        self._online_learner = learner

    def set_model_registry(self, registry):
        self._model_registry = registry

    def set_drift_detector(self, detector):
        self._drift_detector = detector

    def set_validation_gate(self, gate):
        self._validation_gate = gate

    def handle_message(self, msg: AgentMessage) -> Optional[AgentMessage]:
        if msg.message_type == "drift_detected":
            symbol = msg.payload.get("symbol", "")
            drift_count = msg.payload.get("count", 0)
            if self._online_learner:
                self._online_learner.on_drift_detected(symbol, drift_count)
                if self._online_learner.should_retrain(
                    symbol, total_trades=msg.payload.get("trades", 0)
                ):
                    self._online_learner.request_retrain(
                        symbol,
                        msg.payload.get("fetch_fn"),
                        msg.payload.get("feature_pipeline"),
                    )
            return AgentMessage(
                source=self.name,
                target="orchestrator",
                message_type="drift_processed",
                payload={"symbol": symbol, "drift_count": drift_count},
            )

        elif msg.message_type == "check_model":
            model_name = msg.payload.get("model_name", "")
            wf_results = msg.payload.get("walk_forward_results", {})
            stress_results = msg.payload.get("stress_test_results", {})
            if self._validation_gate:
                result = self._validation_gate.evaluate(
                    model_name, wf_results, stress_results
                )
                return AgentMessage(
                    source=self.name,
                    target=msg.source,
                    message_type="model_check_result",
                    payload={
                        "model_name": model_name,
                        "decision": result.decision.value,
                        "reason": result.reason,
                    },
                )
        return None


class DataManagerAgent(BaseConsolidatedAgent):
    """Data ingestion, connectivity, health.

    Consolidates: DataAgent + ConnectionAgent
    """

    def __init__(self):
        super().__init__("data_manager", AgentRole.DATA_MANAGER)
        self._data_manager = None
        self._client = None
        self._connected = False
        self._last_tick: Dict[str, float] = {}

    def set_data_manager(self, dm):
        self._data_manager = dm

    def set_client(self, client):
        self._client = client

    def handle_message(self, msg: AgentMessage) -> Optional[AgentMessage]:
        if msg.message_type == "get_data":
            symbol = msg.payload.get("symbol", "EURUSD")
            lookback = msg.payload.get("lookback", 100)
            if self._data_manager:
                data = self._data_manager.get_rates(symbol, lookback)
                return AgentMessage(
                    source=self.name,
                    target=msg.source,
                    message_type="data_result",
                    payload={"symbol": symbol, "data": data},
                )
        elif msg.message_type == "health_check":
            return AgentMessage(
                source=self.name,
                target=msg.source,
                message_type="health_status",
                payload={
                    "connected": self._connected,
                    "last_tick": self._last_tick,
                    "data_manager": self._data_manager is not None,
                },
            )
        elif msg.message_type == "tick_update":
            symbol = msg.payload.get("symbol", "")
            price = msg.payload.get("price", 0.0)
            self._last_tick[symbol] = price
        return None


class AgentSwarm:
    """Lightweight container for the 6-agent swarm with message routing."""

    def __init__(self):
        self.orchestrator = OrchestratorAgent()
        self.signal_engine = SignalEngineAgent()
        self.risk_manager = RiskManagerAgent()
        self.execution = ExecutionManagerAgent()
        self.learning = LearningManagerAgent()
        self.data = DataManagerAgent()
        self._agents = {
            "orchestrator": self.orchestrator,
            "signal_engine": self.signal_engine,
            "risk_manager": self.risk_manager,
            "execution_manager": self.execution,
            "learning_manager": self.learning,
            "data_manager": self.data,
        }

    def start_all(self):
        for name, agent in self._agents.items():
            agent.start()
        logger.info("All 6 consolidated agents started")

    def stop_all(self):
        for name, agent in self._agents.items():
            agent.stop()
        logger.info("All 6 consolidated agents stopped")

    def send_message(self, msg: AgentMessage):
        """Route a message to the target agent."""
        if msg.target == "*":
            for agent in self._agents.values():
                response = agent.handle_message(msg)
                if response:
                    self.send_message(response)
        elif msg.target in self._agents:
            response = self._agents[msg.target].handle_message(msg)
            if response:
                self.send_message(response)

    def get_agent(self, name: str) -> Optional[BaseConsolidatedAgent]:
        return self._agents.get(name)

    def get_all_states(self) -> Dict[str, Dict]:
        return {
            name: {
                "role": agent.role.value,
                "status": agent.status.value,
            }
            for name, agent in self._agents.items()
        }
