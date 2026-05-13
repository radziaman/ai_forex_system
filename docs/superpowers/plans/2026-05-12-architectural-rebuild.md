# Architectural Rebuild: Service-Oriented Trading System

> **For agentic workers:** Use subagent-driven-development or executing-plans.

**Goal:** Decompose the 2176-line `RTSMoneybotSystem` god object into 5 independent services connected by an event bus, with typed config, pure signal generation, and clean separation.

**Architecture:**
```
TradingOrchestrator (~50 lines)
  ├── DataService     — ingests ticks, caches features, emits events
  ├── SignalEngine    — pure function: features → Signal (no I/O, no risk)
  ├── RiskManager     — gatekeeper: Signal → TradeDecision or None
  ├── ExecutionEngine — dumb executor: TradeDecision → broker order
  └── Monitoring      — dashboard + notifications
```

**Do not delete existing files — add alongside. Import & wrap existing code.**

---

## File Map

```
src/
├── main_v2.py                     # NEW entry point
├── infrastructure/
│   ├── config_v2.py               # NEW: typed dataclass config
│   ├── service_base.py            # NEW: abstract base
│   └── service_registry.py        # NEW: lifecycle manager
├── services/
│   ├── data_service.py            # NEW: clean data + features
│   ├── signal_engine.py           # NEW: features → Signal
│   ├── risk_service.py            # NEW: gatekeeper
│   └── execution_service.py       # NEW: executor
tests/
└── test_services/                 # NEW tests
```

---

## Tasks

### Task 1: `config_v2.py` — Typed Configuration

**File:** `src/infrastructure/config_v2.py`

Typed dataclasses for all config domains. One `AppConfig.from_yaml()` classmethod. `validate()` method returns errors. Override with env vars.

### Task 2: `service_base.py` — Service Lifecycle

**File:** `src/infrastructure/service_base.py`

```python
from abc import ABC, abstractmethod
from typing import Optional
from loguru import logger


class TradingService(ABC):
    """Base class for all trading system services."""

    def __init__(self, name: str):
        self.name = name
        self._running = False

    @abstractmethod
    async def start(self) -> None:
        ...

    @abstractmethod
    async def stop(self) -> None:
        ...

    @property
    def is_running(self) -> bool:
        return self._running

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *args):
        await self.stop()
```

### Task 3: `service_registry.py` — Service Lifecycle Manager

**File:** `src/infrastructure/service_registry.py`

```python
from typing import Dict, List
from .service_base import TradingService
from loguru import logger


class ServiceRegistry:
    """Manages lifecycle of all services."""

    def __init__(self):
        self._services: Dict[str, TradingService] = {}

    def register(self, service: TradingService) -> None:
        self._services[service.name] = service

    def get(self, name: str) -> TradingService:
        return self._services[name]

    @property
    def all(self) -> List[TradingService]:
        return list(self._services.values())

    async def start_all(self) -> None:
        for svc in self._services.values():
            logger.info(f"Starting {svc.name}...")
            await svc.start()
            logger.info(f"{svc.name} started")

    async def stop_all(self) -> None:
        for svc in reversed(list(self._services.values())):
            try:
                await svc.stop()
            except Exception as e:
                logger.error(f"Error stopping {svc.name}: {e}")
```

### Task 4: `services/__init__.py` + Domain Types

**File:** `src/services/__init__.py`

```python
"""Domain types shared across services."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
import time


class Regime(Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CRISIS = "crisis"


class SignalDirection(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    """Pure output of SignalEngine — no risk info, no execution details."""
    symbol: str
    direction: SignalDirection
    confidence: float
    regime: Regime
    price: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeDecision:
    """Risk-approved decision ready for execution."""
    signal: Signal
    volume: float
    sl_price: float
    tp_price: float
    max_slippage: float = 0.0
    execution_algo: str = "market"


@dataclass
class ExecutionResult:
    success: bool
    position_id: Optional[str] = None
    filled_price: Optional[float] = None
    filled_volume: Optional[float] = None
    error: Optional[str] = None
```

### Task 5: `data_service.py` — Data Pipeline Service

**File:** `src/services/data_service.py`

```python
"""Data service: tick ingestion, OHLCV aggregation, feature caching."""
import asyncio
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from loguru import logger

from infrastructure.service_base import TradingService
from infrastructure.event_bus import TradingEventBus, EventType, Event, get_event_bus
from infrastructure.config_v2 import AppConfig
from data.data_manager import DataManager, SYMBOLS
from api.base import PriceTick
from rts_ai_fx.features_unified import FeaturePipeline, compute_features


@dataclass
class FeatureUpdate:
    """Published when new features are ready for a symbol."""
    symbol: str
    timeframe: str
    features: np.ndarray
    price: float
    timestamp: float


class DataService(TradingService):
    """Ingests market data, aggregates OHLCV, computes + caches features."""

    def __init__(self, config: AppConfig):
        super().__init__("data_service")
        self.config = config
        self.data_manager = DataManager(historical_path=config.data.historical_path)
        self.feature_pipeline = FeaturePipeline(
            lookback=config.features.lookback,
            timeframes=config.features.timeframes,
            use_microstructure=config.features.use_microstructure,
        )
        self.event_bus: TradingEventBus = get_event_bus()
        self._features_dirty: Dict[str, bool] = {}
        self._last_bar_ts: Dict[str, float] = {}

    async def start(self) -> None:
        self._running = True
        logger.info(f"DataService: {len(SYMBOLS)} symbols x {self.config.features.timeframes}")

    async def stop(self) -> None:
        self._running = False

    def ingest_tick(self, tick: PriceTick) -> None:
        """Process a single tick — updates OHLCV, marks features dirty on bar close."""
        if not self._running:
            return
        sym = tick.symbol
        self.data_manager.update_tick(sym, tick.bid, tick.ask, tick.volume, tick.timestamp)

        # Check if 1m bar advanced
        df = self.data_manager.get_ohlcv(sym, "1m")
        if df is not None and not df.empty:
            current_bar_ts = float(df["timestamp"].iloc[-1])
            if self._last_bar_ts.get(sym, 0) != current_bar_ts:
                self._last_bar_ts[sym] = current_bar_ts
                self._features_dirty[sym] = True
                self._on_bar_close(sym)

    def _on_bar_close(self, symbol: str) -> None:
        """Called when a new 1m bar closes. Propagate to higher TFs, compute features."""
        features = self.get_features(symbol)
        if features is not None:
            price = self.data_manager.get_price(symbol, "1h")
            update = FeatureUpdate(
                symbol=symbol,
                timeframe="1h",
                features=features,
                price=price,
                timestamp=time.time(),
            )
            asyncio.ensure_future(
                self.event_bus.emit(EventType.FEATURES_READY, update, source="data_service")
            )
            self._features_dirty[symbol] = False

    def get_features(self, symbol: str) -> Optional[np.ndarray]:
        """Get cached or freshly computed features for a symbol."""
        # Check cache
        cached = self.data_manager.get_cached_features(symbol, "1h")
        if cached is not None and not self._features_dirty.get(symbol, True):
            return cached

        df = self.data_manager.get_ohlcv(symbol, "1h")
        if df is None or len(df) < self.config.features.lookback + 10:
            return None

        try:
            features = self.feature_pipeline.transform(
                self.data_manager.ohlcv, symbol=symbol,
            )
            if features is not None:
                self.data_manager.set_cached_features(symbol, "1h", features)
            return features
        except Exception as e:
            logger.debug(f"Feature computation failed for {symbol}: {e}")
            return None

    def get_price(self, symbol: str) -> float:
        return self.data_manager.get_price(symbol, "1h")

    def get_atr(self, symbol: str, period: int = 14) -> float:
        return self.data_manager.get_atr(symbol, "1h", period)

    def get_open_positions(self) -> List[Dict]:
        """Delegate to execution service — stub for now."""
        return []

    def listen(self, event_bus: TradingEventBus) -> None:
        """Wire up subscriptions."""
        event_bus.subscribe(EventType.TICK, lambda e: self.ingest_tick(e.data))
```

### Task 6: `signal_engine.py` — Pure Signal Generation

**File:** `src/services/signal_engine.py`

```python
"""Pure signal generation: features → Signal. No I/O, no risk awareness."""
import numpy as np
from typing import Optional, Dict
from loguru import logger

from infrastructure.service_base import TradingService
from infrastructure.event_bus import TradingEventBus, EventType, Event, get_event_bus
from infrastructure.config_v2 import AppConfig
from services import Signal, SignalDirection, Regime
from rts_ai_fx.regime_detector import HMMRegimeDetector
from rts_ai_fx.ensemble import MoEEnsemble
from rts_ai_fx.drift_detector import DriftMonitor
from rts_ai_fx.features_unified import compute_features
from ai.sentiment import SentimentAnalyzer


class SignalEngine(TradingService):
    """Transforms features into trading signals. Pure — no side effects."""

    def __init__(self, config: AppConfig):
        super().__init__("signal_engine")
        self.config = config
        self.ensemble = MoEEnsemble()
        self.regime_detector = HMMRegimeDetector(n_regimes=4, lookback=60)
        self.sentiment = SentimentAnalyzer(use_finbert=True, cache_ttl=300)
        self.drift_monitors: Dict[str, DriftMonitor] = {}
        self.event_bus: TradingEventBus = get_event_bus()

    async def start(self) -> None:
        # Register built-in experts
        self.ensemble.add_expert(
            name="rule_based",
            predict_fn=self._rule_prediction,
            confidence_fn=lambda X: 0.7,
            regime="ranging",
        )

    async def stop(self) -> None:
        pass

    def on_features(self, symbol: str, features: np.ndarray, price: float) -> Optional[Signal]:
        """Process new features, return Signal or None."""
        if features is None:
            return None

        # Regime detection
        regime = self.regime_detector.detect_regime_from_features(features)
        if not self.regime_detector.should_trade(regime):
            return None

        # Ensemble inference
        pred = self.ensemble.predict(features, regime=regime.value)
        should_trade, direction_str, agreement = self.ensemble.should_trade(
            pred, price, min_confidence=0.40,
        )

        if not should_trade:
            return None

        direction = SignalDirection.BUY if direction_str == "BUY" else SignalDirection.SELL

        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=pred.confidence,
            regime=regime,
            price=price,
            metadata={
                "agreement": agreement,
                "expert_outputs": pred.expert_outputs,
                "ensemble_price": pred.price,
            },
        )

    def on_trade_result(self, symbol: str, prediction: float, actual: float) -> bool:
        """Update drift monitors. Returns True if drift detected."""
        monitor = self.drift_monitors.get(symbol)
        if monitor is None:
            monitor = DriftMonitor()
            self.drift_monitors[symbol] = monitor
        return monitor.update(prediction, actual)

    def _rule_prediction(self, X: np.ndarray, symbol: str = "EURUSD") -> float:
        """Fallback rule-based prediction."""
        return 1.12  # Minimal — ensemble handles real logic
```

### Task 7: `risk_service.py` — Risk Gatekeeper

**File:** `src/services/risk_service.py`

```python
"""Risk Gatekeeper: evaluates Signal → TradeDecision or rejects."""
from typing import Optional, Tuple
from loguru import logger

from infrastructure.service_base import TradingService
from infrastructure.config_v2 import AppConfig
from services import Signal, TradeDecision, SignalDirection
from risk.manager import RiskManager, RiskParameters
from execution.cost_model import CostModel


class RiskService(TradingService):
    """Gatekeeper: receives Signal, returns TradeDecision or None."""

    def __init__(self, config: AppConfig, initial_balance: float = 100_000.0):
        super().__init__("risk_service")
        self.config = config
        params = RiskParameters(
            max_risk_per_trade=config.trading.max_risk_per_trade,
            max_drawdown=config.trading.max_drawdown,
            max_margin_usage=config.trading.max_margin_usage,
        )
        self.risk_manager = RiskManager(params, initial_balance)
        self.cost_model = CostModel(commission_per_lot=config.trading.commission_per_lot)

    async def start(self) -> None:
        logger.info("RiskService: Kelly sizing, VaR/CVaR, drawdown limits")

    async def stop(self) -> None:
        pass

    def evaluate(self, signal: Signal, balance: float, equity: float, margin: float,
                 atr: float, open_positions_count: int) -> Optional[TradeDecision]:
        """Gate: returns TradeDecision or None (rejected)."""
        atr = atr or signal.price * 0.001

        # 1. Mode check
        if self.risk_manager.mode == "PAPER":
            pass  # relaxed checks in paper

        # 2. Pre-trade risk checks
        approved, reason = self.risk_manager.pre_trade_checks(balance, equity, margin, balance - equity)
        if not approved:
            logger.info(f"Risk reject {signal.symbol}: {reason}")
            return None

        # 3. Cost check
        vol = 1000  # placeholder — proper sizing below
        cost = self.cost_model.calculate(
            symbol=signal.symbol, direction=signal.direction.value,
            volume=vol, price=signal.price, atr=atr,
        )
        if not cost.is_acceptable:
            logger.info(f"Cost reject {signal.symbol}: {cost.rejection_reason}")
            return None

        # 4. Kelly sizing
        volume = self.risk_manager.calculate_kelly_size(
            balance, signal.price, atr, signal.confidence, symbol=signal.symbol,
        )
        volume = max(int(volume), 1)

        # 5. SL/TP
        if signal.direction == SignalDirection.BUY:
            sl_price = signal.price - atr * self.config.trading.sl_atr_multiplier
            tp_price = signal.price + atr * self.config.trading.tp_atr_multiplier
        else:
            sl_price = signal.price + atr * self.config.trading.sl_atr_multiplier
            tp_price = signal.price - atr * self.config.trading.tp_atr_multiplier

        return TradeDecision(
            signal=signal,
            volume=volume,
            sl_price=sl_price,
            tp_price=tp_price,
        )
```

### Task 8: `execution_service.py` — Dumb Executor

**File:** `src/services/execution_service.py`

```python
"""Execution Engine: executes TradeDecision. Makes no decisions."""
import asyncio
from typing import Optional, Dict, List
from loguru import logger

from infrastructure.service_base import TradingService
from infrastructure.config_v2 import AppConfig
from services import TradeDecision, ExecutionResult
from execution.engine import ExecutionEngine
from risk.manager import TrailingStopManager
from api.provider_factory import create_execution_provider
from infrastructure.secrets import Secrets
from data.data_manager import DataManager, BASE_PRICES
from api.base import PriceTick
from infrastructure.event_bus import get_event_bus, TradingEventBus, EventType


class ExecutionService(TradingService):
    """Dumb executor: receives decisions, sends orders. No deciding."""

    def __init__(self, config: AppConfig, secrets: Secrets, data_service):
        super().__init__("execution_service")
        self.config = config
        self.ctrader, self.data_provider = create_execution_provider(secrets)
        self.engine = ExecutionEngine(self.ctrader, None, data_service.data_manager)
        self.event_bus = get_event_bus()

    async def start(self) -> None:
        result = await self.ctrader.start()
        if result or self.ctrader.is_connected():
            logger.info("ExecutionService: cTrader connected")
        else:
            logger.warning("ExecutionService: simulation mode")
        self._running = True

    async def stop(self) -> None:
        if hasattr(self.ctrader, 'disconnect'):
            await self.ctrader.disconnect()
        self._running = False

    async def execute(self, decision: TradeDecision) -> Optional[ExecutionResult]:
        """Execute a trade decision. Returns result."""
        if not self._running:
            return ExecutionResult(success=False, error="service_not_running")

        trade = await self.engine.open_position(
            symbol=decision.signal.symbol,
            direction=decision.signal.direction.value,
            volume=decision.volume,
            sl=decision.sl_price,
            tp=decision.tp_price,
            reason=f"signal_conf={decision.signal.confidence:.2f}_regime={decision.signal.regime.value}",
        )

        if trade:
            await self.event_bus.emit(
                EventType.POSITION_OPENED,
                {"symbol": decision.signal.symbol, "volume": decision.volume,
                 "entry": trade.entry_price, "position_id": trade.position_id},
                source="execution_service",
            )
            return ExecutionResult(success=True, position_id=str(trade.position_id),
                                    filled_price=trade.entry_price, filled_volume=decision.volume)
        else:
            await self.event_bus.emit(
                EventType.ORDER_REJECTED,
                {"symbol": decision.signal.symbol},
                source="execution_service",
            )
            return ExecutionResult(success=False, error="order_failed")

    async def close_position(self, position_id: str, reason: str = "AI signal") -> bool:
        return await self.engine.close_position(int(position_id), reason)

    async def close_all(self, reason: str = "system") -> None:
        await self.engine.close_all_positions(reason)

    def get_open_positions(self) -> List[Dict]:
        return self.engine.get_open_positions()

    def get_account_info(self) -> Dict:
        return self.engine.get_account_info()  # not async — engine caches

    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        return self.engine.get_trade_history(limit)
```

### Task 9: `main_v2.py` — New Entry Point

**File:** `src/main_v2.py`

```python
"""RTS: Agentic Moneybot System Elite — v2 Service Architecture Entry Point."""
import asyncio
import signal
import os
import sys
from loguru import logger

from infrastructure.config_v2 import AppConfig
from infrastructure.secrets import Secrets
from infrastructure.event_bus import get_event_bus, EventType
from infrastructure.service_registry import ServiceRegistry
from services.data_service import DataService
from services.signal_engine import SignalEngine
from services.risk_service import RiskService
from services.execution_service import ExecutionService

from data.data_manager import SYMBOLS
from api.base import PriceTick


class TradingOrchestrator:
    """Thin loop: manages services, health checks, graceful shutdown."""

    def __init__(self, config: AppConfig, secrets: Secrets):
        self.config = config
        self.secrets = secrets
        self.registry = ServiceRegistry()
        self.event_bus = get_event_bus()
        self.running = False
        self._init_services()

    def _init_services(self):
        self.data_service = DataService(self.config)
        self.signal_engine = SignalEngine(self.config)
        self.risk_service = RiskService(self.config)
        self.execution_service = ExecutionService(self.config, self.secrets, self.data_service)

        for svc in [self.data_service, self.signal_engine, self.risk_service, self.execution_service]:
            self.registry.register(svc)

        # Wire event-driven signal flow
        self.event_bus.subscribe(EventType.FEATURES_READY, self._on_features_ready)

    async def _on_features_ready(self, event):
        """Features → Signal → Risk → Execution pipeline."""
        update = event.data
        if update is None:
            return

        # 1. Signal
        signal = self.signal_engine.on_features(update.symbol, update.features, update.price)
        if signal is None:
            return

        # 2. Risk
        acc = self.execution_service.get_account_info()
        atr = self.data_service.get_atr(update.symbol)
        open_positions = self.execution_service.get_open_positions()
        decision = self.risk_service.evaluate(
            signal, acc.get("balance", 100_000), acc.get("equity", 100_000),
            acc.get("margin", 0), atr, len(open_positions),
        )
        if decision is None:
            return

        # 3. Execute
        result = await self.execution_service.execute(decision)
        if result and result.success:
            logger.success(f"TRADE: {decision.signal.direction.value} {decision.volume:.0f} "
                          f"{decision.signal.symbol} @ {result.filled_price:.5f}")

    async def start(self):
        logger.info("=" * 50)
        logger.info("  RTS: Agentic Moneybot System Elite v2 — Service Architecture")
        logger.info("=" * 50)

        errors = self.config.validate()
        if errors:
            logger.error(f"Config errors: {errors}")
            return

        await self.event_bus.start()
        await self.registry.start_all()
        self.running = True

        logger.info(f"Services: {[s.name for s in self.registry.all]}")
        logger.info("Trading loop running (event-driven)...")

        while self.running:
            await asyncio.sleep(5)

    async def stop(self):
        logger.info("Shutting down...")
        self.running = False
        await self.execution_service.close_all("shutdown")
        await self.registry.stop_all()
        await self.event_bus.stop()
        logger.info("Shutdown complete.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="RTS: Agentic Moneybot System Elite v2")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--capital", type=float, default=100_000.0)
    args = parser.parse_args()

    config = AppConfig.from_yaml(args.config)
    secrets = Secrets()
    orchestrator = TradingOrchestrator(config, secrets)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.ensure_future(orchestrator.stop()))

    try:
        loop.run_until_complete(orchestrator.start())
    except KeyboardInterrupt:
        loop.run_until_complete(orchestrator.stop())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
```

---

### Task 10: Tests

**Files:** `tests/test_services/test_*.py`

Tests validate:
- `SignalEngine.on_features` returns correct Signal/None
- `RiskService.evaluate` gates correctly on drawdown/costs
- `ExecutionService.execute` delegates to `ExecutionEngine`
- `DataService.ingest_tick` triggers `FEATURES_READY` on bar close
- Config validation catches bad parameters
- End-to-end: fake tick → orchestrator walks full pipeline
