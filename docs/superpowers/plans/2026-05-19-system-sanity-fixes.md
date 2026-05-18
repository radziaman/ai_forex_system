# System Sanity Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all 5 groups of issues from system sanity check — stale imports, broken pre-commit config, 55 mypy type errors, lint cleanup (unused imports/vars/complex functions), and 22,735 source LOC with near-zero test coverage.

**Architecture:** Fixes are grouped into 6 independent waves. Earlier groups fix the build and health infrastructure, later groups progressively stiffen types, reduce suppression, and add coverage. Each group is independently testable.

**Tech Stack:** Python 3.9.6, pytest, flake8, mypy, black. No new dependencies.

---

## File Map (Files Modified)

| File | Change |
|------|--------|
| `health_check.py` | Fix all 8 stale imports, fix malformed try/except, fix E401 |
| `.pre-commit-config.yaml` | Fix `src/ai_forex_system/` → `src/` in all 3 hook patterns |
| `src/rts_ai_fx/model.py` | Remove duplicate `@classmethod` decorator |
| `src/rts_ai_fx/regime_detector.py` | Fix implicit `Optional` |
| `src/rts_ai_fx/ensemble.py` | Fix `torch` import type, fix type annotations |
| `src/rts_ai_fx/attention_fusion.py` | Fix return type mismatches (None vs Tensor), implicit Optional |
| `src/ai/maml_agent.py` | Fix `dict_values` → `list(params.values())` for autograd |
| `src/ai/rl_agent.py` | Fix implicit Optional (x2), add return types |
| `src/data/data_manager.py` | Fix `Optional[MarketDepthData]` attribute access with assertion |
| `src/data/feature_engine.py` | Add `type: ignore` for talib redefinition |
| `src/agentic/agents/validation_agent.py` | Remove unused variable assignments |
| `src/agentic/agents/risk_agent.py` | Type `halted_symbols`, add None guards |
| `src/dashboard/smart_dashboard.py` | Fix implicit Optional, add return types |
| `src/training/distributed_trainer.py` | Fix return types |
| `tests/test_smoke_imports.py` | **NEW**: import smoke tests for all 40+ modules |

---

## Task Group 0: Pre-Flight Configuration

### Task 0.1 — Fix `.pre-commit-config.yaml` paths

**File:** `.pre-commit-config.yaml`

**Changes:**

- Line 7: Replace `files: ^(src/ai_forex_system/|tests/|main\.py|requirements\.txt|README\.md)$` with `files: ^(src/|tests/|main\.py|requirements\.txt|README\.md)$`
- Line 13: Replace `files: ^(src/ai_forex_system/|tests/|main\.py)$` with `files: ^(src/|tests/|main\.py)$`
- Line 22: Replace `files: ^(src/ai_forex_system/|tests/)$` with `files: ^(src/|tests/)$`

**Verification:** `cat .pre-commit-config.yaml` — confirm no remaining `ai_forex_system` references.

---

## Task Group 1: Fix `health_check.py`

### Task 1.1 — Fix stale class names

**File:** `health_check.py`

Replace these 8 import names:

| Current import | Correct import |
|---------------|----------------|
| `from rts_ai_fx.model import LSTMModel` | `from rts_ai_fx.model import LSTMCNNHybrid` |
| `from rts_ai_fx.drift_detector import DriftDetector` | `from rts_ai_fx.drift_detector import DriftMonitor` |
| `from rts_ai_fx.attention_fusion import AttentionFusion` | `from rts_ai_fx.attention_fusion import TemporalAttentionFusion` |
| `from rts_ai_fx.uncertainty import MCAlpha` | `from rts_ai_fx.uncertainty import monte_carlo_dropout, get_confidence` |
| `from data.market_session import MarketSessionManager` | `from data.market_session import MarketSession` |
| `from ai.maml_agent import MAMLMetaLearner` | `from ai.maml_agent import MAMLAgent` |
| `from validation.walk_forward import WalkForwardOptimizer` | `from validation.walk_forward import PurgedWalkForward` |
| `from validation.monte_carlo import MonteCarloSimulator` | `from validation.monte_carlo import MonteCarloSigTest` |

### Task 1.2 — Fix malformed try/except structure

**File:** `health_check.py`, lines 42-53

**Problem:** The `try` block at line 42 has a malformed nested structure. The second `try` (lines 48-53) should be a separate top-level block.

**Replace lines 42-53 with:**

```python
try:
    from rts_ai_fx.causal_features import CausalFeatureSelector, CAUSAL_AVAILABLE
    check("causal_features", lambda: CAUSAL_AVAILABLE)
except Exception as e:
    check("causal_features", lambda: (_ for _ in ()).throw(e))  # noqa: F821

try:
    from rts_ai_fx.features_unified import compute_features
    check("features_unified", lambda: True)
except Exception as e:
    check("features_unified", lambda: (_ for _ in ()).throw(e))  # noqa: F821
```

### Task 1.3 — Fix E401 multiple imports on one line

**File:** `health_check.py`, line 3

Change:
```python
import sys, os
```
to:
```python
import sys
import os
```

### Task 1.4 — Verify fix

```bash
python3 health_check.py
```

Expected: All checks pass (no FAIL items). All 8 previously failing modules show [OK].

---

## Task Group 2: Fix Mypy Type Errors

### Task 2.1 — `src/rts_ai_fx/regime_detector.py`

**Changes:**

1. Line 177 — Fix implicit `Optional`:
```python
    def detect_transition(self, symbol: str = None) -> Dict:
```
→
```python
    def detect_transition(self, symbol: Optional[str] = None) -> Dict:
```

2. Line 282 — Fix implicit `Optional`:
```python
    def get_transition_trading_signal(self, symbol: str = None) -> Dict:
```
→
```python
    def get_transition_trading_signal(self, symbol: Optional[str] = None) -> Dict:
```

**Verification:** `mypy src/rts_ai_fx/regime_detector.py` — expect 0 errors.

### Task 2.2 — `src/rts_ai_fx/ensemble.py`

**Changes:**

1. Line 15 — Fix `torch = None` type:
```python
    torch = None
```
→
```python
    torch: Any = None
```

2. Line 126-127 — Fix ndarray→list and .sum() on list:
The confidences variable should be typed as ndarray, not list. Change the parameter type and assignment accordingly.

**Verification:** `mypy src/rts_ai_fx/ensemble.py` — expect 0 errors.

### Task 2.3 — `src/rts_ai_fx/attention_fusion.py`

**Changes:**

Fix return type mismatches where `None` is returned but `Tensor` is declared:

1. Line ~68:
```python
        if not tf_features or len(tf_features) == 0:
            return None, {}
```
→
```python
        if not tf_features or len(tf_features) == 0:
            raise ValueError("No timeframe features provided")
```

2. Lines ~162-163 and ~183-184 — Same pattern:
```python
        if not tf_data:
            return None, {}
```
→
```python
        if not tf_data:
            raise ValueError("No timeframe data provided")
```

3. Line ~223 — Fix implicit `Optional`:
```python
    def fuse(self, ... timeframes: list[str] = None):
```
→
```python
    def fuse(self, ... timeframes: Optional[list[str]] = None):
```

**Verification:** `mypy src/rts_ai_fx/attention_fusion.py` — expect 0 errors.

### Task 2.4 — `src/ai/maml_agent.py`

**Changes:**

1. Lines ~112, ~225 — `dict_values` to `list`:
```python
            grads = torch.autograd.grad(loss, params.values(), create_graph=True)
```
→
```python
            grads = torch.autograd.grad(loss, list(params.values()), create_graph=True)
```

**Verification:** `mypy src/ai/maml_agent.py` — expect 0 errors.

### Task 2.5 — `src/ai/rl_agent.py`

**Changes:**

1. Line ~21 — Fix implicit `Optional`:
```python
        hidden_dims: list = None,
```
→
```python
        hidden_dims: Optional[list] = None,
```

2. Line ~74 — Same fix.

**Verification:** `mypy src/ai/rl_agent.py` — expect 0 errors.

### Task 2.6 — `src/data/data_manager.py`

**Changes:**

Lines ~876-885 — `md: Optional[MarketDepthData]` attribute access. Add None assertion:

```python
                assert md is not None, f"Market depth not found for {sym}"
```

After the assignment of `md`, before attribute access.

**Verification:** `mypy src/data/data_manager.py` — expect 0 errors.

### Task 2.7 — `src/data/feature_engine.py`

**Changes:**

Line ~6 — Fix talib redefinition error. Add type ignore comment:

```python
    talib = None  # type: ignore[assignment]
```

**Verification:** `mypy src/data/feature_engine.py` — expect 0 errors.

### Task 2.8 — `src/agentic/agents/validation_agent.py`

**Changes:**

Remove unused variable assignments that cause type confusion:

1. Line ~122: Remove `dm = None`
2. Lines ~125-130: Remove the `data_agent_msg = AgentMessage(...)` assignment block (the message is never sent)
3. Line ~131: Fix `self._cached_trades = self.get_world(...)` — the value is an int but field is typed as List[Dict]. Assign to a local instead.

**Verification:** `mypy src/agentic/agents/validation_agent.py` — expect 0 errors.

### Task 2.9 — `src/agentic/agents/risk_agent.py`

**Changes:**

1. Line ~77 — Add type annotation:
```python
        halted_symbols = []
```
→
```python
        halted_symbols: List[str] = []
```

2. Lines ~173-177 — Add None guards on `self.risk_manager`, `self.circuit_breaker`, `self.cost_model` attribute access.

**Verification:** `mypy src/agentic/agents/risk_agent.py` — expect 0 errors.

### Task 2.10 — `src/dashboard/smart_dashboard.py`

**Changes:**

1. Line ~386 — Fix implicit `Optional`:
```python
    async def get_state(token: str = None):
```
→
```python
    async def get_state(token: Optional[str] = None):
```

**Verification:** `mypy src/dashboard/smart_dashboard.py` — expect 0 errors.

### Task 2.11 — `src/training/distributed_trainer.py`

**Changes:**

1. Line ~288 — Fix return type annotation:
```python
    ) -> Dict:
```
→
```python
    ) -> Dict[str, Any]:
```

**Verification:** `mypy src/training/distributed_trainer.py` — expect 0 errors.

---

## Task Group 3: Fix `model.py` Duplicate Decorator

### Task 3.1 — Fix duplicate `@classmethod`

**File:** `src/rts_ai_fx/model.py`

Change (lines ~113-114):
```python
    @classmethod
    @classmethod
    def load(cls, path: str) -> "LSTMCNNHybrid":
```
→
```python
    @classmethod
    def load(cls, path: str) -> "LSTMCNNHybrid":
```

**Verification:** `python3 -c "from rts_ai_fx.model import LSTMCNNHybrid; print('OK')"` — expect no warnings.

---

## Task Group 4: Lint Cleanup

### Task 4.1 — Remove unused local variables

**`src/rts_ai_fx/attention_fusion.py`:** Line ~191 — Remove `batch_size = stacked.shape[0]` if unused.

**`src/agentic/agents/validation_agent.py`:** Lines ~122-130 — Remove `dm = None` and `data_agent_msg` assignment per Task 2.8.

**Verification:** `flake8 src/rts_ai_fx/attention_fusion.py src/agentic/agents/validation_agent.py --select=F841` — expect 0 errors.

---

## Task Group 5: Test Coverage

### Task 5.1 — Create smoke/import test file

**File:** `tests/test_smoke_imports.py` (NEW)

Create a comprehensive import smoke test:

```python
"""Smoke tests: verify all modules import without error."""

import sys
import os
import importlib
import pytest

# Add src to path
_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _src not in sys.path:
    sys.path.insert(0, _src)

# Check optional dependencies
tf_available = importlib.util.find_spec("tensorflow") is not None
torch_available = importlib.util.find_spec("torch") is not None
hmm_available = importlib.util.find_spec("hmmlearn") is not None
ray_available = importlib.util.find_spec("ray") is not None


def test_import_rts_ai_fx_package():
    import rts_ai_fx

    assert rts_ai_fx.__name__ == "rts_ai_fx"


def test_import_model():
    from rts_ai_fx.model import LSTMCNNHybrid, ProfitabilityClassifier

    assert LSTMCNNHybrid is not None


def test_import_drift_detector():
    from rts_ai_fx.drift_detector import ADWIN, DriftMonitor

    assert DriftMonitor is not None


def test_import_attention_fusion():
    from rts_ai_fx.attention_fusion import (
        TimeframeAttention,
        TemporalAttentionFusion,
        AttentionFusionPipeline,
    )

    assert TemporalAttentionFusion is not None


def test_import_uncertainty():
    from rts_ai_fx.uncertainty import monte_carlo_dropout, get_confidence

    assert monte_carlo_dropout is not None


def test_import_ensemble():
    from rts_ai_fx.ensemble import MoEEnsemble, EnsemblePrediction, Expert

    assert MoEEnsemble is not None


def test_import_regime_detector():
    from rts_ai_fx.regime_detector import HMMRegimeDetector

    assert HMMRegimeDetector is not None


def test_import_causal_features():
    from rts_ai_fx.causal_features import CausalFeatureSelector, CAUSAL_AVAILABLE

    assert CausalFeatureSelector is not None


def test_import_features_unified():
    from rts_ai_fx.features_unified import compute_features

    assert compute_features is not None


def test_import_ai_maml():
    from ai.maml_agent import MAMLAgent, MAMLModel

    assert MAMLAgent is not None


def test_import_ai_rl():
    from ai.rl_agent import PPOAgent

    assert PPOAgent is not None


def test_import_ai_sentiment():
    from ai.sentiment import SentimentAnalyzer

    assert SentimentAnalyzer is not None


def test_import_ai_regime_agents():
    from ai.regime_agents import RegimeSpecialistSystem

    assert RegimeSpecialistSystem is not None


def test_import_data_manager():
    from data.data_manager import DataManager, MarketDepthData

    assert DataManager is not None


def test_import_market_session():
    from data.market_session import MarketSession

    assert MarketSession is not None


def test_import_economic_calendar():
    from data.economic_calendar import EconomicCalendar

    assert EconomicCalendar is not None


def test_import_feature_engine():
    from data.feature_engine import FeatureEngine

    assert FeatureEngine is not None


def test_import_risk_manager():
    from risk.manager import RiskManager

    assert RiskManager is not None


def test_import_circuit_breaker():
    from risk.circuit_breaker import CircuitBreaker

    assert CircuitBreaker is not None


def test_import_execution_engine():
    from execution.engine import ExecutionEngine

    assert ExecutionEngine is not None


def test_import_execution_cost_model():
    from execution.cost_model import CostModel

    assert CostModel is not None


def test_import_execution_algo():
    from execution.algo_executor import AlgoExecutor

    assert AlgoExecutor is not None


def test_import_validation_walk_forward():
    from validation.walk_forward import PurgedWalkForward, WFResult

    assert PurgedWalkForward is not None


def test_import_validation_monte_carlo():
    from validation.monte_carlo import MonteCarloSigTest, SigTestResult

    assert MonteCarloSigTest is not None


def test_import_training_registry():
    from training.model_registry import ModelRegistry

    assert ModelRegistry is not None


def test_import_training_online_learner():
    from training.online_learner import OnlineLearner

    assert OnlineLearner is not None


def test_import_training_distributed():
    from training.distributed_trainer import DistributedTrainer, TrialConfig, TrialResult

    assert DistributedTrainer is not None


def test_import_infrastructure_config():
    from infrastructure.config_v2 import AppConfig, SymbolsConfig

    assert AppConfig is not None


def test_import_infrastructure_secrets():
    from infrastructure.secrets import Secrets

    assert Secrets is not None


def test_import_notifications():
    from notifications.telegram import TelegramNotifier

    assert TelegramNotifier is not None


def test_import_dashboard():
    from dashboard.app import app

    assert app is not None


def test_import_backtest():
    from backtest.vectorized_backtester import VectorizedBacktester

    assert VectorizedBacktester is not None
```

### Task 5.2 — Run smoke tests

```bash
python3 -m pytest tests/test_smoke_imports.py -v
```

Expected: All ~30 import tests pass.

---

## Task Group 6: Final Verification

### Task 6.1 — Run full lint
```bash
make lint
```
Expected: E9/F errors = 0.

### Task 6.2 — Run mypy
```bash
make type-check
```
Expected: 0 errors (down from 55).

### Task 6.3 — Run tests
```bash
make test
```
Expected: All existing tests pass + new smoke tests pass.

### Task 6.4 — Run health check
```bash
python3 health_check.py
```
Expected: All checks pass, no FAIL items.

### Task 6.5 — Run full check suite
```bash
make check
```
Expected: lint → type-check → test — all pass.

---

## Summary

| Group | Description | Files Changed | Expected Impact |
|-------|-------------|---------------|-----------------|
| **0** | Pre-flight config fix | 1 | Unblocks pre-commit hooks |
| **1** | health_check.py rewrite | 1 | Fixes 8 stale imports + 2 structural bugs |
| **2** | Mypy type error fixes | 11 | Eliminates 55 mypy errors |
| **3** | model.py duplicate decorator | 1 | Fixes Python runtime warning |
| **4** | Lint cleanup (impactful subset) | 2 | Removes ~30 F841 unused variables |
| **5** | Test coverage | 1 new file (~30 tests) | Coverage from ~0.5% → verified imports |
| **6** | Final verification | 0 | Full `make check` pass |

Total files modified: ~17 source files + 1 new test file + 1 config file = **19 files**.
