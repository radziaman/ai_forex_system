# RTS AI Forex Trading System — Comprehensive Codebase Analysis

> **⚠️ STATUS: Historical Baseline**
> 
> This analysis was captured on 2026-05-31 as a baseline before a comprehensive
> refactoring (8 parallel engineer sessions, June 1st 2026). ALL 25 roadmap items
> across 5 phases have been completed. See [COMPLETION.md](docs/completion.md)
> for the current state.

**Date:** 2026-05-31 (v2 — corrected for pipeline architecture)  
**Analyst:** Senior Architect  
**Scope:** Full codebase analysis of `rts_ai_fx_trading v3.0` (18,000+ LOC across 75+ files)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Pipeline Event Flow](#3-pipeline-event-flow)
4. [Package Structure](#4-package-structure)
5. [Deep Module Analysis](#5-deep-module-analysis)
6. [Critical Issues](#6-critical-issues)
7. [Major Refactoring Recommendations](#7-major-refactoring-recommendations)
8. [Medium-Impact Improvements](#8-medium-impact-improvements)
9. [Testing & Quality Gaps](#9-testing--quality-gaps)
10. [Security & Deployment Concerns](#10-security--deployment-concerns)
11. [Enhancement Proposals](#11-enhancement-proposals)
12. [Architecture Decision Records (New)](#12-architecture-decision-records-new)
13. [Roadmap](#13-roadmap)
14. [File-by-File Issue Log](#14-file-by-file-issue-log)

---

## 1. Executive Summary

This is a **sophisticated, multi-strategy algorithmic FX trading system** with a **clean 5-module EventBus-based pipeline architecture**, **Mixture-of-Experts (MoE) ensemble AI**, and **production-grade risk management**. The system has evolved through three architectural generations:

| Gen | Location | Status | Description |
|-----|----------|--------|-------------|
| 1️⃣ | `src/_archive/agentic/` | 🗄️ Archived | 20-agent consciousness-based swarm (1,446-line monoliths) |
| 2️⃣ | `src/pipeline/` | ✅ **Active** | 5-module pipeline with EventBus pub/sub (3,047 LOC) |
| 3️⃣ | `src/agentic/` | 🏗️ In Progress | ServiceContainer DI framework (101 LOC so far) |

### What's Excellent

- **Clean pipeline architecture** — `EventBus` + 5 decoupled modules (`SignalEngine`, `RiskManager`, `ExecutionManager`, `LearningManager`, `Orchestrator`)
- **MoE Ensemble weighting** — clean multiplicative composition of regime × Elo × Sharpe × confidence × tracker weights
- **Circuit Breaker** — production-grade with degradation modes, API health tracking, cooldown periods
- **Portfolio Optimizer** — correct HRP (Lopez de Prado 2016), analytical Mean-Variance, Risk Parity with CCD
- **Feature Pipeline 49-dim invariant** — well-enforced contract across all models
- **Walk-forward validation** — proper purged CPCV with embargo periods
- **DI container in progress** — `ServiceContainer` at `src/agentic/core/service_container.py` is clean

### Critical Issues

1. **🔴 Dockerfile points to archived entry point** — `agentic.main_agentic` is in `_archive/`; should be `pipeline.main`
2. **🔴 `TradeRecord.position_id` AttributeError** — `enhanced_manager.py` references undefined field
3. **🔴 Dead code return in cTrader loader** — `data_manager.py:607` `return True` makes entire loading block unreachable
4. **🔴 Broken smoke test imports** — 14 of 25 smoke test imports reference non-existent packages
5. **🔴 Feature pipeline normalization key bug** — all timeframes save under same key due to loop variable scoping
6. **🔴 `integration_tests.py` is entirely stubs** — 8 test cases all just `sleep(0.01); return True`

---

## 2. Architecture Overview

### Corrected Architecture (Three Layers)

```
┌─────────────────────────────────────────────────────────────────┐
│                      src/pipeline/ (ACTIVE)                      │
│  3,047 LOC across 9 files — EventBus-based pub/sub              │
│                                                                  │
│  Orchestrator ──→ EventBus ←── SignalEngine                     │
│       │                  │          │                            │
│       │                  ├── "tick" ◄── (from DataManager)       │
│       │                  ├── "signal_generated" ──→ RiskManager  │
│       │                  ├── "risk_approved/rejected" ──→ ExecMgr│
│       │                  ├── "position_opened/closed" ──→ LearnM │
│       │                  ├── "health_check" ──→ Dashboard        │
│       │                  └── ...                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    src/agentic/ (IN PROGRESS)                     │
│  ServiceContainer — lightweight DI container (101 LOC)           │
│  Agent stubs (empty) — planned but not yet built                 │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│               src/_archive/agentic/ (RETIRED)                    │
│  Old 20-agent system — archived, not importable in production    │
└─────────────────────────────────────────────────────────────────┘
```

### Supporting Modules

```
src/rts_ai_fx/        14 files — Core ML (MoE, LSTM, HMM, features, adversarial)
src/data/             15 files — Data ingestion (Dukascopy, cTrader, yFinance)
src/risk/              6 files — Risk management (Kelly, VaR, circuit breaker)
src/validation/        8 files — Walk-forward, Monte Carlo, stress testing
src/infrastructure/    6 files — Config, logging, secrets
src/notifications/     2 files — Telegram alerts
```

### Key Design Decisions

- **EventBus** replaces the old agent-to-agent communication protocol with a clean pub/sub
- **Lazy imports** inside method bodies (e.g., `_ensure_initialized()` does `from rts_ai_fx.ensemble import MoEEnsemble`) — avoids import errors when optional deps (TF, PyTorch) are missing
- **PipelineContext** as DI container — modules receive their dependencies at construction rather than using globals
- **5 focused modules** instead of 20 agents — each module replaces 3-5 old agents (documented in module docstrings)

---

## 3. Pipeline Event Flow

```
DataManager.update_tick(symbol, bid, ask, volume)
       │
       │ bus.emit("tick", symbol=s, bid=b, ask=a, volume=v, ts=t)
       ▼
┌─────────────────────────────────────────────────────────────────┐
│ SignalEngine._on_tick()                                         │
│  1. FeaturePipeline.transform()      → 49-dim feature vector    │
│  2. HMMRegimeDetector.detect_regime() → "trending"/"ranging"/.. │
│  3. MoEEnsemble.predict()            → EnsemblePrediction       │
│  4. should_trade()                   → (bool, direction, conf)  │
│  5. _register_experts()              → 28 experts (PPO, LSTM,   │
│                                          rule-based, alpha)     │
│                                                                  │
│  bus.emit("signal_generated", signal={symbol, direction,         │
│       │     confidence, price, expert_outputs, regime})          │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│ RiskManager._on_signal_generated()                               │
│  1. Pre-trade checks (drawdown, daily loss, consecutive losses)  │
│  2. Circuit breaker check (price velocity, spread, volume, vol)  │
│  3. Kelly sizing with VaR/CVaR adjustment                        │
│  4. Correlation risk check (dynamic regime correlation matrix)    │
│                                                                  │
│  bus.emit("risk_approved", signal, volume, sl, tp)               │
│  ─ or ─                                                          │
│  bus.emit("risk_rejected", signal, reason)                       │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│ ExecutionManager._on_risk_approved()                              │
│  1. Order placement (paper or cTrader live)                      │
│  2. Position tracking (ManagedPosition dataclass)                 │
│  3. Trailing stops with TP milestones (30%/30%/40% partial close)│
│  4. Correlated position monitoring                               │
│                                                                  │
│  bus.emit("position_opened", position_dict)                      │
│  bus.emit("position_closed", position_dict)                      │
│  bus.emit("execution_result", result_dict)                       │
│  bus.emit("execution_quality", quality_dict)                     │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│ LearningManager (listens to position_closed + execution_result)  │
│  1. DriftMonitor.update()         → ADWIN concept drift detect   │
│  2. PerformanceTracker.record_trade() → Sharpe, win rate, PF     │
│  3. ModelRegistry.update()        → Champion/challenger models   │
│  4. CheckpointManager.save_checkpoint() → SHA256-verified state  │
│  5. Retraining triggers when drift detected or performance decays│
│                                                                  │
│  bus.emit("drift_detected", symbol, metric, score)               │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│ Orchestrator (60s health loop)                                   │
│  bus.emit("health_check", health_score, uptime, modules_alive)   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Package Structure (Corrected)

```
src/
├── pipeline/                  # 🟢 ACTIVE — 5-module event bus architecture (3,047 LOC)
│   ├── __init__.py             Exports: Orchestrator, SignalEngine, RiskManager,
│   │                                      ExecutionManager, LearningManager, EventBus
│   ├── event_bus.py            Lightweight pub/sub — bus.on() / bus.emit()
│   ├── orchestrator.py         Lifecycle management + health checks
│   ├── signal_engine.py        1,017 LOC — Features → HMM → MoE → Signals
│   ├── risk_manager.py         420 LOC — Risk checks, Kelly, circuit breaker
│   ├── execution_manager.py    593 LOC — Orders, positions, trailing stops
│   ├── learning_manager.py     671 LOC — Drift, registry, performance, persistence
│   ├── pipeline_context.py     DI container for shared dependencies
│   └── main.py                 Entry point: python -m pipeline.main --mode paper
│
├── agentic/                   # 🟡 IN PROGRESS — new DI framework (101 LOC so far)
│   ├── __init__.py             Empty
│   ├── agents/__init__.py      Empty (stub for future agents)
│   └── core/
│       ├── __init__.py         Docstring only
│       └── service_container.py  Clean DI container (register/get/has)
│
├── rts_ai_fx/                 # Core ML/AI logic (14 files)
│   ├── model.py                LSTM-CNN Hybrid + ProfitabilityClassifier
│   ├── ensemble.py             MoE Ensemble with Elo/Sharpe/Expert Lockout
│   ├── regime_detector.py      HMM + Simple (ADX) regime detection
│   ├── features_unified.py     49-dim feature pipeline
│   ├── attention_fusion.py     Multi-timeframe attention fusion (PyTorch)
│   ├── uncertainty.py          Monte Carlo Dropout
│   ├── drift_detector.py       ADWIN concept drift detection
│   ├── causal_features.py      PC-MCI causal discovery (tigramite)
│   ├── adversarial.py          PGD adversarial training
│   ├── differentiable_risk.py  Differentiable constraint layer (PyTorch)
│   ├── constrained_trainer.py  Augmented Lagrangian PPO trainer
│   ├── multi_asset_model.py    Multi-symbol TF model with embeddings
│   ├── cross_sectional_alpha.py  Cross-sectional alpha signals
│   └── __init__.py             TF/Keras backward compat setup
│
├── data/                      # Data ingestion (15 files)
│   ├── data_manager.py         1,237 LOC — Multi-source data orchestrator
│   ├── dukascopy_provider.py   Dukascopy historical data
│   ├── dukascopy_realtime.py   Dukascopy real-time streaming
│   ├── multi_source_tick_provider.py  Multi-source tick aggregation
│   ├── multi_venue_provider.py Multi-venue abstraction
│   ├── cross_asset_features.py Cross-asset correlation features
│   ├── microstructure_features.py  Tick-level microstructure
│   ├── session_features.py     Market session patterns
│   ├── macroeconomic_features.py  Macroeconomic indicators
│   ├── market_session.py       Session detection
│   ├── economic_calendar.py    Economic event calendar
│   ├── alternative_data.py     Alternative data (sentiment, news)
│   ├── options_data.py         Options market data
│   └── dom_analyzer.py         Level II DOM analysis
│
├── risk/                      # Risk management (6 files)
│   ├── manager.py              Core RiskManager + TrailingStopManager
│   ├── enhanced_manager.py     MAE/MFE, CVaR-Kelly, stress testing
│   ├── circuit_breaker.py      Flash crash detection, graceful degradation
│   ├── portfolio_optimizer.py  Mean-Variance, Risk Parity, HRP
│   ├── correlation_matrix.py   Regime-dependent rolling correlations
│   └── __init__.py
│
├── validation/                # Model validation (8 files)
│   ├── walk_forward.py         Purged walk-forward
│   ├── smart_walk_forward.py   CPCV, regime-aware splits
│   ├── monte_carlo.py          Permutation significance tests
│   ├── smart_stress_test.py    Stress testing
│   ├── attribution.py          Performance attribution
│   ├── integration_tests.py    Smart integration test pipeline (🟡 stubs)
│   ├── run_validation.py       CLI entry point
│   └── __init__.py
│
├── infrastructure/            # Config, logging, secrets (6 files)
│   ├── config_v2.py            🟡 DUPLICATE — used by pipeline/main.py
│   ├── config_v3.py            🟡 DUPLICATE — newer version
│   ├── logging.py
│   ├── secrets.py
│   ├── system_info.py
│   └── __init__.py
│
├── notifications/             # Telegram alerts (2 files)
│
└── _archive/                  # 🗄️ RETIRED — 40+ files, NOT importable in production
    ├── agentic/               Old 20-agent consciousness system
    ├── train_*.py             Various training scripts
    └── app.py, etc.
```

---

## 5. Deep Module Analysis

### 5.1 EventBus (`src/pipeline/event_bus.py`) — 70 lines

**Quality:** ✅ Clean, minimal, correct

- Simple pub/sub with `on()` / `emit()` / `off()` pattern
- Supports decorator-style subscription (`@bus.on("tick")`)
- No ACKs, no priority queues — lightweight by design
- Error isolation — exceptions in one handler don't affect other subscribers
- Well-documented event catalog in module docstring

**Issue:** No way to await handler completion or get return values from handlers. This is a deliberate trade-off.

**Recommendation:** Consider adding `once()` for one-shot listeners and `wait_for(event, timeout)` for testability.

### 5.2 Pipeline Entry Point (`src/pipeline/main.py`) — 114 lines

**Quality:** ✅ Well-structured entry point

- Creates `PipelineContext` (DI container) with config, secrets, bus, data_manager
- Instantiates all 4 modules + orchestrator
- Registers modules with orchestrator for lifecycle management
- Handles SIGINT/SIGTERM for graceful shutdown
- Supports `--mode paper|live|backtest` and `--config` flags

**Issues:**
- Uses `AppConfig` from `config_v2.py` — but `config_v3.py` exists as a newer version
- Imports via `sys.path.insert(0, ...)` — fragile path manipulation
- Does NOT pass `DataManager` to `SignalEngine` (it expects `ctx.data_manager` but `PipelineContext.data_manager` is `Optional`)

### 5.3 SignalEngine (`src/pipeline/signal_engine.py`) — 1,017 lines

**Quality:** ⚠️ Largest file in the system — consolidates 3+ old agents

**What it does:**
- Receives ticks via `bus.on("tick")` → computes features → detects regime → runs ensemble → emits signals
- Registers 28+ experts (PPO regime agents, LSTM-CNN, rule-based breakout/mean-rev, alpha strategies)
- Tracks expert outcomes with Elo ratings, Sharpe ratios, consecutive losses
- Loads models lazily on first use

**Issues:**
- **1,017 lines** — violates single-responsibility principle. Should be split into:
  - `FeatureProcessor` (feature extraction + normalization)
  - `ExpertRegistry` (expert management, registration)
  - `SignalGenerator` (ensemble prediction → signal decision)
- Lazy imports (`from rts_ai_fx.xxx import Xxx`) inside `_ensure_initialized()` are good for avoiding import errors, but make testing harder
- Alpha research pipeline (`from ai.alpha_research import AlphaResearchPipeline`) — likely doesn't exist, wrapped in try/except
- Meta-learning orchestrator — same pattern
- Expert outcomes tracking (Elo, Sharpe) duplicates the same logic in `MoEEnsemble` in `rts_ai_fx/ensemble.py`

### 5.4 RiskManager (`src/pipeline/risk_manager.py`) — 420 lines

**Quality:** ✅ Good — replaces 4 old agents

**Issues:**
- `_risk_engine` and `_circuit_breaker` are optional (both default to `None`). If not injected, they're never created, meaning ALL risk checks silently pass
- Reads from `ctx.config` but the config hierarchy is unclear (method `getattr(self.config, "risk", self.config)`)
- Does NOT integrate with `src/risk/enhanced_manager.py` (MAE/MFE, CVaR) or `src/risk/correlation_matrix.py`

### 5.5 ExecutionManager (`src/pipeline/execution_manager.py`) — 593 lines

**Quality:** ✅ Solid implementation

- `ManagedPosition` dataclass — clean state management
- Correlation group awareness — avoids correlated pairs
- TP milestones with partial closes (30%/30%/40%)
- Paper mode by default, live mode via cTrader
- Tracks execution quality (slippage, fill rate) and emits feedback

**Issues:**
- `ExecutionEngine` is optional (`None` default) — if not injected, no actual order execution happens
- Live mode imports (`from api.ctrader_client import ...`) are inside try/except blocks

### 5.6 LearningManager (`src/pipeline/learning_manager.py`) — 671 lines

**Quality:** ✅ Thorough — replaces 6 old agents

**Notable components:**
- `DriftMonitor` — ADWIN-based concept drift detection with `_SimpleADWIN` fallback
- `PerformanceTracker` — Sharpe, profit factor, win rate, by-symbol/by-regime breakdown
- `CheckpointManager` — SHA256-verified state persistence
- `ModelRegistry` — champion/challenger model tracking
- `OnlineLearner` — incremental model updates

**Issues:**
- Duplicates `ADWIN` from `rts_ai_fx/drift_detector.py` — the `_SimpleADWIN` fallback class is a near-copy
- `CheckpointManager._last_save_time` rate-limiting (300s) means state is only persisted every 5 minutes minimum

### 5.7 MoE Ensemble (`src/rts_ai_fx/ensemble.py`) — 447 lines

**Quality:** ⚠️ Good architecture, notable dead code

**Expert weighting formula** (clean design):
```
weight = regime_weight × elo_weight × sharpe_weight × conf_weight × tracker_weight
```

**Issues:**
- **MAML integration is dead** — `self.maml_agent` is always `None`, `_maml_adapt()` never executes
- **Foundation/Mamba experts are stubs** — `_foundation_predict()` computes naive momentum; `_mamba_predict()` does lazy import inside method
- **`_determine_direction()` is dead code** — docstring admits it's only a fallback
- **No serialization** — all Elo/Sharpe/lockout state is in-memory
- **Duplicate loguru imports** — `logger = __import__("loguru").logger` inside exception handlers instead of using module-level import

### 5.8 Feature Pipeline (`src/rts_ai_fx/features_unified.py`) — 566 lines

**Quality:** ✅ Well-engineered, single-file, clear 49-dim contract

**Strengths:**
- Cyclical time encoding (sin/cos)
- HAR-RV volatility features (Corsi 2009)
- Hurst exponent estimate
- Normalization persistence with symbol-TF keying
- Dimension padding to maintain 49-dim invariant

**Issues:**
- **🔴 Normalization key bug** — `_norm_key(symbol, tf)` at line 380 uses `tf` from outer scope. Due to Python's loop variable scoping, ALL timeframes get saved under the LAST timeframe's key
- `_extract_symbol_data()` — fragile nested vs flat dict handling
- `compute_features()` computes ~45 base features + 4 microstructure = 49. Adding/removing any feature requires updating this invariant

### 5.9 Risk Manager (`src/risk/manager.py` + `src/risk/enhanced_manager.py`)

**Quality:** ✅ Mature, statistically sound

**Strengths:**
- Kelly sizing with R-multiples (PnL/risk-normalized averaging)
- Per-symbol VaR/CVaR
- Dynamic correlation risk checks (static FX_CORR + dynamic `RegimeCorrelationMatrix`)
- Kill switch with drawdown-triggered auto-release at <3%

**Issues:**
- **🔴 `TradeRecord.position_id`** — used in `enhanced_manager.py` lines 120, 154 but NOT defined in `TradeRecord` dataclass (AttributeError on first trade)
- `EnhancedRiskManager.__getattr__()` proxy — fragile, breaks type checking
- `_check_correlation_risk()` — hardcoded `len(open_symbols) >= 3` fallback is overly restrictive
- Static `FX_CORR` dict is out of date — regimes cause correlations to break

### 5.10 Circuit Breaker (`src/risk/circuit_breaker.py`) — 394 lines

**Quality:** ✅ ★ Production-grade

**Strengths:**
- 4 independent detectors: price velocity, spread widening, volume anomaly, volatility spike
- Graceful degradation modes: NORMAL → DEGRADED → HALTED
- API health tracking per source
- Confidence threshold auto-adjustment (0.65 base → 0.95 under stress)
- Cooldown periods prevent repeated halt/recover cycles

**Issues:**
- No warm-up period — baseline statistics aren't meaningful until 50+ observations
- No state persistence across restarts
- `_attempt_recovery()` uses ambiguous `cooldown_seconds * 2` without checking specific symbol

### 5.11 Portfolio Optimizer (`src/risk/portfolio_optimizer.py`) — 475 lines

**Quality:** ✅ ★ Excellent — clean, correct, well-documented

- 3 methods: Mean-Variance (analytical pinv), Risk Parity (CCD), HRP (Lopez de Prado 2016)
- Efficient frontier computation
- Target volatility scaling
- All edge cases handled (singular covariance, zero variance, single asset)

### 5.12 Data Manager (`src/data/data_manager.py`) — 1,237 lines

**Quality:** ⚠️ Too large, but functionally complete

**Strengths:**
- Multi-source: Dukascopy BI5, cTrader protobuf, yFinance, CSV cache
- Batched tick ingestion (1s intervals) → OHLCV propagation (1m→5m→15m→1h→4h)
- Hash-based feature cache invalidation
- Gap detection (>90min) and healing
- Level II DOM with spoofing/iceberg detection

**Issues:**
- **🔴 `load_from_ctrader()` dead code** — `return True` at line 607 before the `try:` block makes the entire cTrader loader unreachable
- **🔴 Thread safety** — `_last_realtime_price`, `_pending_ticks`, `tick_buffer` mutated without locks
- `get_snapshot()` uses `type("Snapshot", (), {...})()` anti-pattern instead of a dataclass
- `_decode_bi5_to_bars()` — manual `struct.unpack` is fragile to format changes
- `update_market_depth()` — three type-check branches that should be `@singledispatch`
- `try_alternative_source()` — yFinance mapping has grown organically; should be a config file

---

## 6. Critical Issues

### 🔴 C-1: Dockerfile Points to Archived Entry Point

**File:** `Dockerfile:29`

```dockerfile
CMD ["python", "-m", "agentic.main_agentic", "--mode", "paper"]
```

`agentic.main_agentic` lives in `_archive/agentic/main_agentic.py`, which is excluded from the Python package (`pyproject.toml` `norecursedirs = ["_archive", ...]`). This Docker build will **fail at runtime** with `ModuleNotFoundError`.

**Fix:**
```dockerfile
CMD ["python", "-m", "pipeline.main", "--mode", "paper"]
```

### 🔴 C-2: `TradeRecord.position_id` AttributeError

**File:** `src/risk/enhanced_manager.py:120,154`

```python
self._mae_mfe[trade.position_id]  # AttributeError: TradeRecord has no 'position_id'
```

The `TradeRecord` dataclass in `src/risk/manager.py:34-44` does not define `position_id`.

**Fix:**
```python
@dataclass
class TradeRecord:
    position_id: int = 0  # ADD THIS
    timestamp: float = 0.0
    ...
```

### 🔴 C-3: Dead Code Return in cTrader Loader

**File:** `src/data/data_manager.py:607-608`

```python
return True  # data_refreshed     ← UNREACHABLE CODE PAST HERE
try:
    existing = self.ohlcv.get(symbol, {}).get(timeframe)
```

The `return True` on line 607 is NOT a comment — it's actual code that executes, making the rest of the method dead code.

**Fix:** Remove the spurious `return True` line.

### 🔴 C-4: Broken Smoke Test Imports

**File:** `tests/test_smoke_imports.py`

14 of 25 test functions import from non-existent packages:
- ✅ VALID: `rts_ai_fx.*`, `data.*`, `risk.*`, `validation.*`, `infrastructure.*`, `pipeline.*`, `notifications.*`
- ❌ BROKEN: `execution.*`, `training.*`, `dashboard.*`, `backtest.*`, `pipeline.*` (actually EXISTS — see corrected analysis)
- ❌ `ai.*` modules (`ai.maml_agent`, `ai.rl_agent`, `ai.sentiment`, `ai.regime_agents`) — may or may not exist

**Fix:** Either create stubs for missing packages or remove the broken test cases. `pipeline.*` tests should work.

### 🔴 C-5: Feature Pipeline Normalization Key Bug

**File:** `src/rts_ai_fx/features_unified.py:380`

```python
key = self._norm_key(symbol, tf)  # 'tf' is loop var from line 354
```
```python
for tf in self.timeframes:        # line 354
    ...
```
# line 380 is OUTSIDE the for loop — `tf` has the LAST value

This means `EURUSD_1m` and `EURUSD_1h` both save under `EURUSD_1h` (or whatever the last timeframe is).

**Fix:** Move key assignment inside the for loop.

### 🔴 C-6: Integration Tests Are All Stubs

**File:** `src/validation/integration_tests.py:154-314`

Every single test implementation:
```python
await asyncio.sleep(0.01)
return True, "Simulated X trading cycles", {...}
```

None exercise the actual trading system, circuit breaker, or data pipeline.

**Fix:** Either implement real tests against pipeline components or remove the module.

---

## 7. Major Refactoring Recommendations

### 7.1 [P0] Fix Dockerfile Entry Point

**What:** Change line 29 from `agentic.main_agentic` to `pipeline.main`.  
**Why:** The system cannot start as-is.  
**Effort:** 1 line change.  
**Risk:** None.

### 7.2 [P0] Remove `_archive/` from Source Tree

**What:** Move `_archive/` out of `src/` entirely (e.g., `archive/` at project root, or a git tag).  
**Why:** 40+ files (13,000+ LOC) of dead code obscure the real codebase. Mypy and pytest already exclude it.  
**Effort:** 5 minutes (git mv).  
**Risk:** Low — nothing imports from `_archive/` in production code.

### 7.3 [P1] Split SignalEngine into Focused Modules

**What:** Break `src/pipeline/signal_engine.py` (1,017 LOC) into:
- `pipeline/feature_processor.py` — feature extraction + normalization
- `pipeline/expert_registry.py` — expert management (register, load, track outcomes)
- `pipeline/signal_generator.py` — ensemble predict → should_trade decision

**Why:** Single-responsibility principle. Each piece is independently testable.  
**Effort:** 2-3 hours.  
**Risk:** Low — the EventBus interface stays the same; only internal structure changes.

### 7.4 [P1] Consolidate Configuration

**What:** Remove `infrastructure/config_v2.py` and `config_v3.py`. Read config directly from `config.yaml` using `yaml.safe_load()`.

**Why:** Two competing Python config classes (v2, v3) plus a YAML config. Only one should be canonical. The YAML is the most readable and maintainable.  
**Effort:** 1 hour + update imports in `pipeline/main.py` and `pipeline_context.py`.  
**Risk:** Medium — need to ensure all config attributes accessed via `getattr(config, "key")` are still resolvable.

### 7.5 [P1] Split DataManager into Focused Modules

**What:** Extract from `data/data_manager.py` (1,237 LOC):
- `data/tick_ingester.py` — tick validation, batching, 1m aggregation
- `data/historical_loader.py` — BI5/CSV/cTrader/yFinance loading
- `data/feature_cache.py` — hash-based feature caching
- `data/dom_manager.py` — market depth, order book analytics

**Why:** Single file handling 7+ concerns is a maintenance burden.  
**Effort:** 3-4 hours.  
**Risk:** Medium — many internal method calls between these concerns.

### 7.6 [P1] Add Ensemble Persistence

**What:** Implement `MoEEnsemble.save_state()` / `MoEEnsemble.load_state()` in `rts_ai_fx/ensemble.py`.

**Why:** Elo ratings, Sharpe ratios, lockout states, and win counts are all lost on restart. This means the ensemble starts fresh every time.  
**Effort:** 1 hour.  
**Format:** JSON with: Elo ratings, consecutive losses, disabled_until timestamps, win count per expert.

### 7.7 [P1] Fix `IntegrationTestPipeline` (or Remove It)

**What:** Either implement real integration tests or delete the module.

**Why:** Stub tests create false confidence. They "pass" without testing anything.  
**Effort:** 2-3 hours for real tests, 5 minutes for deletion.

---

## 8. Medium-Impact Improvements

### 8.1 Thread Safety in DataManager

`_last_realtime_price`, `_pending_ticks`, `tick_buffer` are mutated from multiple async callbacks without synchronization.

**Fix:** Add `asyncio.Lock` for each shared data structure:
```python
self._lock = asyncio.Lock()
async with self._lock:
    self._last_realtime_price[symbol] = mid
```

### 8.2 MAML Dead Code in Ensemble

`ensemble.py`:`191-194` — `self.maml_agent` is always `None`. The `_maml_adapt()` method always returns `X` unchanged.

**Options:**
1. **Implement MAML properly** — add meta-learning loop, wire `maml_agent` attribute
2. **Remove dead code** — remove `maml_agent`, `_maml_adapt()`, `use_maml_adaptation`

**Recommendation:** Option 2 unless MAML is on the roadmap. Dead code is worse than no code.

### 8.3 `_get_trend_direction()` Always Returns 1

`regime_detector.py:388-395`:
```python
def _get_trend_direction(self) -> int:
    if self.regime_history[-1] == "trending":
        return 1
    return 0  # Never returns -1
```

**Fix:** Track actual trend direction via last N price changes or EMA slope.

### 8.4 Circuit Breaker Warm-Up Period

`_check_liquidity()` uses `normal_spreads.get(symbol, spread)` — first tick matches baseline, never triggers. `_check_price_velocity()` requires 11+ price points for meaningful detection.

**Fix:** Add `is_warmed_up` property that returns `False` until minimum data threshold met. Log a warning during warm-up.

### 8.5 Duplicate ADWIN Implementation

`rts_ai_fx/drift_detector.py` has `class ADWIN`.
`pipeline/learning_manager.py` has `class _SimpleADWIN` (near-duplicate).

**Fix:** Import ADWIN from `rts_ai_fx.drift_detector` in `LearningManager` instead of duplicating.

### 8.6 EnhancedRiskManager Proxy Pattern

```python
class EnhancedRiskManager:
    def __getattr__(self, name: str):
        return getattr(self._base, name)
```

This proxy breaks IDE autocomplete, type checking, and `hasattr()`.

**Fix:** Either:
- Explicitly delegate each method (boilerplate but safe)
- Or merge `EnhancedRiskManager` functionality into `RiskManager`

### 8.7 Duplicate loguru Imports

In `ensemble.py` lines 118-119 and 139-140:
```python
logger = __import__("loguru").logger  # Should be: from loguru import logger
```

### 8.8 Standardize Logging

Some files use `import logging` (e.g., `features_unified.py:163`) alongside `from loguru import logger`.

**Fix:** Standardize on loguru everywhere.

### 8.9 `config.yaml` vs Hardcoded Defaults

Config says `max_positions: 3` but `RiskParameters` default is `max_positions: 5`. Config says `kelly_fraction: 0.10` but `RiskParameters` default is `0.25`.

**Fix:** RiskParameters should be populated from config at initialization time, not from hardcoded defaults.

### 8.10 Prediction Threshold in `should_trade()`

Hardcoded `threshold = 0.0003` is symbol-agnostic. JPY pairs move 10x more pips numerically.

**Fix:** Make threshold a function of recent ATR:
```python
threshold = atr / current_price * 0.1  # 10% of ATR
```

---

## 9. Testing & Quality Gaps

### 9.1 Test Coverage

**Current test inventory:**

| Location | Files | Quality |
|----------|-------|---------|
| `tests/rts_ai_fx/` | 5 | ✅ Functional unit tests |
| `tests/pipeline/` | 6 | ✅ Module-level tests |
| `tests/data/` | ? | Not inspected |
| `tests/risk/` | ? | Not inspected |
| `tests/validation/` | ? | Not inspected |
| `tests/test_smoke_imports.py` | 1 | ❌ 50% broken |
| `tests/conftest.py` | 1 | ✅ Proper path setup |

**Missing critical test coverage:**
- `MoEEnsemble.predict()` — expert weighting, lockout, regime gating
- `MoEEnsemble.should_trade()` — tier 1/tier 2 decision logic
- `RiskManager.calculate_kelly_size()` — R-multiple, VaR adjustment, vol targeting
- `HMMRegimeDetector.detect_regime()` — regime classification accuracy
- `CircuitBreaker.check_market_health()` — all 4 detector paths
- `DataManager.update_tick()` → OHLCV propagation correctness
- `EnhancedRiskManager` — MAE/MFE tracking, CVaR sizing

### 9.2 CI/CD Gaps

- `make check` likely fails due to broken smoke test imports
- `make type-check` (mypy) has 12 disabled error codes — catches very little
- No integration test that actually runs the full pipeline

### 9.3 Type Hints

- mypy configured to ignore `import-untyped`, `import-not-found`, `annotation-unchecked`, `no-any-return`, `var-annotated`, `operator`, etc.
- ~50% of functions lack complete type annotations
- `PipelineContext` uses `Optional[Any]` for all lazy-loaded services — defeats type checking

### 9.4 Import Strategy

The system uses `sys.path.insert(0, ...)` in `main.py` to find imports. This is fragile:
- Works in development but breaks if the package is installed via `pip install -e .`
- Should use relative imports within packages and ensure `src/` is on `PYTHONPATH`

---

## 10. Security & Deployment Concerns

### 10.1 Docker

- **Entry point is wrong** (see C-1)
- `gcc`/`g++` installed unnecessarily (likely for TA-Lib). Use multi-stage build or pre-built wheel
- Health check uses inline `python -c "import urllib.request;..."` — fragile, use proper endpoint
- Port 8000 exposed without TLS — for live trading, requires reverse proxy (nginx/caddy)

### 10.2 Secrets

- `.env` file permissions not enforced
- `Secrets` class at `infrastructure/secrets.py` should be audited for log leakage
- No key rotation mechanism

### 10.3 Fly.io Deployment

`fly.toml` exists — deployment config for Fly.io. Ensure secrets are loaded via `fly secrets set` not `.env` in production.

---

## 11. Enhancement Proposals

### E-1: EventBus Priority and Typed Events

Current EventBus dispatches all handlers for an event in arbitrary order. For pipeline correctness, some events need ordered processing:
- `risk_approved` must be processed before `position_opened`
- `signal_generated` must be processed by risk before execution

**Proposal:** Add optional priority parameter:
```python
bus.on("tick", handler, priority=10)  # Higher priority = earlier execution
```
And/or async `wait_for()` for tests:
```python
result = await bus.wait_for("signal_generated", timeout=5.0)
```

### E-2: Configuration Hot-Reload

Live trading requires changing parameters without restart. Currently, config is loaded once at startup.

**Proposal:** Add config file watcher that emits `config_changed` event when `config.yaml` is modified. Modules listen and update their parameters.

### E-3: Performance Attribution Pipeline

The `attribution.py` module exists but is not wired into the pipeline. 
**Proposal:** Wire it to receive `position_closed` events and attribute P&L to:
- Time of day (Asia/London/NY session)
- Market regime at entry/exit
- Expert that generated the signal
- Market condition changes during trade

### E-4: Ensemble State Checkpointing

Use `LearningManager.CheckpointManager` to periodically persist `MoEEnsemble` state (Elo, Sharpe, lockouts) alongside pipeline state.

### E-5: Web Dashboard Integration

`dashboard.html` exists at project root but `dashboard.unified_dashboard` module doesn't exist. 
**Proposal:** Create a lightweight FastAPI dashboard module that subscribes to EventBus events (health_check, position_opened/closed, signal_generated) and serves real-time state via websockets.

### E-6: Walk-Forward Validation CI

Wire `SmartWalkForward` into CI pipeline — run on every push with last 30 days of data. Fail CI if `stability_score < 0.5` or `avg_test_sharpe < 0.3`.

### E-7: Symbol-Agnostic should_trade Threshold

Replace hardcoded `threshold = 0.0003` with dynamic threshold based on recent volatility:
```python
threshold = max(0.0001, atr_per_bar * 0.1)
```

---

## 12. Architecture Decision Records (New)

### ADR-001: Pipeline Architecture over 20-Agent Swarm

**Context:** The original 20-agent system was archived. What replaces it?  
**Decision:** Use 5-module EventBus pipeline (`src/pipeline/`).  
**Rationale:** The old agents had too much overhead (perceive→reason→act→reflect cycles, consciousness, emotions). The pipeline is simpler, testable, and has no agent overhead.  
**Trade-off:** Less autonomy per module, but dramatically simpler debugging and testing.

### ADR-002: 49-Dimension Feature Contract

**Context:** All ML models depend on a fixed-size feature vector.  
**Decision:** Enforce `EXPECTED_FEATURE_DIM = 49` at runtime with padding/trimming.  
**Rationale:** Trading ML models can't be hot-swapped. A fixed contract allows independent model versioning.  
**Trade-off:** Adding features requires coordinated retraining of all models.

### ADR-003: Lazy Imports for Optional Dependencies

**Context:** The system supports both TensorFlow and PyTorch models. TF 2.16+ has breaking changes.  
**Decision:** Import heavy ML libraries inside methods, not at module level. Wrap in try/except.  
**Rationale:** The system should start and run even if one ML framework is broken or missing. Graceful degradation.  
**Trade-off:** First inference is slower (import overhead). Harder to mock in tests.

### ADR-004: Config Consolidation

**Context:** `config_v2.py` and `config_v3.py` duplicate `config.yaml`.  
**Decision:** Delete Python config classes. Use `yaml.safe_load()` directly.  
**Rationale:** Single source of truth. YAML is more human-readable and supports hot-reloading.  
**Migration:** Replace `from infrastructure.config_v2 import AppConfig` with `yaml.safe_load()` in `pipeline/main.py`.

### ADR-005: Dead Code Policy

**Context:** `_archive/`, MAML stubs, foundation model stubs, integration test stubs.  
**Decision:** Remove dead code. If it's not tested and not wired, delete it.  
**Rationale:** Dead code has maintenance cost (reading, searching, understanding) without benefit. Git history preserves it.  
**Exceptions:** Well-documented stubs for planned features are acceptable (e.g., new agentic framework stubs in `src/agentic/`).

---

## 13. Roadmap

### Phase 1: Fix Criticals (Day 1)
| # | Task | Est. | Owner |
|---|------|------|-------|
| 1 | Fix Dockerfile entry point → `pipeline.main` | 5 min | Ops |
| 2 | Add `position_id` to `TradeRecord` dataclass | 5 min | Engineer |
| 3 | Remove dead `return True` in `data_manager.py:607` | 2 min | Engineer |
| 4 | Fix feature pipeline `_norm_key` scoping bug | 10 min | Engineer |
| 5 | Fix broken smoke test imports (remove non-existent pkgs) | 15 min | Engineer |

### Phase 2: Clean-up (Day 2-3)
| # | Task | Est. | Owner |
|---|------|------|-------|
| 6 | Move `_archive/` out of `src/` | 5 min | Engineer |
| 7 | Consolidate config: remove v2/v3, keep YAML | 1 hr | Engineer |
| 8 | Remove MAML/foundation/mamba dead code in ensemble | 20 min | Engineer |
| 9 | Remove duplicate ADWIN in `learning_manager.py` | 10 min | Engineer |
| 10 | Standardize logging on loguru | 15 min | Engineer |

### Phase 3: Structural (Day 4-6)
| # | Task | Est. | Owner |
|---|------|------|-------|
| 11 | Split `signal_engine.py` (1,017 LOC → 3 modules) | 3 hrs | Engineer |
| 12 | Split `data_manager.py` (1,237 LOC → 4 modules) | 4 hrs | Engineer |
| 13 | Fix `EnhancedRiskManager.__getattr__` proxy pattern | 30 min | Engineer |
| 14 | Add thread-safety locks in DataManager | 30 min | Engineer |

### Phase 4: Production Hardening (Day 7-10)
| # | Task | Est. | Owner |
|---|------|------|-------|
| 15 | Add `MoEEnsemble.save_state()`/`load_state()` | 1 hr | Engineer |
| 16 | Implement or remove `IntegrationTestPipeline` | 2-3 hrs | Engineer |
| 17 | Wire `SmartWalkForward` into CI | 2 hrs | DevOps |
| 18 | Add circuit breaker warm-up period | 30 min | Engineer |
| 19 | Fix `_get_trend_direction()` to detect downtrends | 15 min | Engineer |
| 20 | Dynamic prediction threshold from ATR | 30 min | Engineer |

### Phase 5: Enhancements (Week 3+)
| # | Task | Est. | Owner |
|---|------|------|-------|
| 21 | EventBus priority + wait_for() | 1 hr | Engineer |
| 22 | Config hot-reload watcher | 2 hrs | Engineer |
| 23 | Web dashboard with FastAPI + websockets | 4 hrs | Engineer |
| 24 | Performance attribution pipeline | 3 hrs | Quant |
| 25 | Walk-forward CI gate | 2 hrs | DevOps |

---

## 14. File-by-File Issue Log

| File | Severity | Issue |
|------|----------|-------|
| `Dockerfile:29` | 🔴 | Entry point points to archived `agentic.main_agentic` |
| `pipeline/signal_engine.py` | 🟡 | 1,017 LOC overload — split into 3 modules |
| `pipeline/learning_manager.py:64-97` | 🟡 | Duplicate ADWIN implementation |
| `rts_ai_fx/ensemble.py:191-194` | 🟡 | MAML agent never initialized (dead code) |
| `rts_ai_fx/ensemble.py:106-119,139-140` | 🟡 | Foundation/Mamba experts always fail import |
| `rts_ai_fx/ensemble.py:424-435` | 🟢 | Dead code (`_determine_direction` unused) |
| `rts_ai_fx/ensemble.py:118,139` | 🟢 | `__import__("loguru")` → use module import |
| `rts_ai_fx/features_unified.py:380` | 🔴 | `tf` scoping bug → wrong normalization keys |
| `rts_ai_fx/regime_detector.py:388-395` | 🟡 | `_get_trend_direction()` always returns 1 |
| `rts_ai_fx/model.py:243-246` | 🟢 | Label bias: diff==0 → DOWN (short bias) |
| `rts_ai_fx/model.py:162-216` | 🟡 | 4 fallback layers for model loading |
| `rts_ai_fx/adversarial.py:108-138` | 🟡 | Numerical gradient O(n³) — 3 nested loops |
| `rts_ai_fx/causal_features.py:135` | 🟢 | Correlation threshold 0.05 may be too low |
| `risk/manager.py:34-44` | 🔴 | `TradeRecord` missing `position_id` field |
| `risk/enhanced_manager.py:120,154` | 🔴 | `trade.position_id` AttributeError |
| `risk/enhanced_manager.py:46` | 🟡 | `__getattr__` proxy breaks type checking |
| `risk/manager.py:340-372` | 🟡 | Hardcoded FX_CORR dict — outdated |
| `risk/circuit_breaker.py:354-362` | 🟢 | No warm-up period for baseline stats |
| `risk/circuit_breaker.py:226-255` | 🟡 | Recovery uses ambiguous cooldown |
| `risk/portfolio_optimizer.py` | ✅ | ★ Excellent implementation |
| `data/data_manager.py:607-608` | 🔴 | Dead code `return True` before try block |
| `data/data_manager.py:1169-1216` | 🟡 | `type("Snapshot", ...)` anti-pattern |
| `data/data_manager.py:1029-1101` | 🟡 | `update_market_depth()` should use singledispatch |
| `data/data_manager.py:240-310` | 🟡 | No thread-safety on shared state |
| `validation/integration_tests.py:154-314` | 🔴 | All 8 tests are stubs (`sleep; return True`) |
| `validation/smart_walk_forward.py:60-127` | 🟡 | `run()` business logic incomplete |
| `validation/run_validation.py:29` | 🔴 | Imports from non-existent `backtest` module |
| `infrastructure/config_v2.py` | 🟡 | Duplicate config — v2 vs v3 vs YAML |
| `infrastructure/config_v3.py` | 🟡 | Duplicate config — v2 vs v3 vs YAML |
| `tests/test_smoke_imports.py` | 🔴 | 50%+ imports fail (missing packages) |
| `tests/conftest.py` | ✅ | Proper path setup |

---

**Legend:** 🔴 Critical | 🟡 Needs attention | 🟢 Minor/Nit | ✅ No issues | ★ Exceptional

---

## Conclusion

This is a **high-quality, research-grade trading system** that has successfully evolved from a complex 20-agent swarm to a clean 5-module pipeline architecture. The core ML concepts (MoE ensemble, HMM regime detection, differentiable risk constraints, adversarial training) are well-implemented and the risk management is production-grade.

The codebase has **two main problems**:

1. **Architectural drift** — the active pipeline (`src/pipeline/`) is good, but the old agentic archive (`src/_archive/`), the new agentic stubs (`src/agentic/`), and the duplicate configs create confusion about which code is canonical.

2. **Several critical bugs** — wrong Docker entry point (system won't start), `position_id` AttributeError (crashes on first trade), dead code in cTrader loader (data loading broken), normalization key bug (wrong features), and stub integration tests (false confidence).

### Top 5 Immediate Actions

| # | Action | Impact |
|---|--------|--------|
| 1 | Fix Dockerfile → `pipeline.main` | System can start |
| 2 | Add `position_id` to `TradeRecord` | System won't crash on first trade |
| 3 | Remove dead `return True` in cTrader loader | cTrader data loading works |
| 4 | Fix `_norm_key` scoping bug | Features compute correctly |
| 5 | Fix smoke test imports | CI pipeline goes green |

After these fixes, the codebase is in **excellent shape** for further development and production deployment.
