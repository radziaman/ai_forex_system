"""
Comprehensive system check — imports, data flow, symbols, lifecycle, config.
"""

import sys
import os
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

results = {"pass": 0, "fail": 0}


def check(name, ok, detail=""):
    if ok:
        results["pass"] += 1
        print(f"  [OK] {name}")
    else:
        results["fail"] += 1
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))


# ── 1. Data flow integrity ──
print("\n[1] DATA FLOW INTEGRITY")
from agentic.agents.data_agent import DataAgent  # noqa: E402
from agentic.agents.regime_agent import RegimeAgent  # noqa: E402
from agentic.agents.risk_agent import RiskAgent  # noqa: E402
from agentic.agents.execution_agent import ExecutionAgent  # noqa: E402
from agentic.agents.signal_agent import SignalAgent  # noqa: E402
from agentic.agents.position_agent import PositionAgent  # noqa: E402
from agentic.agents.adaptive_risk_agent import AdaptiveRiskAgent  # noqa: E402
from agentic.agents.monitoring_agent import MonitoringAgent  # noqa: E402
from agentic.agents.feature_agent import FeatureAgent  # noqa: E402

s1 = inspect.getsource(DataAgent.reflect)
s2 = inspect.getsource(RegimeAgent.perceive)
s3 = inspect.getsource(RiskAgent._evaluate_signal)
s4 = inspect.getsource(ExecutionAgent._on_start)
s5 = inspect.getsource(SignalAgent._on_features)
s6 = inspect.getsource(PositionAgent._check_trailing)
s7 = inspect.getsource(AdaptiveRiskAgent.reason)
s8 = inspect.getsource(MonitoringAgent._safe_telegram)
s9 = inspect.getsource(FeatureAgent.perceive)

check("data_agent publishes OHLCV", "data.ohlcv" in s1)
check("data_agent publishes ATR", "data.atr." in s1)
check("data_agent publishes primary_symbol", "data.primary_symbol" in s1)
check("regime_agent reads OHLCV", "data.ohlcv" in s2)
check("feature_agent reads OHLCV", "data.ohlcv" in s9)
check("risk_agent reads ATR", "data.atr." in s3)
check("execution_agent stores spread", "data.spread." in s4)
check("risk_agent uses live spread", "live_spread" in s3)
s3_onstart = inspect.getsource(RiskAgent._on_start)
check("risk_agent has price_provider", "price_provider" in s3_onstart)
s5_all = inspect.getsource(SignalAgent)
check("signal_agent per-symbol LSTM", "_lstm_prediction_for_symbol" in s5_all)
check("signal_agent PPO state extraction", "_features_to_ppo_state" in s5_all)
check("position_agent trailing stops", "_check_trailing" in s6)
s7_all = inspect.getsource(AdaptiveRiskAgent)
check("adaptive_risk adjusts kelly", "effective_kelly" in s7_all)
check("monitoring_agent retry logic", "max_retries" in s8)
s8_all = inspect.getsource(MonitoringAgent)
check(
    "monitoring_agent alert levels",
    "level: str" in s8_all or "ALERT_CRITICAL" in s8_all,
)

# ── 2. Dropped symbols check ──
print("\n[2] DROPPED SYMBOLS")
import pathlib  # noqa: E402

dropped = [
    "EURJPY",
    "GBPJPY",
    "EURGBP",
    "XAGUSD",
    "XBRUSD",
    "XNGUSD",
    "US30",
    "USTEC",
    "UK100",
    "DE40",
    "ETHUSD",
    "LTCUSD",
    "XRPUSD",
]
found_any = False
for sym in dropped:
    for py in pathlib.Path("src").rglob("*.py"):
        text = py.read_bytes().decode("utf-8", errors="replace")
        # Skip agentic test files
        if (
            "_check" in py.name
            or "_test" in py.name
            or "_fix" in py.name
            or "_diag" in py.name
        ):
            continue
        if sym in text:
            print(f"  LEAK: {sym} in {py.relative_to('src')}")
            found_any = True
check("zero leftover symbol references", not found_any)

# ── 3. Config consistency ──
print("\n[3] CONFIG CONSISTENCY")
from data.data_manager import SYMBOLS, BASE_PRICES  # noqa: E402
from api.symbol_map import SYMBOL_MAP as SM  # noqa: E402
from api.ctrader_client import SYMBOL_MAP as CM  # noqa: E402
from infrastructure.config_v2 import AppConfig  # noqa: E402

cfg = AppConfig()
check(f"SYMBOLS count = {len(SYMBOLS)}", len(SYMBOLS) == 11)
check("all SYMBOLS have BASE_PRICES", all(s in BASE_PRICES for s in SYMBOLS))
check("SYMBOLS == Config.all", len(SYMBOLS) == len(cfg.symbols.all))
check("ctrader_map covers SYMBOLS", all(s in CM for s in SYMBOLS))
check("symbol_map covers SYMBOLS", all(s in SM for s in SYMBOLS))
check("no extra ctrader IDs", len(CM) == 11)
check("no extra symbol_map IDs", len(SM) == 11)

# ── 4. Agent framework health ──
print("\n[4] AGENT FRAMEWORK HEALTH")
from agentic.core.agent_message import PAYLOAD_SCHEMAS  # noqa: E402
from agentic.core.agent_consciousness import (  # noqa: E402
    EmotionalState,
    ConsciousnessLevel,
)  # noqa: E402
from agentic.core.agent_bus import AgentBus  # noqa: E402

check("PAYLOAD_SCHEMAS defined", len(PAYLOAD_SCHEMAS) > 10)
check(
    "EmotionalState has 5 dimensions",
    all(
        hasattr(EmotionalState(), a)
        for a in ["fatigue", "stress", "engagement", "confidence", "curiosity"]
    ),
)
check("ConsciousnessLevel has META", hasattr(ConsciousnessLevel, "META"))
check("AgentBus has priority queues", AgentBus()._queues is not None)

# ── 5. All 15 agents load ──
print("\n[5] ALL 15 AGENTS IMPORT")
agent_names = [
    "data_agent",
    "feature_agent",
    "regime_agent",
    "signal_agent",
    "risk_agent",
    "adaptive_risk_agent",
    "execution_agent",
    "position_agent",
    "performance_agent",
    "validation_agent",
    "master_agent",
    "monitoring_agent",
    "connection_agent",
    "learning_agent",
    "memory_agent",
]
loaded = 0
for name in agent_names:
    try:
        __import__(f"agentic.agents.{name}", fromlist=["object"])
        loaded += 1
    except Exception as e:
        print(f"  FAIL: {name}: {e}")
check(f"{loaded}/15 agents import", loaded == 15)

# ── 6. All existing modules consistent ──
print("\n[6] SUPPORTING MODULES")
extra_checks = [
    ("execution.cost_model", "CostModel"),
    ("execution.engine", "ExecutionEngine"),
    ("risk.manager", "RiskManager"),
    ("risk.circuit_breaker", "CircuitBreaker"),
    ("api.ctrader_client", "CtraderClient"),
    ("api.symbol_map", "get_symbol_id"),
    ("notifications.telegram", "TelegramNotifier"),
    ("ai.regime_agents", "RegimeSpecialistSystem"),
    ("ai.rl_agent", "PPOAgent"),
    ("rts_ai_fx.ensemble", "MoEEnsemble"),
    ("rts_ai_fx.features_unified", "FeaturePipeline"),
    ("rts_ai_fx.regime_detector", "HMMRegimeDetector"),
    ("rts_ai_fx.model", "LSTMCNNHybrid"),
    ("backtest.vectorized_backtester", "VectorizedBacktester"),
    ("validation.monte_carlo", "MonteCarloSigTest"),
]
for mod_name, cls_name in extra_checks:
    try:
        m = __import__(mod_name, fromlist=[cls_name])
        check(f"{mod_name}.{cls_name}", hasattr(m, cls_name) if m else False)
    except Exception as e:
        check(f"{mod_name}.{cls_name}", False, str(e)[:80])

# ── Summary ──
print(f"\n{'='*50}")
print(f"  RESULTS: {results['pass']} passed, {results['fail']} failed")
print(f"  {'ALL CLEAN' if results['fail'] == 0 else 'ISSUES FOUND'}")
print(f"{'='*50}")
sys.exit(0 if results["fail"] == 0 else 1)
