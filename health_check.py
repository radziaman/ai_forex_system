#!/usr/bin/env python3
"""System Health Check for RTS: Agentic Moneybot System Elite."""
import sys
import os

sys.path.insert(0, "src")
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

print("=" * 60)
print("COMPREHENSIVE HEALTH CHECK")
print("=" * 60)

results = {"OK": 0, "WARN": 0, "FAIL": 0}
messages = []


def check(name, fn):
    try:
        result = fn()
        if result is True:
            results["OK"] += 1
            messages.append("[OK] " + name)
        elif result is False:
            results["FAIL"] += 1
            messages.append("[FAIL] " + name)
        else:
            results["WARN"] += 1
            messages.append("[WARN] {}: {}".format(name, result))
    except Exception as e:
        results["FAIL"] += 1
        messages.append("[FAIL] {}: {}".format(name, e))


check("numpy", lambda: __import__("numpy") is not None)
check("pandas", lambda: __import__("pandas") is not None)
check("sklearn", lambda: __import__("sklearn") is not None)
check("tigramite", lambda: __import__("tigramite") is not None)
check("joblib", lambda: __import__("joblib") is not None)
check("loguru", lambda: __import__("loguru") is not None)
check("yaml", lambda: __import__("yaml") is not None)

# RTS AI FX modules
try:
    from rts_ai_fx.causal_features import CausalFeatureSelector as _  # noqa: F811
    from rts_ai_fx.causal_features import CAUSAL_AVAILABLE

    check("causal_features", lambda: CAUSAL_AVAILABLE)
except Exception:
    check("causal_features", lambda: False)

try:
    from rts_ai_fx.features_unified import compute_features as _  # noqa: F811

    check("features_unified", lambda: True)
except Exception:
    check("features_unified", lambda: False)

try:
    from rts_ai_fx.regime_detector import HMMRegimeDetector as _  # noqa: F811

    check("regime_detector", lambda: True)
except Exception:
    check("regime_detector", lambda: False)

try:
    from rts_ai_fx.ensemble import MoEEnsemble as _  # noqa: F811

    check("ensemble", lambda: True)
except Exception:
    check("ensemble", lambda: False)

try:
    from rts_ai_fx.model import LSTMCNNHybrid as _  # noqa: F811

    check("model (AI)", lambda: True)
except Exception:
    check("model (AI)", lambda: False)

try:
    from rts_ai_fx.drift_detector import DriftMonitor as _  # noqa: F811

    check("drift_detector", lambda: True)
except Exception:
    check("drift_detector", lambda: False)

try:
    from rts_ai_fx.attention_fusion import TemporalAttentionFusion as _  # noqa: F811

    check("attention_fusion", lambda: True)
except Exception:
    check("attention_fusion", lambda: False)

try:
    from rts_ai_fx.uncertainty import monte_carlo_dropout as _  # noqa: F811
    from rts_ai_fx.uncertainty import get_confidence as _  # noqa: F811

    check("uncertainty", lambda: True)
except Exception:
    check("uncertainty", lambda: False)

try:
    from infrastructure.config_v2 import AppConfig as _  # noqa: F811

    check("infrastructure.config_v2", lambda: True)
except Exception:
    check("infrastructure.config_v2", lambda: False)

try:
    from infrastructure.secrets import Secrets as _  # noqa: F811

    check("infrastructure.secrets", lambda: True)
except Exception:
    check("infrastructure.secrets", lambda: False)

try:
    from risk.manager import RiskManager as _  # noqa: F811

    check("risk.manager", lambda: True)
except Exception:
    check("risk.manager", lambda: False)

try:
    from risk.circuit_breaker import CircuitBreaker as _  # noqa: F811

    check("risk.circuit_breaker", lambda: True)
except Exception:
    check("risk.circuit_breaker", lambda: False)

try:
    from execution.engine import ExecutionEngine as _  # noqa: F811

    check("execution.engine", lambda: True)
except Exception:
    check("execution.engine", lambda: False)

try:
    from execution.cost_model import CostModel as _  # noqa: F811

    check("execution.cost_model", lambda: True)
except Exception:
    check("execution.cost_model", lambda: False)

try:
    from execution.algo_executor import AlgoExecutor as _  # noqa: F811

    check("execution.algo_executor", lambda: True)
except Exception:
    check("execution.algo_executor", lambda: False)

try:
    from data.data_manager import DataManager as _  # noqa: F811

    check("data.data_manager", lambda: True)
except Exception:
    check("data.data_manager", lambda: False)

try:
    from data.market_session import MarketSession as _  # noqa: F811

    check("data.market_session", lambda: True)
except Exception:
    check("data.market_session", lambda: False)

try:
    from data.economic_calendar import EconomicCalendar as _  # noqa: F811

    check("data.economic_calendar", lambda: True)
except Exception:
    check("data.economic_calendar", lambda: False)

try:
    from ai.sentiment import SentimentAnalyzer as _  # noqa: F811

    check("ai.sentiment", lambda: True)
except Exception:
    check("ai.sentiment", lambda: False)

try:
    from ai.regime_agents import RegimeSpecialistSystem as _  # noqa: F811

    check("ai.regime_agents", lambda: True)
except Exception:
    check("ai.regime_agents", lambda: False)

try:
    from ai.maml_agent import MAMLAgent as _  # noqa: F811

    check("ai.maml_agent", lambda: True)
except Exception:
    check("ai.maml_agent", lambda: False)

try:
    from training.model_registry import ModelRegistry as _  # noqa: F811

    check("training.model_registry", lambda: True)
except Exception:
    check("training.model_registry", lambda: False)

try:
    from training.online_learner import OnlineLearner as _  # noqa: F811

    check("training.online_learner", lambda: True)
except Exception:
    check("training.online_learner", lambda: False)

try:
    from validation.walk_forward import PurgedWalkForward as _  # noqa: F811

    check("validation.walk_forward", lambda: True)
except Exception:
    check("validation.walk_forward", lambda: False)

try:
    from validation.monte_carlo import MonteCarloSigTest as _  # noqa: F811

    check("validation.monte_carlo", lambda: True)
except Exception:
    check("validation.monte_carlo", lambda: False)

try:
    from backtest.vectorized_backtester import VectorizedBacktester as _  # noqa: F811

    check("backtest.vectorized_backtester", lambda: True)
except Exception:
    check("backtest.vectorized_backtester", lambda: False)

try:
    from notifications.telegram import TelegramNotifier as _  # noqa: F811

    check("notifications.telegram", lambda: True)
except Exception:
    check("notifications.telegram", lambda: False)

try:
    from dashboard.app import app as _  # noqa: F401, F811

    check("dashboard.app", lambda: True)
except Exception:
    check("dashboard.app", lambda: False)

# File system checks
check("config.yaml exists", lambda: os.path.exists("config.yaml"))
check(".env exists", lambda: os.path.exists(".env"))
check("data/ directory", lambda: os.path.isdir("data"))
check("models/ directory", lambda: os.path.isdir("models"))

print()
for m in messages:
    print(m)
print()
ok_count = results["OK"]
warn_count = results["WARN"]
fail_count = results["FAIL"]
print("Results: {} OK, {} WARN, {} FAIL".format(ok_count, warn_count, fail_count))
print("=" * 60)

if fail_count > 0:
    print("\nBLOCKING ISSUES FOUND - Review FAIL items above")
    sys.exit(1)
elif warn_count > 0:
    print("\nReady with warnings - Review WARN items before live")
else:
    print("\nAll checks passed - System ready")
