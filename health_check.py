#!/usr/bin/env python3
"""System Health Check for RTS AI Forex Trading System."""
import sys, os
sys.path.insert(0, 'src')
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

print('=' * 60)
print('COMPREHENSIVE HEALTH CHECK')
print('=' * 60)

results = {'OK': 0, 'WARN': 0, 'FAIL': 0}
messages = []

def check(name, fn):
    try:
        result = fn()
        if result is True:
            results['OK'] += 1
            messages.append('[OK] ' + name)
        elif result is False:
            results['FAIL'] += 1
            messages.append('[FAIL] ' + name)
        else:
            results['WARN'] += 1
            messages.append('[WARN] {}: {}'.format(name, result))
    except Exception as e:
        results['FAIL'] += 1
        messages.append('[FAIL] {}: {}'.format(name, e))

check('numpy', lambda: __import__('numpy') is not None)
check('pandas', lambda: __import__('pandas') is not None)
check('sklearn', lambda: __import__('sklearn') is not None)
check('tigramite', lambda: __import__('tigramite') is not None)
check('joblib', lambda: __import__('joblib') is not None)
check('loguru', lambda: __import__('loguru') is not None)
check('yaml', lambda: __import__('yaml') is not None)

# RTS AI FX modules
try:
    from rts_ai_fx.causal_features import CausalFeatureSelector, CAUSAL_AVAILABLE
    check('causal_features', lambda: CAUSAL_AVAILABLE)
except Exception as e:
    check('causal_features', lambda: (_ for _ in ()).throw(e))

try:
    from rts_ai_fx.features_unified import compute_features
    check('features_unified', lambda: True)
except Exception as e:
    check('features_unified', lambda: (_ for _ in ()).throw(e))

try:
    from rts_ai_fx.regime_detector import HMMRegimeDetector
    check('regime_detector', lambda: True)
except Exception as e:
    check('regime_detector', lambda: (_ for _ in ()).throw(e))

try:
    from rts_ai_fx.ensemble import MoEEnsemble
    check('ensemble', lambda: True)
except Exception as e:
    check('ensemble', lambda: (_ for _ in ()).throw(e))

try:
    from rts_ai_fx.model import LSTMCNNHybrid, ProfitabilityClassifier
    check('model (AI)', lambda: True)
except Exception as e:
    check('model (AI)', lambda: (_ for _ in ()).throw(e))

try:
    from rts_ai_fx.drift_detector import DriftMonitor
    check('drift_detector', lambda: True)
except Exception as e:
    check('drift_detector', lambda: (_ for _ in ()).throw(e))

try:
    from rts_ai_fx.attention_fusion import AttentionFusionPipeline
    check('attention_fusion', lambda: True)
except Exception as e:
    check('attention_fusion', lambda: (_ for _ in ()).throw(e))

try:
    from rts_ai_fx.uncertainty import monte_carlo_dropout, get_confidence
    check('uncertainty', lambda: True)
except Exception as e:
    check('uncertainty', lambda: (_ for _ in ()).throw(e))

# Infrastructure
try:
    from infrastructure.config import Config
    check('infrastructure.config', lambda: True)
except Exception as e:
    check('infrastructure.config', lambda: (_ for _ in ()).throw(e))

try:
    from infrastructure.secrets import Secrets
    check('infrastructure.secrets', lambda: True)
except Exception as e:
    check('infrastructure.secrets', lambda: (_ for _ in ()).throw(e))

try:
    from infrastructure.event_bus import TradingEventBus
    check('infrastructure.event_bus', lambda: True)
except Exception as e:
    check('infrastructure.event_bus', lambda: (_ for _ in ()).throw(e))

# Risk
try:
    from risk.manager import RiskManager
    check('risk.manager', lambda: True)
except Exception as e:
    check('risk.manager', lambda: (_ for _ in ()).throw(e))

try:
    from risk.circuit_breaker import CircuitBreaker
    check('risk.circuit_breaker', lambda: True)
except Exception as e:
    check('risk.circuit_breaker', lambda: (_ for _ in ()).throw(e))

# Execution
try:
    from execution.engine import ExecutionEngine
    check('execution.engine', lambda: True)
except Exception as e:
    check('execution.engine', lambda: (_ for _ in ()).throw(e))

try:
    from execution.cost_model import CostModel
    check('execution.cost_model', lambda: True)
except Exception as e:
    check('execution.cost_model', lambda: (_ for _ in ()).throw(e))

try:
    from execution.algo_executor import AlgoExecutor
    check('execution.algo_executor', lambda: True)
except Exception as e:
    check('execution.algo_executor', lambda: (_ for _ in ()).throw(e))

# Data
try:
    from data.data_manager import DataManager
    check('data.data_manager', lambda: True)
except Exception as e:
    check('data.data_manager', lambda: (_ for _ in ()).throw(e))

try:
    from data.market_session import MarketSession
    check('data.market_session', lambda: True)
except Exception as e:
    check('data.market_session', lambda: (_ for _ in ()).throw(e))

try:
    from data.economic_calendar import EconomicCalendar
    check('data.economic_calendar', lambda: True)
except Exception as e:
    check('data.economic_calendar', lambda: (_ for _ in ()).throw(e))

# AI modules
try:
    from ai.sentiment import SentimentAnalyzer
    check('ai.sentiment', lambda: True)
except Exception as e:
    check('ai.sentiment', lambda: (_ for _ in ()).throw(e))

try:
    from ai.regime_agents import RegimeSpecialistSystem
    check('ai.regime_agents', lambda: True)
except Exception as e:
    check('ai.regime_agents', lambda: (_ for _ in ()).throw(e))

try:
    from ai.maml_agent import MAMLAgent
    check('ai.maml_agent', lambda: True)
except Exception as e:
    check('ai.maml_agent', lambda: (_ for _ in ()).throw(e))

# Training/Validation
try:
    from training.model_registry import ModelRegistry
    check('training.model_registry', lambda: True)
except Exception as e:
    check('training.model_registry', lambda: (_ for _ in ()).throw(e))

try:
    from training.online_learner import OnlineLearner
    check('training.online_learner', lambda: True)
except Exception as e:
    check('training.online_learner', lambda: (_ for _ in ()).throw(e))

try:
    from validation.walk_forward import PurgedWalkForward
    check('validation.walk_forward', lambda: True)
except Exception as e:
    check('validation.walk_forward', lambda: (_ for _ in ()).throw(e))

try:
    from validation.monte_carlo import MonteCarloSigTest
    check('validation.monte_carlo', lambda: True)
except Exception as e:
    check('validation.monte_carlo', lambda: (_ for _ in ()).throw(e))

# Backtest
try:
    from backtest.vectorized_backtester import VectorizedBacktester
    check('backtest.vectorized_backtester', lambda: True)
except Exception as e:
    check('backtest.vectorized_backtester', lambda: (_ for _ in ()).throw(e))

# Notifications
try:
    from notifications.telegram import TelegramNotifier
    check('notifications.telegram', lambda: True)
except Exception as e:
    check('notifications.telegram', lambda: (_ for _ in ()).throw(e))

# Dashboard
try:
    from dashboard.app import app
    check('dashboard.app', lambda: True)
except Exception as e:
    check('dashboard.app', lambda: (_ for _ in ()).throw(e))

# File system checks
check('config.yaml exists', lambda: os.path.exists('config.yaml'))
check('.env exists', lambda: os.path.exists('.env'))
check('data/ directory', lambda: os.path.isdir('data'))
check('models/ directory', lambda: os.path.isdir('models'))

print()
for m in messages:
    print(m)
print()
ok_count = results['OK']
warn_count = results['WARN']
fail_count = results['FAIL']
print('Results: {} OK, {} WARN, {} FAIL'.format(ok_count, warn_count, fail_count))
print('=' * 60)

if fail_count > 0:
    print('\nBLOCKING ISSUES FOUND - Review FAIL items above')
    sys.exit(1)
elif warn_count > 0:
    print('\nReady with warnings - Review WARN items before live')
else:
    print('\nAll checks passed - System ready')
