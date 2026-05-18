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
torch_available = importlib.util.find_spec("torch") is not None


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


@pytest.mark.skipif(not torch_available, reason="torch not installed")
def test_import_ai_maml():
    from ai.maml_agent import MAMLAgent, MAMLModel

    assert MAMLAgent is not None


@pytest.mark.skipif(not torch_available, reason="torch not installed")
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
