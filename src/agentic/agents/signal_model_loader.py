"""
Signal Model Loader — loads PPO, LSTM, TFT, and classifier models per symbol.

Encapsulates all model loading logic for the signal agent.
"""

from __future__ import annotations
import os
from typing import Dict, Any, Optional, Set
from loguru import logger


class ModelLoader:
    """Loads and manages all AI models for signal generation.

    Responsibilities:
    - Load PPO regime agents (shared across symbols)
    - Load per-symbol LSTM models with fallback
    - Load per-symbol TFT models with fallback
    - Load per-symbol classifiers
    - Track which models are loaded and report health
    """

    def __init__(self, active_symbols: list, config: Optional[Any] = None):
        self.active_symbols = active_symbols
        self.config = config
        self.regime_manager: Optional[Any] = None
        self.lstm_models: Dict[str, Any] = {}
        self.tft_models: Dict[str, Any] = {}
        self.classifiers: Dict[str, Any] = {}
        self.ppo_state_dim: int = 49
        self.models_loaded: bool = False
        self.fallback_warnings: Set[str] = set()
        self.model_registry: Optional[Any] = None
        self._log_fn = logger.info

    def set_log_fn(self, fn):
        """Override the log function (e.g. for agent.log_state)."""
        self._log_fn = fn

    def load_all(self) -> bool:
        """Load all models. Returns True if at least some models loaded."""
        self._load_ppo_regime_agents()
        self._load_model_registry()
        self._load_lstm_models()
        self._load_tft_models()
        self._load_classifiers()
        self._log_summary()
        self.models_loaded = self.regime_manager is not None and (
            len(self.lstm_models) > 0 or len(self.tft_models) > 0
        )
        return self.models_loaded

    def _load_ppo_regime_agents(self):
        """Load PPO regime agents (shared across all symbols)."""
        try:
            from ai.regime_agents import RegimeSpecialistSystem
            from rts_ai_fx.features_unified import (
                FeaturePipeline,
                EXPECTED_FEATURE_DIM,
            )

            # Compute PPO state_dim dynamically: 1 (price) + n_features from pipeline
            _fp = FeaturePipeline(lookback=30, timeframes=["1h"])
            _fp.load_normalization()  # loads _feature_cols
            _n_features = (
                len(_fp._feature_cols) if _fp._feature_cols else EXPECTED_FEATURE_DIM
            )
            _ppo_state_dim = 1 + _n_features  # price + features
            self.ppo_state_dim = _ppo_state_dim

            self.regime_manager = RegimeSpecialistSystem(
                state_dim=_ppo_state_dim, n_actions=5
            )
            n_agents = len([a for a in self.regime_manager.agents.values() if a])

            has_real_weights = any(
                any(p.norm().item() > 1.0 for p in agent.actor.parameters())
                for agent in self.regime_manager.agents.values()
                if agent
            )
            if not has_real_weights:
                self._log_fn(
                    f"Loaded {n_agents} PPO regime agents "
                    "(untrained — random weights)"
                )
            else:
                self._log_fn(
                    f"Loaded {n_agents} PPO regime agents " "(trained weights loaded)"
                )
        except Exception as e:
            self._log_fn(f"PPO agents not loaded: {e}")

    def _load_model_registry(self):
        """Load per-symbol model registry."""
        try:
            from agentic.agents.model_registry import SymbolModelRegistry

            self.model_registry = SymbolModelRegistry()
            self.model_registry.discover()
            self._log_fn(f"Model registry: {self.model_registry.summary()}")
        except Exception as e:
            self._log_fn(f"Model registry failed: {e}")

    def _load_lstm_models(self):
        """Load per-symbol LSTM models with fallback to EURUSD."""
        from rts_ai_fx.model import LSTMCNNHybrid

        for sym in self.active_symbols:
            entry = self.model_registry.get_lstm(sym) if self.model_registry else None
            if entry and entry.file_path and os.path.exists(entry.file_path):
                try:
                    loaded = LSTMCNNHybrid.load(entry.file_path)
                    if loaded and loaded.model is not None:
                        self.lstm_models[sym] = loaded
                except Exception:
                    pass
            if sym not in self.lstm_models:
                # Fallback: try the generic model
                for p in [
                    "models/lstm_cnn_model_reloaded.keras",
                    "models/lstm_cnn_EURUSD.keras",
                    "models/lstm_cnn_model.keras",
                ]:
                    if os.path.exists(p):
                        try:
                            loaded = LSTMCNNHybrid.load(p)
                            if loaded and loaded.model is not None:
                                self.lstm_models[sym] = loaded
                                self.fallback_warnings.add(sym)
                                break
                        except Exception:
                            pass

    def _load_tft_models(self):
        """Load per-symbol TFT models with fallback."""
        try:
            from ai.tft_model import TemporalFusionTransformer

            for sym in self.active_symbols:
                tft_path = f"models/tft_{sym}.pt"
                if os.path.exists(tft_path):
                    try:
                        loaded = TemporalFusionTransformer.load(tft_path)
                        self.tft_models[sym] = loaded
                    except Exception:
                        pass
                if sym not in self.tft_models:
                    # Fallback: generic model
                    generic = "models/tft_primary.pt"
                    if os.path.exists(generic):
                        try:
                            loaded = TemporalFusionTransformer.load(generic)
                            self.tft_models[sym] = loaded
                            self.fallback_warnings.add(sym)
                        except Exception:
                            pass
        except Exception as e:
            self._log_fn(f"TFT models not loaded: {e}")

    def _load_classifiers(self):
        """Load classifier models per active symbol."""
        from rts_ai_fx.model import ProfitabilityClassifier

        for sym in self.active_symbols:
            entry = (
                self.model_registry.get_classifier(sym) if self.model_registry else None
            )
            if entry and entry.file_path and os.path.exists(entry.file_path):
                try:
                    self.classifiers[sym] = ProfitabilityClassifier.load(
                        entry.file_path
                    )
                except Exception:
                    pass
            if sym not in self.classifiers:
                try:
                    clf = ProfitabilityClassifier(lookback=30, n_features=49)
                    self.classifiers[sym] = clf
                except Exception:
                    pass

    def _log_summary(self):
        """Log model loading summary."""
        lstm_count = len(self.lstm_models)
        tft_count = len(self.tft_models)
        clf_count = len(self.classifiers)
        lstm_unique = len(set(id(m) for m in self.lstm_models.values()))
        if self.fallback_warnings:
            fallback_list = sorted(self.fallback_warnings)[:5]
            self._log_fn(
                f"{lstm_count} symbols have LSTM models "
                f"({lstm_unique} unique instances) "
                f"— {len(self.fallback_warnings)} symbols fall back to EURUSD "
                f"({', '.join(fallback_list)}"
                f"{'...' if len(self.fallback_warnings) > 5 else ''})"
            )
        else:
            self._log_fn(
                f"{lstm_count} symbols have LSTM models "
                f"({lstm_unique} unique instances)"
            )
        self._log_fn(f"{tft_count} symbols have TFT models")
        self._log_fn(f"{clf_count} symbols have classifiers")
