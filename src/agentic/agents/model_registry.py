"""
Symbol-Specific Model Registry — tracks which trained models exist per symbol.

Currently only EURUSD models exist. Other symbols fall back to EURUSD.
When new models are trained for other symbols, they register here automatically.
"""

from __future__ import annotations
import os
import re
import glob
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class ModelEntry:
    symbol: str
    model_type: str  # lstm, classifier, ppo_trending, etc.
    file_path: str
    file_size_kb: float
    is_fallback: bool = False
    source_symbol: str = ""  # Which symbol this model was actually trained on


class SymbolModelRegistry:
    """
    Maps symbols to their best available trained models.

    Convention: files named like `lstm_cnn_SYMBOL.keras` or `classifier_SYMBOL.keras`
    are automatically discovered. Symbols without dedicated models use EURUSD as fallback.
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self._lstm_models: Dict[str, ModelEntry] = {}
        self._classifier_models: Dict[str, ModelEntry] = {}
        self._all_symbols: set = set()

    def discover(self):
        """Scan models directory for all per-symbol model files."""
        self._lstm_models.clear()
        self._classifier_models.clear()

        # Match per-symbol models like lstm_cnn_EURUSD.keras but NOT generic lstm_cnn_model.keras
        pattern = r"(lstm_cnn|classifier)_([A-Z]{2,6})\.keras$"
        for fpath in glob.glob(os.path.join(self.models_dir, "*.keras")):
            fname = os.path.basename(fpath)
            match = re.match(pattern, fname)
            if match:
                model_type = match.group(1)  # lstm_cnn or classifier
                symbol = match.group(2)  # EURUSD, GBPUSD, etc.
                size_kb = os.path.getsize(fpath) / 1024
                entry = ModelEntry(
                    symbol=symbol,
                    model_type=model_type,
                    file_path=fpath,
                    file_size_kb=size_kb,
                )
                if model_type == "lstm_cnn":
                    self._lstm_models[symbol] = entry
                elif model_type == "classifier":
                    self._classifier_models[symbol] = entry
                self._all_symbols.add(symbol)

        # Also register the generic lstm_cnn_model.keras if it exists
        generic_path = os.path.join(self.models_dir, "lstm_cnn_model.keras")
        if os.path.exists(generic_path) and "EURUSD" not in self._lstm_models:
            try:
                # Check what symbol it was last saved from
                import tensorflow as tf

                size_kb = os.path.getsize(generic_path) / 1024
                # We know this was EURUSD
                entry = ModelEntry(
                    symbol="EURUSD",
                    model_type="lstm_cnn",
                    file_path=generic_path,
                    file_size_kb=size_kb,
                )
                self._lstm_models["EURUSD"] = entry
                self._all_symbols.add("EURUSD")
            except Exception:
                pass

        logger.info(
            f"Model registry: {len(self._lstm_models)} LSTM, "
            f"{len(self._classifier_models)} classifier models found"
        )

    def get_lstm(self, symbol: str) -> ModelEntry:
        """Get the best LSTM model for a symbol. Falls back to EURUSD."""
        symbol = symbol.upper()
        if symbol in self._lstm_models:
            entry = self._lstm_models[symbol]
            entry.is_fallback = False
            return entry
        # Fallback to EURUSD
        if "EURUSD" in self._lstm_models:
            entry = self._lstm_models["EURUSD"]
            entry.is_fallback = True
            entry.source_symbol = "EURUSD"
            logger.debug(
                f"[model_registry] No LSTM model for {symbol}, falling back to EURUSD"
            )
            return entry
        return ModelEntry(
            symbol=symbol,
            model_type="lstm_cnn",
            file_path="",
            file_size_kb=0,
            is_fallback=True,
            source_symbol="none",
        )

    def get_classifier(self, symbol: str) -> ModelEntry:
        """Get the best classifier for a symbol. Falls back to EURUSD."""
        symbol = symbol.upper()
        if symbol in self._classifier_models:
            entry = self._classifier_models[symbol]
            entry.is_fallback = False
            return entry
        if "EURUSD" in self._classifier_models:
            entry = self._classifier_models["EURUSD"]
            entry.is_fallback = True
            entry.source_symbol = "EURUSD"
            return entry
        return ModelEntry(
            symbol=symbol,
            model_type="classifier",
            file_path="",
            file_size_kb=0,
            is_fallback=True,
            source_symbol="none",
        )

    def has_dedicated_lstm(self, symbol: str) -> bool:
        return symbol.upper() in self._lstm_models

    def has_dedicated_classifier(self, symbol: str) -> bool:
        return symbol.upper() in self._classifier_models

    def symbols_with_lstm(self) -> list:
        return sorted(self._lstm_models.keys())

    def symbols_with_classifier(self) -> list:
        return sorted(self._classifier_models.keys())

    def summary(self) -> str:
        lstm_syms = self.symbols_with_lstm()
        clf_syms = self.symbols_with_classifier()
        return (
            f"LSTM models: {len(lstm_syms)} symbols ({', '.join(lstm_syms[:5])}"
            f"{'...' if len(lstm_syms) > 5 else ''}) | "
            f"Classifiers: {len(clf_syms)} symbols ({', '.join(clf_syms[:5])}"
            f"{'...' if len(clf_syms) > 5 else ''})"
        )
