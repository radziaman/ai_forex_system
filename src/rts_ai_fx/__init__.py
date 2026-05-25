"""
RTS: Agentic FX System Elite — LSTM-CNN Hybrid with Reinforcement Learning.

Environment variables for TensorFlow/Keras compatibility are set here,
at the package root, to ensure they take effect before any module
imports tensorflow.
"""

import os

# TF 2.16+ uses Keras 3 by default. Our models were saved with Keras 2.
# These env vars ensure backward compatibility for model loading.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_USE_LEGACY_ADAM", "1")

# Suppress TF informational logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

__version__ = "3.0.0"
