"""Tests for data fetching and preprocessing"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rts_ai_fx.data import DataFetcher, DataPreprocessor

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from rts_ai_fx.data import DataFetcher, DataPreprocessor


class TestDataFetcher:
    """Test data fetching functionality"""

    def test_init(self):
        """Test DataFetcher initialization"""
        fetcher = DataFetcher(source="yfinance")
        assert fetcher.source == "yfinance"

    @patch("yfinance.Ticker")
    def test_fetch_yfinance(self, mock_ticker):
        """Test yfinance data fetching"""
        mock_instance = MagicMock()
        mock_instance.history.return_value = pd.DataFrame(
            {
                "Open": [1.1, 1.2, 1.3],
                "High": [1.15, 1.25, 1.35],
                "Low": [1.05, 1.15, 1.25],
                "Close": [1.12, 1.22, 1.32],
                "Volume": [1000, 2000, 3000],
            }
        )
        mock_ticker.return_value = mock_instance

        fetcher = DataFetcher(source="yfinance")
        result = fetcher.fetch_ohlcv("EURUSD=X", "1h", "2025-01-01")

        assert not result.empty
        assert "close" in result.columns

    def test_preprocessor_sequences(self):
        """Test sequence creation for LSTM"""
        preprocessor = DataPreprocessor(lookback=3)

        data = np.random.randn(10, 5)
        X, y = preprocessor.create_sequences(data)

        assert X.shape == (7, 3, 5)
        assert y.shape == (7,)


class TestDataPreprocessor:
    """Test data preprocessing"""

    def test_normalize_features(self):
        """Test feature normalization"""
        preprocessor = DataPreprocessor()

        df = pd.DataFrame(
            {
                "open": [1.1, 1.2, 1.3],
                "high": [1.15, 1.25, 1.35],
                "close": [1.12, 1.22, 1.32],
                "rsi_14": [30, 50, 70],
                "atr_14": [0.001, 0.002, 0.0015],
            }
        )

        result = preprocessor.normalize_features(df)

        assert not result.empty
        assert "rsi_14" in result.columns
