"""Main trading bot that integrates all components"""

import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
import sys

from ai_forex_system.data import DataFetcher, DataPreprocessor
from ai_forex_system.features import FeatureEngineer
from ai_forex_system.model import LSTMCNNHybrid, ProfitabilityClassifier
from ai_forex_system.risk import RiskManager, TrailingStopManager

sys.path.append("..")


class AITrader:
    """AI-powered forex trading bot (Sentinel AI + Zenox + Aurum hybrid)"""

    def __init__(
        self,
        initial_balance: float = 10000,
        pairs: Optional[List[str]] = None,
        timeframe: str = "1h",
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.pairs: List[str] = (
            pairs if pairs is not None else ["EURUSD=X", "GBPUSD=X", "XAUUSD=X"]
        )
        self.timeframe = timeframe

        self.feature_engineer = FeatureEngineer(lookback=30)
        self.model = LSTMCNNHybrid(lookback=30, n_features=51)
        self.classifier = ProfitabilityClassifier(lookback=30, n_features=51)
        self.risk_manager = RiskManager(account_balance=initial_balance)
        self.trailing_manager = TrailingStopManager()

        self.data_fetcher = DataFetcher(source="yfinance")
        self.preprocessor = DataPreprocessor(lookback=30)

        self.positions: List[Dict] = []
        self.closed_positions: List[Dict] = []
        self.model_trained = False

    def train_models(self, symbol: str = "EURUSD=X", start: str = "2015-01-01"):
        """Train LSTM-CNN hybrid model (25-year training like Zenox)"""
        print(f"Fetching data for {symbol}...")
        df = self.data_fetcher.fetch_ohlcv(symbol, self.timeframe, start)

        print("Engineering features (51+ features)...")
        df = self.feature_engineer.generate_all_features(df)

        print("Normalizing and creating sequences...")
        df_normalized = self.preprocessor.normalize_features(df)

        (X_train, X_test), (y_train, y_test), (train_df, test_df) = (
            self.preprocessor.train_test_split(df_normalized, train_end="2020-01-01")
        )

        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")

        print("Building LSTM-CNN hybrid model...")
        self.model.build()

        print("Training model...")
        history = self.model.train(
            X_train, y_train, X_test, y_test, epochs=50, batch_size=32
        )

        print("Training profitability classifier...")
        self.classifier.build()
        lookback = self.preprocessor.lookback
        y_train_binary = (y_train > train_df["close"].values[lookback:]).astype(int)
        self.classifier.model.fit(
            X_train, y_train_binary, epochs=30, batch_size=32, verbose=0
        )

        self.model_trained = True
        print("Training complete!")

        return history

    def generate_signal(self, features_window: pd.DataFrame) -> Dict[str, float]:
        """Generate trading signal (buy/sell/hold) with confidence"""
        if not self.model_trained:
            raise ValueError("Model not trained. Call train_models() first.")

        prediction = self.model.predict(features_window.values.reshape(1, 30, 51))[0][0]
        confidence = self.classifier.predict_proba(
            features_window.values.reshape(1, 30, 51)
        )[0][0]

        current_price = features_window.iloc[-1]["close"]

        signal = "HOLD"
        if prediction > current_price * 1.0005 and confidence > 0.6:
            signal = "BUY"
        elif prediction < current_price * 0.9995 and confidence > 0.6:
            signal = "SELL"

        return {
            "signal": signal,
            "predicted_price": prediction,
            "confidence": confidence,
            "current_price": current_price,
        }

    def open_position(
        self,
        pair: str,
        direction: str,
        entry_price: float,
        atr: float,
        size: Optional[float] = None,
    ):
        """Open a new position with risk management"""
        if not self.risk_manager.can_open_position(
            pair, self.balance - self.initial_balance
        ):
            return None

        sl, tp = self.risk_manager.calculate_dynamic_sl_tp(entry_price, atr, direction)

        if size is None:
            size = self.risk_manager.calculate_position_size(entry_price, sl)

        position = {
            "pair": pair,
            "direction": direction,
            "entry": entry_price,
            "sl": sl,
            "tp": tp,
            "size": size,
            "entry_time": datetime.now(),
            "trailing_activated": False,
        }

        self.positions.append(position)
        print(f"Opened {direction} position on {pair} at {entry_price:.4f}")
        return position

    def manage_positions(
        self, current_prices: Dict[str, float], atr_values: Dict[str, float]
    ):
        """Manage open positions (trailing stops, partial closes)"""
        for pos in self.positions[:]:
            pair = pos["pair"]
            current_price = current_prices.get(pair, pos["entry"])
            atr = atr_values.get(pair, 0.001)

            if pos["direction"] == "BUY":
                if current_price <= pos["sl"]:
                    self._close_position(pos, current_price, "Stop Loss")
                elif current_price >= pos["tp"]:
                    self._close_position(pos, current_price, "Take Profit")
                else:
                    new_sl = self.trailing_manager.update_trailing_stop(
                        pos["entry"], current_price, atr, "long"
                    )
                    if new_sl:
                        pos["sl"] = max(pos["sl"], new_sl)
            else:
                if current_price >= pos["sl"]:
                    self._close_position(pos, current_price, "Stop Loss")
                elif current_price <= pos["tp"]:
                    self._close_position(pos, current_price, "Take Profit")
                else:
                    new_sl = self.trailing_manager.update_trailing_stop(
                        pos["entry"], current_price, atr, "short"
                    )
                    if new_sl:
                        pos["sl"] = min(pos["sl"], new_sl)

    def _close_position(self, position: Dict, exit_price: float, reason: str):
        """Close a position and calculate PnL"""
        if position["direction"] == "BUY":
            pnl = (exit_price - position["entry"]) * position["size"]
        else:
            pnl = (position["entry"] - exit_price) * position["size"]

        self.balance += pnl
        position["exit"] = exit_price
        position["pnl"] = pnl
        position["reason"] = reason
        position["exit_time"] = datetime.now()

        self.closed_positions.append(position)
        self.positions.remove(position)

        print(
            f"Closed {position['direction']} on {position['pair']}"
            f" at {exit_price:.4f} | PnL: ${pnl:.2f}"
        )

    def get_portfolio_stats(self) -> Dict[str, float]:
        """Get current portfolio statistics"""
        total_pnl = sum(p["pnl"] for p in self.closed_positions)
        winning_trades = [p for p in self.closed_positions if p["pnl"] > 0]

        return {
            "balance": float(self.balance),
            "total_pnl": float(total_pnl),
            "total_return_pct": (self.balance - self.initial_balance)
            / self.initial_balance
            * 100,
            "open_positions": len(self.positions),
            "closed_positions": len(self.closed_positions),
            "win_rate": (
                len(winning_trades) / len(self.closed_positions) * 100
                if self.closed_positions
                else 0.0
            ),
            "winning_trades": len(winning_trades),
        }
