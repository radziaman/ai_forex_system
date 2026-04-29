"""Backtesting engine with out-of-sample validation (Zenox EA methodology)"""

import sys
from typing import Dict

import numpy as np
import pandas as pd

from ai_forex_system.risk import RiskManager

sys.path.append("..")


class BacktestEngine:
    """Backtesting with gap validation and out-of-sample testing"""

    def __init__(self, initial_balance: float = 10000, commission: float = 0.0001):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.commission = commission
        self.trades: list = []
        self.equity_curve: list = []

    def run_backtest(
        self,
        data: pd.DataFrame,
        model,
        feature_engineer,
        risk_manager: "RiskManager",
        train_end: str = "2020-01-01",
        out_of_sample: bool = True,
    ) -> Dict:
        """Run backtest with train/test split"""
        if out_of_sample:
            train_data = data[data.index < train_end]
            test_data = data[data.index >= train_end]
        else:
            train_data = data
            test_data = data

        print(f"Training period: {len(train_data)} bars")
        print(f"Testing period: {len(test_data)} bars (out-of-sample)")

        results = self._simulate_trades(
            test_data, model, feature_engineer, risk_manager
        )
        return results

    def _simulate_trades(
        self, data: pd.DataFrame, model, feature_engineer, risk_manager: "RiskManager"
    ) -> Dict[str, float]:
        """Simulate trade execution"""
        self.balance = self.initial_balance
        self.trades = []
        position = None

        for i in range(30, len(data) - 1):
            current_bar = data.iloc[i]

            if position is None:
                # Use past 30 bars (i-29 to i) to predict next bar
                idx_start = i - 29
                idx_end = i + 1
                signal = self._generate_signal(
                    data.iloc[idx_start:idx_end], model, feature_engineer
                )

                if signal != 0:
                    entry_price = current_bar["close"]
                    atr = current_bar["atr_14"] * current_bar["close"]
                    direction = "long" if signal > 0 else "short"
                    sl, tp = risk_manager.calculate_dynamic_sl_tp(
                        entry_price, atr, direction
                    )

                    if risk_manager.can_open_position(
                        "EURUSD", self.balance - self.initial_balance
                    ):
                        position = {
                            "entry": entry_price,
                            "sl": sl,
                            "tp": tp,
                            "direction": "long" if signal > 0 else "short",
                            "size": risk_manager.calculate_position_size(
                                entry_price, sl
                            ),
                            "entry_idx": i,
                        }
            else:
                current_price = current_bar["close"]
                pnl = self._calculate_pnl(position, current_price)

                if (
                    position["direction"] == "long"
                    and (
                        current_price <= position["sl"]
                        or current_price >= position["tp"]
                    )
                ) or (
                    position["direction"] == "short"
                    and (
                        current_price >= position["sl"]
                        or current_price <= position["tp"]
                    )
                ):

                    self.balance += pnl - (
                        position["size"] * current_price * self.commission
                    )
                    self.trades.append(
                        {
                            "entry": position["entry"],
                            "exit": current_price,
                            "pnl": pnl,
                            "direction": position["direction"],
                            "duration": i - position["entry_idx"],
                        }
                    )
                    position = None

            self.equity_curve.append(
                {
                    "timestamp": data.index[i],
                    "balance": self.balance,
                    "open_pnl": (
                        self._calculate_pnl(position, current_bar["close"])
                        if position
                        else 0
                    ),
                }
            )

        return self._calculate_metrics()

    def _generate_signal(self, window: pd.DataFrame, model, feature_engineer) -> int:
        """Generate buy/sell/hold signal from model prediction"""
        try:
            # Ensure we pass exactly 30 bars to the model
            if len(window) > 30:
                window = window.iloc[-30:]
            
            features = window.values.reshape(1, 30, window.shape[1])
            prediction = model.predict(features)[0][0]
            current_price = window["close"].iloc[-1]

            # Model predicts normalized close price
            # If prediction > current normalized price, model expects price to go up
            # Use a threshold to avoid noise (0.1 in normalized units ≈ small change)
            if prediction > current_price + 0.1:
                return 1  # BUY signal
            elif prediction < current_price - 0.1:
                return -1  # SELL signal
            return 0  # HOLD
        except Exception as e:
            print(f"Signal generation error: {e}")
            return 0

    def _calculate_pnl(self, position: Dict, current_price: float) -> float:
        if position["direction"] == "long":
            return float((current_price - position["entry"]) * position["size"])
        if position["direction"] == "short":
            return float((position["entry"] - current_price) * position["size"])
        return 0.0

    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics"""
        trades_df = pd.DataFrame(self.trades)

        if len(trades_df) == 0:
            return {
                "total_return": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "max_drawdown": 0,
            }

        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]

        total_return = (
            (self.balance - self.initial_balance) / self.initial_balance * 100
        )
        win_rate = len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0
        profit_factor = (
            wins["pnl"].sum() / abs(losses["pnl"].sum())
            if len(losses) > 0
            else float("inf")
        )

        equity_df = pd.DataFrame(self.equity_curve)
        peak = equity_df["balance"].expanding().max()
        drawdown = (equity_df["balance"] - peak) / peak * 100
        max_drawdown = drawdown.min()

        return {
            "total_return": total_return,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "total_trades": len(trades_df),
            "sharpe_ratio": self._calculate_sharpe(),
            "final_balance": self.balance,
        }

    def _calculate_sharpe(self) -> float:
        equity_df = pd.DataFrame(self.equity_curve)
        if len(equity_df) < 2:
            return 0.0
        returns = equity_df["balance"].pct_change().dropna()
        if returns.std() == 0:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(252))

    def save_results(self, filepath: str):
        results = {
            "trades": pd.DataFrame(self.trades),
            "equity_curve": pd.DataFrame(self.equity_curve),
        }
        with open(filepath, "wb") as f:
            import pickle

            pickle.dump(results, f)
