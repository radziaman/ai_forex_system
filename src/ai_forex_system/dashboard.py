"""Real-time dashboard for monitoring AI forex trading system"""

import numpy as np
from typing import Dict, List
import json
from datetime import datetime

from ai_forex_system.trader import AITrader


class TradingDashboard:
    """Dashboard for monitoring trades, positions, and performance"""

    def __init__(self, trader: "AITrader"):
        self.trader = trader
        self.logs: List[Dict[str, float]] = []

    def display_status(self):
        """Display current trading status (console-based)"""
        stats = self.trader.get_portfolio_stats()

        print("\n" + "=" * 60)
        print(
            f"AI FOREX TRADING SYSTEM - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print("=" * 60)
        print(f"Balance: ${stats['balance']:,.2f}")
        print(f"Total Return: {stats['total_return_pct']:.2f}%")
        print(f"Open Positions: {stats['open_positions']}")
        print(f"Closed Positions: {stats['closed_positions']}")
        print(f"Win Rate: {stats['win_rate']:.1f}%")
        print(f"Total PnL: ${stats['total_pnl']:,.2f}")
        print("-" * 60)

        if self.trader.positions:
            print("OPEN POSITIONS:")
            for pos in self.trader.positions:
                print(
                    f"  {pos['pair']} | {pos['direction']}"
                    f" | Entry: {pos['entry']:.4f}"
                    f" | SL: {pos['sl']:.4f}"
                    f" | TP: {pos['tp']:.4f}"
                )
        print("=" * 60 + "\n")

    def log_trade(self, trade: Dict[str, float]):
        """Log trade information"""
        log_entry: Dict[str, float] = {"timestamp": datetime.now().isoformat(), "trade": trade}
        self.logs.append(log_entry)

    def export_logs(self, filepath: str):
        """Export trade logs to JSON"""
        with open(filepath, "w") as f:
            json.dump(self.logs, f, indent=2)

    def generate_performance_report(self) -> Dict[str, float]:
        """Generate comprehensive performance report"""
        stats: Dict[str, float] = self.trader.get_portfolio_stats()
        closed = self.trader.closed_positions

        if not closed:
            return stats

        pnls = [float(p["pnl"]) for p in closed]

        avg_win = (
            float(np.mean([p for p in pnls if p > 0]))
            if any(p > 0 for p in pnls)
            else 0.0
        )
        avg_loss = (
            float(np.mean([p for p in pnls if p < 0]))
            if any(p < 0 for p in pnls)
            else 0.0
        )

        return {
            **stats,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "largest_win": max(pnls),
            "largest_loss": min(pnls),
            "profit_factor": (
                abs(sum(p for p in pnls if p > 0) / sum(p for p in pnls if p < 0))
                if any(p < 0 for p in pnls)
                else float("inf")
            ),
        }


class AlertSystem:
    """Alert system for important trading events"""

    def __init__(self, max_drawdown_pct: float = 10.0):
        self.max_drawdown_pct = max_drawdown_pct
        self.peak_balance = 0

    def check_alerts(self, balance: float, positions: List) -> List[str]:
        """Check for alert conditions"""
        alerts: List[str] = []

        if balance > self.peak_balance:
            self.peak_balance = int(balance)

        drawdown = (self.peak_balance - balance) / self.peak_balance * 100
        if drawdown > self.max_drawdown_pct:
            alerts.append(
                f"WARNING: Drawdown {drawdown:.1f}% exceeds {self.max_drawdown_pct}%!"
            )

        if len(positions) >= 5:
            alerts.append(f"WARNING: {len(positions)} open positions (max recommended)")

        return alerts
