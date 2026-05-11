"""
Position Persistence Service — saves/loads open positions and trade history to/from disk.
Prevents state loss on bot restart.
"""
import json
import os
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
from loguru import logger


class PositionPersistence:
    """Saves and loads trading state to/from disk for crash recovery."""

    def __init__(self, base_path: str = "data/trades"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._positions_path = self.base_path / "open_positions.json"
        self._history_path = self.base_path / "trade_history.json"

    def save_positions(self, positions: List[Dict]) -> bool:
        """Save open positions to disk."""
        try:
            serializable = []
            for p in positions:
                entry = {}
                for k, v in p.items():
                    if isinstance(v, (int, float, str, bool)):
                        entry[k] = v
                    elif v is None:
                        entry[k] = None
                    else:
                        entry[k] = str(v)
                serializable.append(entry)
            with open(self._positions_path, "w") as f:
                json.dump(serializable, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save positions: {e}")
            return False

    def load_positions(self) -> List[Dict]:
        """Load open positions from disk."""
        try:
            if not self._positions_path.exists():
                return []
            with open(self._positions_path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load positions: {e}")
            return []

    def save_trade_history(self, trades: List[Dict]) -> bool:
        """Save trade history to disk."""
        try:
            with open(self._history_path, "w") as f:
                json.dump(trades[-500:], f, indent=2)  # Keep last 500
            return True
        except Exception as e:
            logger.error(f"Failed to save trade history: {e}")
            return False

    def load_trade_history(self) -> List[Dict]:
        """Load trade history from disk."""
        try:
            if not self._history_path.exists():
                return []
            with open(self._history_path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load trade history: {e}")
            return []

    def save_risk_state(self, risk_manager) -> bool:
        """Save risk manager state (daily stats, trade counts)."""
        try:
            state = {
                "daily_pnl": risk_manager.daily_pnl,
                "daily_trades": risk_manager.daily_trades,
                "consecutive_losses": risk_manager.consecutive_losses,
                "total_trades": risk_manager.total_trades,
                "wins": risk_manager.wins,
                "losses": risk_manager.losses,
                "peak_balance": risk_manager.peak_balance,
                "kill_switch_triggered": risk_manager.kill_switch_triggered,
                "timestamp": time.time(),
            }
            with open(self.base_path / "risk_state.json", "w") as f:
                json.dump(state, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save risk state: {e}")
            return False

    def load_risk_state(self, risk_manager) -> bool:
        """Load risk manager state from disk."""
        try:
            path = self.base_path / "risk_state.json"
            if not path.exists():
                return False
            with open(path) as f:
                state = json.load(f)
            risk_manager.daily_pnl = state.get("daily_pnl", 0.0)
            risk_manager.daily_trades = state.get("daily_trades", 0)
            risk_manager.consecutive_losses = state.get("consecutive_losses", 0)
            risk_manager.total_trades = state.get("total_trades", 0)
            risk_manager.wins = state.get("wins", 0)
            risk_manager.losses = state.get("losses", 0)
            risk_manager.peak_balance = state.get("peak_balance", risk_manager.initial_balance)
            risk_manager.kill_switch_triggered = state.get("kill_switch_triggered", False)
            return True
        except Exception as e:
            logger.error(f"Failed to load risk state: {e}")
            return False

    def save_all(self, execution_engine, risk_manager) -> bool:
        """Save all trading state."""
        try:
            positions = execution_engine.get_open_positions() if hasattr(execution_engine, 'get_open_positions') else []
            history = execution_engine.get_trade_history(500) if hasattr(execution_engine, 'get_trade_history') else []
            self.save_positions(positions)
            self.save_trade_history(history)
            self.save_risk_state(risk_manager)
            logger.debug(f"State saved: {len(positions)} positions, {len(history)} trades")
            return True
        except Exception as e:
            logger.warning(f"State save failed: {e}")
            return False
