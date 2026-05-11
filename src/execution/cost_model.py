"""
Transaction cost model — spread, commission, slippage.
Applies realistic costs to every simulated/executed trade.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional
import time


@dataclass
class CostResult:
    spread_cost: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    total: float = 0.0
    actual_spread_pips: float = 0.0
    is_acceptable: bool = True
    rejection_reason: str = ""


# Typical spreads in pips for major forex pairs (variable by liquidity)
SPREADS: Dict[str, float] = {
    "EURUSD": 0.5, "GBPUSD": 0.8, "USDJPY": 0.6, "AUDUSD": 0.7,
    "USDCAD": 0.7, "USDCHF": 0.8, "NZDUSD": 0.9, "XAUUSD": 2.0,
    "BTCUSD": 5.0, "EURJPY": 0.9, "GBPJPY": 1.5, "EURGBP": 0.7,
}
DEFAULT_SPREAD = 1.0  # pips
MAX_SPREAD_MULT = 3.0  # Max spread = 3x normal


class CostModel:
    """
    Realistic transaction cost model with real-time spread verification.
    
    - Spread: variable per pair, plus widening during high volatility
    - Commission: $7 per lot (standard for IC Markets/Raw Spread)
    - Slippage: function of trade size relative to ATR
    - Real-time spread check: rejects trades if spread too wide
    """

    def __init__(self, commission_per_lot: float = 7.0, default_spread: float = 1.0):
        self.commission_per_lot = commission_per_lot
        self.default_spread = default_spread
        self._spread_history: Dict[str, list] = {}
        self._max_spread_multiplier = MAX_SPREAD_MULT

    def calculate(
        self,
        symbol: str,
        direction: str,
        volume: float,  # in units
        price: float,
        atr: float = 0.0,
        volatility_mult: float = 1.0,
        actual_spread_pips: Optional[float] = None,
    ) -> CostResult:
        normal_spread = SPREADS.get(symbol.upper(), self.default_spread)
        
        # Use actual spread if provided, otherwise estimate
        if actual_spread_pips is not None:
            spread_pips = actual_spread_pips
            # Track spread history
            if symbol not in self._spread_history:
                self._spread_history[symbol] = []
            self._spread_history[symbol].append(spread_pips)
            if len(self._spread_history[symbol]) > 100:
                self._spread_history[symbol] = self._spread_history[symbol][-100:]
            
            # Check if spread is acceptable
            avg_spread = self._get_average_spread(symbol)
            is_acceptable = spread_pips <= avg_spread * self._max_spread_multiplier
            rejection_reason = "" if is_acceptable else f"Spread {spread_pips:.1f} > {avg_spread * self._max_spread_multiplier:.1f} (3x avg)"
        else:
            spread_pips = normal_spread * volatility_mult
            is_acceptable = True
            rejection_reason = ""

        # Convert pips to price units (pip size varies by symbol)
        pip_size = self._pip_size(symbol)
        spread_cost = spread_pips * pip_size * volume

        # Commission (1 lot = 100,000 units)
        lots = volume / 100_000
        commission = lots * self.commission_per_lot

        # Slippage: fraction of spread, increases with trade size
        size_factor = min(volume / 100_000 / 10, 1.0)  # 0-1 for 0-10 lots
        slippage_pips = spread_pips * 0.5 * size_factor
        slippage = slippage_pips * pip_size * volume

        return CostResult(
            spread_cost=spread_cost,
            commission=commission,
            slippage=slippage,
            total=spread_cost + commission + slippage,
            actual_spread_pips=spread_pips,
            is_acceptable=is_acceptable,
            rejection_reason=rejection_reason,
        )

    def _get_average_spread(self, symbol: str) -> float:
        """Get average spread from history, fallback to normal spread."""
        if symbol in self._spread_history and self._spread_history[symbol]:
            return sum(self._spread_history[symbol]) / len(self._spread_history[symbol])
        return SPREADS.get(symbol.upper(), self.default_spread)

    def get_spread_warning_level(self, symbol: str, current_spread: float) -> str:
        """Returns 'normal', 'wide', or 'extreme'."""
        avg = self._get_average_spread(symbol)
        if current_spread > avg * 3:
            return "extreme"
        elif current_spread > avg * 2:
            return "wide"
        return "normal"

    def apply_to_pnl(self, pnl: float, cost: CostResult) -> float:
        return pnl - cost.total

    @staticmethod
    def _pip_size(symbol: str) -> float:
        sym = symbol.upper()
        if sym in ("XAUUSD", "XAGUSD", "XTIUSD", "XBRUSD"):
            return 0.01
        if sym in ("XNGUSD",):
            return 0.001
        if sym in ("BTCUSD", "ETHUSD", "LTCUSD"):
            return 1.0
        if sym in ("XRPUSD",):
            return 0.0001
        if sym in ("US500", "US30", "USTEC", "UK100", "DE40"):
            return 1.0
        if "JPY" in sym:
            return 0.01
        return 0.0001

    @staticmethod
    def pip_to_price(symbol: str) -> float:
        return CostModel._pip_size(symbol)

    @staticmethod
    def calculate_pips(symbol: str, price_diff: float) -> float:
        pip_size = CostModel._pip_size(symbol)
        return price_diff / pip_size
