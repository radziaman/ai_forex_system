"""
Transaction cost model — spread, commission, slippage.
Applies realistic costs to every simulated/executed trade.
"""
from dataclasses import dataclass
from typing import Dict


@dataclass
class CostResult:
    spread_cost: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    total: float = 0.0


# Typical spreads in pips for major forex pairs (variable by liquidity)
SPREADS: Dict[str, float] = {
    "EURUSD": 0.5, "GBPUSD": 0.8, "USDJPY": 0.6, "AUDUSD": 0.7,
    "USDCAD": 0.7, "USDCHF": 0.8, "NZDUSD": 0.9, "XAUUSD": 2.0,
    "BTCUSD": 5.0, "EURJPY": 0.9, "GBPJPY": 1.5, "EURGBP": 0.7,
}

DEFAULT_SPREAD = 1.0  # pips


class CostModel:
    """
    Realistic transaction cost model.
    
    - Spread: variable per pair, plus widening during high volatility
    - Commission: $7 per lot (standard for IC Markets/Raw Spread)
    - Slippage: function of trade size relative to ATR
    """

    def __init__(self, commission_per_lot: float = 7.0, default_spread: float = 1.0):
        self.commission_per_lot = commission_per_lot
        self.default_spread = default_spread

    def calculate(
        self,
        symbol: str,
        direction: str,
        volume: float,  # in units
        price: float,
        atr: float = 0.0,
        volatility_mult: float = 1.0,
    ) -> CostResult:
        spread_pips = SPREADS.get(symbol.upper(), self.default_spread)
        # Widen spread during high volatility
        spread_pips *= volatility_mult

        # Convert pips to price units (pip = 0.0001 for most pairs)
        pip_size = 0.0001 if "JPY" not in symbol.upper() else 0.01
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
        )

    def apply_to_pnl(self, pnl: float, cost: CostResult) -> float:
        return pnl - cost.total

    @staticmethod
    def pip_to_price(symbol: str) -> float:
        return 0.0001 if "JPY" not in symbol.upper() else 0.01
