"""
Centralized symbol ID mapping — 11 actively traded symbols.
"""

SYMBOL_MAP: dict = {
    "EURUSD": 1,
    "GBPUSD": 2,
    "USDJPY": 4,
    "AUDUSD": 5,
    "USDCHF": 6,
    "USDCAD": 8,
    "NZDUSD": 12,
    "XAUUSD": 41,
    "XTIUSD": 99,
    "BTCUSD": 114,
    "US500": 115,
}

SYMBOLS_BY_ID: dict = {v: k for k, v in SYMBOL_MAP.items()}


def get_symbol_id(symbol: str) -> int:
    return SYMBOL_MAP.get(symbol.upper(), 1)


def get_symbol_name(symbol_id: int) -> str:
    return SYMBOLS_BY_ID.get(symbol_id, "EURUSD")
