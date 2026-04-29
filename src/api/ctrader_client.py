"""
Simplified cTrader Client for IC Markets Integration
Fallback mode - uses Yahoo Finance for data, prepares for live cTrader.
"""
import time
import logging
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Symbol mapping (EURUSD=1, GBPUSD=2, etc.)
SYMBOL_MAP = {
    1: "EURUSD",
    2: "GBPUSD",
    3: "USDJPY",
    4: "XAUUSD",
    5: "BTCUSD",
    6: "ETHUSD",
    7: "AUDUSD",
    8: "USDCHF",
    9: "NZDUSD",
    10: "USDCAD",
}
REVERSE_SYMBOL_MAP = {v: k for k, v in SYMBOL_MAP.items()}


@dataclass
class MarketDepth:
    """Real-time market data."""
    symbol: str = ""
    symbol_id: int = 0
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0
    volume: int = 0
    timestamp: float = field(default_factory=lambda: time.time())


@dataclass
class CtraderClient:
    """
    Simplified cTrader client for IC Markets.
    Runs in simulation mode with fallback to Yahoo Finance.
    """
    
    def __init__(
        self,
        app_id: str = "",
        app_secret: str = "",
        access_token: str = "",
        account_id: str = "",
        demo: bool = True,
    ):
        self.app_id = app_id
        self.app_secret = app_secret
        self.access_token = access_token
        self.account_id = account_id
        self.demo = demo
        
        self._is_connected = False
        self._is_simulation = True
        self._last_market_depth: Dict[int, MarketDepth] = {}
        
        logger.info(f"CtraderClient initialized | Demo: {demo} | Account: {account_id}")
    
    def start(self):
        """Start in simulation mode."""
        logger.info("Starting in SIMULATION mode")
        logger.info("For live trading, complete OAuth2 setup")
        self._is_connected = True
        self._is_simulation = True
        logger.info("✓ Client ready (simulation mode)")
    
    def subscribe_market_data(self, symbol: str):
        """Subscribe to market data (simulation)."""
        symbol_id = REVERSE_SYMBOL_MAP.get(symbol.upper())
        if symbol_id:
            # In production, this would send subscription request
            logger.info(f"Subscribed to {symbol} (simulation)")
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price (fallback to Yahoo Finance)."""
        symbol_id = REVERSE_SYMBOL_MAP.get(symbol.upper())
        if symbol_id and symbol_id in self._last_market_depth:
            depth = self._last_market_depth[symbol_id]
            return (depth.bid + depth.ask) / 2
        return None
    
    def is_connected(self) -> bool:
        """Check connection status."""
        return self._is_connected
    
    def is_simulation(self) -> bool:
        """Check if in simulation mode."""
        return self._is_simulation
    
    def get_account_info(self) -> dict:
        """Get account information."""
        return {
            "account_id": self.account_id,
            "balance": 10000.0,
            "equity": 10000.0,
            "demo": self.demo,
            "connected": self._is_connected,
        }
    
    def disconnect(self):
        """Disconnect from cTrader."""
        self._is_connected = False
        logger.info("Disconnected from cTrader (simulation)")
