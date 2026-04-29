"""
REAL cTrader Client - Pulls ACTUAL Data from IC Markets
Uses ctrader_open_api with Twisted for real TCP connection
"""
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

# IC Markets cTrader Endpoints
ICMARKETS_DEMO_HOST = "demo.ctrader.com"
ICMARKETS_LIVE_HOST = "live.ctrader.com"
CTADER_PORT = 5035

# Your IC Markets Credentials
APP_ID = "15217_h8WxunXX70m6O6qsnIx9ZO3GZraTdO0wnLjL3dTKyYG6fkbUca"
APP_SECRET = "Zb8tEW4Axzq0AJqCNS8ubniYzsgp2kxuRkYBRD9XXOcLAj5aOT"
ACCESS_TOKEN = "7d574e0bbcf30419cc752a4ea17bd627f8b46bdf79b5a3f34cf38db8285ce17f43592d3cb5bbd067ca3d2d"
ACCOUNT_ID = "6100830"


@dataclass
class RealAccountInfo:
    """REAL account information from IC Markets cTrader."""
    account_id: str = ""
    ctid_trader_account_id: int = 0
    balance: float = 0.0
    equity: float = 0.0
    margin: float = 0.0
    free_margin: float = 0.0
    margin_level: float = 0.0
    currency: str = "USD"
    leverage: str = "1:30"
    is_live: bool = False
    broker_name: str = "IC Markets"
    account_type: str = "Demo"
    deposit_asset_id: int = 1
    # Real data fields
    unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    swap: float = 0.0
    commission: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class RealSymbolInfo:
    """REAL symbol information from cTrader."""
    symbol_id: int = 0
    symbol_name: str = ""
    description: str = ""
    digits: int = 5
    pip_value: float = 0.0001
    min_volume: int = 1000
    max_volume: int = 10000000
    step_volume: int = 1000
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0
    volume: int = 0


class RealCtraderClient:
    """
    REAL cTrader client that connects to IC Markets and pulls ACTUAL data.
    Uses Twisted + ctrader_open_api for TCP connection.
    """
    
    def __init__(self, demo: bool = True):
        self.demo = demo
        self.host = ICMARKETS_DEMO_HOST if demo else ICMARKETS_LIVE_HOST
        self.port = CTADER_PORT
        
        # Connection state
        self._is_connected = False
        self._is_authenticated = False
        self._client = None
        self._reactor_thread = None
        
        # Real data storage
        self._account_info: Optional[RealAccountInfo] = None
        self._symbols: Dict[int, RealSymbolInfo] = {}
        self._spot_prices: Dict[str, Dict] = {}
        
        # Twisted reactor reference
        self._reactor = None
        
        logger.info(f"RealCtraderClient initialized | Host: {self.host}:{self.port}")
        logger.info(f"Account ID: {ACCOUNT_ID} | Demo: {demo}")
    
    def start(self) -> bool:
        """
        Start the client and connect to IC Markets.
        Returns True if connection and authentication successful.
        """
        logger.info("=" * 70)
        logger.info("CONNECTING TO IC MARKETS CTRADER (REAL DATA)")
        logger.info("=" * 70)
        
        try:
            from twisted.internet import reactor
            from ctrader_open_api import Client
            from ctrader_open_api import messages
            from ctrader_open_api.protobuf import Protobuf
            import threading
            
            self._reactor = reactor
            
            # Create client
            self._client = Client(self.host, self.port, None)
            
            # Set up callbacks
            self._client.setConnectedCallback(self._on_connected)
            self._client.setDisconnectedCallback(self._on_disconnected)
            self._client.setMessageReceivedCallback(self._on_message)
            
            # Start reactor in background thread
            def run_reactor():
                reactor.run(installSignalHandlers=False)
            
            self._reactor_thread = threading.Thread(target=run_reactor, daemon=True)
            self._reactor_thread.start()
            
            logger.info("✓ Twisted reactor started in background thread")
            logger.info("✓ Attempting to connect to IC Markets...")
            
            # Wait for connection
            timeout = 10
            start_time = time.time()
            while not self._is_connected and time.time() - start_time < timeout:
                time.sleep(0.5)
            
            if not self._is_connected:
                logger.error("✗ Connection timeout")
                return False
            
            # Wait for authentication
            start_time = time.time()
            while not self._is_authenticated and time.time() - start_time < timeout:
                time.sleep(0.5)
            
            if not self._is_authenticated:
                logger.error("✗ Authentication timeout")
                return False
            
            logger.info("=" * 70)
            logger.info("✓ SUCCESSFULLY CONNECTED TO IC MARKETS!")
            logger.info("=" * 70)
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to start: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _on_connected(self, client):
        """Called when TCP connection established."""
        logger.info("✓ TCP connection established to IC Markets!")
        self._is_connected = True
        
        # Now authenticate with application credentials
        self._authenticate_app()
    
    def _on_disconnected(self, client, reason):
        """Called when disconnected."""
        logger.warning(f"Disconnected from IC Markets: {reason}")
        self._is_connected = False
        self._is_authenticated = False
    
    def _on_message(self, client, message):
        """Called when a message is received from cTrader."""
        try:
            from ctrader_open_api.protobuf import Protobuf
            from ctrader_open_api import messages
            
            # Extract the protobuf message
            payload = Protobuf.extract(message)
            msg_type = type(payload).__name__
            
            logger.info(f"← Received: {msg_type}")
            
            # Handle different message types
            if msg_type == "ProtoOAApplicationAuthRes":
                logger.info("✓ Application authenticated!")
                self._authenticate_account()
                
            elif msg_type == "ProtoOAAccountAuthRes":
                logger.info("✓ Account authenticated!")
                self._is_authenticated = True
                self._get_account_info()
                self._get_symbols()
                
            elif msg_type == "ProtoOAGetAccountInfoRes":
                logger.info("✓ Received REAL account info from IC Markets!")
                self._parse_account_info(payload)
                
            elif msg_type == "ProtoOAGetSymbolsByBrokerRes":
                logger.info("✓ Received symbol list from IC Markets!")
                self._parse_symbols(payload)
                
            elif msg_type == "ProtoOASpotEvent":
                self._parse_spot_data(payload)
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            import traceback
            traceback.print_exc()
    
    def _authenticate_app(self):
        """Authenticate application with cTrader."""
        from ctrader_open_api.protobuf import Protobuf
        
        msg = Protobuf.get("ProtoOAApplicationAuthReq", 
                         clientId=APP_ID, 
                         clientSecret=APP_SECRET)
        self._client.send(msg)
        logger.info("→ Sent application auth request...")
    
    def _authenticate_account(self):
        """Authenticate account with access token."""
        from ctrader_open_api.protobuf import Protobuf
        
        msg = Protobuf.get("ProtoOAAccountAuthReq", 
                         accessToken=ACCESS_TOKEN,
                         ctidTraderAccountId=int(ACCOUNT_ID))
        self._client.send(msg)
        logger.info("→ Sent account auth request...")
        logger.info(f"   Account ID: {ACCOUNT_ID}")
        logger.info(f"   Access Token: {ACCESS_TOKEN[:30]}...")
    
    def _get_account_info(self):
        """Request REAL account information from IC Markets."""
        from ctrader_open_api.protobuf import Protobuf
        
        msg = Protobuf.get("ProtoOAGetAccountInfoReq", 
                         ctidTraderAccountId=int(ACCOUNT_ID))
        self._client.send(msg)
        logger.info("→ Requesting REAL account info...")
    
    def _get_symbols(self):
        """Request symbol list from IC Markets."""
        from ctrader_open_api.protobuf import Protobuf
        
        msg = Protobuf.get("ProtoOAGetSymbolsByBrokerReq")
        self._client.send(msg)
        logger.info("→ Requesting symbol list...")
    
    def _parse_account_info(self, payload):
        """Parse REAL account info from cTrader response."""
        try:
            self._account_info = RealAccountInfo(
                account_id=ACCOUNT_ID,
                ctid_trader_account_id=int(ACCOUNT_ID),
                balance=float(payload.balance) / 100 if hasattr(payload, 'balance') else 0.0,
                equity=float(payload.equity) / 100 if hasattr(payload, 'equity') else 0.0,
                margin=float(payload.margin) / 100 if hasattr(payload, 'margin') else 0.0,
                free_margin=float(payload.freeMargin) / 100 if hasattr(payload, 'freeMargin') else 0.0,
                margin_level=float(payload.marginLevel) if hasattr(payload, 'marginLevel') else 0.0,
                currency=payload.currency if hasattr(payload, 'currency') else "USD",
                leverage=f"1:{payload.leverage}" if hasattr(payload, 'leverage') else "1:30",
                is_live=not self.demo,
                account_type="Live" if not self.demo else "Demo",
                timestamp=time.time(),
            )
            
            logger.info("=" * 70)
            logger.info("REAL ACCOUNT DATA RECEIVED FROM IC MARKETS!")
            logger.info("=" * 70)
            logger.info(f"Account ID: {self._account_info.account_id}")
            logger.info(f"Balance: ${self._account_info.balance:,.2f}")
            logger.info(f"Equity: ${self._account_info.equity:,.2f}")
            logger.info(f"Margin: ${self._account_info.margin:,.2f}")
            logger.info(f"Free Margin: ${self._account_info.free_margin:,.2f}")
            logger.info(f"Margin Level: {self._account_info.margin_level:.2f}%")
            logger.info(f"Leverage: {self._account_info.leverage}")
            logger.info(f"Currency: {self._account_info.currency}")
            logger.info(f"Type: {self._account_info.account_type}")
            logger.info("=" * 70)
            
        except Exception as e:
            logger.error(f"Error parsing account info: {e}")
            import traceback
            traceback.print_exc()
    
    def _parse_symbols(self, payload):
        """Parse symbol list from IC Markets."""
        try:
            if hasattr(payload, 'symbol'):
                for symbol in payload.symbol:
                    s = RealSymbolInfo(
                        symbol_id=symbol.symbolId,
                        symbol_name=symbol.symbolName if hasattr(symbol, 'symbolName') else f"Symbol{symbol.symbolId}",
                        digits=symbol.digits if hasattr(symbol, 'digits') else 5,
                    )
                    self._symbols[s.symbol_id] = s
                
                logger.info(f"✓ Received {len(self._symbols)} symbols from IC Markets")
        except Exception as e:
            logger.error(f"Error parsing symbols: {e}")
    
    def _parse_spot_data(self, payload):
        """Parse real-time spot data."""
        try:
            symbol_id = payload.symbolId
            bid = payload.bid if hasattr(payload, 'bid') else 0.0
            ask = payload.ask if hasattr(payload, 'ask') else 0.0
            
            self._spot_prices[symbol_id] = {
                'bid': float(bid) / 100 if bid > 1000 else float(bid),  # Adjust for decimal places
                'ask': float(ask) / 100 if ask > 1000 else float(ask),
                'spread': (float(ask) - float(bid)) / 100 if bid > 1000 else float(ask) - float(bid),
                'timestamp': time.time(),
            }
        except Exception as e:
            logger.error(f"Error parsing spot data: {e}")
    
    def get_account_info(self) -> Optional[RealAccountInfo]:
        """Get REAL account information from IC Markets."""
        return self._account_info
    
    def is_connected(self) -> bool:
        """Check if connected and authenticated with IC Markets."""
        return self._is_connected and self._is_authenticated
    
    def get_all_data(self) -> Dict[str, Any]:
        """Get ALL data for dashboard from IC Markets."""
        account = self.get_account_info()
        
        return {
            "connected": self._is_connected and self._is_authenticated,
            "demo": self.demo,
            "account_id": ACCOUNT_ID,
            "real_data": True if self._is_authenticated else False,
            "account": {
                "account_id": account.account_id if account else ACCOUNT_ID,
                "balance": account.balance if account else 0.0,
                "equity": account.equity if account else 0.0,
                "margin": account.margin if account else 0.0,
                "free_margin": account.free_margin if account else 0.0,
                "margin_level": account.margin_level if account else 0.0,
                "leverage": account.leverage if account else "1:30",
                "currency": account.currency if account else "USD",
                "type": account.account_type if account else "Unknown",
            } if account else {},
            "symbols_count": len(self._symbols),
            "spot_prices": self._spot_prices,
            "broker": {
                "name": "IC Markets",
                "regulation": "ASIC, CySEC",
                "max_leverage": "1:500",
            },
            "timestamp": time.time(),
        }
    
    def disconnect(self):
        """Disconnect from IC Markets."""
        if self._client:
            self._client.stopService()
        self._is_connected = False
        self._is_authenticated = False
        logger.info("Disconnected from IC Markets")


if __name__ == "__main__":
    # Test the REAL client
    print("=" * 70)
    print("AI FOREX TRADING SYSTEM v3.0 - REAL CTRADER CONNECTION")
    print("=" * 70)
    print()
    
    client = RealCtraderClient(demo=True)
    success = client.start()
    
    if success and client.is_connected():
        print("\n" + "=" * 70)
        print("✓ SUCCESS! CONNECTED TO IC MARKETS WITH REAL DATA")
        print("=" * 70)
        
        # Get all data
        data = client.get_all_data()
        print(f"\nAccount Data from IC Markets:")
        print(f"  Balance: ${data['account'].get('balance', 0):,.2f}")
        print(f"  Equity: ${data['account'].get('equity', 0):,.2f}")
        print(f"  Margin: ${data['account'].get('margin', 0):,.2f}")
        print(f"  Free Margin: ${data['account'].get('free_margin', 0):,.2f}")
        print(f"  Type: {data['account'].get('type', 'Unknown')}")
        print(f"  Real Data: {data['real_data']}")
        
        # Keep running for a bit to receive data
        print("\nListening for real-time updates for 10 seconds...")
        time.sleep(10)
        
        client.disconnect()
    else:
        print("\n✗ Failed to connect to IC Markets.")
        print("\nPossible reasons:")
        print("1. Invalid credentials")
        print("2. Access token expired")
        print("3. Network connection issues")
        print("4. IC Markets server maintenance")
    
    print("\n" + "=" * 70)
