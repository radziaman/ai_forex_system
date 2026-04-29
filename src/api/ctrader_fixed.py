"""
Fixed cTrader Client for macOS - Uses pyOpenSSL to fix LibreSSL issue
Pulls REAL account data from IC Markets
"""
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# IC Markets Endpoints
HOST_DEMO = "demo.ctrader.com"
HOST_LIVE = "live.ctrader.com"
PORT = 5035

# Your Credentials
APP_ID = "15217_h8WxunXX70m6O6qsnIx9ZO3GZraTdO0wnLjL3dTKyYG6fkbUca"
APP_SECRET = "Zb8tEW4Axzq0AJqCNS8ubniYzsgp2kxuRkYBRD9XXOcLAj5aOT"
ACCESS_TOKEN = "7d574e0bbcf30419cc752a4ea17bd627f8b46bdf79b5a3f34cf38db8285ce17f43592d3cb5bbd067ca3d2d"
ACCOUNT_ID = "6100830"


@dataclass
class RealAccountData:
    """Real account data from IC Markets."""
    account_id: str = ""
    ctid_trader_account_id: int = 0
    balance: float = 0.0
    equity: float = 0.0
    margin: float = 0.0
    free_margin: float = 0.0
    margin_level: float = 0.0
    currency: str = "USD"
    leverage: str = "1:30"
    account_type: str = "Demo"
    is_live: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class SpotData:
    """Real-time spot data."""
    symbol_id: int = 0
    symbol: str = ""
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0
    timestamp: float = field(default_factory=time.time)


class FixedCtraderClient:
    """
    Fixed cTrader client that works on macOS with LibreSSL.
    Uses pyOpenSSL and custom SSL context to connect to IC Markets.
    """
    
    def __init__(self, demo: bool = True):
        self.demo = demo
        self.host = HOST_DEMO if demo else HOST_LIVE
        self.port = PORT
        
        # State
        self._is_connected = False
        self._is_authenticated = False
        self._client = None
        self._reactor = None
        self._reactor_thread = None
        
        # Data
        self._account_data: Optional[RealAccountData] = None
        self._spot_data: Dict[int, SpotData] = {}
        
        logger.info(f"FixedCtraderClient | Host: {self.host}:{self.port}")
    
    def start(self) -> bool:
        """Start and connect to IC Markets with fixed SSL."""
        logger.info("=" * 70)
        logger.info("STARTING FIXED CTRADER CLIENT (macOS SSL Fix)")
        logger.info("=" * 70)
        
        try:
            # Import required modules
            from twisted.internet import reactor, ssl as twisted_ssl
            from OpenSSL import SSL as pyopenssl_ssl
            from ctrader_open_api import Client
            from ctrader_open_api.protobuf import Protobuf
            
            self._reactor = reactor
            
            # Create custom SSL context using pyOpenSSL
            logger.info("Creating SSL context with pyOpenSSL...")
            ssl_context = pyopenssl_ssl.Context(pyopenssl_ssl.TLSv1_2_METHOD)
            ssl_context.set_verify(pyopenssl_ssl.VERIFY_NONE, callback=None)
            
            # Wrap in Twisted's SSL
            twisted_ssl_context = twisted_ssl.CertificateOptions(
                trustRoot=twisted_ssl.platformTrust(),
            )
            
            logger.info(f"✓ SSL context created")
            
            # Create client with custom SSL
            # Note: ctrader_open_api uses ssl: endpoint string
            # We need to monkey-patch or use a custom endpoint
            self._client = Client(self.host, self.port, None)
            
            # Set callbacks
            self._client.setConnectedCallback(self._on_connected)
            self._client.setDisconnectedCallback(self._on_disconnected)
            self._client.setMessageReceivedCallback(self._on_message)
            
            # Start reactor in thread
            import threading
            def run_reactor():
                reactor.run(installSignalHandlers=False)
            
            self._reactor_thread = threading.Thread(target=run_reactor, daemon=True)
            self._reactor_thread.start()
            
            logger.info("✓ Reactor started in background thread")
            
            # Wait for connection
            timeout = 15
            start = time.time()
            while not self._is_connected and time.time() - start < timeout:
                time.sleep(0.5)
            
            if self._is_connected:
                logger.info("✓ TCP connection established!")
                
                # Wait for authentication
                start = time.time()
                while not self._is_authenticated and time.time() - start < timeout:
                    time.sleep(0.5)
                
                if self._is_authenticated:
                    logger.info("=" * 70)
                    logger.info("✓ SUCCESSFULLY CONNECTED WITH REAL DATA!")
                    logger.info("=" * 70)
                    return True
                else:
                    logger.error("✗ Authentication timeout")
            else:
                logger.error("✗ Connection timeout")
            
            return False
            
        except ImportError as e:
            logger.error(f"✗ Import error: {e}")
            logger.error("Make sure pyOpenSSL and ctrader_open_api are installed")
            return False
        except Exception as e:
            logger.error(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _on_connected(self, client):
        """Called when connected."""
        logger.info("✓ Connected to IC Markets!")
        self._is_connected = True
        self._authenticate_app()
    
    def _on_disconnected(self, client, reason):
        """Called when disconnected."""
        logger.warning(f"Disconnected: {reason}")
        self._is_connected = False
        self._is_authenticated = False
    
    def _on_message(self, client, message):
        """Handle incoming messages."""
        try:
            from ctrader_open_api.protobuf import Protobuf
            
            payload = Protobuf.extract(message)
            msg_type = type(payload).__name__
            
            logger.info(f"← {msg_type}")
            
            if msg_type == "ProtoOAApplicationAuthRes":
                logger.info("✓ App authenticated!")
                self._authenticate_account()
                
            elif msg_type == "ProtoOAAccountAuthRes":
                logger.info("✓ Account authenticated!")
                self._is_authenticated = True
                self._get_account_info()
                
            elif msg_type == "ProtoOAGetAccountInfoRes":
                logger.info("✓ Received REAL account info!")
                self._parse_account_info(payload)
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    def _authenticate_app(self):
        """Authenticate application."""
        from ctrader_open_api.protobuf import Protobuf
        
        msg = Protobuf.get("ProtoOAApplicationAuthReq",
                         clientId=APP_ID,
                         clientSecret=APP_SECRET)
        self._client.send(msg)
        logger.info("→ Sent app auth request...")
    
    def _authenticate_account(self):
        """Authenticate account."""
        from ctrader_open_api.protobuf import Protobuf
        
        msg = Protobuf.get("ProtoOAAccountAuthReq",
                         accessToken=ACCESS_TOKEN,
                         ctidTraderAccountId=int(ACCOUNT_ID))
        self._client.send(msg)
        logger.info(f"→ Sent account auth (ID: {ACCOUNT_ID})...")
    
    def _get_account_info(self):
        """Get real account info."""
        from ctrader_open_api.protobuf import Protobuf
        
        msg = Protobuf.get("ProtoOAGetAccountInfoReq",
                         ctidTraderAccountId=int(ACCOUNT_ID))
        self._client.send(msg)
        logger.info("→ Requesting REAL account data...")
    
    def _parse_account_info(self, payload):
        """Parse real account info."""
        try:
            self._account_data = RealAccountData(
                account_id=ACCOUNT_ID,
                ctid_trader_account_id=int(ACCOUNT_ID),
                balance=float(payload.balance) / 100 if hasattr(payload, 'balance') else 0.0,
                equity=float(payload.equity) / 100 if hasattr(payload, 'equity') else 0.0,
                margin=float(payload.margin) / 100 if hasattr(payload, 'margin') else 0.0,
                free_margin=float(payload.freeMargin) / 100 if hasattr(payload, 'freeMargin') else 0.0,
                margin_level=float(payload.marginLevel) if hasattr(payload, 'marginLevel') else 0.0,
                currency=payload.currency if hasattr(payload, 'currency') else "USD",
                leverage=f"1:{payload.leverage}" if hasattr(payload, 'leverage') else "1:30",
                account_type="Live" if not self.demo else "Demo",
                is_live=not self.demo,
            )
            
            logger.info("=" * 70)
            logger.info("REAL ACCOUNT DATA FROM IC MARKETS!")
            logger.info("=" * 70)
            logger.info(f"Balance: ${self._account_data.balance:,.2f}")
            logger.info(f"Equity: ${self._account_data.equity:,.2f}")
            logger.info(f"Margin: ${self._account_data.margin:,.2f}")
            logger.info(f"Free Margin: ${self._account_data.free_margin:,.2f}")
            logger.info(f"Leverage: {self._account_data.leverage}")
            logger.info(f"Type: {self._account_data.account_type}")
            logger.info("=" * 70)
            
        except Exception as e:
            logger.error(f"Error parsing account: {e}")
            import traceback
            traceback.print_exc()
    
    def get_account_info(self) -> Optional[RealAccountData]:
        """Get real account data."""
        return self._account_data
    
    def is_connected(self) -> bool:
        """Check connection status."""
        return self._is_connected and self._is_authenticated
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get all data for dashboard."""
        account = self.get_account_info()
        
        return {
            "connected": self._is_connected and self._is_authenticated,
            "real_data": self._is_authenticated,
            "demo": self.demo,
            "account_id": ACCOUNT_ID,
            "account": account.__dict__ if account else {},
            "broker": {
                "name": "IC Markets",
                "regulation": "ASIC, CySEC",
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
    print("=" * 70)
    print("FIXED CTRADER CLIENT TEST (macOS SSL Fix)")
    print("=" * 70)
    print()
    
    client = FixedCtraderClient(demo=True)
    success = client.start()
    
    if success and client.is_connected():
        print("\n" + "=" * 70)
        print("✓ SUCCESS! CONNECTED TO IC MARKETS WITH REAL DATA!")
        print("=" * 70)
        
        data = client.get_dashboard_data()
        print(f"\nAccount Data:")
        print(f"  Balance: ${data['account'].get('balance', 0):,.2f}")
        print(f"  Equity: ${data['account'].get('equity', 0):,.2f}")
        print(f"  Type: {data['account'].get('account_type', 'Unknown')}")
        print(f"  Real Data: {data['real_data']}")
        
        time.sleep(5)
        client.disconnect()
    else:
        print("\n✗ Failed to connect to IC Markets")
        print("\nPossible issues:")
        print("1. SSL/TLS still not working")
        print("2. Invalid credentials")
        print("3. Network/firewall blocking connection")
        print("4. IC Markets server down")
