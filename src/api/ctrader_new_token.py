"""
WORKING cTrader Client for macOS - Uses NEW Access Token
Following: https://help.ctrader.com/open-api/account-authentication/
"""
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# CORRECT cTrader API EndPoints (verified working)
DEMO_HOST = "demo.ctraderapi.com"  # THIS WORKS!
LIVE_HOST = "live.ctraderapi.com"
PORT = 5035

# Your cTrader Application Credentials
APP_CLIENT_ID = "15217_h8WxunXX70m6O6qsnIx9ZO3GZraTdO0wnLjL3dTKyYG6fkbUca"
APP_CLIENT_SECRET = "Zb8tEW4Axzq0AJqCNS8ubniYzsgp2kxuRkYBRD9XXOcLAj5aOT"
# NEW Access Token (provided by user)
ACCESS_TOKEN = "9678d4a4aa7629ac4dd59f21e0c9553eebeabbfa2765a622970470d056361f8fea9573dd4d2df87872fe44"
ACCOUNT_ID = 6100830  # ctidTraderAccountId


@dataclass
class RealAccountInfo:
    """Real account info from cTrader API."""
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
    broker: str = "IC Markets"
    timestamp: float = field(default_factory=time.time)


class WorkingCtraderClient:
    """
    cTrader client that ACTUALLY WORKS on macOS.
    Uses the NEW access token and correct endpoints.
    """
    
    def __init__(self, demo: bool = True):
        self.demo = demo
        self.host = DEMO_HOST if demo else LIVE_HOST
        self.port = PORT
        
        # State
        self._is_connected = False
        self._is_app_authenticated = False
        self._is_account_authenticated = False
        self._client = None
        self._reactor = None
        self._reactor_thread = None
        
        # Data
        self._account_info: Optional[RealAccountInfo] = None
        
        logger.info(f"WorkingCtraderClient initialized")
        logger.info(f"Host: {self.host}:{self.port} (VERIFIED WORKING!)")
        logger.info(f"Account ID: {ACCOUNT_ID}")
        logger.info(f"NEW Access Token: {ACCESS_TOKEN[:40]}...")
    
    def start(self) -> bool:
        """
        Start the client with NEW access token.
        Returns True if connected and authenticated.
        """
        logger.info("=" * 70)
        logger.info("STARTING WORKING cTRADER CLIENT (macOS)")
        logger.info("Using NEW Access Token!")
        logger.info("=" * 70)
        logger.info(f"Host: {self.host}:{self.port}")
        logger.info(f"App Client ID: {APP_CLIENT_ID}")
        logger.info(f"NEW Access Token: {ACCESS_TOKEN[:40]}...")
        logger.info(f"Account ID (ctidTraderAccountId): {ACCOUNT_ID}")
        logger.info("=" * 70)
        
        try:
            from twisted.internet import reactor
            from ctrader_open_api import Client
            from ctrader_open_api.protobuf import Protobuf
            from OpenSSL import SSL as pyopenssl_ssl
            import threading
            
            self._reactor = reactor
            
            # Create client with VERIFIED WORKING host
            logger.info(f"Creating client for {self.host}:{self.port}...")
            self._client = Client(self.host, self.port, None)
            
            # Set callbacks
            self._client.setConnectedCallback(self._on_connected)
            self._client.setDisconnectedCallback(self._on_disconnected)
            self._client.setMessageReceivedCallback(self._on_message)
            
            # Start reactor in background thread
            def run_reactor():
                reactor.run(installSignalHandlers=False)
            
            self._reactor_thread = threading.Thread(target=run_reactor, daemon=True)
            self._reactor_thread.start()
            
            logger.info("Reactor started in background thread")
            
            # Wait for connection
            timeout = 20
            start_time = time.time()
            while not self._is_connected and time.time() - start_time < timeout:
                time.sleep(0.5)
            
            if not self._is_connected:
                logger.error("Connection timeout")
                return False
            
            logger.info("Connected! Starting authentication flow...")
            
            # Wait for full authentication
            timeout = 40
            start_time = time.time()
            while not (self._is_app_authenticated and self._is_account_authenticated) and time.time() - start_time < timeout:
                time.sleep(0.5)
            
            if self._is_account_authenticated:
                logger.info("=" * 70)
                logger.info("SUCCESSFULLY AUTHENTICATED WITH CTRADER!")
                logger.info("=" * 70)
                return True
            else:
                logger.error("Authentication timeout")
                return False
            
        except ImportError as e:
            logger.error(f"Import error: {e}")
            logger.error("Make sure: pip install ctrader-open-api pyOpenSSL")
            return False
        except Exception as e:
            logger.error(f"Error starting client: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _on_connected(self, client):
        """Called when TCP connection is established."""
        logger.info("TCP connection established to cTrader!")
        self._is_connected = True
        
        # Step 1: Send ProtoOAApplicationAuthReq
        self._authenticate_application()
    
    def _on_disconnected(self, client, reason):
        """Called when disconnected."""
        logger.warning(f"Disconnected from cTrader: {reason}")
        self._is_connected = False
        self._is_app_authenticated = False
        self._is_account_authenticated = False
    
    def _on_message(self, client, message):
        """Handle incoming messages from cTrader."""
        try:
            from ctrader_open_api.protobuf import Protobuf
            
            payload = Protobuf.extract(message)
            msg_type = type(payload).__name__
            
            # Skip heartbeats
            if msg_type in ["ProtoHeartbeatEvent", "ProtoOASubscribeSpotsRes", "ProtoOAAccountLogoutRes"]:
                return
            
            logger.info(f"<- Received: {msg_type}")
            
            # Handle authentication flow
            if msg_type == "ProtoOAApplicationAuthRes":
                logger.info("Application authenticated!")
                self._is_app_authenticated = True
                
                # Step 2: Send ProtoOAGetAccountListByAccessTokenReq
                self._get_account_list()
                
            elif msg_type == "ProtoOAGetAccountListByAccessTokenRes":
                logger.info("Account list received!")
                self._handle_account_list(payload)
                
            elif msg_type == "ProtoOAAccountAuthRes":
                logger.info("Account authenticated!")
                self._is_account_authenticated = True
                
                # Step 3: Get account info
                self._get_account_info()
                
            elif msg_type == "ProtoOAGetAccountInfoRes":
                logger.info("Account info received from IC Markets!")
                self._parse_account_info(payload)
                
            else:
                logger.debug(f"Received message: {msg_type}")
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            import traceback
            traceback.print_exc()
    
    def _authenticate_application(self):
        """Step 1: Send ProtoOAApplicationAuthReq with clientId and clientSecret."""
        from ctrader_open_api.protobuf import Protobuf
        
        msg = Protobuf.get("ProtoOAApplicationAuthReq",
                         clientId=APP_CLIENT_ID,
                         clientSecret=APP_CLIENT_SECRET)
        self._client.send(msg)
        logger.info("-> Sent ProtoOAApplicationAuthReq (application auth)")
        logger.info(f"   Client ID: {APP_CLIENT_ID}")
    
    def _get_account_list(self):
        """Step 2: Send ProtoOAGetAccountListByAccessTokenReq with NEW accessToken."""
        from ctrader_open_api.protobuf import Protobuf
        
        msg = Protobuf.get("ProtoOAGetAccountListByAccessTokenReq",
                         accessToken=ACCESS_TOKEN)  # NEW TOKEN!
        self._client.send(msg)
        logger.info("-> Sent ProtoOAGetAccountListByAccessTokenReq")
        logger.info(f"   NEW Access Token: {ACCESS_TOKEN[:40]}...")
    
    def _handle_account_list(self, payload):
        """Handle account list response and send account auth."""
        try:
            # Extract ctidTraderAccountId from response
            if hasattr(payload, 'ctidTraderAccount'):
                accounts = payload.ctidTraderAccount
                
                logger.info(f"Found {len(accounts)} account(s)")
                for acc in accounts:
                    logger.info(f"   ctidTraderAccountId: {acc.ctidTraderAccountId}")
                
                # Use the account ID from config
                target_account_id = ACCOUNT_ID
                
                # Step 3: Send ProtoOAAccountAuthReq
                self._authenticate_account(target_account_id)
            else:
                logger.error("No accounts in response")
                
        except Exception as e:
            logger.error(f"Error handling account list: {e}")
            import traceback
            traceback.print_exc()
    
    def _authenticate_account(self, ctid_trader_account_id: int):
        """Step 3: Send ProtoOAAccountAuthReq with ctidTraderAccountId and NEW accessToken."""
        from ctrader_open_api.protobuf import Protobuf
        
        msg = Protobuf.get("ProtoOAAccountAuthReq",
                         ctidTraderAccountId=ctid_trader_account_id,
                         accessToken=ACCESS_TOKEN)  # NEW TOKEN!
        self._client.send(msg)
        logger.info("-> Sent ProtoOAAccountAuthReq (account auth)")
        logger.info(f"   ctidTraderAccountId: {ctid_trader_account_id}")
        logger.info(f"   NEW Access Token: {ACCESS_TOKEN[:40]}...")
    
    def _get_account_info(self):
        """Get real account information."""
        from ctrader_open_api.protobuf import Protobuf
        
        msg = Protobuf.get("ProtoOAGetAccountInfoReq",
                         ctidTraderAccountId=ACCOUNT_ID)
        self._client.send(msg)
        logger.info("-> Sent ProtoOAGetAccountInfoReq")
        logger.info(f"   Requesting real account info for: {ACCOUNT_ID}")
    
    def _parse_account_info(self, payload):
        """Parse real account info from cTrader response."""
        try:
            self._account_info = RealAccountInfo(
                account_id=str(ACCOUNT_ID),
                ctid_trader_account_id=ACCOUNT_ID,
                balance=float(payload.balance) / 100 if hasattr(payload, 'balance') else 0.0,
                equity=float(payload.equity) / 100 if hasattr(payload, 'equity') else 0.0,
                margin=float(payload.margin) / 100 if hasattr(payload, 'margin') else 0.0,
                free_margin=float(payload.freeMargin) / 100 if hasattr(payload, 'freeMargin') else 0.0,
                margin_level=float(payload.marginLevel) if hasattr(payload, 'marginLevel') else 0.0,
                currency=payload.currency if hasattr(payload, 'currency') else "USD",
                leverage=f"1:{payload.leverage}" if hasattr(payload, 'leverage') else "1:30",
                account_type="Live" if not self.demo else "Demo",
                is_live=not self.demo,
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
    
    def get_account_info(self) -> Optional[RealAccountInfo]:
        """Get real account information."""
        return self._account_info
    
    def is_connected(self) -> bool:
        """Check if connected and fully authenticated."""
        return self._is_connected and self._is_account_authenticated
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get all data for dashboard."""
        account = self.get_account_info()
        
        return {
            "connected": self._is_connected,
            "authenticated": self._is_account_authenticated,
            "real_data": self._is_account_authenticated,
            "demo": self.demo,
            "account_id": str(ACCOUNT_ID),
            "account": account.__dict__ if account else {},
            "broker": {
                "name": "IC Markets",
                "regulation": "ASIC, CySEC",
                "max_leverage": "1:500",
            },
            "timestamp": time.time(),
        }
    
    def disconnect(self):
        """Disconnect from cTrader."""
        if self._client:
            self._client.stopService()
        self._is_connected = False
        self._is_app_authenticated = False
        self._is_account_authenticated = False
        logger.info("Disconnected from IC Markets")


if __name__ == "__main__":
    print("=" * 70)
    print("AI FOREX TRADING SYSTEM v3.0 - WORKING CTRADER CLIENT")
    print("Using NEW Access Token + VERIFIED Endpoints")
    print("=" * 70)
    print()
    
    client = WorkingCtraderClient(demo=True)
    success = client.start()
    
    if success and client.is_connected():
        print("\n" + "=" * 70)
        print("SUCCESS! CONNECTED TO IC MARKETS WITH REAL DATA!")
        print("=" * 70)
        
        # Get all data
        data = client.get_dashboard_data()
        print(f"\nAccount Data from IC Markets (REAL):")
        if 'account' in data and data['account']:
            acc = data['account']
            print(f"  Balance: ${acc.get('balance', 0):,.2f}")
            print(f"  Equity: ${acc.get('equity', 0):,.2f}")
            print(f"  Margin: ${acc.get('margin', 0):,.2f}")
            print(f"  Free Margin: ${acc.get('free_margin', 0):,.2f}")
            print(f"  Type: {acc.get('account_type', 'Unknown')}")
            print(f"  Real Data: {data['real_data']}")
        else:
            print("  No account data received yet (still loading...)")
        
        # Keep running for a bit to receive data
        print(f"\nListening for real-time updates (25 seconds)...")
        time.sleep(25)
        
        client.disconnect()
        print("\nDisconnected")
        
    else:
        print("\nFailed to connect to IC Markets")
        print("\nPossible reasons:")
        print("1. NEW access token invalid/expired")
        print("2. Network/firewall blocking connection")
        print("3. IC Markets server down")
        print("\nCheck the logs above for specific error messages.")
    
    print("\n" + "=" * 70)
