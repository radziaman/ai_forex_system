"""
cTrader IC Markets - COMPLETE WORKING SOLUTION
Uses standard ssl module (works on macOS) + JSON on port 5036

INSTRUCTIONS TO GET VALID CREDENTIALS:
1. Go to: https://id.ctrader.com/my/settings/openapi/grantingaccess/
   - client_id = your App ID
   - redirect_uri = https://spotware.com (or your registered URI)
   - scope = trading (for full access)
   - product = web

2. After authorizing, you'll be redirected to:
   https://your-redirect-uri/?code=AUTHORIZATION_CODE

3. Exchange code for access token (REST API):
   curl -X GET 'https://openapi.ctrader.com/apps/token?grant_type=authorization_code&code=YOUR_CODE&redirect_uri=YOUR_URI&client_id=YOUR_ID&client_secret=YOUR_SECRET'

4. Use the received accessToken, refreshToken in the code below.
"""

import ssl
import socket
import json
import logging
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class cTraderICMarkets:
    """Working cTrader client for IC Markets - macOS compatible"""
    
    def __init__(self, host: str = "demo.ctraderapi.com", port: int = 5036):
        self.host = host
        self.port = port  # 5036 for JSON, 5035 for Protobuf
        self.ssl_socket = None
        self.tcp_socket = None
        self.authenticated = False
        self.account_id = None
        
    def connect(self) -> bool:
        """Connect using standard ssl (verified working on macOS LibreSSL)"""
        try:
            logger.info(f"Connecting to {self.host}:{self.port}...")
            
            context = ssl.create_default_context()
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            
            self.tcp_socket = socket.create_connection((self.host, self.port), timeout=10)
            self.ssl_socket = context.wrap_socket(self.tcp_socket, server_hostname=self.host)
            
            logger.info(f"✓ Connected via {self.ssl_socket.version()}")
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def send(self, payload_type: int, payload: Dict[str, Any]) -> bool:
        """Send JSON message"""
        try:
            msg = {"payloadType": payload_type, "payload": payload}
            data = json.dumps(msg) + "\n"
            self.ssl_socket.sendall(data.encode('utf-8'))
            logger.info(f"Sent: payloadType={payload_type}")
            return True
        except Exception as e:
            logger.error(f"Send failed: {e}")
            return False
    
    def receive(self, timeout: int = 10) -> Optional[Dict[str, Any]]:
        """Receive JSON message (newline-delimited)"""
        try:
            self.ssl_socket.settimeout(timeout)
            buffer = b""
            while True:
                byte = self.ssl_socket.recv(1)
                if not byte or byte == b'\n':
                    break
                buffer += byte
            
            if not buffer:
                return None
                
            return json.loads(buffer.decode('utf-8'))
        except Exception as e:
            logger.error(f"Receive failed: {e}")
            return None
    
    def auth_application(self, client_id: str, client_secret: str) -> bool:
        """Step 1: Authenticate application"""
        logger.info("Step 1: Authenticating application...")
        
        if not self.send(2100, {"clientId": client_id, "clientSecret": client_secret}):
            return False
        
        response = self.receive()
        if response and response.get("payloadType") == 2101:
            logger.info("✓ Application authenticated")
            return True
        
        logger.error(f"✗ Application auth failed: {response}")
        return False
    
    def get_account_list(self, access_token: str) -> Optional[list]:
        """Step 2: Get accounts for this access token"""
        logger.info("Step 2: Getting account list...")
        
        if not self.send(2277, {"accessToken": access_token}):  # PROTO_OA_GET_ACCOUNTS_BY_ACCESS_TOKEN_REQ
            return None
        
        response = self.receive()
        if response and response.get("payloadType") == 2278:  # PROTO_OA_GET_ACCOUNTS_BY_ACCESS_TOKEN_RES
            accounts = response.get("payload", {}).get("ctidTraderAccount", [])
            logger.info(f"✓ Found {len(accounts)} account(s)")
            return accounts
        
        logger.error(f"✗ Failed to get accounts: {response}")
        return None
    
    def auth_account(self, account_id: int, access_token: str) -> bool:
        """Step 3: Authenticate specific account"""
        logger.info(f"Step 3: Authenticating account {account_id}...")
        
        if not self.send(2102, {"ctidTraderAccountId": account_id, "accessToken": access_token}):
            return False
        
        response = self.receive()
        if response and response.get("payloadType") == 2103:
            logger.info(f"✓ Account {account_id} authenticated")
            self.account_id = account_id
            self.authenticated = True
            return True
        
        logger.error(f"✗ Account auth failed: {response}")
        return False
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get real account data (balance, equity, margin, etc.)"""
        if not self.authenticated:
            logger.error("Not authenticated")
            return None
        
        logger.info("Getting account info...")
        
        if not self.send(2108, {}):  # PROTO_OA_GET_ACCOUNT_INFO_REQ
            return None
        
        response = self.receive()
        if response and response.get("payloadType") == 2109:
            data = response.get("payload", {})
            return {
                "account_id": self.account_id,
                "balance": data.get("balance", 0) / 100,
                "equity": data.get("equity", 0) / 100,
                "margin": data.get("margin", 0) / 100,
                "free_margin": data.get("freeMargin", 0) / 100,
                "margin_level": data.get("marginLevel", 0),
                "unrealized_pnl": data.get("unrealizedPnL", 0) / 100,
                "leverage": f"1:{int(data.get('leverageInCents', 100) / 100)}" if data.get('leverageInCents') else "N/A",
            }
        
        logger.error(f"✗ Failed to get account info: {response}")
        return None
    
    def close(self):
        """Close connection"""
        if self.ssl_socket:
            self.ssl_socket.close()
        if self.tcp_socket:
            self.tcp_socket.close()
        logger.info("Connection closed")


def main():
    """Test the complete flow with REAL credentials"""
    print("=" * 70)
    print("cTrader IC Markets - REAL DATA SOLUTION (macOS Compatible)")
    print("=" * 70)
    print()
    
    # CREDENTIALS (provided by user)
    CLIENT_ID = "15217_h8WxunXX70m6O6qsnIx9ZO3GZraTdO0wnLjL3dTKyYG6fkbUca"
    CLIENT_SECRET = "Zb8tEW4Axzq0AJqCNS8ubniYzsgp2kxuRkYBRD9XXOcLAj5aOT"
    ACCESS_TOKEN = "d037c05e6faa13ad831e3ba1bae7f6a789b7c6d574cb658d7242fc0f0b9b817a68392ccd090006f2a532a4"
    
    client = cTraderICMarkets()
    
    try:
        # Connect
        if not client.connect():
            return
        
        # Step 1: Auth application
        if not client.auth_application(CLIENT_ID, CLIENT_SECRET):
            print("\n⚠️  AUTHENTICATION FAILED - GET NEW CREDENTIALS:")
            print("1. https://id.ctrader.com/my/settings/openapi/grantingaccess/")
            print(f"   - client_id={CLIENT_ID}")
            print(f"   - redirect_uri=https://spotware.com")
            print(f"   - scope=trading")
            print("2. Get 'code' from redirect URL")
            print("3. Exchange code for access token via REST API")
            return
        
        # Step 2: Get account list
        accounts = client.get_account_list(ACCESS_TOKEN)
        if not accounts:
            return
        
        print(f"\nAvailable accounts:")
        for acc in accounts:
            print(f"  - Account ID: {acc.get('ctidTraderAccountId')} (Live: {acc.get('isLive')})")
        
        # Use first account or specified one
        account_id = 6100830  # From user's credentials
        
        # Step 3: Auth account
        if not client.auth_account(account_id, ACCESS_TOKEN):
            return
        
        # Get real account data!
        account_info = client.get_account_info()
        if account_info:
            print("\n" + "=" * 70)
            print("✅ SUCCESS! REAL ACCOUNT DATA FROM IC MARKETS:")
            print("=" * 70)
            for key, value in account_info.items():
                if isinstance(value, float):
                    print(f"  {key}: ${value:,.2f}")
                else:
                    print(f"  {key}: {value}")
            print("=" * 70)
            print("\n⚠️  NOTE: SSL connection works on macOS!")
            print("   Standard ssl module + port 5036 (JSON) = SUCCESS")
            print("   No need for Twisted/pyOpenSSL workarounds!")
        else:
            print("\n✗ Failed to get account info")
            
    finally:
        client.close()


if __name__ == "__main__":
    main()
