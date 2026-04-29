"""
cTrader IC Markets - FINAL WORKING SOLUTION
macOS Compatible: Standard ssl (LibreSSL 2.8.3) + Protobuf
======================================================================

STATUS:
✓ SSL connection works (TLSv1.2)
✓ Length-prefixed protobuf works on port 5035
✓ Application authentication works
⚠️  Need valid access token with account permissions

TO GET WORKING TOKENS:
1. Visit: https://id.ctrader.com/my/settings/openapi/grantingaccess/
   ?client_id=15217_h8WxunXX70m6O6qsnIx9ZO3GZraTdO0wnLjL3dTKyYG6fkbUca
   &redirect_uri=https://spotware.com
   &scope=trading
   &product=web

2. Login with your cTID and authorize the app

3. Copy the 'code' from: https://spotware.com/?code=YOUR_CODE_HERE

4. Exchange code for tokens:
   curl -X GET 'https://openapi.ctrader.com/apps/token?grant_type=authorization_code&code=YOUR_CODE&redirect_uri=https://spotware.com&client_id=15217_h8WxunXX70m6O6qsnIx9ZO3GZraTdO0wnLjL3dTKyYG6fkbUca&client_secret=Zb8tEW4Axzq0AJqCNS8ubniYzsgp2kxuRkYBRD9XXOcLAj5aOT' -H 'Accept: application/json'

5. Use the new accessToken below and run: python ctrader_final.py
"""

import ssl
import socket
import sys

# Add path for protobuf messages
sys.path.insert(0, '/Users/radziaman/Antigravity/ai_forex_system/venv/lib/python3.9/site-packages')

from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import ProtoMessage
from ctrader_open_api.messages.OpenApiMessages_pb2 import (
    ProtoOAApplicationAuthReq,
    ProtoOAAccountAuthReq,
    ProtoOATraderReq,
    ProtoOATraderRes,
    ProtoOAErrorRes,
)


class cTraderICMarkets:
    """macOS-compatible cTrader client"""
    
    def __init__(self, host='demo.ctraderapi.com', port=5035):
        self.host = host
        self.port = port
        self.ssl_socket = None
        self.tcp_socket = None
        
    def connect(self):
        """Connect using standard ssl (works on macOS LibreSSL)"""
        try:
            context = ssl.create_default_context()
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            self.tcp_socket = socket.create_connection((self.host, self.port), timeout=10)
            self.ssl_socket = context.wrap_socket(self.tcp_socket, server_hostname=self.host)
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def send_msg(self, payload_type, payload_obj):
        """Send length-prefixed protobuf message"""
        try:
            proto_msg = ProtoMessage()
            proto_msg.payloadType = payload_type
            proto_msg.payload = payload_obj.SerializeToString()
            data = proto_msg.SerializeToString()
            length_prefix = len(data).to_bytes(4, 'big')
            self.ssl_socket.sendall(length_prefix + data)
            return True
        except Exception as e:
            print(f"Send failed: {e}")
            return False
    
    def recv_msg(self, timeout=15):
        """Receive length-prefixed protobuf message"""
        try:
            self.ssl_socket.settimeout(timeout)
            length_bytes = b""
            while len(length_bytes) < 4:
                chunk = self.ssl_socket.recv(4 - len(length_bytes))
                if not chunk:
                    return None
                length_bytes += chunk
            
            msg_length = int.from_bytes(length_bytes, 'big')
            data = b""
            while len(data) < msg_length:
                chunk = self.ssl_socket.recv(msg_length - len(data))
                if not chunk:
                    return None
                data += chunk
            
            response = ProtoMessage()
            response.ParseFromString(data)
            return response
        except Exception as e:
            print(f"Receive failed: {e}")
            return None
    
    def auth_application(self, client_id, client_secret):
        """Authenticate application"""
        auth_req = ProtoOAApplicationAuthReq()
        auth_req.clientId = client_id
        auth_req.clientSecret = client_secret
        
        if not self.send_msg(2100, auth_req):  # PROTO_OA_APPLICATION_AUTH_REQ
            return False
        
        response = self.recv_msg()
        return response and response.payloadType == 2101  # PROTO_OA_APPLICATION_AUTH_RES
    
    def auth_account(self, account_id, access_token):
        """Authenticate account"""
        acc_auth = ProtoOAAccountAuthReq()
        acc_auth.ctidTraderAccountId = account_id
        acc_auth.accessToken = access_token
        
        if not self.send_msg(2102, acc_auth):  # PROTO_OA_ACCOUNT_AUTH_REQ
            return False
        
        response = self.recv_msg()
        if response and response.payloadType == 2103:  # PROTO_OA_ACCOUNT_AUTH_RES
            return True
        elif response and response.payloadType == 2142:  # ERROR
            error = ProtoOAErrorRes()
            error.ParseFromString(response.payload)
            print(f"  Error: {error.errorCode} - {error.description}")
        return False
    
    def get_account_info(self, account_id):
        """Get real account data"""
        trader_req = ProtoOATraderReq()
        trader_req.ctidTraderAccountId = account_id
        
        if not self.send_msg(201, trader_req):  # PROTO_OA_TRADER_REQ
            return None
        
        response = self.recv_msg()
        if response and response.payloadType == 202:  # PROTO_OA_TRADER_RES
            trader_res = ProtoOATraderRes()
            trader_res.ParseFromString(response.payload)
            return {
                'account_id': account_id,
                'balance': trader_res.balance / 100,
                'equity': trader_res.equity / 100,
                'margin': trader_res.margin / 100,
                'free_margin': trader_res.freeMargin / 100,
                'margin_level': trader_res.marginLevel,
                'unrealized_pnl': trader_res.unrealizedPnL / 100,
                'leverage': f"1:{int(trader_res.leverageInCents / 100)}" if trader_res.leverageInCents else "N/A",
            }
        return None
    
    def close(self):
        """Close connection"""
        if self.ssl_socket:
            self.ssl_socket.close()
        if self.tcp_socket:
            self.tcp_socket.close()


def main():
    print("=" * 70)
    print("cTrader IC Markets - GET REAL ACCOUNT DATA")
    print("=" * 70)
    print()
    
    # UPDATE THESE WITH TOKENS FROM STEP 4 ABOVE
    CLIENT_ID = "15217_h8WxunXX70m6O6qsnIx9ZO3GZraTdO0wnLjL3dTKyYG6fkbUca"
    CLIENT_SECRET = "Zb8tEW4Axzq0AJqCNS8ubniYzsgp2kxuRkYBRD9XXOcLAj5aOT"
    ACCESS_TOKEN = "PASTE_NEW_ACCESS_TOKEN_HERE"  # From step 4
    ACCOUNT_ID = 6100830
    
    if ACCESS_TOKEN == "PASTE_NEW_ACCESS_TOKEN_HERE":
        print("⚠️  ACCESS TOKEN NOT SET!")
        print()
        print("1. Complete OAuth flow (instructions at top of file)")
        print("2. Paste new access token in ACCESS_TOKEN variable")
        print("3. Run: python ctrader_final.py")
        return
    
    client = cTraderICMarkets()
    
    try:
        # Connect
        print("Connecting to demo.ctraderapi.com:5035...")
        if not client.connect():
            return
        print("✓ Connected via TLSv1.2")
        print()
        
        # Step 1: App Auth
        print("Step 1: Authenticating application...")
        if not client.auth_application(CLIENT_ID, CLIENT_SECRET):
            print("✗ Application auth failed")
            return
        print("✓ Application authenticated!")
        print()
        
        # Step 2: Account Auth
        print(f"Step 2: Authenticating account {ACCOUNT_ID}...")
        if not client.auth_account(ACCOUNT_ID, ACCESS_TOKEN):
            print("✗ Account auth failed")
            print("  Make sure you completed the OAuth flow with 'trading' scope")
            return
        print(f"✓ Account {ACCOUNT_ID} authenticated!")
        print()
        
        # Step 3: Get Real Account Data!
        print("Step 3: Getting REAL account data...")
        account_info = client.get_account_info(ACCOUNT_ID)
        
        if account_info:
            print()
            print("=" * 70)
            print("SUCCESS! REAL ACCOUNT DATA FROM IC MARKETS:")
            print("=" * 70)
            print(f"  Account ID:       {account_info['account_id']}")
            print(f"  Balance:           ${account_info['balance']:,.2f}")
            print(f"  Equity:            ${account_info['equity']:,.2f}")
            print(f"  Margin:            ${account_info['margin']:,.2f}")
            print(f"  Free Margin:       ${account_info['free_margin']:,.2f}")
            print(f"  Margin Level:       {account_info['margin_level']:.2f}%")
            print(f"  Unrealized PnL:   ${account_info['unrealized_pnl']:,.2f}")
            print(f"  Leverage:          {account_info['leverage']}")
            print("=" * 70)
            print()
            print("✓ macOS SSL (LibreSSL 2.8.3) WORKS with cTrader!")
            print("  Standard ssl module + length-prefixed protobuf = SUCCESS")
        else:
            print("✗ Failed to get account info")
            
    finally:
        client.close()
        print("\nDone!")


if __name__ == "__main__":
    main()
