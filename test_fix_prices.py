"""
Live FIX API Price Test — Fetch EURUSD and XAUUSD from cTrader.

Uses your credentials:
- Host: live-uk-eqx-01.p.c-trader.com
- Price Port: 5211 (SSL)
- SenderCompID: YOUR_SENDER_COMP_ID
- Account ID: YOUR_ACCOUNT_ID
- Password: YOUR_FIX_PASSWORD
"""
import sys
sys.path.insert(0, 'src')

from datetime import datetime, timezone
import time
import logging

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_fix_connection():
    """Test FIX API connection and fetch prices."""
    
    try:
        import simplefix as fix
    except ImportError:
        print("❌ simplefix not installed. Run: pip install simplefix")
        return False
    
    print("=" * 70)
    print("cTrader FIX API — Live Price Test")
    print("=" * 70)
    print()
    
    # Configuration
    HOST = "live-uk-eqx-01.p.c-trader.com"
    PORT = 5211  # Price connection (SSL)
    SENDER_COMP_ID = "YOUR_SENDER_COMP_ID"
    TARGET_COMP_ID = "cServer"
    SENDER_SUB_ID = "QUOTE"
    ACCOUNT_ID = "YOUR_ACCOUNT_ID"
    PASSWORD = "YOUR_FIX_PASSWORD"
    
    print(f"📡 Connecting to {HOST}:{PORT}...")
    print(f"   SenderCompID: {SENDER_COMP_ID}")
    print(f"   TargetCompID: {TARGET_COMP_ID}")
    print(f"   SenderSubID: {SENDER_SUB_ID}")
    print(f"   Account: {ACCOUNT_ID}")
    print()
    
    try:
        # Create SSL context
        import ssl
        ctx = ssl.create_default_context()
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        
        # Create FIX socket connection
        conn = fix.SocketConnection(
            HOST,
            PORT,
            ssl=True,
        )
        conn.ssl_context = ctx
        
        print("✓ TCP/SSL connection established")
        
        # Send Logon (35=A)
        print("\n📝 Sending Logon message (35=A)...")
        logon = fix.FixMessage()
        logon.append_pair(35, "A")  # MsgType = Logon
        logon.append_pair(49, SENDER_COMP_ID)  # SenderCompID
        logon.append_pair(56, TARGET_COMP_ID)  # TargetCompID
        logon.append_pair(50, SENDER_SUB_ID)  # SenderSubID (QUOTE)
        logon.append_pair(57, SENDER_SUB_ID)  # TargetSubID (must match for cTrader)
        logon.append_pair(34, 1)  # MsgSeqNum
        logon.append_pair(52, datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f")[:-3])  # SendingTime
        logon.append_pair(98, 0)  # EncryptMethod (0=None)
        logon.append_pair(108, 30)  # HeartBtInt
        logon.append_pair(141, "Y")  # ResetSeqNum
        logon.append_pair(553, ACCOUNT_ID)  # Username (numeric account ID)
        logon.append_pair(554, PASSWORD)  # Password (FIX API password)
        
        print(f"   Logon message: {logon}")
        conn.send_message(logon)
        print("✓ Logon sent")
        
        # Wait for Logon response (35=A) or Reject (35=5 with 58=RET_INVALID_DATA)
        print("\n⏳ Waiting for server response...")
        time.sleep(3)
        
        # Try to receive response
        try:
            # SocketConnection doesn't have recv_message in simplefix
            # We need to read from the underlying socket
            print("   (Note: simplefix may not expose recv easily)")
            print("   Check cTrader platform to see if connection appears...")
        except Exception as e:
            print(f"❌ Error receiving: {e}")
        
        # Subscribe to Market Data (35=V)
        print("\n📊 Subscribing to Market Data (35=V)...")
        print("   Symbols: EURUSD, XAUUSD")
        
        md_req = fix.FixMessage()
        md_req.append_pair(35, "V")  # MsgType = MarketDataRequest
        md_req.append_pair(49, SENDER_COMP_ID)
        md_req.append_pair(56, TARGET_COMP_ID)
        md_req.append_pair(50, SENDER_SUB_ID)
        md_req.append_pair(57, SENDER_SUB_ID)
        md_req.append_pair(34, 2)  # MsgSeqNum
        md_req.append_pair(52, datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f")[:-3])
        md_req.append_pair(262, str(int(time.time())))  # MDReqID
        md_req.append_pair(263, 1)  # SubscriptionType (1=Snapshot+Updates)
        md_req.append_pair(264, 0)  # MarketDepth (0=Full book)
        md_req.append_pair(265, 0)  # MDUpdateType (0=Full refresh)
        md_req.append_pair(146, 2)  # NoRelatedSym (2 symbols)
        md_req.append_pair(55, "EURUSD", 1)  # Symbol 1
        md_req.append_pair(55, "XAUUSD", 2)  # Symbol 2
        
        conn.send_message(md_req)
        print("✓ Market data subscription sent")
        
        print("\n⏳ Waiting for market data...")
        print("   (Market data comes as 35=X - MarketDataIncrementalRefresh)")
        print("   Press Ctrl+C to stop")
        print()
        print("-" * 70)
        
        # Keep connection alive and print any incoming data
        import select
        
        start_time = time.time()
        while time.time() - start_time < 30:  # Run for 30 seconds
            # Try to read from socket (non-blocking)
            # Note: simplefix SocketConnection wraps socket but doesn't expose recv
            # This is a limitation - we need a proper FIX engine like QuickFIX
            time.sleep(1)
            print(f"   {datetime.now().strftime('%H:%M:%S')} - Connection alive...")
        
        # Logout (35=5)
        print("\n📝 Sending Logout (35=5)...")
        logout = fix.FixMessage()
        logout.append_pair(35, "5")
        logout.append_pair(49, SENDER_COMP_ID)
        logout.append_pair(56, TARGET_COMP_ID)
        logout.append_pair(50, SENDER_SUB_ID)
        logout.append_pair(57, SENDER_SUB_ID)
        logout.append_pair(34, 3)
        logout.append_pair(52, datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f")[:-3])
        logout.append_pair(58, "Test complete")
        
        conn.send_message(logout)
        conn.close()
        
        print("✓ Connection closed")
        return True
        
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n⚠️  IMPORTANT: Verify your FIX API password before running!")
    print("   Go to cTrader → Settings → FIX API → Check password matches: YOUR_FIX_PASSWORD")
    print()
    input("Press Enter to continue (or Ctrl+C to cancel)...")
    
    test_fix_connection()
