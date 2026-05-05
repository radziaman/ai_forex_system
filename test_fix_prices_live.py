#!/usr/bin/env python3
"""
Simplified FIX API Test for cTrader.
Uses simplefix library to connect and subscribe to EURUSD + XAUUSD prices.
"""
import ssl
import socket
import time
from datetime import datetime, timezone

# Try to import simplefix
try:
    import simplefix as fix
    print("✓ simplefix imported successfully")
except ImportError:
    print("❌ simplefix not installed. Run: pip install simplefix")
    exit(1)

# Your cTrader FIX API Credentials
HOST = "live-uk-eqx-01.p.c-trader.com"
PORT = 5211  # Price connection (SSL)
SENDER_COMP_ID = "YOUR_SENDER_COMP_ID"
TARGET_COMP_ID = "cServer"
SENDER_SUB_ID = "QUOTE"
ACCOUNT_ID = "YOUR_ACCOUNT_ID"
PASSWORD = "YOUR_FIX_PASSWORD"

def build_fix_logon():
    """Build FIX Logon message following cTrader specification."""
    msg = fix.FixMessage()
    msg.append_pair(8, "FIX.4.4")  # BeginString
    msg.append_pair(9, "0")  # BodyLength (placeholder)
    msg.append_pair(35, "A")  # MsgType = Logon
    msg.append_pair(49, SENDER_COMP_ID)  # SenderCompID
    msg.append_pair(56, TARGET_COMP_ID)  # TargetCompID
    msg.append_pair(50, SENDER_SUB_ID)  # SenderSubID (QUOTE)
    msg.append_pair(57, SENDER_SUB_ID)  # TargetSubID (must match for cTrader)
    msg.append_pair(34, 1)  # MsgSeqNum
    msg.append_pair(52, datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f")[:-3])  # SendingTime
    msg.append_pair(98, 0)  # EncryptMethod (0 = None)
    msg.append_pair(108, 30)  # HeartBtInt
    msg.append_pair(141, "Y")  # ResetSeqNum
    msg.append_pair(553, ACCOUNT_ID)  # Username (numeric account ID only)
    msg.append_pair(554, PASSWORD)  # Password (FIX API password)
    return msg

def build_market_data_request():
    """Build MarketDataRequest (35=V) for EURUSD and XAUUSD."""
    msg = fix.FixMessage()
    msg.append_pair(8, "FIX.4.4")
    msg.append_pair(9, "0")
    msg.append_pair(35, "V")  # MsgType = MarketDataRequest
    msg.append_pair(49, SENDER_COMP_ID)
    msg.append_pair(56, TARGET_COMP_ID)
    msg.append_pair(50, SENDER_SUB_ID)
    msg.append_pair(57, SENDER_SUB_ID)
    msg.append_pair(34, 2)  # MsgSeqNum
    msg.append_pair(52, datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f")[:-3])
    msg.append_pair(262, str(int(time.time())))  # MDReqID
    msg.append_pair(263, 1)  # SubscriptionType (1 = Snapshot + Updates)
    msg.append_pair(264, 0)  # MarketDepth (0 = Full book)
    msg.append_pair(265, 0)  # MDUpdateType (0 = Full refresh)
    msg.append_pair(146, 2)  # NoRelatedSym (2 symbols)
    msg.append_pair(55, "EURUSD", 1)  # Symbol 1
    msg.append_pair(55, "XAUUSD", 2)  # Symbol 2
    return msg

def build_logout():
    """Build Logout message (35=5)."""
    msg = fix.FixMessage()
    msg.append_pair(8, "FIX.4.4")
    msg.append_pair(9, "0")
    msg.append_pair(35, "5")  # MsgType = Logout
    msg.append_pair(49, SENDER_COMP_ID)
    msg.append_pair(56, TARGET_COMP_ID)
    msg.append_pair(50, SENDER_SUB_ID)
    msg.append_pair(57, SENDER_SUB_ID)
    msg.append_pair(34, 3)  # MsgSeqNum
    msg.append_pair(52, datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f")[:-3])
    msg.append_pair(58, "Test complete")
    return msg

def test_connection():
    """Test FIX API connection and fetch prices."""
    print("=" * 70)
    print("cTrader FIX API — Live Price Test")
    print("=" * 70)
    print()
    print(f"🔌 Connecting to {HOST}:{PORT} (SSL)...")
    print(f"   SenderCompID: {SENDER_COMP_ID}")
    print(f"   TargetCompID: {TARGET_COMP_ID}")
    print(f"   SenderSubID: {SENDER_SUB_ID}")
    print(f"   Account ID: {ACCOUNT_ID}")
    print()

    try:
        # Create SSL context
        ctx = ssl.create_default_context()
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2

        # Connect via TCP
        sock = socket.create_connection((HOST, PORT))
        ssl_sock = ctx.wrap_socket(sock, server_hostname=HOST)

        print("✓ TCP/SSL connection established")
        print()

        # Send Logon
        print("📝 Sending Logon (35=A)...")
        logon_msg = build_fix_logon()
        logon_str = str(logon_msg)
        print(f"   Message: {logon_str[:150]}...")
        ssl_sock.sendall(logon_str.encode())
        print("✓ Logon sent")
        print()

        # Wait for response
        print("⏳ Waiting for server response...")
        time.sleep(3)

        try:
            ssl_sock.settimeout(5)
            response = ssl_sock.recv(4096).decode(errors='ignore')
            print(f"   Raw response: {response[:200]}...")

            if "35=5" in response:
                print("❌ Logout received (35=5)")
                if "58=" in response:
                    start = response.find("58=") + 3
                    end = response.find("\x01", start)
                    reason = response[start:end]
                    print(f"   Reason: {reason}")
                return False
            elif "35=A" in response or "34=1" in response:
                print("✓ Logon successful!")
            else:
                print(f"✓ Received: {response[:100]}")
        except socket.timeout:
            print("⚠️  No immediate response (might be OK)")

        print()

        # Subscribe to Market Data
        print("📊 Subscribing to Market Data (35=V)...")
        print("   Symbols: EURUSD, XAUUSD")
        md_msg = build_market_data_request()
        md_str = str(md_msg)
        print(f"   Message: {md_str[:150]}...")
        ssl_sock.sendall(md_str.encode())
        print("✓ Market data subscription sent")
        print()

        # Listen for market data
        print("⏳ Listening for market data (35=X)...")
        print("   (Press Ctrl+C to stop)")
        print("-" * 70)

        ssl_sock.settimeout(1)
        buffer = ""
        start_time = time.time()

        while time.time() - start_time < 30:  # Run for 30 seconds
            try:
                data = ssl_sock.recv(4096).decode(errors='ignore')
                buffer += data

                # Parse messages
                while "\x01" in buffer:
                    msg_end = buffer.find("\x01") + 1
                    msg_str = buffer[:msg_end]
                    buffer = buffer[msg_end:]

                    # Parse fields
                    fields = {}
                    for part in msg_str.split("\x01"):
                        if "=" in part:
                            tag, value = part.split("=", 1)
                            fields[tag] = value

                    if "35" in fields:
                        msg_type = fields["35"]

                        if msg_type == "X" or msg_type == "Y":  # MarketDataIncrementalRefresh or Snapshot
                            symbol = fields.get("55", "Unknown")
                            bid = fields.get("132", "N/A")  # BidPx
                            ask = fields.get("133", "N/A")  # OfferPx
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] {symbol}: Bid={bid}, Ask={ask}")

                        elif msg_type == "5":  # Logout
                            reason = fields.get("58", "No reason")
                            print(f"❌ Logout received: {reason}")
                            return False

                        elif msg_type == "0":  # Heartbeat
                            pass  # Ignore heartbeats

                        else:
                            print(f"   Received: 35={msg_type}")

            except socket.timeout:
                time.sleep(0.1)
                continue
            except Exception as e:
                print(f"Error: {e}")
                break

        # Logout
        print()
        print("📝 Sending Logout (35=5)...")
        logout_msg = build_logout()
        ssl_sock.sendall(str(logout_msg).encode())
        print("✓ Logout sent")

        ssl_sock.close()
        print()
        print("=" * 70)
        print("✅ Test complete!")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n⚠️  IMPORTANT: Verify your FIX API password before running!")
    print("   Go to cTrader → Settings → FIX API → Check password")
    print()

    test_connection()
