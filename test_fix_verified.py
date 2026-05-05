#!/usr/bin/env python3
"""
Test cTrader FIX API with VERIFIED settings.

Settings from user:
- Host: live-uk-eqx-01.p.c-trader.com (IP: 76.223.4.250)
- Port: 5211 (SSL), 5201 (Plain)
- SenderCompID: YOUR_SENDER_COMP_ID
- TargetCompID: cServer
- SenderSubID: QUOTE
- Account: YOUR_ACCOUNT_ID
- Password: YOUR_FIX_PASSWORD
"""
import socket
import ssl
import time
from datetime import datetime, timezone

# FIX Message Builder (proper implementation)
class FIXMessage:
    def __init__(self):
        self.fields = []  # List of (tag, value) tuples
    
    def append_pair(self, tag, value):
        self.fields.append((str(tag), str(value)))
    
    def __str__(self):
        # Build message without checksum
        msg_parts = []
        for tag, value in self.fields:
            msg_parts.append(f"{tag}={value}\x01")
        return "".join(msg_parts)
    
    def encode(self):
        return str(self).encode('utf-8')

def build_fix_message(msg_type, sender_comp_id, target_comp_id, sender_sub_id, account_id, password, seq_num):
    """Build a complete FIX message with correct header, body, and trailer."""
    msg = FIXMessage()
    
    # Header
    msg.append_pair(8, "FIX.4.4")  # BeginString
    # BodyLength (9) will be calculated later
    msg.append_pair(35, msg_type)  # MsgType
    msg.append_pair(49, sender_comp_id)  # SenderCompID
    msg.append_pair(56, target_comp_id)  # TargetCompID
    msg.append_pair(50, sender_sub_id)  # SenderSubID
    msg.append_pair(57, sender_sub_id)  # TargetSubID (must match for cTrader)
    msg.append_pair(34, seq_num)  # MsgSeqNum
    msg.append_pair(52, datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f")[:-3])  # SendingTime
    
    # Body for Logon (35=A)
    if msg_type == "A":
        msg.append_pair(98, 0)  # EncryptMethod (0=None)
        msg.append_pair(108, 30)  # HeartBtInt
        msg.append_pair(141, "Y")  # ResetSeqNum
        msg.append_pair(553, str(account_id))  # Username (numeric only)
        msg.append_pair(554, password)  # Password
    
    # Calculate BodyLength (9)
    msg_str = str(msg)
    # Find start of body (after BeginString and BodyLength)
    # Actually, BodyLength = length from tag 35 to end of message body
    body_start = msg_str.find("35=")
    if body_start == -1:
        body_start = 0
    body = msg_str[body_start:]
    body_length = len(body.encode('utf-8'))
    
    # Rebuild with correct BodyLength
    final_msg = FIXMessage()
    final_msg.append_pair(8, "FIX.4.4")
    final_msg.append_pair(9, body_length)
    for tag, value in msg.fields:
        if tag not in ["8", "9"]:
            final_msg.append_pair(tag, value)
    
    # Calculate Checksum (10)
    msg_for_checksum = str(final_msg)
    checksum = sum(bytearray(msg_for_checksum.encode('utf-8'))) % 256
    final_msg.append_pair(10, f"{checksum:03d}")
    
    return str(final_msg)

def test_fix_connection():
    """Test FIX API connection with user's verified settings."""
    
    # Your settings
    HOST = "live-uk-eqx-01.p.c-trader.com"
    PORT = 5201  # Try plain text first (easier to debug)
    USE_SSL = False  # Set to True to test SSL
    
    SENDER_COMP_ID = "YOUR_SENDER_COMP_ID"
    TARGET_COMP_ID = "cServer"
    SENDER_SUB_ID = "QUOTE"
    ACCOUNT_ID = YOUR_ACCOUNT_ID
    PASSWORD = "YOUR_FIX_PASSWORD"
    
    print("=" * 70)
    print("cTrader FIX API Test — Plain Text (Port 5201)")
    print("=" * 70)
    print()
    print(f"Host: {HOST}")
    print(f"Port: {PORT} (Plain text)")
    print(f"SenderCompID: {SENDER_COMP_ID}")
    print(f"TargetCompID: {TARGET_COMP_ID}")
    print(f"SenderSubID: {SENDER_SUB_ID}")
    print(f"Account ID: {ACCOUNT_ID}")
    print(f"Password: {PASSWORD}")
    print()
    
    try:
        # Connect
        print(f"🔌 Connecting to {HOST}:{PORT}...")
        sock = socket.create_connection((HOST, PORT))
        
        if USE_SSL:
            ctx = ssl.create_default_context()
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            sock = ctx.wrap_socket(sock, server_hostname=HOST)
            print("✓ SSL connection established")
        else:
            print("✓ Plain text connection established")
        
        print()
        
        # Build and send Logon (35=A)
        print("📝 Sending Logon (35=A)...")
        logon_msg = build_fix_message(
            msg_type="A",
            sender_comp_id=SENDER_COMP_ID,
            target_comp_id=TARGET_COMP_ID,
            sender_sub_id=SENDER_SUB_ID,
            account_id=ACCOUNT_ID,
            password=PASSWORD,
            seq_num=1
        )
        print(f"   Message: {logon_msg[:150]}...")
        print()
        
        sock.sendall(logon_msg.encode('utf-8'))
        print("✓ Logon sent")
        print()
        
        # Wait for response
        print("⏳ Waiting for server response (5 seconds)...")
        sock.settimeout(5)
        
        try:
            response = sock.recv(4096).decode('utf-8', errors='ignore')
            print(f"   Raw response: {response[:300]}...")
            print()
            
            if "35=5" in response:
                print("❌ Logout received (35=5)")
                # Parse reason
                if "58=" in response:
                    start = response.find("58=") + 3
                    end = response.find("\x01", start)
                    reason = response[start:end] if end > start else "Unknown"
                    print(f"   Reason: {reason}")
                return False
            elif "35=A" in response:
                print("✓ Logon successful! (35=A received)")
                print()
                
                # Subscribe to Market Data (35=V)
                print("📊 Subscribing to Market Data (35=V)...")
                print("   Symbols: EURUSD, XAUUSD")
                
                md_msg = FIXMessage()
                md_msg.append_pair(8, "FIX.4.4")
                md_msg.append_pair(9, "0")  # Placeholder
                md_msg.append_pair(35, "V")  # MarketDataRequest
                md_msg.append_pair(49, SENDER_COMP_ID)
                md_msg.append_pair(56, TARGET_COMP_ID)
                md_msg.append_pair(50, SENDER_SUB_ID)
                md_msg.append_pair(57, SENDER_SUB_ID)
                md_msg.append_pair(34, 2)  # SeqNum
                md_msg.append_pair(52, datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f")[:-3])
                md_msg.append_pair(262, str(int(time.time())))  # MDReqID
                md_msg.append_pair(263, 1)  # SubscriptionType (1=Snapshot+Updates)
                md_msg.append_pair(264, 0)  # MarketDepth (0=Full book)
                md_msg.append_pair(265, 0)  # MDUpdateType (0=Full refresh)
                md_msg.append_pair(146, 2)  # NoRelatedSym (2 symbols)
                md_msg.append_pair(55, "EURUSD")  # Symbol 1
                md_msg.append_pair(55, "XAUUSD")  # Symbol 2
                
                # Calculate length and checksum
                md_str = str(md_msg)
                body_start = md_str.find("35=")
                body = md_str[body_start:]
                body_length = len(body.encode('utf-8'))
                
                # Rebuild with correct length
                final_md = FIXMessage()
                final_md.append_pair(8, "FIX.4.4")
                final_md.append_pair(9, body_length)
                for tag, value in md_msg.fields:
                    if tag not in ["8", "9"]:
                        final_md.append_pair(tag, value)
                
                checksum = sum(bytearray(str(final_md).encode('utf-8'))) % 256
                final_md.append_pair(10, f"{checksum:03d}")
                
                print(f"   Sending: {str(final_md)[:150]}...")
                sock.sendall(str(final_md).encode('utf-8'))
                print("✓ Market data subscription sent")
                print()
                
                # Listen for market data
                print("⏳ Listening for market data (35=X)...")
                print("   (Press Ctrl+C to stop)")
                print("-" * 70)
                
                sock.settimeout(1)
                buffer = ""
                start_time = time.time()
                
                while time.time() - start_time < 30:  # Run for 30 seconds
                    try:
                        data = sock.recv(4096).decode('utf-8', errors='ignore')
                        buffer += data
                        
                        # Parse complete messages
                        while "\x01" in buffer:
                            msg_end = buffer.find("\x01", 1) + 1
                            if msg_end <= 0:
                                break
                            msg_str = buffer[:msg_end]
                            buffer = buffer[msg_end:]
                            
                            # Parse fields
                            fields = {}
                            for part in msg_str.split("\x01"):
                                if "=" in part:
                                    tag, value = part.split("=", 1)
                                    fields[tag] = value
                            
                            msg_type = fields.get("35", "Unknown")
                            
                            if msg_type == "X":  # MarketDataIncrementalRefresh
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] Market Data (35=X)")
                                if "55" in fields:
                                    print(f"   Symbol: {fields['55']}")
                                if "132" in fields:  # BidPx
                                    print(f"   Bid: {fields['132']}")
                                if "133" in fields:  # OfferPx
                                    print(f"   Ask: {fields['133']}")
                                if "31" in fields:  # LastPx
                                    print(f"   Last: {fields['31']}")
                                print()
                            
                            elif msg_type == "Y":  # MarketDataSnapshotFullRefresh
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] Snapshot (35=Y)")
                                if "55" in fields:
                                    print(f"   Symbol: {fields['55']}")
                                print()
                            
                            elif msg_type == "5":  # Logout
                                print(f"❌ Logout received: {fields.get('58', 'No reason')}")
                                return False
                            
                            elif msg_type == "0":  # Heartbeat
                                pass  # Ignore
                            
                            else:
                                print(f"   Received: 35={msg_type}")
                    
                    except socket.timeout:
                        time.sleep(0.1)
                        continue
                    except Exception as e:
                        print(f"Error: {e}")
                        break
                
            else:
                print(f"✓ Received response: {response[:100]}")
        
        except socket.timeout:
            print("⚠️  No response received (timeout)")
            print("   Possible issues:")
            print("   - Wrong password")
            print("   - Hostname changed")
            print("   - Account not provisioned for FIX")
            return False
        
        # Logout
        print()
        print("📝 Sending Logout (35=5)...")
        logout = FIXMessage()
        logout.append_pair(8, "FIX.4.4")
        logout.append_pair(9, "0")
        logout.append_pair(35, "5")
        logout.append_pair(49, SENDER_COMP_ID)
        logout.append_pair(56, TARGET_COMP_ID)
        logout.append_pair(50, SENDER_SUB_ID)
        logout.append_pair(57, SENDER_SUB_ID)
        logout.append_pair(34, 3)
        logout.append_pair(52, datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f")[:-3])
        logout.append_pair(58, "Test complete")
        
        logout_str = str(logout)
        checksum = sum(bytearray(logout_str.encode('utf-8'))) % 256
        logout.append_pair(10, f"{checksum:03d}")
        
        sock.sendall(str(logout).encode('utf-8'))
        print("✓ Logout sent")
        
        sock.close()
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
    print("\n⚠️  IMPORTANT: Verify your FIX API password in cTrader settings!")
    print("   Go to cTrader → Settings → FIX API → Check password matches: YOUR_FIX_PASSWORD")
    print()
    
    test_fix_connection()
