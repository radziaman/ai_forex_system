#!/usr/bin/env python3
"""
cTrader FIX API Test — CORRECTED version based on official docs.

Key fixes from https://help.ctrader.com/fix/specification/:
1. Symbol ID MUST be numeric (e.g., 1 for EURUSD, not "EURUSD")
2. For Depth subscription: 264=0 (Full book), 265=1 (Incremental refresh)
3. For Spot subscription: 264=1 (Spot), 265=1 (Incremental refresh)
4. Tag 267=2 (both Bid and Ask), Tag 269 in repeating group
"""
import socket
import ssl
import time
from datetime import datetime, timezone

# Your VERIFIED credentials
HOST = "live-uk-eqx-01.p.c-trader.com"
PORT = 5201  # Plain text (easier debugging)
SENDER_COMP_ID = "YOUR_SENDER_COMP_ID"
TARGET_COMP_ID = "cServer"
SENDER_SUB_ID = "QUOTE"
ACCOUNT_ID = "YOUR_ACCOUNT_ID"
PASSWORD = "YOUR_FIX_PASSWORD"

def build_fix_message(fields_dict):
    """Build FIX message with proper header, body, trailer."""
    # Start with BeginString
    msg = "8=FIX.4.4\x01"
    
    # Build body (all fields except 8 and 9)
    body_parts = []
    for tag, value in fields_dict.items():
        if tag not in ["8", "9", "10"]:
            body_parts.append(f"{tag}={value}")
    body = "\x01".join(body_parts) + "\x01"
    
    # Calculate BodyLength (tag 9)
    body_length = len(body.encode('utf-8'))
    
    # Rebuild with correct length
    full_msg = f"8=FIX.4.4\x019={body_length}\x01{body}"
    
    # Calculate Checksum (tag 10)
    checksum = sum(full_msg.encode('utf-8')) % 256
    full_msg += f"10={checksum:03d}\x01"
    
    return full_msg

def test_connection():
    """Test FIX API with CORRECTED format."""
    
    print("=" * 70)
    print("cTrader FIX API Test — CORRECTED (Official Spec)")
    print("=" * 70)
    print()
    print(f"Host: {HOST}:{PORT}")
    print(f"SenderCompID: {SENDER_COMP_ID}")
    print(f"TargetCompID: {TARGET_COMP_ID}")
    print(f"SenderSubID: {SENDER_SUB_ID}")
    print(f"Account ID (Tag 553): {ACCOUNT_ID}")
    print(f"Password (Tag 554): {PASSWORD}")
    print()
    
    # IMPORTANT: Get numeric Symbol IDs!
    print("⚠️  CRITICAL: You MUST use NUMERIC Symbol IDs!")
    print("   In cTrader: Active Symbol Panel → Symbol Info → FIX Symbol ID")
    print("   Common IDs (verify yours):")
    print("     EURUSD = 1, GBPUSD = 2, USDJPY = 3")
    print("     XAUUSD = 4, BTCUSD = 5, AUDUSD = 6")
    print()
    
    # Using YOUR verified numeric Symbol IDs!
    EURUSD_ID = 1    # ← From cTrader: EURUSD FIX Symbol ID = 1
    XAUUSD_ID = 41   # ← From cTrader: XAUUSD FIX Symbol ID = 41
    
    try:
        # Connect
        print(f"🔌 Connecting to {HOST}:{PORT} (plain text)...")
        sock = socket.create_connection((HOST, PORT))
        print("✓ TCP connection established")
        print()
        
        # Send Logon (35=A)
        print("📝 Sending Logon (35=A)...")
        logon_fields = {
            "35": "A",
            "49": SENDER_COMP_ID,
            "56": TARGET_COMP_ID,
            "50": SENDER_SUB_ID,
            "57": SENDER_SUB_ID,
            "34": 1,
            "52": datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f")[:-3],
            "98": 0,
            "108": 30,
            "141": "Y",
            "553": ACCOUNT_ID,
            "554": PASSWORD,
        }
        logon_msg = build_fix_message(logon_fields)
        print(f"   Message: {logon_msg[:150]}...")
        sock.sendall(logon_msg.encode('utf-8'))
        print("✓ Logon sent")
        print()
        
        # Wait for Logon response
        print("⏳ Waiting for Logon response...")
        sock.settimeout(5)
        
        try:
            response = sock.recv(4096).decode('utf-8', errors='ignore')
            print(f"   Raw response: {response[:200]}...")
            print()
            
            if "35=5" in response:
                print("❌ Logout received (35=5)")
                # Parse reason
                if "58=" in response:
                    start = response.find("58=") + 3
                    end = response.find("\x01", start)
                    if end == -1:
                        end = len(response)
                    reason = response[start:end]
                    print(f"   Reason (58): {reason}")
                return False
            
            elif "35=A" in response:
                print("✓ Logon successful! (35=A received)")
                print()
                
                # Subscribe to Market Data (35=V) — CORRECTED
                print("📊 Subscribing to Market Data (35=V)...")
                print("   Type: Depth (Full book)")
                print(f"   EURUSD Symbol ID: {EURUSD_ID}")
                print(f"   XAUUSD Symbol ID: {XAUUSD_ID}")
                print()
                
                # Market Data Request — Depth Subscription
                # According to docs: 264=0 (Depth), 265=1 (Incremental), 267=2 (Bid+Ask)
                md_fields = {
                    "35": "V",
                    "49": SENDER_COMP_ID,
                    "56": TARGET_COMP_ID,
                    "50": SENDER_SUB_ID,
                    "57": SENDER_SUB_ID,
                    "34": 2,
                    "52": datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f")[:-3],
                    "262": str(int(time.time())),  # MDReqID
                    "263": 1,  # SubscriptionType (1=Snapshot+Updates)
                    "264": 0,  # MarketDepth (0=Depth/Full book)
                    "265": 1,  # MDUpdateType (1=Incremental refresh)
                    "267": 2,  # NoMDEntryTypes (2=Both Bid and Ask)
                    "269": 0,  # MDEntryType (0=Bid) — first in group
                    "269": 1,  # MDEntryType (1=Offer) — second in group
                    "146": 2,  # NoRelatedSym (2 symbols)
                    "55": EURUSD_ID,  # Symbol ID (NOT "EURUSD"!)
                    "55": XAUUSD_ID,  # Symbol ID (NOT "XAUUSD"!)
                }
                
                # Note: For repeating groups, we need to handle specially
                # Let me rebuild manually
                md_msg = (
                    "8=FIX.4.4\x01"
                    f"9=000\x01"
                    "35=V\x01"
                    f"49={SENDER_COMP_ID}\x01"
                    f"56={TARGET_COMP_ID}\x01"
                    f"50={SENDER_SUB_ID}\x01"
                    f"57={SENDER_SUB_ID}\x01"
                    "34=2\x01"
                    f"52={datetime.now(timezone.utc).strftime('%Y%m%d-%H:%M:%S.%f')[:-3]}\x01"
                    f"262={int(time.time())}\x01"
                    "263=1\x01"  # Snapshot+Updates
                    "264=0\x01"  # Depth/Full book
                    "265=1\x01"  # Incremental refresh
                    "267=2\x01"  # Both Bid and Ask
                    "269=0\x01"  # Bid (first)
                    "269=1\x01"  # Ask (second)
                    "146=2\x01"  # 2 symbols
                    f"55={EURUSD_ID}\x01"  # EURUSD numeric ID!
                    f"55={XAUUSD_ID}\x01"  # XAUUSD numeric ID!
                )
                
                # Calculate length and checksum
                body_start = md_msg.find("35=")
                body = md_msg[body_start:]
                body_length = len(body.encode('utf-8'))
                
                full_md = (
                    f"8=FIX.4.4\x019={body_length}\x01"
                    "35=V\x01"
                    f"49={SENDER_COMP_ID}\x01"
                    f"56={TARGET_COMP_ID}\x01"
                    f"50={SENDER_SUB_ID}\x01"
                    f"57={SENDER_SUB_ID}\x01"
                    "34=2\x01"
                    f"52={datetime.now(timezone.utc).strftime('%Y%m%d-%H:%M:%S.%f')[:-3]}\x01"
                    f"262={int(time.time())}\x01"
                    "263=1\x01"
                    "264=0\x01"
                    "265=1\x01"
                    "267=2\x01"
                    "269=0\x01"
                    "269=1\x01"
                    "146=2\x01"
                    f"55={EURUSD_ID}\x01"
                    f"55={XAUUSD_ID}\x01"
                )
                
                checksum = sum(full_md.encode('utf-8')) % 256
                full_md += f"10={checksum:03d}\x01"
                
                print(f"   Sending: {full_md[:200]}...")
                sock.sendall(full_md.encode('utf-8'))
                print("✓ Market data subscription sent")
                print()
                
                # Listen for market data
                print("⏳ Listening for market data (35=X or 35=Y)...")
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
                                    print(f"   Symbol ID: {fields['55']}")
                                if "270" in fields:  # MDEntryPx
                                    print(f"   Price: {fields['270']}")
                                if "271" in fields:  # MDEntrySize
                                    print(f"   Size: {fields['271']}")
                                print()
                            
                            elif msg_type == "Y":  # MarketDataSnapshotFullRefresh
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] Snapshot (35=Y)")
                                if "55" in fields:
                                    print(f"   Symbol ID: {fields['55']}")
                                print()
                            
                            elif msg_type == "5":  # Logout
                                print(f"❌ Logout received: {fields.get('58', 'No reason')}")
                                return False
                            
                            elif msg_type == "3":  # Reject
                                print(f"❌ Reject (35=3)")
                                if "58" in fields:
                                    print(f"   Reason (58): {fields['58']}")
                                if "371" in fields:  # RefTagID
                                    print(f"   RefTagID (371): {fields['371']}")
                                if "372" in fields:  # RefMsgType
                                    print(f"   RefMsgType (372): {fields['372']}")
                                print()
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
                
                # Logout
                print()
                print("📝 Sending Logout (35=5)...")
                logout_msg = (
                    "8=FIX.4.4\x01"
                    "9=000\x01"
                    "35=5\x01"
                    f"49={SENDER_COMP_ID}\x01"
                    f"56={TARGET_COMP_ID}\x01"
                    f"50={SENDER_SUB_ID}\x01"
                    f"57={SENDER_SUB_ID}\x01"
                    "34=3\x01"
                    f"52={datetime.now(timezone.utc).strftime('%Y%m%d-%H:%M:%S.%f')[:-3]}\x01"
                    "58=Test complete\x01"
                )
                
                body_start = logout_msg.find("35=")
                body = logout_msg[body_start:]
                body_length = len(body.encode('utf-8'))
                
                full_logout = (
                    f"8=FIX.4.4\x019={body_length}\x01"
                    "35=5\x01"
                    f"49={SENDER_COMP_ID}\x01"
                    f"56={TARGET_COMP_ID}\x01"
                    f"50={SENDER_SUB_ID}\x01"
                    f"57={SENDER_SUB_ID}\x01"
                    "34=3\x01"
                    f"52={datetime.now(timezone.utc).strftime('%Y%m%d-%H:%M:%S.%f')[:-3]}\x01"
                    "58=Test complete\x01"
                )
                
                checksum = sum(full_logout.encode('utf-8')) % 256
                full_logout += f"10={checksum:03d}\x01"
                
                sock.sendall(full_logout.encode('utf-8'))
                print("✓ Logout sent")
                
                sock.close()
                print()
                print("=" * 70)
                print("✅ Test complete!")
                print("=" * 70)
                return True
            
        except socket.timeout:
            print("⚠️  No response received (timeout)")
            print("   Possible issues:")
            print("   - Wrong password (verify in cTrader → FIX API settings)")
            print("   - Hostname changed (check cTrader Web → FIX API)")
            print("   - Invalid Symbol IDs (must be numeric!)")
            return False
        
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n⚠️  IMPORTANT: Verify your settings before running!")
    print("   1. Open cTrader Web/Desktop")
    print("   2. Settings → FIX API")
    print("   3. Verify password matches: YOUR_FIX_PASSWORD")
    print("   4. Copy the HOSTNAME (may have changed!)")
    print("   5. Get FIX Symbol IDs for EURUSD and XAUUSD:")
    print("      (Right-click symbol → Symbol Info → FIX Symbol ID)")
    print("   6. Update EURUSD_ID and XAUUSD_ID in this script")
    print()
    
    test_connection()
