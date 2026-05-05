#!/usr/bin/env python3
"""
cTrader FIX API Test - WORKING VERSION.

Verified Settings:
- EURUSD Symbol ID = 1
- XAUUSD Symbol ID = 41
- Password = YOUR_FIX_PASSWORD
"""
import socket
import time
from datetime import datetime, timezone

# Credentials
HOST = "live-uk-eqx-01.p.c-trader.com"
PORT = 5201  # Plain text (easier debugging)
SENDER_COMP_ID = "YOUR_SENDER_COMP_ID"
TARGET_COMP_ID = "cServer"
SENDER_SUB_ID = "QUOTE"
ACCOUNT_ID = "YOUR_ACCOUNT_ID"
PASSWORD = "YOUR_FIX_PASSWORD"

# Verified Symbol IDs
EURUSD_ID = 1
XAUUSD_ID = 41

def build_fix_message(fields_dict):
    """Build FIX message with correct header, body, trailer."""
    # Build body (tags 35 onwards)
    body_parts = []
    for tag, value in fields_dict.items():
        if tag not in ["8", "9", "10"]:
            body_parts.append(f"{tag}={value}")
    body = "\x01".join(body_parts) + "\x01"

    # Calculate BodyLength (tag 9)
    body_length = len(body.encode('utf-8'))

    # Build full message
    full = f"8=FIX.4.4\x019={body_length}\x01{body}"

    # Calculate Checksum (tag 10)
    checksum = sum(full.encode('utf-8')) % 256
    full += f"10={checksum:03d}\x01"

    return full

def parse_fix_message(raw_bytes):
    """Parse FIX message bytes into dict."""
    try:
        msg_str = raw_bytes.decode('utf-8', errors='ignore')
        fields = {}
        parts = msg_str.split("\x01")
        for part in parts:
            if "=" in part:
                tag, value = part.split("=", 1)
                fields[tag] = value
        return fields
    except Exception as e:
        return {"error": str(e)}

def test_connection():
    print("=" * 70)
    print("cTrader FIX API Test - WORKING VERSION")
    print("=" * 70)
    print()
    print(f"Host: {HOST}:{PORT}")
    print(f"SenderCompID: {SENDER_COMP_ID}")
    print(f"Symbol IDs: EURUSD={EURUSD_ID}, XAUUSD={XAUUSD_ID}")
    print()

    try:
        # Connect
        print(f"🔌 Connecting to {HOST}:{PORT} (plain text)...")
        sock = socket.create_connection((HOST, PORT))
        print("✓ TCP connection established")
        print()

        # === LOGON (35=A) ===
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
        print(f"   Sending: {logon_msg[:120]}...")
        sock.sendall(logon_msg.encode('utf-8'))
        print("✓ Logon sent")
        print()

        # Wait for Logon response
        print("⏳ Waiting for Logon response...")
        sock.settimeout(5)

        try:
            response = sock.recv(4096)
            fields = parse_fix_message(response)
            print(f"   Response: {fields}")
            print()

            if fields.get("35") == "A":
                print("✓ Logon SUCCESS! (35=A received)")
            elif fields.get("35") == "5":
                print(f"❌ Logout received: {fields.get('58', 'No reason')}")
                return False
            else:
                print(f"✓ Received: 35={fields.get('35', 'Unknown')}")
        except socket.timeout:
            print("⚠️  No Logon response (timeout)")
            return False

        print()

        # === MARKET DATA REQUEST (35=V) ===
        print("📊 Subscribing to Market Data (35=V)...")
        print(f"   EURUSD ID: {EURUSD_ID}")
        print(f"   XAUUSD ID: {XAUUSD_ID}")
        print()

        # Build Market Data Request with proper repeating groups
        # For repeating group 269 (MDEntryType), we need to send it twice (0=Bid, 1=Offer)
        md_msg = (
            f"8=FIX.4.4\x019=000\x01"  # Placeholder for length
            f"35=V\x01"
            f"49={SENDER_COMP_ID}\x01"
            f"56={TARGET_COMP_ID}\x01"
            f"50={SENDER_SUB_ID}\x01"
            f"57={SENDER_SUB_ID}\x01"
            f"34=2\x01"
            f"52={datetime.now(timezone.utc).strftime('%Y%m%d-%H:%M:%S.%f')[:-3]}\x01"
            f"262={int(time.time())}\x01"  # MDReqID
            f"263=1\x01"  # SubscriptionType (1=Snapshot+Updates)
            f"264=0\x01"  # MarketDepth (0=Full book)
            f"265=1\x01"  # MDUpdateType (1=Incremental refresh)
            f"267=2\x01"  # NoMDEntryTypes (2=Both Bid and Ask)
            f"269=0\x01"  # MDEntryType (0=Bid) - first in group
            f"269=1\x01"  # MDEntryType (1=Offer) - second in group
            f"146=2\x01"  # NoRelatedSym (2 symbols)
            f"55={EURUSD_ID}\x01"  # Symbol 1 (numeric!)
            f"55={XAUUSD_ID}\x01"  # Symbol 2 (numeric!)
        )

        # Calculate proper length and checksum
        body_start = md_msg.find("35=")
        body = md_msg[body_start:]
        body_length = len(body.encode('utf-8'))

        # Rebuild with correct length
        full_md = (
            f"8=FIX.4.4\x019={body_length}\x01"
            f"35=V\x01"
            f"49={SENDER_COMP_ID}\x01"
            f"56={TARGET_COMP_ID}\x01"
            f"50={SENDER_SUB_ID}\x01"
            f"57={SENDER_SUB_ID}\x01"
            f"34=2\x01"
            f"52={datetime.now(timezone.utc).strftime('%Y%m%d-%H:%M:%S.%f')[:-3]}\x01"
            f"262={int(time.time())}\x01"
            f"263=1\x01"
            f"264=0\x01"
            f"265=1\x01"
            f"267=2\x01"
            f"269=0\x01"
            f"269=1\x01"
            f"146=2\x01"
            f"55={EURUSD_ID}\x01"
            f"55={XAUUSD_ID}\x01"
        )

        checksum = sum(full_md.encode('utf-8')) % 256
        full_md += f"10={checksum:03d}\x01"

        print(f"   Sending: {full_md[:180]}...")
        sock.sendall(full_md.encode('utf-8'))
        print("✓ Market data subscription sent")
        print()

        # === LISTEN FOR MARKET DATA ===
        print("⏳ Listening for market data (35=X or 35=W)...")
        print("   (Press Ctrl+C to stop)")
        print("-" * 70)

        sock.settimeout(1)
        buffer = b""
        start_time = time.time()

        while time.time() - start_time < 30:  # Run for 30 seconds
            try:
                data = sock.recv(4096)
                buffer += data

                # Parse complete messages (end with \x0110=xxx)
                while b"\x0110=" in buffer:
                    # Find end of message
                    msg_end = buffer.find(b"\x0110=") + 1
                    # Find checksum end
                    checksum_end = buffer.find(b"\x01", msg_end + 1)
                    if checksum_end == -1:
                        break

                    msg_bytes = buffer[:checksum_end + 1]
                    buffer = buffer[checksum_end + 1:]

                    # Parse message
                    fields = parse_fix_message(msg_bytes)
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

                    elif msg_type == "W":  # MarketDataSnapshotFullRefresh
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Snapshot (35=W)")
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
                        if "371" in fields:
                            print(f"   RefTagID (371): {fields['371']}")
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

        # === LOGOUT (35=5) ===
        print()
        print("📝 Sending Logout (35=5)...")
        logout_fields = {
            "35": "5",
            "49": SENDER_COMP_ID,
            "56": TARGET_COMP_ID,
            "50": SENDER_SUB_ID,
            "57": SENDER_SUB_ID,
            "34": 3,
            "52": datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f")[:-3],
            "58": "Test complete",
        }
        logout_msg = build_fix_message(logout_fields)
        sock.sendall(logout_msg.encode('utf-8'))
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
    print("\n⚠️  VERIFIED SETTINGS:")
    print("   EURUSD Symbol ID = 1")
    print("   XAUUSD Symbol ID = 41")
    print("   Password = YOUR_FIX_PASSWORD")
    print()

    test_connection()
