"""
cTrader FIX API Client — Institutional-grade connection for live trading.

Uses FIX 4.4 protocol with raw sockets (VERIFIED WORKING).
This implementation matches test_fix_working.py which successfully
received live prices from cTrader FIX API.

VERIFIED SETTINGS:
- Host: live-uk-eqx-01.p.c-trader.com
- Price Port: 5201 (plain text)
- SenderCompID: YOUR_SENDER_COMP_ID
- Password: YOUR_FIX_PASSWORD
- EURUSD Symbol ID: 1
- XAUUSD Symbol ID: 41
"""

import socket
import time
import threading
import logging
from typing import Optional, Dict, Callable, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

SOH = "\x01"  # FIX field delimiter


@dataclass
class FIXConfig:
    """FIX API configuration."""
    host: str = "live-uk-eqx-01.p.c-trader.com"
    price_port_plain: int = 5201
    price_port_ssl: int = 5211
    trade_port_plain: int = 5202
    trade_port_ssl: int = 5212
    sender_comp_id: str = ""  # e.g., "YOUR_SENDER_COMP_ID"
    target_comp_id: str = "cServer"
    sender_sub_id: str = "QUOTE"  # QUOTE for price, TRADE for trading
    password: str = ""  # Set via environment variable FIX_PASSWORD
    account_id: str = ""  # Set via environment variable FIX_ACCOUNT_ID
    use_ssl: bool = False
    heartbeat_secs: int = 30


@dataclass
class FIXMarketData:
    """Market data from FIX feed."""
    symbol: str = ""
    symbol_id: int = 0
    bid: float = 0.0
    ask: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0
    last_trade: float = 0.0
    volume: float = 0.0
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class CtraderFIXClient:
    """
    cTrader FIX API client - VERIFIED WORKING.

    This implementation matches the exact message format from test_fix_working.py
    which successfully connected and received live prices.

    Threading model:
    - Main thread: sends messages
    - Background thread: receives and parses messages
    """

    def __init__(self, config: FIXConfig):
        self.config = config
        self.sock: Optional[socket.socket] = None
        self.connected: bool = False
        self.seq_num: int = 1
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._buffer: bytes = b""

        # Symbol mapping: numeric ID -> symbol name (VERIFIED IDs)
        self._symbol_id_map: Dict[int, str] = {
            1: "EURUSD",
            2: "GBPUSD",
            3: "EURJPY",
            4: "USDJPY",
            5: "AUDUSD",
            6: "USDCHF",
            7: "GBPJPY",
            8: "USDCAD",
            9: "EURGBP",
            12: "NZDUSD",
            41: "XAUUSD",
            42: "XAGUSD",
            99: "XTIUSD",
            100: "XBRUSD",
            105: "ETHUSD",
            108: "USTEC",
            112: "LTCUSD",
            114: "BTCUSD",
            115: "US500",
            116: "UK100",
            121: "XNGUSD",
            125: "US30",
            139: "DE40",
            215: "XRPUSD",
        }

        # Reverse mapping: symbol name -> numeric ID
        self._symbol_name_map: Dict[str, int] = {v: k for k, v in self._symbol_id_map.items()}

        # Callbacks
        self.on_market_data: Optional[Callable[[FIXMarketData], None]] = None
        self.on_logon: Optional[Callable] = None
        self.on_logout: Optional[Callable] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.on_order_update: Optional[Callable[[Dict], None]] = None
        self.on_positions_update: Optional[Callable[[List[Dict]], None]] = None
        self.on_security_list: Optional[Callable[[Dict[int, str]], None]] = None

        # Track order-to-position mappings
        self._order_positions: Dict[str, str] = {}  # ClOrdID -> PosMaintRptID
        self._position_orders: Dict[str, List[str]] = {}  # PosMaintRptID -> [ClOrdID, ...]

        logger.info(f"CtraderFIXClient initialized for {config.sender_comp_id}")

    def _build_fix_message(self, fields: List[Tuple[str, str]]) -> str:
        """
        Build FIX message with correct header, body, trailer.

        Uses a list of (tag, value) tuples to support repeating groups
        (duplicate tags like 269=0, 269=1 for MDEntryType).

        Uses SOH (\x01) as delimiter - CRITICAL for cTrader FIX API.
        """
        body_parts = []
        for tag, value in fields:
            if tag not in ["8", "9", "10"]:
                body_parts.append(f"{tag}={value}")
        body = SOH.join(body_parts) + SOH

        body_length = len(body.encode('utf-8'))
        full = f"8=FIX.4.4{SOH}9={body_length}{SOH}{body}"
        checksum = sum(full.encode('utf-8')) % 256
        full += f"10={checksum:03d}{SOH}"
        return full

    def _parse_fix_message(self, raw_bytes: bytes) -> Dict[str, str]:
        """Parse FIX message bytes into dict."""
        try:
            msg_str = raw_bytes.decode('utf-8', errors='ignore')
            fields = {}
            parts = msg_str.split(SOH)
            for part in parts:
                if "=" in part:
                    tag, value = part.split("=", 1)
                    fields[tag] = value
            return fields
        except Exception as e:
            logger.error(f"Parse error: {e}")
            return {"error": str(e)}

    def connect(self) -> bool:
        """
        Connect to Price feed and logon.

        Returns:
            True if connection and logon successful
        """
        try:
            port = self.config.price_port_plain  # Use plain text for now

            logger.info(f"Connecting to {self.config.host}:{port}...")
            self.sock = socket.create_connection((self.config.host, port), timeout=10)
            logger.info("✓ TCP connection established")

            # Send Logon (35=A)
            logon_fields = {
                "35": "A",
                "49": self.config.sender_comp_id,
                "56": self.config.target_comp_id,
                "50": self.config.sender_sub_id,
                "57": self.config.sender_sub_id,
                "34": self.seq_num,
                "52": datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f")[:-3],
                "98": 0,
                "108": self.config.heartbeat_secs,
                "141": "Y",
                "553": self.config.account_id,
                "554": self.config.password,
            }
            self.seq_num += 1

            logon_msg = self._build_fix_message(list(logon_fields.items()))
            logger.debug(f"Sending Logon: {logon_msg[:120]}...")

            self.sock.sendall(logon_msg.encode('utf-8'))
            logger.info("✓ Logon sent")

            # Wait for Logon response
            self.sock.settimeout(5)
            try:
                response = self.sock.recv(4096)
                fields = self._parse_fix_message(response)
                logger.info(f"Logon response: {fields}")

                if fields.get("35") == "A":
                    logger.info("✓ Logon SUCCESS! (35=A received)")
                    self.connected = True

                    # Start background listener
                    self._thread = threading.Thread(target=self._listener, daemon=True)
                    self._thread.start()

                    if self.on_logon:
                        self.on_logon()
                    return True
                elif fields.get("35") == "5":
                    logger.error(f"✗ Logout received: {fields.get('58', 'No reason')}")
                    self.sock.close()
                    return False
                else:
                    logger.warning(f"Received: 35={fields.get('35', 'Unknown')}")

            except socket.timeout:
                logger.error("✗ No Logon response (timeout)")
                self.sock.close()
                return False

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            if self.on_error:
                self.on_error(str(e))
            return False

    def _listener(self):
        """Background thread to listen for messages."""
        logger.info("Listener started")
        self.sock.settimeout(1.0)

        while not self._stop_event.is_set() and self.sock:
            try:
                data = self.sock.recv(4096)
                if not data:
                    break

                self._buffer += data

                # Parse complete messages (end with SOH + "10=")
                while b"\x0110=" in self._buffer:
                    # Find end of message
                    msg_end = self._buffer.find(b"\x0110=") + 1
                    checksum_end = self._buffer.find(b"\x01", msg_end + 1)

                    if checksum_end == -1:
                        break

                    msg_bytes = self._buffer[:checksum_end + 1]
                    self._buffer = self._buffer[checksum_end + 1:]

                    # Parse message
                    fields = self._parse_fix_message(msg_bytes)
                    self._handle_message(fields)

            except socket.timeout:
                continue
            except Exception as e:
                if not self._stop_event.is_set():
                    logger.error(f"Listener error: {e}")
                break

        logger.info("Listener stopped")

    def _handle_message(self, fields: Dict[str, str]):
        """Handle incoming FIX messages."""
        msg_type = fields.get("35", "Unknown")

        if msg_type == "A":
            logger.debug("Logon response received")

        elif msg_type == "W":  # MarketDataSnapshotFullRefresh
            self._handle_snapshot(fields)

        elif msg_type == "X":  # MarketDataIncrementalRefresh - LIVE PRICES!
            self._handle_incremental_refresh(fields)

        elif msg_type == "5":
            logger.info(f"Logout received: {fields.get('58', 'No reason')}")
            if self.on_logout:
                self.on_logout()

        elif msg_type == "3":  # Reject
            logger.error(f"Reject (35=3): {fields.get('58', 'Unknown reason')}")

        elif msg_type == "0":  # Heartbeat
            logger.debug("Heartbeat received")

        else:
            logger.debug(f"Unknown message: 35={msg_type}")

    def _handle_snapshot(self, fields: Dict[str, str]):
        """Handle Market Data Snapshot (35=W)."""
        symbol_id = int(fields.get("55", 0))
        symbol = self._symbol_id_map.get(symbol_id, f"UNKNOWN_{symbol_id}")

        # Parse MD entries
        no_entries = int(fields.get("268", 0))
        for i in range(1, no_entries + 1):
            entry_type = fields.get(f"269_{i}", fields.get("269"))
            price = float(fields.get(f"270_{i}", fields.get("270", 0)))
            size = float(fields.get(f"271_{i}", fields.get("271", 0)))

            market_data = FIXMarketData(
                symbol=symbol,
                symbol_id=symbol_id,
            )

            if entry_type == "0":  # Bid
                market_data.bid = price
                market_data.bid_size = size
            elif entry_type == "1":  # Offer/Ask
                market_data.ask = price
                market_data.ask_size = size
            elif entry_type == "2":  # Trade
                market_data.last_trade = price
                market_data.volume = size

            if self.on_market_data:
                self.on_market_data(market_data)

    def _handle_incremental_refresh(self, fields: Dict[str, str]):
        """Handle Market Data Incremental Refresh (35=X) - LIVE PRICES!"""
        symbol_id = int(fields.get("55", 0))
        symbol = self._symbol_id_map.get(symbol_id, f"UNKNOWN_{symbol_id}")

        # Parse MD entries (repeating group)
        no_entries = int(fields.get("268", 0))
        for i in range(1, no_entries + 1):
            entry_type = fields.get(f"269_{i}", fields.get("269"))
            update_action = fields.get(f"279_{i}", fields.get("279", "0"))  # NEW TAG 279!
            md_entry_id = fields.get(f"278_{i}", fields.get("278"))  # NEW TAG 278!
            price = float(fields.get(f"270_{i}", fields.get("270", 0)))
            size = float(fields.get(f"271_{i}", fields.get("271", 0)))

            # Handle MDUpdateAction (tag 279) - 0=New, 2=Delete
            if update_action == "2":  # Delete
                logger.debug(f"Deleting entry {md_entry_id} for {symbol}")
                continue  # Skip processing deleted entries

            market_data = FIXMarketData(
                symbol=symbol,
                symbol_id=symbol_id,
            )

            if entry_type == "0":  # Bid
                market_data.bid = price
                market_data.bid_size = size
            elif entry_type == "1":  # Offer/Ask
                market_data.ask = price
                market_data.ask_size = size
            elif entry_type == "2":  # Trade
                market_data.last_trade = price
                market_data.volume = size

            if self.on_market_data:
                self.on_market_data(market_data)

    def subscribe_market_data(self, symbol_ids: List[int]) -> bool:
        """
        Subscribe to market data for symbols using numeric Symbol IDs.

        Args:
            symbol_ids: List of numeric cTrader Symbol IDs (e.g., [1, 41])

        Returns:
            True if subscription request sent
        """
        if not self.connected or not self.sock:
            logger.error("Not connected")
            return False

        try:
            # Build Market Data Request (35=V) with correct tag order per cTrader spec
            # Spec example: ...262|263|264=1|265|146|55|267|269|269
            md_msg = (
                f"8=FIX.4.4{SOH}9=000{SOH}"
                f"35=V{SOH}"
                f"49={self.config.sender_comp_id}{SOH}"
                f"56={self.config.target_comp_id}{SOH}"
                f"50={self.config.sender_sub_id}{SOH}"
                f"57={self.config.sender_sub_id}{SOH}"
                f"34={self.seq_num}{SOH}"
                f"52={datetime.now(timezone.utc).strftime('%Y%m%d-%H:%M:%S.%f')[:-3]}{SOH}"
                f"262={int(time.time())}{SOH}"  # MDReqID
                f"263=1{SOH}"  # SubscriptionType (1=Snapshot+Updates)
                f"264=1{SOH}"  # MarketDepth (1=Spot, not 0=Depth!)
                f"265=1{SOH}"  # MDUpdateType (1=Incremental refresh)
                f"146={len(symbol_ids)}{SOH}"  # NoRelatedSym BEFORE symbols
            )

            # Add symbols (numeric IDs!)
            for sym_id in symbol_ids:
                md_msg += f"55={sym_id}{SOH}"

            # Then MDEntryTypes group
            md_msg += (
                f"267=2{SOH}"  # NoMDEntryTypes (2=Both Bid and Ask)
                f"269=0{SOH}"  # MDEntryType (0=Bid)
                f"269=1{SOH}"  # MDEntryType (1=Offer)
            )

            # Calculate proper length and checksum
            body_start = md_msg.find("35=")
            body = md_msg[body_start:]
            body_length = len(body.encode('utf-8'))

            # Rebuild with correct length
            full_md = md_msg.replace("9=000", f"9={body_length}")

            checksum = sum(full_md.encode('utf-8')) % 256
            full_md += f"10={checksum:03d}{SOH}"

            self.seq_num += 1

            logger.info(f"Subscribing to {len(symbol_ids)} symbols: {symbol_ids}")
            self.sock.sendall(full_md.encode('utf-8'))
            logger.info("✓ Market data subscription sent")
            return True

        except Exception as e:
            logger.error(f"Subscription failed: {e}")
            return False

    def disconnect(self):
        """Disconnect and send Logout."""
        self._stop_event.set()

        if self.sock:
            try:
                # Send Logout (35=5)
                logout_fields = {
                    "35": "5",
                    "49": self.config.sender_comp_id,
                    "56": self.config.target_comp_id,
                    "50": self.config.sender_sub_id,
                    "57": self.config.sender_sub_id,
                    "34": self.seq_num,
                    "52": datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f")[:-3],
                    "58": "Client disconnect",
                }
                logout_msg = self._build_fix_message(list(logout_fields.items()))
                self.sock.sendall(logout_msg.encode('utf-8'))
                logger.info("✓ Logout sent")
            except Exception:
                pass

            try:
                self.sock.close()
            except Exception:
                pass

        self.connected = False

        if self._thread:
            self._thread.join(timeout=2)

        logger.info("Disconnected")

    def place_order(
        self,
        symbol_id: int,
        side: str,  # "BUY" or "SELL"
        volume: float,
        order_type: str = "MARKET",  # "MARKET", "LIMIT", or "STOP"
        price: float = 0.0,
        stop_px: float = 0.0,  # For STOP orders (tag 99)
        stop_loss: float = 0.0,  # Absolute SL (tag 1002)
        take_profit: float = 0.0,  # Absolute TP (tag 1000)
        relative_sl: float = 0.0,  # Relative SL in pips (tag 1003)
        relative_tp: float = 0.0,  # Relative TP in pips (tag 1001)
        trailing_sl: bool = False,  # Trailing stop loss (tag 1004)
        trigger_method_sl: int = 1,  # SL trigger method (tag 1005)
        guaranteed_sl: bool = False,  # Guaranteed SL (tag 1006)
        position_id: Optional[str] = None,  # For closing/modifying (tag 721)
        expire_time: Optional[str] = None,  # GTD orders (tag 126)
    ) -> Optional[str]:
        """
        Send New Order Single (35=D) via FIX API.
        
        IMPORTANT: This implementation follows the cTrader FIX API specification
        for tags 1000-1006 (SL/TP) as documented at:
        https://help.ctrader.com/fix/specification/
        """
        if not self.connected or not self.sock:
            logger.error("Not connected to FIX")
            return None
        
        try:
            cl_ord_id = f"{int(time.time() * 1000)}"
            
            # Map order type
            ord_type_map = {
                "MARKET": "1",  # Market (IOC)
                "LIMIT": "2",   # Limit (GTC)
                "STOP": "3",    # Stop (GTC)
            }
            ord_type = ord_type_map.get(order_type.upper(), "1")
            
            order_fields = {
                "35": "D",
                "49": self.config.sender_comp_id,
                "56": self.config.target_comp_id,
                "50": "TRADE",  # Must be TRADE for order operations
                "57": "TRADE",
                "34": self.seq_num,
                "52": datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f")[:-3],
                "11": cl_ord_id,
                "55": symbol_id,
                "54": "1" if side.upper() == "BUY" else "2",
                "38": str(int(volume)),
                "40": ord_type,
            }
            self.seq_num += 1
            
            # Price for LIMIT orders (tag 44)
            if ord_type == "2" and price > 0:
                order_fields["44"] = f"{price:.5f}"
            
            # StopPx for STOP orders (tag 99)
            if ord_type == "3" and stop_px > 0:
                order_fields["99"] = f"{stop_px:.5f}"
            
            # Stop Loss - Absolute (tag 1002) - FIXED from tag 432!
            if stop_loss > 0:
                order_fields["1002"] = f"{stop_loss:.5f}"
            
            # Take Profit - Absolute (tag 1000)
            if take_profit > 0:
                order_fields["1000"] = f"{take_profit:.5f}"
            
            # Relative SL (tag 1003) - distance in pips from entry
            if relative_sl > 0:
                order_fields["1003"] = f"{relative_sl:.5f}"
            
            # Relative TP (tag 1001)
            if relative_tp > 0:
                order_fields["1001"] = f"{relative_tp:.5f}"
            
            # Trailing SL (tag 1004)
            if trailing_sl:
                order_fields["1004"] = "Y"
            
            # SL Trigger Method (tag 1005)
            if trigger_method_sl in [1, 2, 3, 4]:
                order_fields["1005"] = str(trigger_method_sl)
            
            # Guaranteed SL (tag 1006)
            if guaranteed_sl:
                order_fields["1006"] = "Y"
            
            # Position ID for placing order to existing position (tag 721)
            if position_id:
                order_fields["721"] = position_id
            
            # ExpireTime for GTD orders (tag 126)
            if expire_time:
                order_fields["126"] = expire_time
            
            order_msg = self._build_fix_message(list(order_fields.items()))
            logger.debug(f"Sending order: {order_msg[:150]}...")
            
            self.sock.sendall(order_msg.encode('utf-8'))
            logger.info(f"✓ Order sent: {side} {volume} {symbol_id} @ {price} (type={order_type})")
            return cl_ord_id
            
        except Exception as e:
            logger.error(f"Order send failed: {e}")
            return None

    def cancel_order(
        self,
        orig_cl_ord_id: str,
        order_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Send Order Cancel Request (35=F) to cancel/close a position.
        
        Args:
            orig_cl_ord_id: Client Order ID of the order to cancel (tag 41)
            order_id: Optional server OrderID (tag 37) - preferred if available
            
        Returns:
            New ClOrdID for the cancel request, or None if failed
        """
        if not self.connected or not self.sock:
            logger.error("Not connected")
            return None
        
        try:
            new_cl_ord_id = f"CANC_{int(time.time())}"
            
            fields = {
                "35": "F",
                "49": self.config.sender_comp_id,
                "56": self.config.target_comp_id,
                "50": "TRADE",
                "57": "TRADE",
                "34": self.seq_num,
                "52": datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f")[:-3],
                "11": new_cl_ord_id,  # New ClOrdID for this cancel request
                "41": orig_cl_ord_id,  # OrigClOrdID - order to cancel
            }
            
            if order_id:
                fields["37"] = order_id  # OrderID (server-assigned)
            
            self.seq_num += 1
            
            msg = self._build_fix_message(list(fields.items()))
            self.sock.sendall(msg.encode('utf-8'))
            
            logger.info(f"Order Cancel Request sent: {new_cl_ord_id} for orig={orig_cl_ord_id}")
            return new_cl_ord_id
            
        except Exception as e:
            logger.error(f"Cancel request failed: {e}")
            return None

    def close_position(self, position_id: str, orig_cl_ord_id: str) -> bool:
        """
        Close position via Order Cancel Request (35=F).
        
        Args:
            position_id: Position ID (PosMaintRptID, tag 721)
            orig_cl_ord_id: Original ClOrdID of the opening order
        """
        result = self.cancel_order(orig_cl_ord_id=orig_cl_ord_id)
        if result:
            logger.info(f"Position close requested: {position_id}")
            return True
        return False

    def request_positions(self, position_id: Optional[str] = None) -> str:
        """
        Send Request for Positions (35=AN) to get all open positions.
        
        Args:
            position_id: Optional specific position to query (tag 721)
            
        Returns:
            PosReqID for tracking the response
        """
        if not self.connected or not self.sock:
            logger.error("Not connected")
            return ""
        
        pos_req_id = f"POS_{int(time.time())}"
        
        fields = {
            "35": "AN",
            "49": self.config.sender_comp_id,
            "56": self.config.target_comp_id,
            "50": "TRADE",
            "57": "TRADE",
            "34": self.seq_num,
            "52": datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f")[:-3],
            "710": pos_req_id,  # PosReqID
        }
        
        if position_id:
            fields["721"] = position_id  # PosMaintRptID - specific position
        
        self.seq_num += 1
        
        msg = self._build_fix_message(list(fields.items()))
        self.sock.sendall(msg.encode('utf-8'))
        
        logger.info(f"Request for Positions sent: {pos_req_id}")
        return pos_req_id

    def _handle_execution_report(self, fields: Dict[str, str]):
        """
        Handle Execution Report (35=8) - CRITICAL for order management.

        ExecType (150):
            0 = New (order accepted)
            4 = Canceled
            5 = Replaced (amended)
            8 = Rejected
            C = Expired
            F = Trade (order filled)
            I = Order Status (response to 35=H or 35=AF)
        """
        cl_ord_id = fields.get("11", "")
        order_id = fields.get("37", "")
        exec_type = fields.get("150", "")
        ord_status = fields.get("39", "")
        pos_maint_rpt_id = fields.get("721", "")
        mass_status_req_id = fields.get("584", "")
        tot_num_reports = fields.get("911", "0")

        # Symbol info
        symbol_id = int(fields.get("55", 0))
        symbol = self._symbol_id_map.get(symbol_id, f"UNKNOWN_{symbol_id}")
        side = "BUY" if fields.get("54") == "1" else "SELL"

        # Order details
        order_qty = float(fields.get("38", 0))
        price = float(fields.get("44", 0))
        stop_px = float(fields.get("99", 0))

        # Fill information
        last_qty = float(fields.get("32", 0))
        last_px = float(fields.get("31", 0))
        leaves_qty = float(fields.get("151", 0))  # Remaining quantity
        cum_qty = float(fields.get("14", 0))  # Cumulative filled quantity
        avg_px = float(fields.get("6", 0))  # Average fill price

        # SL/TP from execution report
        abs_tp = fields.get("1000")
        abs_sl = fields.get("1002")
        trailing_sl = fields.get("1004")

        # Rejection reason
        ord_rej_reason = fields.get("103", "")
        text = fields.get("58", "")

        # Store position mapping
        if pos_maint_rpt_id and cl_ord_id:
            self._order_positions[cl_ord_id] = pos_maint_rpt_id
            if pos_maint_rpt_id not in self._position_orders:
                self._position_orders[pos_maint_rpt_id] = []
            self._position_orders[pos_maint_rpt_id].append(cl_ord_id)

        # Log based on ExecType
        if exec_type == "0":  # New order accepted
            logger.info(
                f"✓ Order NEW: ClOrdID={cl_ord_id}, OrderID={order_id}, "
                f"Symbol={symbol}, Side={side}, Qty={order_qty}, "
                f"Price={price}, PosID={pos_maint_rpt_id}"
            )
        elif exec_type == "F":  # Trade (filled)
            logger.info(
                f"✓ Order FILLED: ClOrdID={cl_ord_id}, OrderID={order_id}, "
                f"Symbol={symbol}, LastQty={last_qty}, LastPx={last_px}, "
                f"LeavesQty={leaves_qty}, AvgPx={avg_px}, "
                f"PosID={pos_maint_rpt_id}"
            )
        elif exec_type == "4":  # Canceled
            logger.info(
                f"✓ Order CANCELED: ClOrdID={cl_ord_id}, OrderID={order_id}, "
                f"Symbol={symbol}, LeavesQty={leaves_qty}"
            )
        elif exec_type == "5":  # Replaced (amended)
            logger.info(
                f"✓ Order REPLACED: ClOrdID={cl_ord_id}, OrderID={order_id}, "
                f"Symbol={symbol}, NewQty={order_qty}, NewPrice={price}"
            )
        elif exec_type == "8":  # Rejected
            logger.error(
                f"✗ Order REJECTED: ClOrdID={cl_ord_id}, Reason={ord_rej_reason}, "
                f"Text={text}"
            )
        elif exec_type == "C":  # Expired
            logger.warning(
                f"⚠ Order EXPIRED: ClOrdID={cl_ord_id}, OrderID={order_id}"
            )
        elif exec_type == "I":  # Order Status (response to 35=H or 35=AF)
            logger.info(
                f"Order STATUS: ClOrdID={cl_ord_id}, OrderID={order_id}, "
                f"OrdStatus={ord_status}, LeavesQty={leaves_qty}, "
                f"CumQty={cum_qty}, TotNumReports={tot_num_reports}"
            )

        # Call callback with parsed data
        if self.on_order_update:
            order_update = {
                "msg_type": "8",
                "exec_type": exec_type,
                "ord_status": ord_status,
                "cl_ord_id": cl_ord_id,
                "order_id": order_id,
                "pos_maint_rpt_id": pos_maint_rpt_id,
                "symbol": symbol,
                "symbol_id": symbol_id,
                "side": side,
                "order_qty": order_qty,
                "price": price,
                "last_qty": last_qty,
                "last_px": last_px,
                "leaves_qty": leaves_qty,
                "cum_qty": cum_qty,
                "avg_px": avg_px,
                "text": text,
            }
            self.on_order_update(order_update)

    def _handle_business_reject(self, fields: Dict[str, str]):
        """Handle Business Message Reject (35=j)."""
        ref_seq_num = fields.get("45", "")
        ref_msg_type = fields.get("372", "")  # Original message type
        business_reject_ref_id = fields.get("379", "")  # Original ClOrdID
        business_reject_reason = fields.get("380", "0")
        text = fields.get("58", "No reason")

        reason_map = {"0": "Other"}
        reason_text = reason_map.get(business_reject_reason, business_reject_reason)

        logger.error(
            f"Business Message Reject (35=j): "
            f"RefMsgType={ref_msg_type}, RefID={business_reject_ref_id}, "
            f"Reason={reason_text}, Text={text}"
        )

        # Call error callback
        if self.on_error:
            self.on_error(f"Business Reject: {reason_text} - {text}")

        # Also call order update callback if this was an order-related reject
        if ref_msg_type == "D" and self.on_order_update:
            self.on_order_update({
                "exec_type": "8",  # Treat as rejection
                "ord_status": "8",  # Rejected
                "cl_ord_id": business_reject_ref_id,
                "text": text,
            })

    def _handle_position_report(self, fields: Dict[str, str]):
        """Handle Position Report (35=AP)."""
        pos_req_id = fields.get("710", "")
        pos_maint_rpt_id = fields.get("721", "")
        total_num_reports = fields.get("727", "0")
        pos_req_result = fields.get("728", "")

        if pos_req_result != "0":  # NOT_VALID_REQUEST
            logger.info(f"Position Report: No positions found (result={pos_req_result})")
            if self.on_positions_update:
                self.on_positions_update([])
            return

        symbol_id = int(fields.get("55", 0))
        symbol = self._symbol_id_map.get(symbol_id, f"UNKNOWN_{symbol_id}")
        long_qty = float(fields.get("704", 0))
        short_qty = float(fields.get("705", 0))
        settl_price = float(fields.get("730", 0))

        # SL/TP info
        abs_tp = fields.get("1000")
        abs_sl = fields.get("1002")
        trailing_sl = fields.get("1004")

        position = {
            "position_id": pos_maint_rpt_id,
            "symbol": symbol,
            "symbol_id": symbol_id,
            "long_qty": long_qty,
            "short_qty": short_qty,
            "settl_price": settl_price,
            "abs_tp": float(abs_tp) if abs_tp else None,
            "abs_sl": float(abs_sl) if abs_sl else None,
            "trailing_sl": trailing_sl == "Y",
        }

        logger.info(f"Position Report: {symbol} Long={long_qty} Short={short_qty} @ {settl_price}")

        if self.on_positions_update:
            self.on_positions_update([position])
