"""
Telegram notification integration.
Sends trade alerts, daily summaries, and risk warnings to a Telegram chat.
Uses threading to avoid blocking the async trading loop.
"""
import asyncio
import threading
import time
from typing import Optional, Dict, List
from collections import deque
from dataclasses import dataclass, field
from loguru import logger

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

BOT_API = "https://api.telegram.org/bot{token}/sendMessage"


@dataclass
class Notification:
    message: str
    level: str = "info"  # info, warning, error, success
    timestamp: float = field(default_factory=time.time)
    html: Optional[str] = None


class TelegramNotifier:
    def __init__(
        self,
        bot_token: str = "",
        chat_id: str = "",
        send_trade_alerts: bool = True,
        send_daily_summary: bool = True,
        send_risk_alerts: bool = True,
        rate_limit_per_second: float = 1.0,
        max_queue: int = 100,
    ):
        self.bot_token = bot_token or ""
        self.chat_id = chat_id or ""
        self.send_trade_alerts = send_trade_alerts
        self.send_daily_summary = send_daily_summary
        self.send_risk_alerts = send_risk_alerts
        self._min_interval = 1.0 / max(rate_limit_per_second, 0.5)
        self._last_send = 0.0
        self._queue: deque = deque(maxlen=max_queue)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._enabled = bool(self.bot_token and self.chat_id)

        if self._enabled:
            self._start_worker()
            logger.info(f"Telegram notifier initialized (chat_id={chat_id[:6]}...)")
        else:
            logger.info("Telegram notifier disabled (no token or chat_id)")

    def _start_worker(self):
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def _worker_loop(self):
        while self._running:
            try:
                if self._queue:
                    notif = self._queue.popleft()
                    self._send_sync(notif)
                else:
                    time.sleep(0.1)
            except Exception:
                time.sleep(1)

    def _send_sync(self, notif: Notification):
        if not self._enabled or not REQUESTS_AVAILABLE:
            return
        elapsed = time.time() - self._last_send
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        try:
            text = notif.html or notif.message
            resp = requests.post(
                BOT_API.format(token=self.bot_token),
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=10,
            )
            self._last_send = time.time()
            if resp.status_code != 200:
                logger.warning(f"Telegram API error: {resp.status_code} {resp.text[:100]}")
        except Exception as e:
            logger.debug(f"Telegram send failed: {e}")

    def send(self, message: str, level: str = "info", html: Optional[str] = None):
        if not self._enabled:
            return
        self._queue.append(Notification(message=message, level=level, html=html))

    # Convenience methods
    def trade_opened(self, symbol: str, direction: str, volume: float, price: float,
                     regime: str, confidence: float, atr: float = 0.0):
        emoji = "\U0001F7E2" if direction == "BUY" else "\U0001F534"
        html = (
            f"{emoji} <b>TRADE OPENED</b>\n"
            f"Pair: {symbol}\n"
            f"Direction: {direction}\n"
            f"Volume: {volume:.0f}\n"
            f"Price: {price:.5f}\n"
            f"Regime: {regime}\n"
            f"Confidence: {confidence:.1%}\n"
            f"ATR: {atr:.5f}"
        )
        self.send(message=f"TRADE OPENED {direction} {symbol}", level="success", html=html)

    def trade_closed(self, symbol: str, direction: str, entry: float, exit: float,
                     pnl: float, reason: str, hold_time: float = 0.0):
        emoji = "\U0001F7E2" if pnl > 0 else "\U0001F534"
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        pct = ((exit - entry) / entry) * (1 if direction == "BUY" else -1) * 100
        minutes = int(hold_time / 60)
        html = (
            f"{emoji} <b>TRADE CLOSED</b>\n"
            f"Pair: {symbol}\n"
            f"Direction: {direction}\n"
            f"Entry: {entry:.5f} → Exit: {exit:.5f}\n"
            f"PnL: {pnl_str} ({pct:+.2f}%)\n"
            f"Reason: {reason}\n"
            f"Held: {minutes}min"
        )
        self.send(message=f"TRADE CLOSED {symbol} {pnl_str}", level="success" if pnl > 0 else "warning", html=html)

    def risk_warning(self, message: str, details: Optional[Dict] = None):
        detail_str = ""
        if details:
            detail_str = "\n" + "\n".join(f"  {k}: {v}" for k, v in details.items())
        html = (
            f"\u26A0\uFE0F <b>RISK WARNING</b>\n"
            f"{message}{detail_str}"
        )
        self.send(message=f"RISK: {message}", level="warning", html=html)

    def daily_summary(self, trades_today: int, wins: int, losses: int, pnl: float,
                      balance: float, open_positions: int, regime_summary: str = ""):
        win_rate = wins / max(wins + losses, 1)
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        html = (
            f"\U0001F4CA <b>DAILY SUMMARY</b>\n"
            f"Trades: {trades_today} | W/L: {wins}/{losses}\n"
            f"Win Rate: {win_rate:.0%}\n"
            f"PnL: {pnl_str}\n"
            f"Balance: ${balance:,.0f}\n"
            f"Open Positions: {open_positions}\n"
            f"Regimes: {regime_summary}"
        )
        self.send(message=f"DAILY SUMMARY: {pnl_str}", level="info", html=html)

    def system_alert(self, message: str):
        html = f"\U0001F6A8 <b>SYSTEM ALERT</b>\n{message}"
        self.send(message=f"ALERT: {message}", level="error", html=html)

    def shutdown(self):
        self._running = False
        if self._queue:
            time.sleep(1)
