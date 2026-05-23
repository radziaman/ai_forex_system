"""
Telegram notification integration.
Sends trade alerts, daily summaries, and risk warnings to a Telegram chat.
Uses threading to avoid blocking the async trading loop.
"""

import threading
import time
from typing import Optional, Dict
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
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
    level: str = "info"
    timestamp: float = field(default_factory=time.time)
    html: Optional[str] = None


def _fmt_time(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S UTC")


def _fmt_pnl(pnl: float) -> str:
    if pnl >= 0:
        return f"<code>+${pnl:.2f}</code>"
    return f"<code>-${abs(pnl):.2f}</code>"


def _fmt_pct(pct: float) -> str:
    sign = "+" if pct >= 0 else ""
    return f"<code>{sign}{pct:.2f}%</code>"


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
        last_error = None
        for attempt in range(3):
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
                if resp.status_code == 200:
                    self._last_send = time.time()
                    return
                if resp.status_code == 429:
                    retry_after = (
                        resp.json().get("parameters", {}).get("retry_after", 5)
                    )
                    logger.warning(f"Telegram rate-limited, waiting {retry_after}s")
                    time.sleep(retry_after)
                    self._min_interval = max(self._min_interval, 1.1)
                    last_error = "rate_limited"
                    continue
                logger.warning(
                    f"Telegram API error (attempt {attempt+1}): {resp.status_code} {resp.text[:100]}"  # noqa: E501
                )
                last_error = f"HTTP {resp.status_code}"
                time.sleep(2)
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Telegram send failed (attempt {attempt+1}): {e}")
                time.sleep(2)
        if last_error:
            logger.error(f"Telegram send failed after 3 attempts: {last_error}")

    def send(self, message: str, level: str = "info", html: Optional[str] = None):
        if not self._enabled:
            return
        self._queue.append(Notification(message=message, level=level, html=html))

    def trade_opened(
        self,
        symbol: str,
        direction: str,
        volume: float,
        price: float,
        regime: str,
        confidence: float,
        atr: float = 0.0,
    ):
        badge = "\U0001f7e2 BUY" if direction == "BUY" else "\U0001f534 SELL"
        html = (
            f"<b>| TRADE OPENED |</b>\n"
            f"{'─' * 30}\n"
            f"Direction:  {badge}\n"
            f"Symbol:     <code>{symbol}</code>\n"
            f"Volume:     <code>{volume:.0f}</code>\n"
            f"Price:      <code>{price:.5f}</code>\n"
            f"Regime:     <code>{regime}</code>\n"
            f"Confidence: <code>{confidence:.1%}</code>\n"
            f"ATR:        <code>{atr:.5f}</code>\n"
            f"{'─' * 30}"
        )
        self.send(message=f"OPEN {direction} {symbol}", level="success", html=html)

    def trade_closed(
        self,
        symbol: str,
        direction: str,
        entry: float,
        exit: float,
        pnl: float,
        reason: str,
        hold_time: float = 0.0,
    ):
        pct = ((exit - entry) / entry) * (1 if direction == "BUY" else -1) * 100
        minutes = int(hold_time / 60)
        badge = "\U0001f7e2" if pnl > 0 else "\U0001f534"
        html = (
            f"<b>| TRADE CLOSED |</b>\n"
            f"{'─' * 30}\n"
            f"Symbol:  <code>{symbol}</code>\n"
            f"Result:  {badge} {_fmt_pnl(pnl)} ({_fmt_pct(pct)})\n"
            f"Entry:   <code>{entry:.5f}</code>\n"
            f"Exit:    <code>{exit:.5f}</code>\n"
            f"Reason:  <code>{reason}</code>\n"
            f"Held:    <code>{minutes}m</code>\n"
            f"{'─' * 30}"
        )
        self.send(
            message=f"CLOSE {symbol} {_fmt_pnl(pnl)}",
            level="success" if pnl > 0 else "warning",
            html=html,
        )

    def risk_warning(self, message: str, details: Optional[Dict] = None):
        detail_str = ""
        if details:
            detail_str = "\n" + "\n".join(
                f"  <code>{k}: {v}</code>" for k, v in details.items()
            )
        html = (
            f"<b>| RISK WARNING |</b>\n"
            f"{'─' * 30}\n"
            f"{message}{detail_str}\n"
            f"{'─' * 30}"
        )
        self.send(message=f"RISK: {message}", level="warning", html=html)

    def daily_summary(
        self,
        trades_today: int,
        wins: int,
        losses: int,
        pnl: float,
        balance: float,
        open_positions: int,
        regime_summary: str = "",
    ):
        win_rate = wins / max(wins + losses, 1)
        html = (
            f"<b>| DAILY SUMMARY |</b>\n"
            f"{'─' * 30}\n"
            f"Date:    <code>{datetime.now().strftime('%Y-%m-%d')}</code>\n"
            f"Trades:  <code>{trades_today}</code> | W/L: <code>{wins}/{losses}</code>\n"  # noqa: E501
            f"WinRate: <code>{win_rate:.0%}</code>\n"
            f"PnL:     {_fmt_pnl(pnl)}\n"
            f"Balance: <code>${balance:,.0f}</code>\n"
            f"Open:    <code>{open_positions}</code>\n"
            f"Regimes: <code>{regime_summary}</code>\n"
            f"{'─' * 30}"
        )
        self.send(message=f"SUMMARY: {_fmt_pnl(pnl)}", level="info", html=html)

    def system_alert(self, message: str):
        html = (
            f"<b>| SYSTEM ALERT |</b>\n"
            f"{'─' * 30}\n"
            f"{message}\n"
            f"Time: {_fmt_time(time.time())}\n"
            f"{'─' * 30}"
        )
        self.send(message=f"ALERT: {message}", level="error", html=html)

    def heartbeat(self, cycle: int, positions: int, regimes: str):
        html = (
            f"<b>| HEARTBEAT |</b>\n"
            f"{'─' * 30}\n"
            f"Cycle:     <code>{cycle}</code>\n"
            f"Positions: <code>{positions}</code>\n"
            f"Regimes:   <code>{regimes}</code>\n"
            f"Time:      {_fmt_time(time.time())}\n"
            f"{'─' * 30}"
        )
        self.send(message=f"HEARTBEAT cycle={cycle}", level="info", html=html)

    def data_loaded(self, symbol: str, bars: int, source: str):
        html = (
            f"<b>| DATA LOADED |</b>\n"
            f"{'─' * 30}\n"
            f"Symbol: <code>{symbol}</code>\n"
            f"Bars:   <code>{bars}</code>\n"
            f"Source: <code>{source}</code>\n"
            f"{'─' * 30}"
        )
        self.send(message=f"DATA {symbol} {bars}b {source}", level="info", html=html)

    def shutdown(self):
        self._running = False
        if self._queue:
            time.sleep(1)

    def trade_blocked(self, symbol: str, reason: str, details: Optional[Dict] = None):
        detail_str = ""
        if details:
            detail_str = "\n" + "\n".join(
                f"  <code>{k}: {v}</code>" for k, v in details.items()
            )
        html = (
            f"<b>| TRADE BLOCKED |</b>\n"
            f"{'─' * 30}\n"
            f"Symbol: <code>{symbol}</code>\n"
            f"Reason: <code>{reason}</code>{detail_str}\n"
            f"{'─' * 30}"
        )
        self.send(message=f"BLOCKED {symbol}: {reason}", level="warning", html=html)
