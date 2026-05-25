"""
Monitoring Agent — best-practice notifications with structured formatting,
alert aggregation, rate limiting, escalation, and daily summaries.
"""

from __future__ import annotations
import time
from typing import Dict, List, Any
from collections import defaultdict, deque
from datetime import datetime, timezone
from loguru import logger

from agentic.core.base_agent import BaseAgent
from agentic.core.agent_message import (
    AgentMessage,
    MessageType,
)
from agentic.core.agent_consciousness import ConsciousnessLevel
from infrastructure.logging import SystemHeartbeat

# ── Alert severity levels ──
ALERT_CRITICAL = "CRITICAL"
ALERT_HIGH = "HIGH"
ALERT_MEDIUM = "MEDIUM"
ALERT_LOW = "LOW"
ALERT_DEBUG = "DEBUG"

# ── Emoji / icon per level ──
ICON = {
    ALERT_CRITICAL: "\u26a0\ufe0f",
    ALERT_HIGH: "\u26a1",
    ALERT_MEDIUM: "\u2139\ufe0f",
    ALERT_LOW: "\u2705",
    ALERT_DEBUG: "\U0001f50d",
}

# ── Alert categories for rate limiting ──
CATEGORY_TRADE = "trade"
CATEGORY_RISK = "risk"
CATEGORY_SYSTEM = "system"
CATEGORY_PERFORMANCE = "performance"
CATEGORY_HEARTBEAT = "heartbeat"


class MonitoringAgent(BaseAgent):
    def __init__(self, secrets=None):
        super().__init__(
            name="monitoring_agent",
            role="System Monitor & Notifier",
            purpose="Best-practice alerting: structured, aggregated, rate-limited, escalated",  # noqa: E501
            domain="monitoring",
            capabilities={
                "telegram_notifications",
                "alert_aggregation",
                "rate_limiting",
                "structured_formatting",
                "daily_summaries",
                "uptime_tracking",
            },
            tick_interval=15.0,
            consciousness_level=ConsciousnessLevel.REFLECTIVE,
        )
        self.secrets = secrets
        self.notifier = None
        self._event_log: deque = deque(maxlen=1000)
        self._last_telegram_ok: float = 0.0
        self._telegram_health: str = "unknown"

        # ── Rate limiting: category -> [(timestamp, message)] ──
        self._rate_limit_buckets: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        self._rate_limits = {
            CATEGORY_TRADE: {"max_per_hour": 20},
            CATEGORY_RISK: {"max_per_hour": 10},
            CATEGORY_SYSTEM: {"max_per_hour": 5},
            CATEGORY_PERFORMANCE: {"max_per_hour": 2},
            CATEGORY_HEARTBEAT: {"max_per_hour": 1},
        }

        # ── Alert aggregation: dedup identical alerts within window ──
        self._recent_alerts: Dict[str, Dict] = {}  # key -> {count, last_ts, first_ts}
        self._aggregation_window = 300  # 5 min

        # ── Daily summary ──
        self._last_summary_date: int = 0
        self._daily_trades: List[Dict] = []
        self._daily_alerts: List[str] = []

        # Structured system heartbeat (5-min interval, JSON format)
        self._heartbeat = SystemHeartbeat(interval=300, logger=logger)

        self.subscribe(MessageType.AGENT_HEARTBEAT)
        self.subscribe(MessageType.RISK_ALERT)
        self.subscribe(MessageType.CIRCUIT_BREAKER)
        self.subscribe(MessageType.SYSTEM_STATE_CHANGE)
        self.subscribe(MessageType.SIGNAL_GENERATED)
        self.subscribe(MessageType.POSITION_OPENED)
        self.subscribe(MessageType.POSITION_CLOSED)
        self.subscribe(MessageType.EXECUTION_RESULT)
        self.subscribe(MessageType.DIAGNOSTIC_REQUEST)

    # ── Start ──────────────────────────────────────────────────

    async def _on_start(self):
        self.consciousness.current_intention = "initializing notification system"
        if self.secrets:
            try:
                from notifications.telegram import TelegramNotifier

                self.notifier = TelegramNotifier(
                    bot_token=getattr(self.secrets, "telegram_bot_token", ""),
                    chat_id=getattr(self.secrets, "telegram_chat_id", ""),
                    send_trade_alerts=True,
                    send_daily_summary=True,
                    send_risk_alerts=True,
                )
                self._safe_telegram(
                    "\U0001f916 Agentic FX System Elite — online\n"
                    f"Mode: {self.get_world('config.mode', 'paper')}",
                    ALERT_LOW,
                    CATEGORY_SYSTEM,
                )
            except Exception as e:
                self.log_state(f"Telegram not available: {e}", "warning")
                self._telegram_health = "unconfigured"
        self.set_world("monitoring.telegram_health", self._telegram_health)

    # ── Cycle ───────────────────────────────────────────────────

    async def perceive(self) -> Dict[str, Any]:
        health = self.get_world("master.system_health", 1.0)
        perf = self.get_world("performance.stats", {})
        regime = self.get_world("regime.current", "?")
        now = datetime.now(timezone.utc)
        new_day = now.day != self._last_summary_date

        return {
            "health": health,
            "performance": perf,
            "regime": regime,
            "new_day": new_day,
            "hour": now.hour,
        }

    async def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        actions = []

        # ── Daily summary at 00:05 UTC ──
        if perception.get("new_day") and perception.get("hour") == 0:
            actions.append("daily_summary")

        # ── Health degradation alert ──
        if perception.get("health", 1.0) < 0.5:
            actions.append("health_critical")
        elif perception.get("health", 1.0) < 0.8:
            actions.append("health_degraded")

        # ── Negative Sharpe ──
        perf = perception.get("performance", {})
        sharpe = perf.get("sharpe", 0)
        if sharpe < -1.0:
            actions.append(f"sharpe_critical:{sharpe:.2f}")
        elif sharpe < 0.5 and perf.get("total_trades", 0) > 20:
            actions.append(f"sharpe_low:{sharpe:.2f}")

        return {"actions": actions}

    async def act(self, decision: Dict[str, Any]):
        for action in decision.get("actions", []):
            if action == "daily_summary":
                self._send_daily_summary()
            elif action == "health_critical":
                self._send_alert("HEALTH CRITICAL", ALERT_CRITICAL, CATEGORY_SYSTEM)
            elif action == "health_degraded":
                self._send_alert("Health degraded", ALERT_HIGH, CATEGORY_SYSTEM)
            elif action.startswith("sharpe_critical"):
                s = action.split(":")[1]
                self._send_alert(
                    f"Sharpe ratio {s} — review strategy",
                    ALERT_CRITICAL,
                    CATEGORY_PERFORMANCE,
                )
            elif action.startswith("sharpe_low"):
                s = action.split(":")[1]
                self._send_alert(
                    f"Sharpe ratio {s} below 0.5", ALERT_HIGH, CATEGORY_PERFORMANCE
                )

        # Structured system heartbeat (5-min interval, JSON format)
        self._heartbeat.emit(self.get_world)

    async def reflect(self, outcome: Dict[str, Any]):
        pass

    # ── Message handling ────────────────────────────────────────

    async def on_message(self, message: AgentMessage):
        self._event_log.append(
            {
                "t": time.time(),
                "type": message.msg_type.name,
                "src": message.source_agent,
            }
        )

        payload = message.payload if isinstance(message.payload, dict) else {}

        if message.msg_type == MessageType.POSITION_OPENED:
            self._daily_trades.append(payload)
            self._send_alert(
                self._fmt_trade_opened(payload), ALERT_MEDIUM, CATEGORY_TRADE
            )

        elif message.msg_type == MessageType.POSITION_CLOSED:
            pnl = payload.get("pnl", 0)
            reason = payload.get("reason", "closed")
            direction = payload.get("direction", "")
            symbol = payload.get("symbol", "?")
            entries = payload.get("entry", 0)
            self._daily_trades.append(payload)
            level = ALERT_MEDIUM if pnl >= 0 else ALERT_HIGH
            self._send_alert(
                self._fmt_trade_closed(symbol, direction, pnl, reason, entries),
                level,
                CATEGORY_TRADE,
            )

        elif message.msg_type == MessageType.RISK_ALERT:
            self._daily_alerts.append(str(payload))
            alert_type = payload.get("type", "?")
            reason = payload.get("reason", "?")
            self._send_alert(
                self._fmt_risk_alert(alert_type, reason, payload),
                ALERT_CRITICAL,
                CATEGORY_RISK,
            )

        elif message.msg_type == MessageType.DIAGNOSTIC_REQUEST:
            await self.send(
                MessageType.DIAGNOSTIC_RESULT,
                payload={
                    "agent": self.name,
                    "events_logged": len(self._event_log),
                    "telegram_health": self._telegram_health,
                    "uptime": time.time() - self.consciousness.started_at,
                },
                target=message.source_agent,
            )

    # ── Alert sending with aggregation + rate limiting ──────────

    def _send_alert(self, text: str, level: str, category: str):
        """Send alert with aggregation, rate limiting, and structured formatting."""
        if not self.notifier:
            return

        # Aggregation: dedup identical alerts within window
        dedup_key = f"{category}:{text[:80]}"
        now = time.time()
        if dedup_key in self._recent_alerts:
            entry = self._recent_alerts[dedup_key]
            entry["count"] += 1
            entry["last_ts"] = now
            if now - entry["first_ts"] < self._aggregation_window:
                # Suppress: within aggregation window, send only every N
                if entry["count"] % 10 != 0:
                    return
                text = f"{text} (x{entry['count']})"
        else:
            self._recent_alerts[dedup_key] = {
                "count": 1,
                "first_ts": now,
                "last_ts": now,
            }

        # Rate limiting: max per hour per category
        bucket = self._rate_limit_buckets[category]
        bucket.append(now)
        limit = self._rate_limits.get(category, {}).get("max_per_hour", 100)
        recent = sum(1 for t in bucket if now - t < 3600)
        if recent > limit:
            if recent == limit + 1:
                logger.warning(f"[{self.name}] Rate limit hit for {category}")
            return

        # Structured formatting
        _ = ICON.get(level, "")
        emoji_level = {
            ALERT_CRITICAL: "\u26a0\ufe0f",
            ALERT_HIGH: "\u26a1",
            ALERT_MEDIUM: "\u2139\ufe0f",
            ALERT_LOW: "\u2705",
        }.get(level, "")
        header = f"{emoji_level} [{level}]"
        regime = self.get_world("regime.current", "?")
        positions = self.get_world("account.open_positions", 0)
        perf = self.get_world("performance.stats", {})
        n_trades = perf.get("total_trades", 0)
        if n_trades > 0:
            sharpe = perf.get("sharpe", 0)
            pnl = perf.get("total_pnl", 0)
            footer = f"\n\U0001f4c8 {positions} pos | Sharpe {sharpe:.2f} | PnL ${pnl:+.2f} | {regime}"  # noqa: E501
        else:
            footer = f"\n\U0001f4c8 {positions} pos | Sharpe N/A | PnL N/A | {regime}"

        full_text = f"{header}\n{text}{footer}"
        self._safe_telegram(full_text, level, category)

    # ── Structured formatters ────────────────────────────────────

    def _fmt_trade_opened(self, p: Dict) -> str:
        session = p.get("session", "?")
        strategy = p.get("strategy", "?")
        sl = float(p.get("sl_price", 0))
        tp = float(p.get("tp_price", 0))
        entry = float(p.get("entry", 0))
        bal = self.get_world("account.balance", 0)
        lines = [
            f"\U0001f4c8 *{p.get('direction', '?')} {p.get('symbol', '?')}*",
            f"Volume: {p.get('volume', 0):.2f} lots | Strategy: {strategy}",
            f"Session: {session} | Confidence: {p.get('confidence', 0):.0%}",
        ]
        if entry:
            lines.append(f"Entry: {entry:.5f} | SL: {sl:.5f} | TP: {tp:.5f}")
        if bal:
            lines.append(f"Balance: ${bal:.2f}")
        return "\n".join(lines)

    def _fmt_trade_closed(
        self, symbol: str, direction: str, pnl: float, reason: str, entry: float
    ) -> str:
        emoji = "\U0001f4b0" if pnl >= 0 else "\U0001f4a5"
        bal = self.get_world("account.balance", 0)
        exit_price = self.get_world(f"data.price.{symbol}", 0)
        lines = [
            f"{emoji} *{direction} {symbol}* closed",
            f"PnL: *${pnl:+.2f}* | Balance: ${bal:.2f}",
            f"Reason: {reason}",
        ]
        if entry:
            lines.append(f"Entry: {entry:.5f} | Exit: {exit_price:.5f}")
        # Add strategy performance context if available
        if pnl > 0:
            lines.append("\U0001f525 Winning trade")
        else:
            lines.append("\U0001f4aa Holding — next trade will recover")
        return "\n".join(lines)

    def _fmt_risk_alert(self, alert_type: str, reason: str, payload: Dict) -> str:
        text = f"\u26a0\ufe0f *Risk Alert: {alert_type}*\n{reason}"
        agent = payload.get("agent", "")
        if agent:
            text += f"\nSource: {agent}"
        consec = payload.get("consecutive_errors", 0)
        if consec:
            text += f"\nConsecutive errors: {consec}"
        return text

    def _fmt_daily_summary(self) -> str:
        perf = self.get_world("performance.stats", {})
        regime = self.get_world("regime.current", "?")
        balance = self.get_world("account.balance", 0)
        uptime = time.time() - self.consciousness.started_at
        uptime_h = uptime / 3600
        t = len(self._daily_trades)
        wins = sum(1 for d in self._daily_trades if d.get("pnl", 0) > 0)
        pnl = sum(d.get("pnl", 0) for d in self._daily_trades)

        # Get per-symbol strategy performance
        tracker_data = self.get_world("signal.tradeable_symbols", {})
        active_symbols = []
        blocked_symbols = []
        if isinstance(tracker_data, dict):
            for sym, info in tracker_data.items():
                if info.get("tradeable"):
                    active_symbols.append(sym)
                else:
                    blocked_symbols.append(sym)

        lines = [
            "\U0001f4ca *Daily Summary*",
            f"Uptime: {uptime_h:.1f}h | Balance: ${balance:.2f}",
            f"Regime: {regime} | Trades: {t} | Wins: {wins} ({wins/max(t,1):.0%})",
            f"PnL: *${pnl:+.2f}* | Sharpe: {perf.get('sharpe', 0):.2f}",
        ]
        if active_symbols:
            lines.append(f"Active: {', '.join(active_symbols)}")
        if blocked_symbols:
            lines.append(f"Learning: {', '.join(blocked_symbols)}")
        return "\n".join(lines)

    def _send_daily_summary(self):
        self._last_summary_date = datetime.now(timezone.utc).day
        summary = self._fmt_daily_summary()
        self._safe_telegram(summary, ALERT_LOW, CATEGORY_PERFORMANCE)
        self._daily_trades.clear()
        self._daily_alerts.clear()
        self.log_state("Daily summary sent")

    # ── Telegram delivery ──────────────────────────────────────

    def _safe_telegram(
        self,
        text: str,
        level: str = "info",
        category: str = "system",
        max_retries: int = 2,
    ):
        if not self.notifier:
            return
        for attempt in range(max_retries):
            try:
                self.notifier.send(text, level=level.lower())
                self._last_telegram_ok = time.time()
                self._telegram_health = "ok"
                self._telegram_retries = 0
                return
            except Exception as e:
                self._telegram_retries += 1
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    self._telegram_health = "error"
                    logger.warning(f"[{self.name}] Telegram failed: {e}")
        self.set_world("monitoring.telegram_health", self._telegram_health)

    def _build_status(self) -> Dict:
        return {
            "timestamp": time.time(),
            "uptime": time.time() - self.consciousness.started_at,
            "health": self.get_world("master.system_health", 1.0),
            "performance": self.get_world("performance.stats", {}),
            "regime": self.get_world("regime.current", "unknown"),
            "telegram": self._telegram_health,
            "alerts_today": len(self._daily_alerts),
            "trades_today": len(self._daily_trades),
        }
