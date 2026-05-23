"""Tests for pre-trade kill switch activation and attribution publishing."""

from unittest.mock import MagicMock

import pytest

from agentic.core.world_state import get_world_state
from risk.enhanced_manager import EnhancedRiskManager, RiskParameters


class TestEnhancedManagerKillSwitch:
    def test_pre_trade_sets_kill_switch_world_state(self):
        params = RiskParameters(max_drawdown=0.05)
        mgr = EnhancedRiskManager(params, initial_balance=100_000.0)
        approved, reason = mgr.pre_trade_checks(
            balance=100_000.0,
            equity=90_000.0,
            margin=0.0,
            current_pnl=-10_000.0,
        )
        assert approved is False
        ws = get_world_state()
        assert ws.get("risk.kill_switch") is True
        assert ws.get("risk.drawdown") == pytest.approx(0.10, abs=1e-9)

    def test_release_kill_switch_releases_when_drawdown_low(self):
        params = RiskParameters(max_drawdown=0.05)
        mgr = EnhancedRiskManager(params, initial_balance=100_000.0)
        # Trigger kill switch first
        mgr.pre_trade_checks(
            balance=100_000.0,
            equity=90_000.0,
            margin=0.0,
            current_pnl=-10_000.0,
        )
        assert mgr.kill_switch_triggered is True
        # Now recover equity to 1% drawdown
        mgr._base.peak_balance = 100_000.0
        released = mgr.release_kill_switch(equity=99_000.0)
        assert released is True
        assert mgr.kill_switch_triggered is False
        ws = get_world_state()
        assert ws.get("risk.kill_switch") is False

    def test_release_kill_switch_keeps_active_when_drawdown_high(self):
        params = RiskParameters()
        mgr = EnhancedRiskManager(params, initial_balance=100_000.0)
        mgr._base.peak_balance = 100_000.0
        mgr._base.kill_switch_triggered = True
        released = mgr.release_kill_switch(equity=95_000.0)  # 5% drawdown
        assert released is False
        assert mgr.kill_switch_triggered is True


class TestSignalAgentKillSwitch:
    @pytest.fixture(autouse=True)
    def reset_world_state(self):
        ws = get_world_state()
        ws.delete("risk.kill_switch")
        ws.delete("risk.drawdown")
        ws.delete("strategy_attribution")

    def test_blocks_signals_when_kill_switch_active(self):
        from agentic.agents.signal_agent import SignalAgent

        agent = SignalAgent(config={})
        agent._kill_switch_since = None
        agent._kill_switch_alerted = False
        ws = get_world_state()
        ws.set("risk.kill_switch", True)

        assert agent._check_kill_switch() is True
        assert agent._kill_switch_since is not None

    def test_publishes_attribution_every_5_trades(self):
        from agentic.agents.signal_agent import SignalAgent

        agent = SignalAgent(config={})
        agent._attribution_engine = MagicMock()
        agent._attribution_engine.get_report.return_value = {"strat1": {"trades": 5}}
        agent.set_world = MagicMock()
        agent._attribution_trade_counter = 0

        # 4 trades — no publish
        for _ in range(4):
            agent._maybe_publish_attribution()
        agent.set_world.assert_not_called()

        # 5th trade — publish
        agent._maybe_publish_attribution()
        agent.set_world.assert_called_once_with(
            "strategy_attribution", {"strat1": {"trades": 5}}
        )
