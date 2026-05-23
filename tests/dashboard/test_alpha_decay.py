"""Tests for real-time alpha decay dashboard."""

from dashboard.smart_dashboard import get_dashboard_html, _compute_strategy_status


class TestAlphaDecayDashboard:
    def test_dashboard_has_alpha_decay_section(self):
        html = get_dashboard_html()
        assert "Strategy Alpha Decay Monitor" in html
        assert 'id="alphaDecayGrid"' in html
        assert "renderAlphaDecay" in html

    def test_strategy_status_green(self):
        report = {
            "good_strat": {
                "alpha_decay": {"sharpe": 0.6},
                "win_rate": 0.55,
                "should_disable": False,
            }
        }
        statuses = _compute_strategy_status(report)
        assert statuses["good_strat"] == "🟢 Active"

    def test_strategy_status_yellow(self):
        report = {
            "warn_strat": {
                "alpha_decay": {"sharpe": 0.3},
                "win_rate": 0.55,
                "should_disable": False,
            }
        }
        statuses = _compute_strategy_status(report)
        assert statuses["warn_strat"] == "🟡 Warning"

    def test_strategy_status_red(self):
        report = {
            "bad_strat": {
                "alpha_decay": {"sharpe": -0.1},
                "win_rate": 0.55,
                "should_disable": False,
            }
        }
        statuses = _compute_strategy_status(report)
        assert statuses["bad_strat"] == "🔴 Disabled"
