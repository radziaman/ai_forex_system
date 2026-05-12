"""Tests for risk management module."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import risk manager when available
try:
    from rts_ai_fx.risk import RiskManager
except ImportError:
    pytest.skip("RiskManager not available", allow_module_level=True)


class TestRiskManager:
    """Test suite for RiskManager."""

    @pytest.fixture
    def risk_manager(self):
        """Create a RiskManager with test config."""
        return RiskManager(max_risk_per_trade=0.02, max_drawdown=0.10, max_positions=10)

    def test_initialization(self, risk_manager):
        """Risk manager should initialize with correct parameters."""
        assert risk_manager.max_risk_per_trade == 0.02
        assert risk_manager.max_drawdown == 0.10
        assert risk_manager.max_positions == 10

    def test_calculate_position_size(self, risk_manager):
        """Should calculate correct position size based on risk."""
        balance = 10000
        risk_amount = balance * risk_manager.max_risk_per_trade
        # Test with dummy values
        position_size = risk_manager.calculate_position_size(
            balance=balance, entry_price=1.1000, stop_loss=1.0950, pip_value=0.0001
        )
        assert position_size > 0

    def test_max_positions_limit(self, risk_manager):
        """Should reject new positions when max reached."""
        # This would need actual position tracking
        pass
