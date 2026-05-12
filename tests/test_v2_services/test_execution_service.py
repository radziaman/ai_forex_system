"""Tests for ExecutionService."""

from unittest.mock import MagicMock, AsyncMock, patch
import asyncio
from services.execution_service import ExecutionService
from services import Signal, SignalDirection, Regime, TradeDecision, ExecutionResult


class TestExecutionService:
    def setup_method(self):
        self.config = MagicMock()
        self.config.trading.sl_atr_multiplier = 2.0
        self.config.trading.tp_atr_multiplier = 4.0

    @patch("services.execution_service.create_execution_provider")
    @patch("services.execution_service.ExecutionEngine")
    def test_execute_returns_result(self, mock_engine_cls, mock_provider):
        mock_ctrader = MagicMock()
        mock_ctrader.start = AsyncMock(return_value=True)
        mock_ctrader.is_connected = MagicMock(return_value=True)
        mock_ctrader.disconnect = AsyncMock()
        mock_ctrader.subscribe_depth = AsyncMock(return_value=True)
        mock_ctrader.raw = mock_ctrader
        mock_provider.return_value = (mock_ctrader, None)

        mock_engine = MagicMock()
        mock_engine.open_position = AsyncMock(
            return_value=MagicMock(
                entry_price=1.1205,
                position_id=42,
            )
        )
        mock_engine_cls.return_value = mock_engine

        mock_pipeline = MagicMock()
        mock_pipeline.data_manager = MagicMock()

        mock_secrets = MagicMock()
        mock_secrets.is_demo = True
        svc = ExecutionService(self.config, mock_secrets, mock_pipeline)
        asyncio.run(svc.start())

        signal = Signal(
            symbol="EURUSD",
            direction=SignalDirection.BUY,
            confidence=0.75,
            regime=Regime.RANGING,
            price=1.12,
        )
        decision = TradeDecision(
            signal=signal, volume=1000, sl_price=1.11, tp_price=1.14
        )

        result = asyncio.run(svc.execute(decision))
        assert result is not None
        assert result.success
        assert result.position_id == "42"
        assert result.filled_price == 1.1205

        asyncio.run(svc.stop())

    @patch("services.execution_service.create_execution_provider")
    def test_execute_returns_failure_on_order_error(self, mock_provider):
        mock_ctrader = MagicMock()
        mock_ctrader.start = AsyncMock(return_value=True)
        mock_ctrader.is_connected = MagicMock(return_value=True)
        mock_ctrader.disconnect = AsyncMock()
        mock_ctrader.subscribe_depth = AsyncMock(return_value=True)
        mock_ctrader.raw = mock_ctrader
        mock_provider.return_value = (mock_ctrader, None)

        mock_engine = MagicMock()
        mock_engine.open_position = AsyncMock(return_value=None)
        mock_pipeline = MagicMock()
        mock_pipeline.data_manager = MagicMock()

        mock_secrets = MagicMock()
        mock_secrets.is_demo = True
        with patch(
            "services.execution_service.ExecutionEngine", return_value=mock_engine
        ):
            svc = ExecutionService(self.config, mock_secrets, mock_pipeline)
            asyncio.run(svc.start())

            signal = Signal(
                symbol="EURUSD",
                direction=SignalDirection.BUY,
                confidence=0.75,
                regime=Regime.RANGING,
                price=1.12,
            )
            decision = TradeDecision(
                signal=signal, volume=1000, sl_price=1.11, tp_price=1.14
            )
            result = asyncio.run(svc.execute(decision))
            assert result is not None
            assert not result.success

            asyncio.run(svc.stop())
