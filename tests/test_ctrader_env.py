#!/usr/bin/env python3
"""
Tests for cTrader environment variable handling for RTS - AI FX Trading System.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from api.ctrader_env import cTraderEnvClient


class TestCTraderEnvClient:
    """Test cases for cTraderEnvClient."""

    def test_init_without_env(self):
        """Test initialization without environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            client = cTraderEnvClient()
            assert client.access_token is None
            assert client.client_id == "15217_h8WxunXX70m6O6qsnIx9ZO3GZraTdO0wnLjL3dTKyYG6fkbUca"
            assert client.host == "demo.ctraderapi.com"

    def test_init_with_env(self):
        """Test initialization with environment variables."""
        test_token = "test_access_token_12345"
        test_account = "9999999"

        with patch.dict(os.environ, {
            "CTRADER_ACCESS_TOKEN": test_token,
            "CTRADER_ACCOUNT_ID": test_account
        }):
            client = cTraderEnvClient()
            assert client.access_token == test_token
            assert client.account_id == test_account

    def test_load_env_from_file(self, tmp_path):
        """Test loading environment variables from .env file."""
        env_file = tmp_path / ".env"
        env_file.write_text("""
CTRADER_ACCESS_TOKEN=token_from_file
CTRADER_REFRESH_TOKEN=refresh_from_file
""")

        client = cTraderEnvClient(env_file=str(env_file))
        assert client.access_token == "token_from_file"
        assert client.refresh_token == "refresh_from_file"

    @patch('socket.socket')
    @patch('ssl.create_default_context')
    def test_connect_success(self, mock_ssl_context, mock_socket):
        """Test successful SSL connection."""
        mock_sock = MagicMock()
        mock_ssl_socket = MagicMock()
        mock_socket.return_value = mock_sock
        mock_ssl_context.return_value.wrap_socket.return_value = mock_ssl_socket

        client = cTraderEnvClient()
        result = client.connect()

        assert result is True
        assert client.connection_status == "connected"
        mock_ssl_socket.connect.assert_called_once_with(("demo.ctraderapi.com", 5035))

    @patch('socket.socket')
    @patch('ssl.create_default_context')
    def test_connect_failure(self, mock_ssl_context, mock_socket):
        """Test connection failure."""
        mock_ssl_context.return_value.wrap_socket.side_effect = Exception("SSL Error")

        client = cTraderEnvClient()
        result = client.connect()

        assert result is False
        assert client.connection_status == "error"
        assert "SSL Error" in client.last_error

    def test_get_account_list_no_token(self):
        """Test get_account_list returns None when no token."""
        client = cTraderEnvClient()
        client.access_token = None
        result = client.get_account_list()
        assert result is None
        assert "No access token" in client.last_error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
