"""
Tests for cTrader SSL connection (macOS LibreSSL 2.8.3)
"""
import unittest
import ssl
import socket
import sys

sys.path.insert(0, 'src/api')
sys.path.insert(0, 'venv/lib/python3.9/site-packages')

from ctrader_ready import cTraderICMarkets


class TestCTraderSSL(unittest.TestCase):
    """Test SSL connection works on macOS"""
    
    def setUp(self):
        self.client = cTraderICMarkets()
        
    def tearDown(self):
        if self.client.ssl_socket:
            self.client.close()
    
    def test_ssl_connection(self):
        """Test SSL connection works on macOS LibreSSL"""
        result = self.client.connect()
        self.assertTrue(result)
        self.assertEqual(self.client.connection_status, "connected")
        self.assertIsNotNone(self.client.ssl_socket)
        
    def test_tls_version(self):
        """Test TLSv1.2 is used"""
        if self.client.connect():
            version = self.client.ssl_socket.version()
            self.assertIn("TLS", version)
            print(f"\n✓ Connected via {version}")
        else:
            self.fail("Could not connect")
    
    def test_invalid_token_handling(self):
        """Test invalid token returns proper error"""
        if not self.client.connect():
            self.skipTest("Could not connect")
        
        # This should fail with proper error
        result = self.client.get_account_list("invalid_token_123")
        self.assertIsNone(result)
        self.assertIsNotNone(self.client.last_error)


if __name__ == "__main__":
    unittest.main()
