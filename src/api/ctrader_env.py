#!/usr/bin/env python3
"""
cTrader client with environment variable support for RTS - AI FX Trading System.
Loads credentials from .env file or environment variables.

Usage:
    from src.api.ctrader_env import cTraderEnvClient
    client = cTraderEnvClient()
    client.connect()
"""

import os
import ssl
import socket
from google.protobuf.json_format import MessageToDict
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import ProtoMessage
from ctrader_open_api.messages.OpenApiMessages_pb2 import (
    ProtoOAApplicationAuthReq,
    ProtoOAGetAccountListByAccessTokenReq,
    ProtoOAGetAccountListByAccessTokenRes,
)

# Payload type constants (ctrader_open_api uses raw integers)
PAYLOAD_OA_APPLICATION_AUTH_REQ = 2100
PAYLOAD_OA_APPLICATION_AUTH_RES = 2101
PAYLOAD_OA_GET_ACCOUNT_LIST_BY_ACCESS_TOKEN_REQ = 2149
PAYLOAD_OA_GET_ACCOUNT_LIST_BY_ACCESS_TOKEN_RES = 2150


class cTraderEnvClient:
    """cTrader client that loads credentials from environment variables."""

    def __init__(self, env_file=".env"):
        """Initialize with credentials from environment or .env file."""
        self.load_env(env_file)

        self.access_token = os.getenv("CTRADER_ACCESS_TOKEN")
        self.refresh_token = os.getenv("CTRADER_REFRESH_TOKEN")
        self.client_id = os.getenv("CTRADER_CLIENT_ID", "15217_h8WxunXX70m6O6qsnIx9ZO3GZraTdO0wnLjL3dTKyYG6fkbUca")  # noqa: E501
        self.client_secret = os.getenv("CTRADER_CLIENT_SECRET", "Zb8tEW4Axzq0AJqCNS8ubniYzsgp2kxuRkYBRD9XXOcLAj5aOT")  # noqa: E501
        self.account_id = os.getenv("CTRADER_ACCOUNT_ID", "6100830")

        self.host = "demo.ctraderapi.com"
        self.port = 5035
        self.sock = None
        self.last_error = None
        self.connection_status = "disconnected"

    def load_env(self, env_file):
        """Load environment variables from .env file."""
        if os.path.exists(env_file):
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()

    def connect(self):
        """Connect to cTrader API with SSL."""
        try:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

            raw_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock = context.wrap_socket(raw_sock, server_hostname=self.host)

            self.sock.connect((self.host, self.port))
            self.connection_status = "connected"
            self.last_error = None
            return True
        except Exception as e:
            self.last_error = f"Connection failed: {e}"
            self.connection_status = "error"
            return False

    def send_message(self, payload_type, message):
        """Send a protobuf message with length prefix."""
        payload = message.SerializeToString()
        header = len(payload).to_bytes(4, byteorder='big')
        proto_msg = ProtoMessage(payloadType=payload_type, payload=payload)
        full_msg = header + proto_msg.SerializeToString()

        try:
            self.sock.sendall(full_msg)
            return True
        except Exception as e:
            self.last_error = f"Send failed: {e}"
            return False

    def receive_message(self):
        """Receive a protobuf message with length prefix."""
        try:
            header = self.sock.recv(4)
            if not header:
                return None, None, "Connection closed"

            msg_len = int.from_bytes(header, byteorder='big')
            data = b""
            while len(data) < msg_len:
                chunk = self.sock.recv(msg_len - len(data))
                if not chunk:
                    return None, None, "Connection closed"
                data += chunk

            proto_msg = ProtoMessage()
            proto_msg.ParseFromString(data)
            return proto_msg.payloadType, proto_msg.payload, None
        except Exception as e:
            return None, None, f"Receive failed: {e}"

    def authenticate_application(self):
        """Authenticate with application credentials."""
        auth_req = ProtoOAApplicationAuthReq()
        auth_req.clientId = self.client_id
        auth_req.clientSecret = self.client_secret

        if not self.send_message(PAYLOAD_OA_APPLICATION_AUTH_REQ, auth_req):
            return False

        payload_type, payload, error = self.receive_message()
        if error:
            self.last_error = error
            return False

        if payload_type == PAYLOAD_OA_APPLICATION_AUTH_RES:
            return True
        else:
            self.last_error = f"Unexpected response: {payload_type}"
            return False

    def get_account_list(self):
        """Get account list using access token."""
        if not self.access_token:
            self.last_error = "No access token available"
            return None

        req = ProtoOAGetAccountListByAccessTokenReq()
        req.accessToken = self.access_token

        if not self.send_message(PAYLOAD_OA_GET_ACCOUNT_LIST_BY_ACCESS_TOKEN_REQ, req):
            return None

        payload_type, payload, error = self.receive_message()
        if error:
            self.last_error = error
            return None

        if payload_type == PAYLOAD_OA_GET_ACCOUNT_LIST_BY_ACCESS_TOKEN_RES:
            res = ProtoOAGetAccountListByAccessTokenRes()
            res.ParseFromString(payload)
            return MessageToDict(res)
        else:
            self.last_error = f"Unexpected response type: {payload_type}"
            return None

    def close(self):
        """Close the connection."""
        if self.sock:
            self.sock.close()
            self.connection_status = "disconnected"


if __name__ == "__main__":
    print("RTS - AI FX Trading System: cTrader Environment Client")
    print("=" * 60)

    client = cTraderEnvClient()

    if not client.access_token:
        print("ERROR: No CTRADER_ACCESS_TOKEN found!")
        print("Run: python scripts/oauth_helper.py")
        exit(1)

    print(f"Token loaded: {client.access_token[:20]}...")

    if client.connect():
        print("✓ Connected to cTrader")

        if client.authenticate_application():
            print("✓ Application authenticated")

            accounts = client.get_account_list()
            if accounts:
                print(f"✓ Account list received: {accounts}")
            else:
                print(f"✗ Failed to get accounts: {client.last_error}")
        else:
            print(f"✗ Application auth failed: {client.last_error}")

        client.close()
    else:
        print(f"✗ Connection failed: {client.last_error}")
