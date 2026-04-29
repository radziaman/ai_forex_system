"""
cTrader OAuth2 Automatic Setup for IC Markets
Automatically fetches access token via OAuth2 flow.
"""
import json
import sys
from pathlib import Path
from typing import Optional
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs


# OAuth2 configuration
REDIRECT_URI = "http://localhost:8080"
AUTH_URL_TEMPLATE = "https://connect.ctrader.com/apps/{app_id}/authorize"
TOKEN_URL = "https://connect.ctrader.com/apps/token"


class OAuth2Handler(BaseHTTPRequestHandler):
    """Handle OAuth2 redirect callback."""

    authorization_code = None
    server_error = None

    def do_GET(self):
        """Handle GET request with authorization code."""
        parsed_path = urlparse(self.path)
        query_params = parse_qs(parsed_path.query)

        if "code" in query_params:
            OAuth2Handler.authorization_code = query_params["code"][0]
            response = """
            <html><body>
            <h1>Authorization Successful!</h1>
            <p>You can close this window and return to the setup script.</p>
            </body></html>
            """
        elif "error" in query_params:
            OAuth2Handler.server_error = query_params["error"][0]
            response = f"""
            <html><body>
            <h1>Authorization Failed!</h1>
            <p>Error: {query_params['error'][0]}</p>
            </body></html>
            """
        else:
            response = "<html><body><h1>No code received</h1></body></html>"

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(response.encode())

    def log_message(self, format, *args):
        """Suppress log messages."""
        pass


def get_authorization_code(app_id: str) -> Optional[str]:
    """Start local server and open browser for authorization."""
    # Start local server
    server = HTTPServer(("localhost", 8080), OAuth2Handler)
    server_thread = threading.Thread(target=server.handle_request)
    server_thread.daemon = True
    server_thread.start()

    # Build authorization URL
    auth_url = f"https://connect.ctrader.com/apps/{app_id}/authorize"
    auth_url += f"?client_id={app_id}&redirect_uri={REDIRECT_URI}&response_type=code"

    print(f"\nOpening browser for authorization...")
    print(f"If browser doesn't open, visit:")
    print(f"{auth_url}\n")

    # Try to open browser
    try:
        import webbrowser
        webbrowser.open(auth_url)
    except:
        pass

    # Wait for callback
    print("Waiting for authorization...")
    server_thread.join(timeout=120)

    if OAuth2Handler.server_error:
        print(f"Error: {OAuth2Handler.server_error}")
        return None

    return OAuth2Handler.authorization_code


def exchange_code_for_token(app_id: str, app_secret: str, code: str) -> Optional[str]:
    """Exchange authorization code for access token."""
    try:
        import urllib.request
        import base64

        # Build token request
        credentials = base64.b64encode(f"{app_id}:{app_secret}".encode()).decode()
        data = f"grant_type=authorization_code&code={code}&redirect_uri={REDIRECT_URI}".encode()

        req = urllib.request.Request(TOKEN_URL, data=data)
        req.add_header("Authorization", f"Basic {credentials}")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")

        # Make request
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode())
            return result.get("access_token")

    except Exception as e:
        print(f"Error exchanging code for token: {e}")
        return None


def get_oauth2_credentials() -> dict:
    """Get credentials via OAuth2 flow."""
    print("=" * 60)
    print("cTrader API Setup for IC Markets (OAuth2 Mode)")
    print("=" * 60)

    config = {}

    # App credentials
    print("\nCreate a cTrader app at: https://connect.ctrader.com/")
    print("Make sure to add redirect URI: http://localhost:8080\n")

    config["app_id"] = input("Enter your App ID: ").strip()
    config["app_secret"] = input("Enter your App Secret: ").strip()

    # Get authorization code
    code = get_authorization_code(config["app_id"])
    if not code:
        print("Failed to get authorization code.")
        return None

    # Exchange for access token
    print("\nExchanging code for access token...")
    access_token = exchange_code_for_token(
        config["app_id"], config["app_secret"], code
    )

    if not access_token:
        print("Failed to get access token.")
        return None

    config["access_token"] = access_token
    print("Access token received!")

    # Get account ID
    config["account_id"] = input("\nEnter your cTrader Account ID: ").strip()

    # Settings
    demo_input = input("Use Demo account? (y/n): ").strip().lower()
    config["demo"] = demo_input == 'y'
    config["use_websocket"] = False

    config["ic_markets"] = {
        "name": "IC Markets",
        "server_type": "live" if not config["demo"] else "demo",
        "host": "live.ctrader.com" if not config["demo"] else "demo.ctrader.com",
        "port": 5035,
    }

    return config


def save_config(config: dict, config_path: Path):
    """Save configuration to JSON file."""
    config_path.parent.mkdir(exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nConfig saved to {config_path}")
    print("\nYou can now run:")
    print("  - python scripts/live_trading.py")
    print("  - python scripts/ic_markets_signals.py")


def setup_ctrader_oauth():
    """Interactive OAuth2 setup for cTrader API credentials"""
    try:
        config = get_oauth2_credentials()
        if config:
            config_path = Path("config/ctrader_config.json")
            save_config(config, config_path)
            return True
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        return False
    except Exception as e:
        print(f"\nError during setup: {e}")
        return False


if __name__ == "__main__":
    setup_ctrader_oauth()
